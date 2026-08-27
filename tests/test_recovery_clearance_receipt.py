from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_clearance", ROOT / "scripts" / "create_recovery_clearance_receipt.py"
)
assert SPEC and SPEC.loader
clearance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clearance)


def fixture_result() -> dict:
    domain = {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "mismatch": None}
    durable = {
        "gate_id": clearance.GATE_ID,
        "source_sha256": clearance.SOURCE_SHA256,
        "step": 300,
        "checkpoint_sha256": clearance.FINAL_CHECKPOINT_SHA256,
    }
    return {
        "schema": clearance.GATE_SCHEMA,
        "gate_id": clearance.GATE_ID,
        "passed": False,
        "source_sha256": clearance.SOURCE_SHA256,
        "preregistration_sha256": clearance.PREREGISTRATION_SHA256,
        "confirmatory_data_used": False,
        "confirmatory_seed_used": False,
        "shared_lineage_checkpoint_sha256": clearance.SHARED_PARENT_SHA256,
        "reference_checkpoint": {
            "path": "/reference/ckpt_iter_300.pt",
            "sha256": clearance.FINAL_CHECKPOINT_SHA256,
            "size_bytes": 256_007_203,
        },
        "recovered_checkpoint": {
            "path": "/recovered/ckpt_iter_300.pt",
            "sha256": clearance.FINAL_CHECKPOINT_SHA256,
            "size_bytes": 256_007_203,
        },
        "checks": {
            "final_step": True,
            "durable_progress": {"ok": True, "errors": [], "cadence_seconds": 60},
            **{name: dict(domain) for name in clearance.EXACT_DOMAINS},
            "checkpoint_lineage": {
                "ok": True,
                "shared_parent_sha256": clearance.SHARED_PARENT_SHA256,
                "recovered_final_sha256": clearance.FINAL_CHECKPOINT_SHA256,
            },
            "data_position": {
                "ok": False,
                "batches_per_epoch": 8,
                "final_epoch": 37,
                "final_cursor": 4,
                "resume_fast_forward_step": 150,
                "reference_fast_forward_observed": False,
                "recovered_fast_forward_observed": False,
            },
        },
        "durable_reference_final": dict(durable),
        "durable_recovered_final": dict(durable),
    }


def frozen_payload(monkeypatch: pytest.MonkeyPatch, document: dict) -> bytes:
    payload = (json.dumps(document, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(clearance, "RESULT_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_audit_preserves_false_result_and_issues_narrow_go(monkeypatch) -> None:
    payload = frozen_payload(monkeypatch, fixture_result())
    receipt = clearance.audit_result(payload, {"path": "/immutable/result.json"})

    assert receipt["original_result"]["passed"] is False
    assert receipt["disposition"] == {
        "engineering_recovery_equivalence": "GO",
        "repeat_t4_required_for_this_engineering_question": False,
        "original_schema_v2_gate_passed": False,
        "scientific_or_confirmatory_result": False,
    }
    assert receipt["exact_replay_evidence"]["final_checkpoint_byte_identical"] is True
    assert set(receipt["exact_replay_evidence"]["domains"]) == set(clearance.EXACT_DOMAINS)
    assert receipt["sole_failed_original_check"]["classification"] == (
        "observation-harness false negative"
    )
    assert receipt["caveats_and_follow_up"]["future_latch_hardening_required"] is True


@pytest.mark.parametrize("domain", clearance.EXACT_DOMAINS)
def test_audit_refuses_any_nonexact_domain(monkeypatch, domain: str) -> None:
    result = fixture_result()
    result["checks"][domain]["max_abs"] = 1e-30
    payload = frozen_payload(monkeypatch, result)
    with pytest.raises(clearance.ClearanceError, match="exact zero"):
        clearance.audit_result(payload, {"path": "/immutable/result.json"})


def test_audit_refuses_a_second_failed_check(monkeypatch) -> None:
    result = fixture_result()
    result["checks"]["weights"]["ok"] = False
    payload = frozen_payload(monkeypatch, result)
    with pytest.raises(clearance.ClearanceError, match="weights check did not pass"):
        clearance.audit_result(payload, {"path": "/immutable/result.json"})


def test_audit_refuses_an_unreviewed_extra_check(monkeypatch) -> None:
    result = fixture_result()
    result["checks"]["new_check"] = {"ok": False}
    payload = frozen_payload(monkeypatch, result)
    with pytest.raises(clearance.ClearanceError, match="unexpected check set"):
        clearance.audit_result(payload, {"path": "/immutable/result.json"})


def test_audit_refuses_changed_checkpoint_bytes(monkeypatch) -> None:
    result = fixture_result()
    result["recovered_checkpoint"]["sha256"] = "0" * 64
    payload = frozen_payload(monkeypatch, result)
    with pytest.raises(clearance.ClearanceError, match="recovered final checkpoint hash"):
        clearance.audit_result(payload, {"path": "/immutable/result.json"})


def test_audit_refuses_rewriting_original_pass_state(monkeypatch) -> None:
    result = fixture_result()
    result["passed"] = True
    payload = frozen_payload(monkeypatch, result)
    with pytest.raises(clearance.ClearanceError, match="must remain passed:false"):
        clearance.audit_result(payload, {"path": "/immutable/result.json"})


def test_write_once_is_content_addressed_idempotent_and_read_only(tmp_path) -> None:
    receipt = {"schema": clearance.SCHEMA, "answer": "GO"}
    first = clearance.write_once(tmp_path, receipt)
    second = clearance.write_once(tmp_path, receipt)

    assert first == second
    payload = first.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest in first.name
    assert first.with_suffix(".json.sha256").read_text() == "%s  %s\n" % (
        digest, first.name
    )
    assert first.stat().st_mode & 0o222 == 0
