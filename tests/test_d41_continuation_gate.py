"""D41 must continue a started confirmatory lifecycle without outcome inspection."""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import tarfile

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d41_continuation_gate_under_test", ROOT / "scripts" / "d41_continuation_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _object(uri: str, *, sha256: str = "a" * 64, generation: str = "123") -> dict:
    return {
        "uri": uri,
        "generation": generation,
        "sha256": sha256,
        "size_bytes": 7,
    }


def _exact_ten_document() -> dict:
    runtime_patch_projection = {"schema": "nextlat_forgetting/runtime_patch/1"}
    runtime = {
        "predecessor_runtime_evidence": {
            "session_id": gate.PREDECESSOR_SESSION_ID,
            "source_archive_object": _object(
                "gs://bucket/source.tar.gz",
                sha256=gate.PREDECESSOR_SOURCE_SHA256,
                generation=gate.PREDECESSOR_SOURCE_GENERATION,
            ),
            "observed_preflight": dict(gate.EXPECTED_RUNTIME),
            "observation_basis": {"kind": "host_controller_verified_preflight_console"},
            "per_run_runtime_patch_receipt_retained": False,
        },
        "expected_successor_contract": dict(gate.EXPECTED_RUNTIME),
        "runtime_patch": {
            "source_path": "scripts/runtime_bootstrap.py",
            "source_sha256": "b" * 64,
            "historical_reference_receipt_object": _object("gs://bucket/patch.json"),
            "historical_reference_receipt_sha256": "a" * 64,
            "historical_reference_receipt_projection": runtime_patch_projection,
            "historical_reference_receipt_projection_sha256": gate.canonical_sha256(
                runtime_patch_projection
            ),
            "expected_receipt_projection": runtime_patch_projection,
            "expected_receipt_projection_sha256": gate.canonical_sha256(
                runtime_patch_projection
            ),
            "successor_must_emit_and_verify_own_receipt": True,
        },
    }
    jobs = []
    for job_id in gate.EXACT_TEN_JOB_IDS:
        model, seed = job_id.split("-seed", 1)[0], int(job_id.split("-seed", 1)[1][:4])
        jobs.append({
            "job_id": job_id,
            "model": model,
            "seed": seed,
            "regime": gate.REGIME,
            "target_step": gate.TARGET_STEP,
            "predecessor_source_sha256": gate.PREDECESSOR_SOURCE_SHA256,
            "state_object": _object(f"gs://bucket/{job_id}/state.json"),
            "checkpoint_object": _object(f"gs://bucket/{job_id}/checkpoint.pt"),
            "sidecar_object": _object(f"gs://bucket/{job_id}/checkpoint.pt.meta.json"),
            "checkpoint_semantics": {
                "filename_step": gate.TARGET_STEP,
                "sidecar_step": gate.TARGET_STEP,
                "sidecar_step_field": "step",
                "sidecar_path": f"/content/{job_id}/checkpoint.pt.meta.json",
                "sidecar_run_id": job_id,
                "payload_training_steps": gate.TARGET_STEP,
            },
            "verification": {
                "state_trained_exact_target": True,
                "checkpoint_bytes_sha256_verified": True,
                "sidecar_bytes_sha256_verified": True,
                "sidecar_binds_checkpoint": True,
                "source_identity_verified": True,
                "payload_training_steps_verified": True,
            },
        })
    return {
        "schema": gate.EXACT_TEN_SCHEMA,
        "status": "PASS",
        "predecessor_source_sha256": gate.PREDECESSOR_SOURCE_SHA256,
        "target_step": gate.TARGET_STEP,
        "required_job_ids": list(gate.EXACT_TEN_JOB_IDS),
        "jobs": jobs,
        "checkpoint_semantic_contract": {
            "verifier_path": "src/lurestar/durable_checkpoint.py",
            "verifier_symbol": "exact_sidecar_step",
            "verifier_source_sha256": "d" * 64,
            "target_step": gate.TARGET_STEP,
            "predecessor_sidecar_schema": "legacy_step_only",
            "checkpoint_payload_step_field": "training_steps",
            "all_checkpoint_payloads_deserialized_on_host": True,
            "conflicting_dual_sidecar_step_fields_rejected": True,
            "migration_scope": "receipt_bound_source_migrated_only",
            "successor_sidecar_contract": "canonical_training_steps_required",
            "successor_retains_legacy_step": True,
            "successor_canonicalization_requires_from_to_hash_provenance": True,
            "current_source_retry_sidecar_contract": "training_steps_required",
        },
        "recovery_plan_and_receipts": [
            {"path": path.as_posix(), "sha256": "c" * 64, "schema": f"schema/{index}"}
            for index, path in enumerate(gate.RECOVERY_EVIDENCE_PATHS)
        ],
        "runtime_equivalence": runtime,
        "scientific_metrics_inspected": False,
        "confirmatory_lifecycle": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
        },
    }


def test_exact_ten_receipt_is_ordered_complete_and_lifecycle_truthful() -> None:
    document = _exact_ten_document()
    gate.validate_exact_ten_document(document)
    assert len(document["jobs"]) == 10
    assert document["confirmatory_lifecycle"] == {
        "compute_started": True,
        "scientific_evaluations_inspected": False,
    }


def test_checkpoint_payload_verifier_requires_literal_training_steps() -> None:
    torch = pytest.importorskip("torch")
    valid = io.BytesIO()
    torch.save({"training_steps": gate.TARGET_STEP, "model": {}}, valid)
    assert gate._checkpoint_payload_training_steps(
        valid.getvalue(), "gpt-seed1234-hmm-persistent_moderate"
    ) == gate.TARGET_STEP

    for invalid_state in (
        {"step": gate.TARGET_STEP},
        {"training_steps": True},
        {"training_steps": "3000"},
        [gate.TARGET_STEP],
    ):
        payload = io.BytesIO()
        torch.save(invalid_state, payload)
        with pytest.raises(gate.D41GateError, match="literal integer training_steps|not a mapping"):
            gate._checkpoint_payload_training_steps(payload.getvalue(), "invalid-job")


def test_gate_loads_the_exact_shared_sidecar_parser() -> None:
    parser = gate._exact_sidecar_step_parser(ROOT)
    assert parser({"step": gate.TARGET_STEP}) == gate.TARGET_STEP
    assert parser({"step": gate.TARGET_STEP, "training_steps": gate.TARGET_STEP}) == gate.TARGET_STEP
    with pytest.raises(ValueError, match="conflict"):
        parser({"step": gate.TARGET_STEP, "training_steps": gate.TARGET_STEP - 1})


@pytest.mark.parametrize("mutation,message", [
    ("drop", "incomplete or out of order"),
    ("step", "job identity mismatch"),
    ("checkpoint", "checkpoint_object identity mismatch"),
    ("inspection", "outcome-blind"),
    ("fresh", "lifecycle is false"),
    ("environment", "runtime contract mismatch"),
    ("patch", "runtime-patch source/receipt contract mismatch"),
    ("payload", "verification is nonpassing"),
    ("semantic", "checkpoint semantics mismatch"),
    ("contract", "semantic contract is missing or stale"),
])
def test_exact_ten_receipt_fails_closed(mutation: str, message: str) -> None:
    document = _exact_ten_document()
    if mutation == "drop":
        document["jobs"].pop()
    elif mutation == "step":
        document["jobs"][0]["target_step"] = 2999
    elif mutation == "checkpoint":
        document["jobs"][0]["checkpoint_object"]["generation"] = "latest"
    elif mutation == "inspection":
        document["scientific_metrics_inspected"] = True
    elif mutation == "fresh":
        document["confirmatory_lifecycle"]["compute_started"] = False
    elif mutation == "environment":
        document["runtime_equivalence"]["expected_successor_contract"]["cuda_version"] = "changed"
    elif mutation == "payload":
        document["jobs"][0]["verification"]["payload_training_steps_verified"] = False
    elif mutation == "semantic":
        document["jobs"][0]["checkpoint_semantics"]["payload_training_steps"] = 2999
    elif mutation == "contract":
        document["checkpoint_semantic_contract"]["migration_scope"] = "all_sources"
    else:
        document["runtime_equivalence"]["runtime_patch"]["source_sha256"] = "bad"
    with pytest.raises(gate.D41GateError, match=message):
        gate.validate_exact_ten_document(document)


def _runner_source(command_value: int = 1) -> bytes:
    assignments = "\n".join(
        f"{name} = {index!r}"
        for index, name in enumerate(sorted(
            value for value in gate._RUNNER_SCIENTIFIC_SYMBOLS if "." not in value
            and value.isupper()
        ))
    )
    functions = "\n".join(
        f"def {name}():\n    return {command_value}\n"
        for name in sorted(
            value for value in gate._RUNNER_SCIENTIFIC_SYMBOLS
            if "." not in value and not value.isupper()
        )
    )
    return (assignments + "\n" + functions +
            f"\nclass HMMFabricLauncher:\n    def command(self):\n        return {command_value}\n").encode()


def _archive(path: pathlib.Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))


def test_source_equivalence_allows_only_operational_delta(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gate, "_mixed_module_projection", lambda _before, _after: {})
    before = {
        "configs/gpt_hmm.yaml": b"frozen\n",
        "scripts/train_hmm.py": b"frozen\n",
        "scripts/runtime_bootstrap.py": b"frozen\n",
        "scripts/evaluate_hmm_checkpoints.py": b"frozen\n",
        "scripts/aggregate_hmm_family.py": b"frozen\n",
        "scripts/materialize_hmm_family.py": b"frozen\n",
        "scripts/run_hmm_matrix.py": _runner_source(),
        "scripts/colab_train_loop.py": b"old operational\n",
    }
    after = {**before, "scripts/colab_train_loop.py": b"new operational\n"}
    predecessor = tmp_path / "predecessor.tar.gz"
    successor = tmp_path / "successor.tar.gz"
    _archive(predecessor, before)
    _archive(successor, after)
    monkeypatch.setattr(gate, "PREDECESSOR_SOURCE_SHA256", gate.sha256_file(predecessor))

    output = gate.create_source_equivalence(tmp_path, predecessor, successor)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "PASS"
    assert receipt["changed_paths"] == ["scripts/colab_train_loop.py"]
    assert all(receipt["equivalence_claims"].values())


@pytest.mark.parametrize("path,payload,message", [
    ("configs/gpt_hmm.yaml", b"changed\n", "exact-byte scientific files"),
    ("scripts/run_hmm_matrix.py", None, "scientific runner symbols"),
])
def test_source_equivalence_rejects_scientific_or_optimizer_delta(
    tmp_path, monkeypatch, path: str, payload: bytes | None, message: str
) -> None:
    monkeypatch.setattr(gate, "_mixed_module_projection", lambda _before, _after: {})
    files = {
        "configs/gpt_hmm.yaml": b"frozen\n",
        "scripts/train_hmm.py": b"frozen\n",
        "scripts/runtime_bootstrap.py": b"frozen\n",
        "scripts/evaluate_hmm_checkpoints.py": b"frozen\n",
        "scripts/aggregate_hmm_family.py": b"frozen\n",
        "scripts/materialize_hmm_family.py": b"frozen\n",
        "scripts/run_hmm_matrix.py": _runner_source(),
    }
    changed = dict(files)
    changed[path] = _runner_source(2) if payload is None else payload
    predecessor = tmp_path / "predecessor.tar.gz"
    successor = tmp_path / "successor.tar.gz"
    _archive(predecessor, files)
    _archive(successor, changed)
    monkeypatch.setattr(gate, "PREDECESSOR_SOURCE_SHA256", gate.sha256_file(predecessor))

    with pytest.raises(gate.D41GateError, match=message):
        gate.create_source_equivalence(tmp_path, predecessor, successor)


def test_mixed_module_projection_allows_only_exact_reviewed_operational_symbols(
        monkeypatch) -> None:
    path = "scripts/run_hmm_matrix.py"
    before = {path: b"FROZEN = 1\ndef scientific():\n    return 1\ndef operational():\n    return 1\n"}
    after = {path: b"FROZEN = 1\ndef scientific():\n    return 1\ndef operational():\n    return 2\n"}
    monkeypatch.setattr(gate, "_ALLOWED_OPERATIONAL_AST_DELTA", {
        path: frozenset({"operational"}),
    })
    monkeypatch.setattr(gate, "_HMM_CLI_SCIENTIFIC_OPTIONS", frozenset())
    projection = gate._mixed_module_projection(before, after)
    assert projection[path]["reviewed_operational_delta_symbols"] == ["operational"]

    changed_science = {
        path: b"FROZEN = 1\ndef scientific():\n    return 3\ndef operational():\n    return 2\n"
    }
    with pytest.raises(gate.D41GateError, match=r"unexpected=\['scientific'\]"):
        gate._mixed_module_projection(before, changed_science)

    changed_import = {path: b"import os\n" + after[path]}
    with pytest.raises(gate.D41GateError, match="non-allowlisted full-module AST"):
        gate._mixed_module_projection(before, changed_import)

    changed_expression = {path: after[path] + b"\nprint('semantic side effect')\n"}
    with pytest.raises(gate.D41GateError, match="non-allowlisted full-module AST"):
        gate._mixed_module_projection(before, changed_expression)


def test_hmm_cli_projection_detects_precision_default_change(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_HMM_CLI_SCIENTIFIC_OPTIONS", frozenset({"--precision"}))
    before = b"def main():\n    parser.add_argument('--precision', default='bf16-mixed')\n"
    after = b"def main():\n    parser.add_argument('--precision', default='16-mixed')\n"
    assert gate._hmm_cli_projection(before) != gate._hmm_cli_projection(after)
