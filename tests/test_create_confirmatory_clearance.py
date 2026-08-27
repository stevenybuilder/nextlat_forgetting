"""Regressions for atomic, preregistration-bound confirmatory clearance."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def _load():
    scripts = str(PROJECT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = PROJECT / "scripts" / "create_confirmatory_clearance.py"
    spec = importlib.util.spec_from_file_location("confirmatory_clearance_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n")


def _evidence(root: Path, module, spec: dict) -> tuple[str, Path, Path]:
    """Build a fully synthetic PASS chain in an isolated test directory."""
    del spec
    for relative in module.driver.CONFIRMATORY_PROTOCOL_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {relative}\n")
    validator = root / "scripts" / "validate_preregistration.py"
    validator.parent.mkdir(parents=True, exist_ok=True)
    validator.write_text(
        "import json, pathlib\n"
        "def validate(evidence, *, amendment, spec):\n"
        "    path = pathlib.Path(evidence).parent / 'synthetic-recomputed-preregistration.json'\n"
        "    return json.loads(path.read_text())\n"
    )
    report = root / "docs" / "INDEPENDENT_CONFIRMATORY_REVIEW.md"
    report.write_text("VERDICT: PASS\n")
    fixture_manifest = root / "manifests" / "synthetic.json"
    fixture_manifest.parent.mkdir(parents=True, exist_ok=True)
    fixture_manifest.write_text('{"fixture":true}\n')
    (fixture_manifest.parent / "manifest_inventory.sha256").write_text(
        f"{module.driver.sha256_file(fixture_manifest)}  manifests/synthetic.json\n"
    )

    # The source archive is created before any source-bound receipts. All evidence
    # generated below lives under .agent_state, which package() intentionally excludes.
    archive = Path(module.driver.package(str(root)))
    source_sha = module.driver.sha256_file(archive)
    state = root / ".agent_state"
    input_inventory = fixture_manifest.parent / "manifest_inventory.sha256"
    input_sha = module.driver.sha256_file(input_inventory)
    input_prefix = f"lurestar/input_bundles/{input_sha}"
    _write_json(state / "input-bundle-upload.json", {
        "schema": module.driver.INPUT_BUNDLE_UPLOAD_SCHEMA,
        "status": "COMPLETE",
        "bucket": module.driver.BUCKET,
        "bundle_prefix": input_prefix,
        "input_bundle_sha256": input_sha,
        "object_count": 1,
        "objects": [{
            "local_path": "manifests/synthetic.json",
            "name": f"{input_prefix}/manifests/synthetic.json",
            "generation": "101",
            "size_bytes": fixture_manifest.stat().st_size,
            "sha256": module.driver.sha256_file(fixture_manifest),
        }],
        "commit": {
            "local_path": "manifests/manifest_inventory.sha256",
            "name": f"{input_prefix}/manifests/manifest_inventory.sha256",
            "generation": "102",
            "size_bytes": input_inventory.stat().st_size,
            "sha256": input_sha,
        },
    })
    test_receipt_path = state / "confirmatory-test-receipt.json"
    review_receipt_path = state / "confirmatory-review-receipt.json"
    _write_json(test_receipt_path, {
        "schema": module.driver.FULL_TEST_SUITE_SCHEMA,
        "outcome": "PASS",
        "exit_code": 0,
        "tests_passed": 800,
        "source_sha256": source_sha,
    })
    module.record_review(root, source_sha, str(report), "independent-agent", "PASS")
    evidence_path = state / "preregistration-evidence.json"
    evidence = {
        "schema": module.driver.PREREGISTRATION_EVIDENCE_SCHEMA,
        "gates": {str(gate): {} for gate in range(1, 12)},
    }
    evidence["gates"]["1"] = {
        "artifacts": [{
            "role": "source_snapshot",
            "path": ".agent_state/project.tar.gz",
            "sha256": source_sha,
            "schema": "binary/source-snapshot",
        }],
    }
    evidence["gates"]["11"] = {
        "artifacts": [
            {
                "role": "full_suite_receipt",
                "path": ".agent_state/confirmatory-test-receipt.json",
                "sha256": module.driver.sha256_file(test_receipt_path),
                "schema": module.driver.FULL_TEST_SUITE_SCHEMA,
            },
            {
                "role": "independent_review_receipt",
                "path": ".agent_state/confirmatory-review-receipt.json",
                "sha256": module.driver.sha256_file(review_receipt_path),
                "schema": module.driver.INDEPENDENT_SCIENTIFIC_REVIEW_SCHEMA,
            },
        ],
    }
    _write_json(evidence_path, evidence)
    receipt_path = state / "preregistration-freeze-receipt.json"
    amendment = root / "docs" / "PREREGISTRATION_AMENDMENT_2026-08-24.md"
    specification = root / "nextlat_v4_predictive_geometry_spec.md"
    receipt_document = {
        "schema": module.driver.PREREGISTRATION_FREEZE_SCHEMA,
        "status": "PASS",
        "all_eleven_gates_pass": True,
        "authority": {
            "amendment": {
                "path": "docs/PREREGISTRATION_AMENDMENT_2026-08-24.md",
                "sha256": module.driver.sha256_file(amendment),
            },
            "spec": {
                "path": "nextlat_v4_predictive_geometry_spec.md",
                "sha256": module.driver.sha256_file(specification),
            },
            "evidence": {
                "path": ".agent_state/preregistration-evidence.json",
                "sha256": module.driver.sha256_file(evidence_path),
            },
            "validator": {
                "path": "scripts/validate_preregistration.py",
                "sha256": module.driver.sha256_file(validator),
            },
        },
        "missing_gate_blocks": [],
        "extra_gate_blocks": [],
        "global_issues": [],
        "gates": [
            {"gate": gate, "status": "PASS", "issues": []}
            for gate in range(1, 12)
        ],
        "meaning": "pre-compute design frozen; no scientific outcome evaluated",
    }
    _write_json(receipt_path, receipt_document)
    _write_json(state / "synthetic-recomputed-preregistration.json", receipt_document)

    return source_sha, archive, receipt_path


def test_issue_writes_only_a_validated_go_receipt(tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, receipt_path = _evidence(tmp_path, module, spec)

    output = module.issue(tmp_path, spec, source_sha)

    clearance = json.loads(output.read_text())
    assert clearance["authorization"] == "GO"
    assert clearance["preregistration"] == {
        "receipt_path": ".agent_state/preregistration-freeze-receipt.json",
        "receipt_sha256": module.driver.sha256_file(receipt_path),
        "receipt_schema": module.driver.PREREGISTRATION_FREEZE_SCHEMA,
        "evidence_sha256": module.driver.sha256_file(
            tmp_path / ".agent_state" / "preregistration-evidence.json"),
        "validator_sha256": module.driver.sha256_file(
            tmp_path / "scripts" / "validate_preregistration.py"),
        "source_archive_sha256": source_sha,
    }
    assert not output.with_name(output.name + ".candidate").exists()


@pytest.mark.parametrize("malicious,message", [
    ({"gpu": "a100", "run_matrix_args": ["--phase", "adapt"]}, "excludes H3"),
    ({"gpu": "a100", "run_matrix_args": ["--phase", "base"],
      "regime": "easy"}, "unknown fields"),
    ({"runner": "hmm", "runner_phase": "train", "family": False,
      "gpu": "a100"}, "complete frozen family"),
])
def test_issue_refuses_semantically_malicious_job_specs_even_with_fresh_source_hash(
        tmp_path, malicious, message) -> None:
    module = _load()
    safe = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, _ = _evidence(tmp_path, module, safe)

    with pytest.raises(SystemExit, match=message):
        module.issue(tmp_path, malicious, source_sha)

    assert not (tmp_path / ".agent_state" / "confirmatory-clearance.json").exists()


def test_structurally_plausible_stored_pass_must_equal_recomputed_validator_output(
        tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, receipt_path = _evidence(tmp_path, module, spec)
    stored = json.loads(receipt_path.read_text())
    stored["meaning"] = "fabricated but structurally plausible PASS"
    _write_json(receipt_path, stored)

    with pytest.raises(SystemExit, match="does not equal hardened validator output"):
        module.issue(tmp_path, spec, source_sha)

    assert not (tmp_path / ".agent_state" / "confirmatory-clearance.json").exists()


def test_gate_11_receipts_use_the_same_schemas_as_clearance(tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    _evidence(tmp_path, module, spec)

    test_receipt = json.loads(
        (tmp_path / ".agent_state" / "confirmatory-test-receipt.json").read_text())
    review_receipt = json.loads(
        (tmp_path / ".agent_state" / "confirmatory-review-receipt.json").read_text())
    assert test_receipt["schema"] == "nextlat_forgetting/full_test_suite_receipt/1"
    assert review_receipt["schema"] == (
        "nextlat_forgetting/independent_scientific_review/1")


@pytest.mark.parametrize("mutation,message", [
    ("missing", "missing or invalid"),
    ("block", "not an all-eleven PASS"),
    ("schema", "schema mismatch"),
    ("stale_archive", "stale for the exact source archive"),
])
def test_issue_refuses_missing_block_stale_or_wrong_schema_preregistration(
        tmp_path, mutation, message) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, archive, receipt_path = _evidence(tmp_path, module, spec)
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "stale_archive":
        archive.write_bytes(archive.read_bytes() + b"tampered")
    else:
        receipt = json.loads(receipt_path.read_text())
        if mutation == "block":
            receipt["status"] = "BLOCK"
            receipt["all_eleven_gates_pass"] = False
        else:
            receipt["schema"] = "wrong/schema"
        _write_json(receipt_path, receipt)

    with pytest.raises(SystemExit, match=message):
        module.issue(tmp_path, spec, source_sha)

    output = tmp_path / ".agent_state" / "confirmatory-clearance.json"
    assert not output.exists()
    assert not output.with_name(output.name + ".candidate").exists()


def test_issue_refuses_preregistration_bound_to_another_source(tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, _ = _evidence(tmp_path, module, spec)

    with pytest.raises(SystemExit, match="stale for the exact source archive"):
        module.issue(tmp_path, spec, "0" * 64)

    assert source_sha != "0" * 64


def test_clearance_refuses_preregistration_evidence_changed_after_issue(tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, _ = _evidence(tmp_path, module, spec)
    module.issue(tmp_path, spec, source_sha)
    evidence = tmp_path / ".agent_state" / "preregistration-evidence.json"
    evidence.write_text(evidence.read_text() + "\n")

    with pytest.raises(SystemExit, match="evidence authority hash mismatch"):
        module.driver.validate_confirmatory_clearance(tmp_path, spec, source_sha)


def test_failed_review_issue_leaves_no_go_or_candidate(tmp_path) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, _ = _evidence(tmp_path, module, spec)
    review = tmp_path / ".agent_state" / "confirmatory-review-receipt.json"
    document = json.loads(review.read_text())
    document["source_sha256"] = "c" * 64
    _write_json(review, document)

    with pytest.raises(SystemExit, match="confirmatory review receipt is not a passing source-bound receipt"):
        module.issue(tmp_path, spec, source_sha)

    output = tmp_path / ".agent_state" / "confirmatory-clearance.json"
    assert not output.exists()
    assert not output.with_name(output.name + ".candidate").exists()


@pytest.mark.parametrize("verdict", ["BLOCK", "FAIL"])
def test_nonpassing_independent_review_explicitly_refuses_clearance(
        tmp_path, verdict) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, _ = _evidence(tmp_path, module, spec)
    review = tmp_path / ".agent_state" / "confirmatory-review-receipt.json"
    document = json.loads(review.read_text())
    document["verdict"] = verdict
    _write_json(review, document)

    with pytest.raises(
            SystemExit,
            match="confirmatory review receipt is not a passing source-bound receipt"):
        module.issue(tmp_path, spec, source_sha)

    output = tmp_path / ".agent_state" / "confirmatory-clearance.json"
    assert not output.exists()
    assert not output.with_name(output.name + ".candidate").exists()


def test_blocked_preregistration_precedes_every_external_colab_action(
        tmp_path, monkeypatch) -> None:
    module = _load()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha, _, receipt_path = _evidence(tmp_path, module, spec)
    module.issue(tmp_path, spec, source_sha)
    _write_json(tmp_path / ".agent_state" / "job_spec.json", spec)
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "BLOCK"
    receipt["all_eleven_gates_pass"] = False
    _write_json(receipt_path, receipt)

    external_calls: list[str] = []

    def forbidden_shell(command, **_kwargs):
        external_calls.append(command)
        raise AssertionError("external command ran before preregistration validation")

    def forbidden_status(*_args, **_kwargs):
        external_calls.append("colab status")
        raise AssertionError("Colab status ran before preregistration validation")

    monkeypatch.setattr(module.driver, "sh", forbidden_shell)
    monkeypatch.setattr(module.driver, "colab_status_pair", forbidden_status)

    with pytest.raises(SystemExit, match="not an all-eleven PASS"):
        module.driver._owned_loop(str(tmp_path))

    assert external_calls == []


def test_d41_issue_preserves_predecessor_freeze_and_binds_truthful_continuation(
        tmp_path, monkeypatch) -> None:
    module = _load()
    spec = {
        "runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
        "predecessor_source_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
        "recovery_job_ids": list(module.d41.EXACT_TEN_JOB_IDS),
        "recovery_receipt_sha256": "a" * 64,
    }
    source_sha = "b" * 64
    for relative in module.driver.CONFIRMATORY_PROTOCOL_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"successor {relative}\n")
    state = tmp_path / ".agent_state"
    _write_json(state / "confirmatory-test-receipt.json", {"outcome": "PASS"})
    _write_json(state / "confirmatory-review-receipt.json", {"verdict": "PASS"})
    predecessor_preregistration = {
        "receipt_path": ".agent_state/preregistration-freeze-receipt.json",
        "receipt_sha256": "1" * 64,
        "receipt_schema": module.driver.PREREGISTRATION_FREEZE_SCHEMA,
        "evidence_sha256": "2" * 64,
        "validator_sha256": "3" * 64,
        "source_archive_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
    }
    _write_json(tmp_path / module.d41.PREDECESSOR_REFERENCE_PATH, {
        "issued_clearance": {"preregistration": predecessor_preregistration},
    })
    continuation = {
        "schema": module.d41.CONTINUATION_SCHEMA,
        "confirmatory_lifecycle": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
        },
        "successor_assurance": {
            "full_test_receipt": {"sha256": "4" * 64},
            "independent_review_receipt": {"sha256": "5" * 64},
        },
    }
    monkeypatch.setattr(module.driver, "validate_confirmatory_job_spec", lambda _spec: None)
    monkeypatch.setattr(module.driver, "validate_input_bundle_receipt", lambda _root: {
        "receipt_schema": module.driver.INPUT_BUNDLE_UPLOAD_SCHEMA,
    })
    validations: list[Path] = []

    def validate_candidate(_root, _spec, _source, candidate=None):
        candidate = Path(candidate or tmp_path / ".agent_state/confirmatory-clearance.json")
        stored = json.loads(candidate.read_text())
        recomputed = module.d41.validate_d41_continuation_bundle(
            tmp_path, spec, source_sha)
        assert stored["continuation"] == recomputed
        validations.append(candidate)
        return stored

    monkeypatch.setattr(module.driver, "validate_confirmatory_clearance", validate_candidate)
    monkeypatch.setattr(
        module.driver, "validate_preregistration_pass_receipt",
        lambda *_args, **_kwargs: pytest.fail("successor must not be relabelled pre-compute"),
    )
    monkeypatch.setattr(
        module.d41, "validate_d41_continuation_bundle",
        lambda _root, _spec, _source: continuation,
    )

    output = module.issue(tmp_path, spec, source_sha)
    clearance = json.loads(output.read_text())
    assert clearance["source_sha256"] == source_sha
    assert clearance["preregistration"] == predecessor_preregistration
    assert clearance["continuation"]["schema"] == module.d41.CONTINUATION_SCHEMA
    assert set(clearance["continuation"]["successor_assurance"]) == {
        "full_test_receipt", "independent_review_receipt",
    }
    assert clearance["continuation"]["confirmatory_lifecycle"] == {
        "compute_started": True,
        "scientific_evaluations_inspected": False,
    }
    assert len(validations) == 2


def test_d43_issue_recomputes_explicit_gate_and_binds_atomic_partition(
        tmp_path, monkeypatch) -> None:
    module = _load()
    receipt_sha = "d" * 64
    spec = {
        "runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
        "predecessor_source_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
        "recovery_job_ids": list(module.driver.D41_RECOVERY_JOB_IDS),
        "recovery_receipt_sha256": "a" * 64,
        "continuation_gate": module.driver.D43_CONTINUATION_GATE,
        "continuation_gate_schema": module.driver.D43_CONTINUATION_SCHEMA,
        "continuation_receipt_sha256": receipt_sha,
    }
    source_sha = "b" * 64
    for relative in module.driver.CONFIRMATORY_PROTOCOL_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"successor {relative}\n")
    state = tmp_path / ".agent_state"
    _write_json(state / "confirmatory-test-receipt.json", {"outcome": "PASS"})
    _write_json(state / "confirmatory-review-receipt.json", {"verdict": "PASS"})
    _write_json(state / module.driver.D43_RECEIPT_PATH.split("/", 1)[1], {
        "schema": module.driver.D43_CONTINUATION_SCHEMA,
    })
    predecessor_preregistration = {
        "receipt_path": ".agent_state/preregistration-freeze-receipt.json",
        "receipt_sha256": "1" * 64,
        "receipt_schema": module.driver.PREREGISTRATION_FREEZE_SCHEMA,
        "evidence_sha256": "2" * 64,
        "validator_sha256": "3" * 64,
        "source_archive_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
    }
    _write_json(tmp_path / module.d41.PREDECESSOR_REFERENCE_PATH, {
        "issued_clearance": {"preregistration": predecessor_preregistration},
    })
    completed = list(module.driver.D41_RECOVERY_JOB_IDS)
    all_jobs = [
        f"{model}-seed{seed}-hmm-{regime}"
        for regime in (
            "persistent_moderate", "fast_mixing_moderate",
            "persistent_high_aliasing",
        )
        for model in ("gpt", "nextlat")
        for seed in range(1234, 1239)
    ]
    continuation = {
        "schema": module.driver.D43_LAUNCH_BINDING_SCHEMA,
        "gate": module.driver.D43_CONTINUATION_GATE,
        "gate_schema": module.driver.D43_CONTINUATION_SCHEMA,
        "receipt_path": module.driver.D43_RECEIPT_PATH,
        "receipt_sha256": receipt_sha,
        "source_sha256": source_sha,
        "predecessor_source_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
        "completed_job_ids": completed,
        "pending_job_ids": [job for job in all_jobs if job not in completed],
        "scientific_evaluations_started": False,
        "scientific_metrics_inspected": False,
    }
    monkeypatch.setattr(module.driver, "validate_confirmatory_job_spec", lambda _spec: None)
    monkeypatch.setattr(module.driver, "validate_input_bundle_receipt", lambda _root: {
        "receipt_schema": module.driver.INPUT_BUNDLE_UPLOAD_SCHEMA,
    })
    d43_calls = []

    def validate_d43(root, requested, requested_source):
        d43_calls.append((root, requested, requested_source))
        return continuation

    monkeypatch.setattr(module.driver, "validate_d43_continuation_bundle", validate_d43)
    monkeypatch.setattr(
        module.d41, "validate_d41_continuation_bundle",
        lambda *_args, **_kwargs: pytest.fail("D43 must not downgrade to D41"),
    )
    validations: list[Path] = []

    def validate_candidate(_root, _spec, _source, candidate=None):
        candidate = Path(candidate or state / "confirmatory-clearance.json")
        stored = json.loads(candidate.read_text())
        assert stored["continuation"] == continuation
        validations.append(candidate)
        return stored

    monkeypatch.setattr(module.driver, "validate_confirmatory_clearance", validate_candidate)

    output = module.issue(tmp_path, spec, source_sha)
    clearance = json.loads(output.read_text())
    assert clearance["continuation"] == continuation
    assert clearance["preregistration"] == predecessor_preregistration
    assert len(clearance["continuation"]["completed_job_ids"]) == 10
    assert len(clearance["continuation"]["pending_job_ids"]) == 20
    assert clearance["continuation"]["scientific_evaluations_started"] is False
    assert clearance["continuation"]["scientific_metrics_inspected"] is False
    assert d43_calls == [(tmp_path, spec, source_sha)]
    assert len(validations) == 2


def test_issue_refuses_d41_fallback_when_d43_receipt_exists(
        tmp_path, monkeypatch) -> None:
    module = _load()
    spec = {
        "runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
        "predecessor_source_sha256": module.d41.PREDECESSOR_SOURCE_SHA256,
        "recovery_job_ids": list(module.driver.D41_RECOVERY_JOB_IDS),
        "recovery_receipt_sha256": "a" * 64,
    }
    receipt = tmp_path / module.driver.D43_RECEIPT_PATH
    _write_json(receipt, {"schema": module.driver.D43_CONTINUATION_SCHEMA})
    monkeypatch.setattr(module.driver, "validate_confirmatory_job_spec", lambda _spec: None)
    monkeypatch.setattr(
        module.d41, "validate_d41_continuation_bundle",
        lambda *_args, **_kwargs: pytest.fail("D41 must not be consulted"),
    )

    with pytest.raises(SystemExit, match="refusing fallback to D41"):
        module.issue(tmp_path, spec, "b" * 64)

    assert not (tmp_path / ".agent_state" / "confirmatory-clearance.json").exists()
