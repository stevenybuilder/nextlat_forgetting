"""Adversarial regressions for the outcome-blind D43 measurement amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pathlib
import tarfile
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d43_gate_under_test", ROOT / "scripts/d43_measurement_amendment_gate.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tar(path: pathlib.Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _runner(version: int, *, training_version: int = 1) -> bytes:
    evaluation = "\n".join(
        f"def {name}():\n    return {version}\n"
        for name in sorted(gate.RUN_HMM_ALLOWED_CHANGED_SYMBOLS)
    )
    imports = ("from run_matrix import DONE\n" if version == 1 else
               "from run_matrix import DONE, COMPLETION_SUMMARY\n")
    return (imports + f"HMM_TRAIN_UPDATES = 3000\n"
            f"def build_hmm_matrix():\n    return {training_version}\n"
            f"class HMMFabricLauncher:\n    def command(self):\n        return {training_version}\n"
            + evaluation).encode()


def _archive_pair(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    critical = {
        "configs/gpt_hmm.yaml": b"updates: 3000\n",
        "manifests/hmm_family.json": b"{}\n",
        "scripts/train_hmm.py": b"TRAINING_STEPS = 3000\n",
        "scripts/run_matrix.py": b"def train(): pass\n",
        "scripts/runtime_bootstrap.py": b"PINNED = True\n",
        "scripts/launch_train.sh": b"python train.py\n",
        "src/lurestar/adaptation.py": b"def objective(): pass\n",
        "src/lurestar/durable_checkpoint.py": b"def save(): pass\n",
        "upstream/models/model_base.py": b"class Model: pass\n",
        "scripts/run_hmm_matrix.py": _runner(1),
    }
    baseline = dict(critical)
    # Existing measurement authorities have old bytes; the decision/gate/materializer are new.
    new_only = {
        "docs/DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md",
        "docs/DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md",
        "scripts/d43_measurement_amendment_gate.py",
        "scripts/materialize_lurestar_evaluation.py",
        "tests/test_d43_measurement_amendment_gate.py",
    }
    for name in gate.REQUIRED_MEASUREMENT_CHANGES - new_only - {"scripts/run_hmm_matrix.py"}:
        baseline[name] = f"old:{name}\n".encode()
    successor = dict(baseline)
    for name in gate.REQUIRED_MEASUREMENT_CHANGES:
        successor[name] = (_runner(2) if name == "scripts/run_hmm_matrix.py"
                           else f"d43:{name}\n".encode())
    base_path, successor_path = tmp_path / "d41.tar.gz", tmp_path / "d43.tar.gz"
    _tar(base_path, baseline)
    _tar(successor_path, successor)
    monkeypatch.setattr(gate, "D41_SUCCESSOR_SHA256", gate.sha256_file(base_path))
    exact = {}
    for name in sorted(set(baseline) | set(successor)):
        if baseline.get(name) != successor.get(name):
            exact[name] = {
                "before_sha256": _sha(baseline[name]) if name in baseline else None,
                "after_sha256": _sha(successor[name]) if name in successor else None,
            }
    declaration = {"successor_archive_sha256": gate.sha256_file(successor_path),
                   "exact_changed_files": exact}
    return baseline, successor, base_path, successor_path, declaration


def test_measurement_delta_accepts_exact_enumerated_evaluation_only_change(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _before, _after, base, successor, declaration = _archive_pair(tmp_path, monkeypatch)
    result = gate._validate_measurement_delta(base, successor, declaration)
    assert set(result["changed_paths"]) == set(declaration["exact_changed_files"])
    assert result["hmm_orchestrator_changed_symbols"] == sorted(
        gate.RUN_HMM_ALLOWED_CHANGED_SYMBOLS)
    assert result["immutable_training_surface_file_count"] >= 8


def test_training_objective_config_model_manifest_or_update_mutation_is_refused(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before, after, base, _successor, declaration = _archive_pair(tmp_path, monkeypatch)
    for target, payload in (
        ("configs/gpt_hmm.yaml", b"updates: 3001\n"),
        ("scripts/train_hmm.py", b"TRAINING_STEPS = 3001\n"),
        ("upstream/models/model_base.py", b"class ChangedModel: pass\n"),
        ("manifests/hmm_family.json", b'{"changed":true}\n'),
    ):
        mutated = dict(after)
        mutated[target] = payload
        path = tmp_path / (target.replace("/", "-") + ".tar.gz")
        _tar(path, mutated)
        changed_declaration = dict(declaration, successor_archive_sha256=gate.sha256_file(path))
        with pytest.raises(gate.D43GateError, match="unenumerated source change"):
            gate._validate_measurement_delta(base, path, changed_declaration)


def test_extra_mutation_of_an_allowed_path_is_refused_even_when_path_is_allowlisted(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _before, after, base, _successor, declaration = _archive_pair(tmp_path, monkeypatch)
    mutated = dict(after)
    mutated["src/lurestar/evaluate.py"] += b"# undeclared second edit\n"
    path = tmp_path / "mutated-allowed.tar.gz"
    _tar(path, mutated)
    declaration = dict(declaration, successor_archive_sha256=gate.sha256_file(path))
    with pytest.raises(gate.D43GateError, match="declared bytes are stale"):
        gate._validate_measurement_delta(base, path, declaration)


def test_mixed_orchestrator_training_change_is_refused_even_if_archive_is_rehashed(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before, after, base, _successor, _declaration = _archive_pair(tmp_path, monkeypatch)
    after["scripts/run_hmm_matrix.py"] = _runner(2, training_version=2)
    path = tmp_path / "runner-training-change.tar.gz"
    _tar(path, after)
    exact = {
        name: {"before_sha256": _sha(before[name]) if name in before else None,
               "after_sha256": _sha(after[name]) if name in after else None}
        for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)
    }
    declaration = {"successor_archive_sha256": gate.sha256_file(path),
                   "exact_changed_files": exact}
    with pytest.raises(gate.D43GateError, match="not evaluation-only"):
        gate._validate_measurement_delta(base, path, declaration)


@pytest.mark.parametrize("mutation", [
    lambda source: source.replace(
        b"from run_matrix import DONE, COMPLETION_SUMMARY\n",
        b"from run_matrix import DONE, COMPLETION_SUMMARY, launch_training\n"),
    lambda source: source + b"\nlaunch_training()\n",
    lambda source: source + b"\nif __name__ == '__main__':\n    launch_training()\n",
])
def test_rehashed_import_or_top_level_side_effect_in_mixed_runner_is_refused(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mutation) -> None:
    before, after, base, _successor, _declaration = _archive_pair(tmp_path, monkeypatch)
    after["scripts/run_hmm_matrix.py"] = mutation(after["scripts/run_hmm_matrix.py"])
    path = tmp_path / "runner-top-level-mutation.tar.gz"
    _tar(path, after)
    exact = {
        name: {"before_sha256": _sha(before[name]) if name in before else None,
               "after_sha256": _sha(after[name]) if name in after else None}
        for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)
    }
    declaration = {"successor_archive_sha256": gate.sha256_file(path),
                   "exact_changed_files": exact}
    with pytest.raises(gate.D43GateError, match="completion import|top-level or training"):
        gate._validate_measurement_delta(base, path, declaration)


def test_successor_archive_must_equal_live_tree_for_every_packaged_file(
        tmp_path: pathlib.Path) -> None:
    files = {"scripts/validator.py": b"good\n", "tests/test_gate.py": b"assert True\n"}
    archive = tmp_path / "source.tar.gz"
    _tar(archive, files)
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    result = gate._validate_successor_live_tree(tmp_path, archive)
    assert result["packaged_file_count"] == 2

    # Rehashing the archive and its declaration cannot borrow witnesses/tests from a good live tree.
    files["scripts/validator.py"] = b"silently broken\n"
    _tar(archive, files)
    with pytest.raises(gate.D43GateError, match="archive/live-tree mismatch"):
        gate._validate_successor_live_tree(tmp_path, archive)


def _declaration(root: pathlib.Path, successor_sha: str) -> dict:
    d42, d43 = root / gate.D42_DECISION, root / gate.D43_DECISION
    d42.parent.mkdir(parents=True, exist_ok=True)
    d42.write_text("D42\n")
    d43.write_text("D43\n")
    return {
        "schema": gate.DECLARATION_SCHEMA, "status": "FROZEN",
        "original_predecessor_sha256": gate.ORIGINAL_PREDECESSOR_SHA256,
        "d41_successor_sha256": gate.D41_SUCCESSOR_SHA256,
        "successor_archive_sha256": successor_sha,
        "d42_decision": {"path": gate.D42_DECISION.as_posix(), "sha256": gate.sha256_file(d42)},
        "d43_decision": {"path": gate.D43_DECISION.as_posix(), "sha256": gate.sha256_file(d43)},
        "allowed_changed_paths": sorted(gate.ALLOWED_MEASUREMENT_PATHS),
        "exact_changed_files": {},
        "confirmatory_lifecycle": {
            "training_started": True, "completed_hmm_training_cells": 10,
            "total_hmm_training_cells": 30, "scientific_evaluations_inspected": False,
        },
    }


def test_declaration_never_claims_training_started_false_and_binds_decision_hashes(
        tmp_path: pathlib.Path) -> None:
    value = _declaration(tmp_path, "f" * 64)
    _write_json(tmp_path / gate.DECLARATION, value)
    gate._validate_declaration(tmp_path, "f" * 64)
    value["confirmatory_lifecycle"]["training_started"] = False
    _write_json(tmp_path / gate.DECLARATION, value)
    with pytest.raises(gate.D43GateError, match="lifecycle-false"):
        gate._validate_declaration(tmp_path, "f" * 64)
    value["confirmatory_lifecycle"]["training_started"] = True
    value["d43_decision"]["sha256"] = "0" * 64
    _write_json(tmp_path / gate.DECLARATION, value)
    with pytest.raises(gate.D43GateError, match="declared amendment file hash"):
        gate._validate_declaration(tmp_path, "f" * 64)


def _review_text() -> str:
    return (
        "D43\nOutcome-blind: PASS\nP0=0\nP1=0\nTraining started: true\n"
        "Completed HMM training cells: 10\nScientific evaluations started: false\n"
        "Evaluator invocations: 0\nScientific metrics inspected: false\n"
    )


def _outcome_and_state(root: pathlib.Path, source_sha: str, reviewer: str) -> dict:
    report = root / "docs/review.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_review_text())
    review_binding = {
        "reviewer": reviewer,
        "report_path": "docs/review.md",
        "report_sha256": gate.sha256_file(report),
    }
    no_outcome = {
        "schema": gate.NO_OUTCOME_SCHEMA, "status": "PASS",
        "original_predecessor_sha256": gate.ORIGINAL_PREDECESSOR_SHA256,
        "successor_archive_sha256": source_sha,
        "training_started": True, "completed_hmm_training_cells": 10,
        "scientific_evaluations_started": False, "scientific_metrics_inspected": False,
        "evaluator_invocations": 0,
        "statement": "ten HMM cells trained; no scientific evaluation was run, opened, or interpreted",
        "independent_review": review_binding,
        "attested_by": {"research_controller": "controller", "independent_reviewer": reviewer},
    }
    pending = [job for job in gate.ALL_HMM_JOB_IDS if job not in gate.EXACT_TEN_JOB_IDS]
    state = {
        "schema": gate.CONTINUATION_SCHEMA,
        "status": "ATOMIC_READY_FOR_TRAINING_CONTINUATION",
        "source_archive_sha256": source_sha,
        "original_predecessor_sha256": gate.ORIGINAL_PREDECESSOR_SHA256,
        "training_started": True, "completed_job_ids": list(gate.EXACT_TEN_JOB_IDS),
        "pending_job_ids": pending, "evaluated_job_ids": [],
        "scientific_evaluations_started": False, "scientific_metrics_inspected": False,
        "evaluator_invocations": 0, "total_hmm_training_cells": 30,
        "next_phase": "train_remaining_20_then_atomic_all_30_preflight_before_evaluation",
    }
    _write_json(root / gate.NO_OUTCOME, no_outcome)
    _write_json(root / gate.CONTINUATION, state)
    return dict(review_binding,
                no_outcome_attestation_sha256=gate.sha256_file(root / gate.NO_OUTCOME))


@pytest.mark.parametrize("artifact,field,value,message", [
    ("outcome", "scientific_metrics_inspected", True, "no-outcome evidence"),
    ("outcome", "scientific_evaluations_started", True, "no-outcome evidence"),
    ("outcome", "training_started", False, "no-outcome evidence"),
    ("outcome", "completed_hmm_training_cells", 9, "no-outcome evidence"),
    ("state", "completed_job_ids", [], "continuation state is non-atomic"),
    ("state", "evaluated_job_ids", [gate.EXACT_TEN_JOB_IDS[0]], "continuation state is non-atomic"),
    ("state", "evaluator_invocations", 1, "continuation state is non-atomic"),
])
def test_missing_forged_outcome_evidence_or_nonatomic_state_is_refused(
        tmp_path: pathlib.Path, artifact: str, field: str, value: object, message: str) -> None:
    review = _outcome_and_state(tmp_path, "e" * 64, "reviewer")
    path = tmp_path / (gate.NO_OUTCOME if artifact == "outcome" else gate.CONTINUATION)
    document = json.loads(path.read_text())
    document[field] = value
    _write_json(path, document)
    with pytest.raises(gate.D43GateError, match=message):
        gate._validate_no_outcome_and_continuation(tmp_path, "e" * 64, review)


def _semantic_evidence(root: pathlib.Path, source_sha: str) -> tuple[pathlib.Path, dict]:
    artifact = (root / ".agent_state" / "preregistration" / source_sha /
                gate.SEMANTIC_EXECUTION_NAME)
    witnesses = {name: {"test": name} for name in gate.REQUIRED_SEMANTIC_WITNESSES}
    payload = {"claim": gate.SEMANTIC_EXECUTION_ROLE, "semantic_witnesses": witnesses}
    _write_json(artifact, {
        "schema": gate.SEMANTIC_EXECUTION_SCHEMA, "status": "PASS",
        "role": gate.SEMANTIC_EXECUTION_ROLE, "source_archive_sha256": source_sha,
        "payload": payload, "payload_sha256": gate.canonical_sha256(payload),
    })
    checks = {
        "lurestar_schema_contract": dict(gate.EXPECTED_LURE_SCHEMAS),
        "schemas": list(gate.EXPECTED_LURE_SCHEMAS.values()) + [
            gate.EXPECTED_HMM_AGGREGATE_SCHEMA],
        "h1_h2_metrics_preserved": True, "missing_metrics_refused": True,
        "extra_metrics_refused": True, "invalid_cells_emitted": True,
        "nulls_emitted": True, "manipulation_failures_emitted": True,
        "multiplicity_fields_emitted": True,
    }
    _write_json(root / gate.SEMANTIC_EVIDENCE, {
        "schema": "nextlat_forgetting/preregistration_evidence/1",
        "gates": {"10": {"checks": checks, "artifacts": [{
            "path": str(artifact), "role": gate.SEMANTIC_EXECUTION_ROLE,
            "schema": gate.SEMANTIC_EXECUTION_SCHEMA,
            "sha256": gate.sha256_file(artifact)}]}},
    })
    return artifact, witnesses


def test_semantic_evidence_requires_current_schemas_witnesses_and_successor_binding(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _artifact, witnesses = _semantic_evidence(tmp_path, "d" * 64)
    monkeypatch.setattr(gate, "_recompute_semantic_witnesses", lambda _root: witnesses)
    gate._validate_semantic_evidence(tmp_path, "d" * 64)
    evidence = json.loads((tmp_path / gate.SEMANTIC_EVIDENCE).read_text())
    evidence["gates"]["10"]["checks"]["lurestar_schema_contract"]["confirmatory_report"] = \
        "nextlat_forgetting/lurestar_confirmatory_report/3"
    _write_json(tmp_path / gate.SEMANTIC_EVIDENCE, evidence)
    with pytest.raises(gate.D43GateError, match="schemas are stale"):
        gate._validate_semantic_evidence(tmp_path, "d" * 64)
    _semantic_evidence(tmp_path, "d" * 64)
    artifact = pathlib.Path(json.loads((tmp_path / gate.SEMANTIC_EVIDENCE).read_text())[
        "gates"]["10"]["artifacts"][0]["path"])
    value = json.loads(artifact.read_text())
    value["source_archive_sha256"] = "c" * 64
    _write_json(artifact, value)
    evidence = json.loads((tmp_path / gate.SEMANTIC_EVIDENCE).read_text())
    evidence["gates"]["10"]["artifacts"][0]["sha256"] = gate.sha256_file(artifact)
    _write_json(tmp_path / gate.SEMANTIC_EVIDENCE, evidence)
    with pytest.raises(gate.D43GateError, match="stale or nonpassing"):
        gate._validate_semantic_evidence(tmp_path, "d" * 64)


def test_semantic_evidence_refuses_wrong_role_or_noncanonical_execution_path(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_sha = "d" * 64
    artifact, witnesses = _semantic_evidence(tmp_path, source_sha)
    monkeypatch.setattr(gate, "_recompute_semantic_witnesses", lambda _root: witnesses)
    evidence_path = tmp_path / gate.SEMANTIC_EVIDENCE
    evidence = json.loads(evidence_path.read_text())
    value = json.loads(artifact.read_text())
    value["role"] = "arbitrary_passing_artifact"
    _write_json(artifact, value)
    record = evidence["gates"]["10"]["artifacts"][0]
    record["role"] = value["role"]
    record["sha256"] = gate.sha256_file(artifact)
    _write_json(evidence_path, evidence)
    with pytest.raises(gate.D43GateError, match="exact semantic execution artifact"):
        gate._validate_semantic_evidence(tmp_path, source_sha)

    artifact, witnesses = _semantic_evidence(tmp_path, source_sha)
    wrong_path = artifact.with_name("arbitrary-g10.json")
    _write_json(wrong_path, json.loads(artifact.read_text()))
    evidence = json.loads(evidence_path.read_text())
    record = evidence["gates"]["10"]["artifacts"][0]
    record["path"] = str(wrong_path)
    record["sha256"] = gate.sha256_file(wrong_path)
    _write_json(evidence_path, evidence)
    with pytest.raises(gate.D43GateError, match="exact semantic execution artifact"):
        gate._validate_semantic_evidence(tmp_path, source_sha)


@pytest.mark.parametrize("mutation,message", [
    ("missing", "incomplete or noncanonical"),
    ("wrong", "differ from source/test recomputation"),
])
def test_semantic_execution_artifact_witness_payload_fails_closed_after_rehash(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        mutation: str, message: str) -> None:
    source_sha = "d" * 64
    artifact, witnesses = _semantic_evidence(tmp_path, source_sha)
    monkeypatch.setattr(gate, "_recompute_semantic_witnesses", lambda _root: witnesses)
    value = json.loads(artifact.read_text())
    stored = value["payload"]["semantic_witnesses"]
    key = sorted(stored)[0]
    if mutation == "missing":
        stored.pop(key)
    else:
        stored[key] = {"test": "forged-but-rehashed"}
    value["payload_sha256"] = gate.canonical_sha256(value["payload"])
    _write_json(artifact, value)
    evidence_path = tmp_path / gate.SEMANTIC_EVIDENCE
    evidence = json.loads(evidence_path.read_text())
    evidence["gates"]["10"]["artifacts"][0]["sha256"] = gate.sha256_file(artifact)
    _write_json(evidence_path, evidence)
    with pytest.raises(gate.D43GateError, match=message):
        gate._validate_semantic_evidence(tmp_path, source_sha)


def _tests_and_review(root: pathlib.Path, source_sha: str) -> None:
    review_binding = _outcome_and_state(root, source_sha, "independent")
    report = root / review_binding["report_path"]
    _write_json(root / gate.TEST_RECEIPT, {
        "schema": "nextlat_forgetting/full_test_suite_receipt/1", "source_sha256": source_sha,
        "outcome": "PASS", "exit_code": 0, "tests_passed": 123,
        "command": ["python", "-m", "pytest", "tests", "-q"],
    })
    _write_json(root / gate.REVIEW_RECEIPT, {
        "schema": "nextlat_forgetting/independent_scientific_review/1",
        "source_sha256": source_sha, "verdict": "PASS", "reviewer": "independent",
        "report_path": "docs/review.md", "report_sha256": gate.sha256_file(report),
        "no_outcome_attestation_sha256": review_binding["no_outcome_attestation_sha256"],
    })


@pytest.mark.parametrize("kind,mutation,message", [
    ("test", ("source_sha256", "0" * 64), "full-suite receipt is stale"),
    ("test", ("outcome", "FAIL"), "full-suite receipt is stale"),
    ("review", ("source_sha256", "0" * 64), "independent review is stale"),
    ("review", ("verdict", "BLOCK"), "independent review is stale"),
])
def test_stale_tests_or_review_refuse_d43(tmp_path: pathlib.Path, kind: str,
                                         mutation: tuple[str, object], message: str) -> None:
    _tests_and_review(tmp_path, "b" * 64)
    path = tmp_path / (gate.TEST_RECEIPT if kind == "test" else gate.REVIEW_RECEIPT)
    value = json.loads(path.read_text())
    value[mutation[0]] = mutation[1]
    _write_json(path, value)
    with pytest.raises(gate.D43GateError, match=message):
        gate._validate_tests_and_review(tmp_path, "b" * 64)


def test_review_receipt_cryptographically_binds_no_outcome_attestation(
        tmp_path: pathlib.Path) -> None:
    _tests_and_review(tmp_path, "b" * 64)
    attestation = json.loads((tmp_path / gate.NO_OUTCOME).read_text())
    attestation["evaluator_invocations"] = 1
    _write_json(tmp_path / gate.NO_OUTCOME, attestation)
    with pytest.raises(gate.D43GateError, match="independent review is stale"):
        gate._validate_tests_and_review(tmp_path, "b" * 64)


@pytest.mark.parametrize("line", [
    "Training started: true\n",
    "Completed HMM training cells: 10\n",
    "Scientific evaluations started: false\n",
    "Evaluator invocations: 0\n",
    "Scientific metrics inspected: false\n",
])
def test_review_report_must_explicitly_attest_every_lifecycle_fact(
        tmp_path: pathlib.Path, line: str) -> None:
    _tests_and_review(tmp_path, "b" * 64)
    report = tmp_path / "docs/review.md"
    report.write_text(report.read_text().replace(line, ""))
    receipt_path = tmp_path / gate.REVIEW_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    receipt["report_sha256"] = gate.sha256_file(report)
    _write_json(receipt_path, receipt)
    with pytest.raises(gate.D43GateError, match="lacks required outcome-blind"):
        gate._validate_tests_and_review(tmp_path, "b" * 64)


def test_no_outcome_attestation_binds_exact_review_report_and_reviewer(
        tmp_path: pathlib.Path) -> None:
    review = _outcome_and_state(tmp_path, "e" * 64, "reviewer")
    attestation_path = tmp_path / gate.NO_OUTCOME
    attestation = json.loads(attestation_path.read_text())
    attestation["independent_review"]["report_sha256"] = "0" * 64
    _write_json(attestation_path, attestation)
    review["no_outcome_attestation_sha256"] = gate.sha256_file(attestation_path)
    with pytest.raises(gate.D43GateError, match="no-outcome evidence"):
        gate._validate_no_outcome_and_continuation(tmp_path, "e" * 64, review)


def test_exact_ten_projection_binds_predecessor_and_successor_checkpoint_provenance(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = []
    for index, job_id in enumerate(gate.EXACT_TEN_JOB_IDS):
        object_record = lambda role: {  # noqa: E731
            "uri": f"gs://bucket/{job_id}/{role}", "generation": str(100 + index),
            "sha256": f"{index + 1:064x}", "size_bytes": 10 + index,
        }
        jobs.append({
            "job_id": job_id, "predecessor_source_sha256": gate.ORIGINAL_PREDECESSOR_SHA256,
            "target_step": gate.TARGET_STEP, "state_object": object_record("state"),
            "checkpoint_object": object_record("checkpoint"), "sidecar_object": object_record("sidecar"),
            "checkpoint_semantics": {"payload_training_steps": gate.TARGET_STEP},
            "verification": {"payload_training_steps_verified": True},
        })
    _write_json(tmp_path / gate.D41_RECEIPT, {"schema": "d41/2", "jobs": jobs})
    monkeypatch.setattr(gate, "D41_RECOVERY_RECEIPT_SHA256",
                        gate.sha256_file(tmp_path / gate.D41_RECEIPT))
    monkeypatch.setattr(gate, "_d41_module", lambda _root: SimpleNamespace(
        validate_exact_ten_document=lambda _document: None))
    result = gate._checkpoint_projection(tmp_path, "a" * 64)
    assert result["reuse_policy"] == "read_only_exact_step_3000_no_retraining_no_evaluation"
    assert all(item["created_under_predecessor_source_sha256"] ==
               gate.ORIGINAL_PREDECESSOR_SHA256 for item in result[
                   "predecessor_to_successor_provenance"])
    assert all(item["consumed_read_only_by_successor_source_sha256"] == "a" * 64
               for item in result["predecessor_to_successor_provenance"])


def test_input_inventory_is_exact_hash_generation_and_atomic_commit_bound(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data/x.bin"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"frozen")
    inventory = tmp_path / "manifests/manifest_inventory.sha256"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(f"{gate.sha256_file(data)}  data/x.bin\n")
    inventory_sha = gate.sha256_file(inventory)
    monkeypatch.setattr(gate, "INPUT_INVENTORY_SHA256", inventory_sha)
    monkeypatch.setattr(gate, "INPUT_PREFIX", f"lurestar/input_bundles/{inventory_sha}")
    receipt = {
        "schema": "nextlat_forgetting/input_bundle_upload/1", "status": "COMPLETE",
        "bucket": gate.INPUT_BUCKET, "bundle_prefix": gate.INPUT_PREFIX,
        "input_bundle_sha256": inventory_sha, "object_count": 1,
        "objects": [{"local_path": "data/x.bin", "name": f"{gate.INPUT_PREFIX}/corpus/x.bin",
                     "generation": "11", "size_bytes": 6, "sha256": gate.sha256_file(data)}],
        "commit": {"local_path": "manifests/manifest_inventory.sha256",
                   "name": f"{gate.INPUT_PREFIX}/manifests/manifest_inventory.sha256",
                   "generation": "12", "size_bytes": inventory.stat().st_size,
                   "sha256": inventory_sha},
    }
    _write_json(tmp_path / gate.INPUT_RECEIPT, receipt)
    assert gate._validate_input_inventory(tmp_path)["commit_generation"] == "12"
    receipt["objects"][0]["generation"] = "latest"
    _write_json(tmp_path / gate.INPUT_RECEIPT, receipt)
    with pytest.raises(gate.D43GateError, match="object provenance"):
        gate._validate_input_inventory(tmp_path)


def test_create_is_atomic_and_validation_recomputes_all_evidence(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archives = [tmp_path / name for name in ("p.tar.gz", "d41.tar.gz", "d43.tar.gz")]
    for path in archives:
        _tar(path, {"x": path.name.encode()})
    expected = {"schema": gate.SCHEMA, "status": "PASS", "proof": "recomputed"}
    monkeypatch.setattr(gate, "build_receipt", lambda *_args: dict(expected))
    path = gate.create_receipt(tmp_path, *archives)
    assert path.name == gate.RECEIPT.name
    assert gate.validate_receipt(tmp_path, *archives) == expected
    value = json.loads(path.read_text())
    value["proof"] = "forged"
    _write_json(path, value)
    with pytest.raises(gate.D43GateError, match="differs from recomputed"):
        gate.validate_receipt(tmp_path, *archives)
