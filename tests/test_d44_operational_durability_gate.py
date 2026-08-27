"""Adversarial regressions for the D44 outcome-blind operational durability gate."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pathlib
import tarfile

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d44_gate_under_test", ROOT / "scripts/d44_operational_durability_gate.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: pathlib.Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _json(path: pathlib.Path, value: object) -> None:
    _write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _tar(path: pathlib.Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _controller(*, sync_value: int = 1, artifact_value: int = 1,
                helper_value: int | None = None, launch_value: int = 1,
                train_value: int = 1) -> bytes:
    helper = "" if helper_value is None else (
        "\n    def _d44_receipt_pinned_pointer_targets(self):\n"
        f"        return {helper_value}\n")
    return (
        "import json\n"
        "CONSTANT = 'unchanged'\n\n"
        "class RuntimeDurability:\n"
        "    def _artifact_paths(self):\n"
        f"        return {artifact_value}\n\n"
        "    def sync_job(self):\n"
        f"        return {sync_value}\n"
        f"{helper}\n"
        "    def train_dispatch(self):\n"
        f"        return {train_value}\n\n"
        "def launch_controller():\n"
        f"    return {launch_value}\n"
    ).encode()


def _d43_predecessor(root: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    d43_source = {
        gate.D43_DECISION_SOURCE_PATH: b"D43 decision\n",
        gate.D43_GATE_SOURCE_PATH: b"D43 gate\n",
        gate.D43_GATE_TEST_SOURCE_PATH: b"D43 tests\n",
        gate.CONTROLLER_PATH: _controller(),
        gate.CONTROLLER_TEST_PATH: b"def test_controller(): pass\n",
        "configs/gpt_hmm.yaml": b"updates: 3000\n",
        "data/frozen.bin": b"frozen\n",
        "manifests/hmm_family.json": b"{}\n",
        "src/lurestar/adaptation.py": b"def objective(): pass\n",
        "upstream/models/model_base.py": b"class Model: pass\n",
        "scripts/train_hmm.py": b"TRAINING_STEPS = 3000\n",
        "scripts/run_hmm_matrix.py": b"def run_train(): pass\n",
        "scripts/run_matrix.py": b"def run(): pass\n",
        "scripts/evaluate_hmm_checkpoints.py": b"def evaluate(): pass\n",
        "scripts/evaluate_lurestar_checkpoints.py": b"def evaluate(): pass\n",
    }
    predecessor = root / gate.D43_PREDECESSOR_ARCHIVE
    _tar(predecessor, d43_source)
    monkeypatch.setattr(gate, "D43_PREDECESSOR_SHA256", gate.sha256_file(predecessor))

    declaration = {
        "schema": "nextlat_forgetting/d43_measurement_amendment_declaration/1",
        "exact_changed_files": {
            gate.D43_GATE_SOURCE_PATH: {"after_sha256": _sha(d43_source[gate.D43_GATE_SOURCE_PATH])},
            gate.D43_GATE_TEST_SOURCE_PATH: {
                "after_sha256": _sha(d43_source[gate.D43_GATE_TEST_SOURCE_PATH])},
        },
    }
    continuation = {"schema": "nextlat_forgetting/d43_atomic_continuation_state/1"}
    no_outcome = {"schema": "nextlat_forgetting/d43_no_outcome_attestation/1"}
    test = {"schema": gate.FULL_TEST_SCHEMA}
    review = {"schema": "nextlat_forgetting/independent_scientific_review/1"}
    for path, value in ((gate.D43_DECLARATION, declaration), (gate.D43_CONTINUATION, continuation),
                        (gate.D43_NO_OUTCOME, no_outcome), (gate.D43_TEST_RECEIPT, test),
                        (gate.D43_REVIEW_RECEIPT, review)):
        _json(root / path, value)
    hashes = {path: gate.sha256_file(root / path) for path in (
        gate.D43_DECLARATION, gate.D43_CONTINUATION, gate.D43_NO_OUTCOME,
        gate.D43_TEST_RECEIPT, gate.D43_REVIEW_RECEIPT,
    )}
    receipt = {
        "schema": gate.D43_SCHEMA, "status": "PASS", "authorization": "MEASUREMENT_AMENDMENT_GO",
        "archives": {"d43_measurement_successor": {"sha256": gate.D43_PREDECESSOR_SHA256}},
        "confirmatory_lifecycle": {
            "training_started": True, "completed_hmm_training_cells": 10,
            "total_hmm_training_cells": 30, "scientific_evaluations_started": False,
            "scientific_evaluations_inspected": False,
        },
        "declaration": {"sha256": hashes[gate.D43_DECLARATION]},
        "outcome_blind_atomic_continuation": {
            "atomic_continuation_state": {"sha256": hashes[gate.D43_CONTINUATION]}},
        "successor_assurance": {
            "full_suite": {"sha256": hashes[gate.D43_TEST_RECEIPT]},
            "independent_review": {"sha256": hashes[gate.D43_REVIEW_RECEIPT]},
        },
        "decisions": {"d43_decision": {"sha256": _sha(d43_source[gate.D43_DECISION_SOURCE_PATH])}},
    }
    _json(root / gate.D43_RECEIPT, receipt)
    all_paths = (gate.D43_RECEIPT, gate.D43_DECLARATION, gate.D43_CONTINUATION,
                 gate.D43_NO_OUTCOME, gate.D43_TEST_RECEIPT, gate.D43_REVIEW_RECEIPT)
    schemas = {
        gate.D43_RECEIPT: gate.D43_SCHEMA,
        gate.D43_DECLARATION: declaration["schema"],
        gate.D43_CONTINUATION: continuation["schema"],
        gate.D43_NO_OUTCOME: no_outcome["schema"],
        gate.D43_TEST_RECEIPT: test["schema"],
        gate.D43_REVIEW_RECEIPT: review["schema"],
    }
    monkeypatch.setattr(gate, "D43_RECORD_BINDINGS", {
        path: {"schema": schemas[path], "sha256": gate.sha256_file(root / path)}
        for path in all_paths
    })
    return predecessor


def _successor(root: pathlib.Path, predecessor: pathlib.Path) -> tuple[pathlib.Path, dict[str, bytes]]:
    before = gate._archive(predecessor)
    after = dict(before)
    after[gate.CONTROLLER_PATH] = _controller(sync_value=2, artifact_value=2, helper_value=2)
    after[gate.CONTROLLER_TEST_PATH] = b"def test_d44_pointer_guard(): assert True\n"
    after[gate.D44_DECISION.as_posix()] = b"D44 decision\n"
    after["scripts/d44_operational_durability_gate.py"] = b"D44 gate\n"
    after["tests/test_d44_operational_durability_gate.py"] = b"def test_d44(): pass\n"
    successor = root / ".agent_state/project.tar.gz"
    _tar(successor, after)
    for name, payload in after.items():
        _write(root / name, payload)
    return successor, after


def _declaration(root: pathlib.Path, predecessor: pathlib.Path,
                 successor: pathlib.Path) -> None:
    before, after = gate._archive(predecessor), gate._archive(successor)
    exact = {
        name: {
            "before_sha256": _sha(before[name]) if name in before else None,
            "after_sha256": _sha(after[name]) if name in after else None,
        }
        for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)
    }
    _json(root / gate.DECLARATION, {
        "schema": gate.DECLARATION_SCHEMA, "status": "FROZEN",
        "d43_predecessor_archive_sha256": gate.D43_PREDECESSOR_SHA256,
        "d43_predecessor_receipt": gate._record_binding(root, gate.D43_RECEIPT),
        "successor_archive_sha256": gate.sha256_file(successor),
        "d44_decision": {"path": gate.D44_DECISION.as_posix(),
                         "sha256": gate.sha256_file(root / gate.D44_DECISION)},
        "allowed_changed_paths": sorted(gate.ALLOWED_CHANGED_PATHS),
        "exact_changed_files": exact,
        "operational_controller_changed_symbols": [
            "RuntimeDurability._artifact_paths",
            "RuntimeDurability._d44_receipt_pinned_pointer_targets",
            "RuntimeDurability.sync_job",
        ],
    })


def _assurance(root: pathlib.Path, successor: pathlib.Path) -> None:
    source_sha = gate.sha256_file(successor)
    report_path = pathlib.Path(".agent_state/d44-independent-review.md")
    report = (
        "D44\nOutcome-blind: PASS\nP0=0\nP1=0\nTraining started: true\n"
        "Completed HMM training cells: 10\nPending HMM training cells: 20\n"
        "New HMM training cells: 0\nScientific evaluations started: false\n"
        "Evaluator invocations: 0\nScientific metrics inspected: false\n"
    )
    _write(root / report_path, report.encode())
    incident = {
        "schema": gate.INCIDENT_SCHEMA, "status": "PASS",
        "d43_predecessor_archive_sha256": gate.D43_PREDECESSOR_SHA256,
        "successor_archive_sha256": source_sha,
        "training_started": True, "completed_hmm_training_cells": 10,
        "pending_hmm_training_cells": 20, "new_hmm_training_cells": 0,
        "scientific_evaluations_started": False, "evaluator_invocations": 0,
        "scientific_metrics_inspected": False,
        "statement": ("metadata-only operational incident; no new HMM cell, scientific evaluation, "
                      "or scientific metric inspection occurred"),
        "independent_review": {
            "reviewer": "independent operational reviewer", "report_path": report_path.as_posix(),
            "report_sha256": gate.sha256_file(root / report_path),
        },
        "attested_by": {
            "research_controller": "controller", "independent_reviewer": "independent operational reviewer"},
    }
    _json(root / gate.INCIDENT, incident)
    _json(root / gate.TEST_RECEIPT, {
        "schema": gate.FULL_TEST_SCHEMA, "source_sha256": source_sha, "outcome": "PASS",
        "exit_code": 0, "tests_passed": 123, "command": ["python", "-m", "pytest", "tests", "-q"],
    })
    _json(root / gate.REVIEW_RECEIPT, {
        "schema": gate.REVIEW_SCHEMA, "status": "PASS", "source_sha256": source_sha,
        "reviewer": "independent operational reviewer", "report_path": report_path.as_posix(),
        "report_sha256": gate.sha256_file(root / report_path),
        "incident_attestation_sha256": gate.sha256_file(root / gate.INCIDENT),
    })


def _fixture(root: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    predecessor = _d43_predecessor(root, monkeypatch)
    successor, _after = _successor(root, predecessor)
    _declaration(root, predecessor, successor)
    _assurance(root, successor)
    return predecessor, successor


def test_exact_closed_delta_accepts_only_durability_methods_and_d44_artifacts(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor, successor = _fixture(tmp_path, monkeypatch)
    declaration, _decision = gate._validate_declaration(tmp_path, gate.sha256_file(successor))
    delta = gate._validate_delta(predecessor, successor, declaration)
    assert delta["changed_paths"] == sorted(gate.ALLOWED_CHANGED_PATHS)
    assert delta["controller_changed_symbols"] == [
        "RuntimeDurability._artifact_paths",
        "RuntimeDurability._d44_receipt_pinned_pointer_targets",
        "RuntimeDurability.sync_job",
    ]
    assert gate._validate_live_tree(tmp_path, successor)["packaged_file_count"] > 10


@pytest.mark.parametrize("path,payload", [
    ("configs/gpt_hmm.yaml", b"updates: 3001\n"),
    ("src/lurestar/adaptation.py", b"def changed_objective(): pass\n"),
    ("scripts/train_hmm.py", b"TRAINING_STEPS = 3001\n"),
    ("scripts/evaluate_hmm_checkpoints.py", b"def changed_evaluate(): pass\n"),
])
def test_config_model_training_or_evaluation_archive_change_is_refused(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, path: str, payload: bytes) -> None:
    predecessor, successor = _fixture(tmp_path, monkeypatch)
    mutated = gate._archive(successor)
    mutated[path] = payload
    _tar(successor, mutated)
    declaration = json.loads((tmp_path / gate.DECLARATION).read_text())
    declaration["successor_archive_sha256"] = gate.sha256_file(successor)
    with pytest.raises(gate.D44GateError, match="not exactly the declared operational repair"):
        gate._validate_delta(predecessor, successor, declaration)


def test_controller_training_dispatch_or_top_level_mutation_is_refused_even_if_rehashed(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor, successor = _fixture(tmp_path, monkeypatch)
    mutated = gate._archive(successor)
    mutated[gate.CONTROLLER_PATH] = _controller(
        sync_value=2, artifact_value=2, helper_value=2, train_value=99, launch_value=99)
    _tar(successor, mutated)
    before = gate._archive(predecessor)
    exact = {
        name: {"before_sha256": _sha(before[name]) if name in before else None,
               "after_sha256": _sha(mutated[name]) if name in mutated else None}
        for name in sorted(set(before) | set(mutated)) if before.get(name) != mutated.get(name)
    }
    declaration = json.loads((tmp_path / gate.DECLARATION).read_text())
    declaration["successor_archive_sha256"] = gate.sha256_file(successor)
    declaration["exact_changed_files"] = exact
    declaration["operational_controller_changed_symbols"] = sorted([
        "RuntimeDurability._artifact_paths",
        "RuntimeDurability._d44_receipt_pinned_pointer_targets",
        "RuntimeDurability.sync_job",
        "RuntimeDurability.train_dispatch",
    ])
    _json(tmp_path / gate.DECLARATION, declaration)
    with pytest.raises(gate.D44GateError, match="permitted durability repair"):
        gate._validate_delta(predecessor, successor, declaration)


def test_live_tree_mismatch_cannot_borrow_tests_or_receipts_from_good_source(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _predecessor, successor = _fixture(tmp_path, monkeypatch)
    _write(tmp_path / gate.CONTROLLER_PATH, b"silently changed live source\n")
    with pytest.raises(gate.D44GateError, match="archive/live-tree mismatch"):
        gate._validate_live_tree(tmp_path, successor)


def test_d43_archive_and_every_preserved_record_are_immutable_inputs(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor, _successor = _fixture(tmp_path, monkeypatch)
    gate._validate_d43_predecessor(tmp_path, predecessor)
    receipt = tmp_path / gate.D43_RECEIPT
    value = json.loads(receipt.read_text())
    value["authorization"] = "FORGED"
    _json(receipt, value)
    with pytest.raises(gate.D44GateError, match="preserved record changed"):
        gate._validate_d43_predecessor(tmp_path, predecessor)


@pytest.mark.parametrize("field,value", [
    ("completed_hmm_training_cells", 9), ("pending_hmm_training_cells", 21),
    ("new_hmm_training_cells", 1), ("evaluator_invocations", 1),
    ("scientific_metrics_inspected", True),
])
def test_metadata_only_incident_attestation_fails_closed(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    _predecessor, successor = _fixture(tmp_path, monkeypatch)
    incident_path = tmp_path / gate.INCIDENT
    incident = json.loads(incident_path.read_text())
    incident[field] = value
    _json(incident_path, incident)
    review = json.loads((tmp_path / gate.REVIEW_RECEIPT).read_text())
    review["incident_attestation_sha256"] = gate.sha256_file(incident_path)
    with pytest.raises(gate.D44GateError, match="incident evidence"):
        gate._validate_incident(tmp_path, gate.sha256_file(successor), review)


def test_review_must_bind_attestation_and_explicitly_state_all_lifecycle_facts(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _predecessor, successor = _fixture(tmp_path, monkeypatch)
    report_path = tmp_path / ".agent_state/d44-independent-review.md"
    report_path.write_text(report_path.read_text().replace("New HMM training cells: 0\n", ""))
    review_path = tmp_path / gate.REVIEW_RECEIPT
    review = json.loads(review_path.read_text())
    review["report_sha256"] = gate.sha256_file(report_path)
    _json(review_path, review)
    with pytest.raises(gate.D44GateError, match="required outcome-blind operational verdict"):
        gate._validate_assurance(tmp_path, gate.sha256_file(successor))


def test_create_is_atomic_and_validate_recomputes_every_d44_binding(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor, successor = _fixture(tmp_path, monkeypatch)
    created = gate.create_receipt(tmp_path, predecessor, successor)
    assert created == tmp_path / gate.RECEIPT
    assert gate.validate_receipt(tmp_path, predecessor, successor)["status"] == "PASS"
    receipt = json.loads(created.read_text())
    receipt["authorization"] = "FORGED"
    _json(created, receipt)
    with pytest.raises(gate.D44GateError, match="differs from recomputed"):
        gate.validate_receipt(tmp_path, predecessor, successor)
