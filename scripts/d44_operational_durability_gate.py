#!/usr/bin/env python
"""Fail-closed, outcome-blind D44 operational-durability continuation gate.

D44 is intentionally narrower than D43.  It accepts one already-frozen D43 source archive and
one exact successor whose only source change is a receipt-pinned durability repair in the Colab
controller.  This module never imports a model, trainer, evaluator, or result reader.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import tarfile
from typing import Any, Mapping


SCHEMA = "nextlat_forgetting/d44_operational_durability_continuation/1"
DECLARATION_SCHEMA = "nextlat_forgetting/d44_operational_durability_declaration/1"
INCIDENT_SCHEMA = "nextlat_forgetting/d44_operational_incident_attestation/1"
REVIEW_SCHEMA = "nextlat_forgetting/d44_independent_operational_review/1"
FULL_TEST_SCHEMA = "nextlat_forgetting/full_test_suite_receipt/1"
D43_SCHEMA = "nextlat_forgetting/d43_measurement_amendment/1"

D43_PREDECESSOR_SHA256 = (
    "baf9fa4986956b4ab8aa3e07b6f1fe74e570a848ac2d550667177560b68b258d"
)
D43_PREDECESSOR_ARCHIVE = pathlib.Path(".agent_state/project-predecessor-d43.tar.gz")
D43_RECEIPT = pathlib.Path(".agent_state/d43-measurement-amendment-receipt.json")
D43_DECLARATION = pathlib.Path(".agent_state/d43-measurement-amendment-declaration.json")
D43_CONTINUATION = pathlib.Path(".agent_state/d43-continuation-state.json")
D43_NO_OUTCOME = pathlib.Path(".agent_state/d43-no-outcome-attestation.json")
D43_TEST_RECEIPT = pathlib.Path(".agent_state/confirmatory-test-receipt.json")
D43_REVIEW_RECEIPT = pathlib.Path(".agent_state/confirmatory-review-receipt.json")

# These records are the immutable D43 witness set.  A D44 issuance must retain their literal
# bytes rather than rewriting their source binding to its successor.  The mapping is deliberately
# data-only so a later independent verifier can substitute a separately archived fixture in tests.
D43_RECORD_BINDINGS: dict[pathlib.Path, dict[str, str]] = {
    D43_RECEIPT: {
        "schema": D43_SCHEMA,
        "sha256": "634519ce4c6434a21676085a90e84e12e6b5bbb2a5b1db25181de8112b9dbc38",
    },
    D43_DECLARATION: {
        "schema": "nextlat_forgetting/d43_measurement_amendment_declaration/1",
        "sha256": "6ce260dac1bfb75abb5a00e7a6ce7bb59d9dab0170f32b67aed2627f7f1dcaee",
    },
    D43_CONTINUATION: {
        "schema": "nextlat_forgetting/d43_atomic_continuation_state/1",
        "sha256": "d5ca7386685b65c0422ccfb0f020cf32115de2f5f1f6a42b3e2ac40aa1b80010",
    },
    D43_NO_OUTCOME: {
        "schema": "nextlat_forgetting/d43_no_outcome_attestation/1",
        "sha256": "297e7075d326ecd473710a02944e61c6037562729693829bb3302360cbec7c59",
    },
    D43_TEST_RECEIPT: {
        "schema": FULL_TEST_SCHEMA,
        "sha256": "bfe38601684f7d47270c4bbaefc4835d346886b02c18eec9a9ea8da75ff82186",
    },
    D43_REVIEW_RECEIPT: {
        "schema": "nextlat_forgetting/independent_scientific_review/1",
        "sha256": "ae0168a4ff11fc0a898197fbfa39309b551847edd54770dedbec05c5c6644302",
    },
}

D44_DECISION = pathlib.Path("docs/DECISION_D44_OPERATIONAL_DURABILITY_CONTINUATION.md")
DECLARATION = pathlib.Path(".agent_state/d44-operational-durability-declaration.json")
INCIDENT = pathlib.Path(".agent_state/d44-operational-incident-attestation.json")
TEST_RECEIPT = pathlib.Path(".agent_state/d44-full-test-suite-receipt.json")
REVIEW_RECEIPT = pathlib.Path(".agent_state/d44-independent-operational-review-receipt.json")
RECEIPT = pathlib.Path(".agent_state/d44-operational-durability-receipt.json")

CONTROLLER_PATH = "scripts/colab_train_loop.py"
CONTROLLER_TEST_PATH = "tests/test_colab_train_loop.py"
ALLOWED_CHANGED_PATHS = frozenset({
    CONTROLLER_PATH,
    CONTROLLER_TEST_PATH,
    D44_DECISION.as_posix(),
    "scripts/d44_operational_durability_gate.py",
    "tests/test_d44_operational_durability_gate.py",
})

# D44 permits no controller-wide rewrite.  The repair may modify the two existing durability
# methods and add narrowly named private helpers; all imports, module constants, launch logic,
# dispatch, training command construction, and evaluator-related code must remain identical.
ALLOWED_CONTROLLER_SYMBOLS = frozenset({
    "RuntimeDurability.sync_job",
    "RuntimeDurability._artifact_paths",
})
_SHA = re.compile(r"[0-9a-f]{64}")


class D44GateError(RuntimeError):
    """D44 evidence is missing, stale, overbroad, or outcome-visible."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D44GateError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise D44GateError(f"{label} must be a JSON object")
    return value


def _binding(root: pathlib.Path, relative: pathlib.Path, schema: str | None = None) -> dict[str, str]:
    path = root / relative
    document = _json(path, relative.as_posix())
    if schema is not None and document.get("schema") != schema:
        raise D44GateError(f"{relative} schema is stale")
    return {"path": relative.as_posix(), "sha256": sha256_file(path),
            "schema": str(document.get("schema"))}


def _archive(path: pathlib.Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        archive = tarfile.open(path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise D44GateError(f"source archive is missing or invalid: {path}") from exc
    with archive:
        for member in archive.getmembers():
            pure = pathlib.PurePosixPath(member.name)
            if member.isdir():
                continue
            if (not member.isfile() or pure.is_absolute() or ".." in pure.parts or
                    member.name in files):
                raise D44GateError(f"unsafe, duplicate, or nonregular archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise D44GateError(f"unreadable archive member: {member.name}")
            files[member.name] = stream.read()
    return files


def _record_binding(root: pathlib.Path, relative: pathlib.Path) -> dict[str, str]:
    expected = D43_RECORD_BINDINGS.get(relative)
    if expected is None:
        raise D44GateError(f"unrecognized D43 preserved record: {relative}")
    path = root / relative
    document = _json(path, f"preserved D43 record {relative}")
    if document.get("schema") != expected["schema"] or sha256_file(path) != expected["sha256"]:
        raise D44GateError(f"D43 preserved record changed or stale: {relative}")
    return {"path": relative.as_posix(), "schema": expected["schema"],
            "sha256": expected["sha256"]}


def _validate_d43_predecessor(root: pathlib.Path, archive_path: pathlib.Path) -> dict[str, Any]:
    """Verify the frozen D43 source/record pair without regenerating D43 evidence."""
    if sha256_file(archive_path) != D43_PREDECESSOR_SHA256:
        raise D44GateError("D44 predecessor archive is not the exact frozen D43 source")
    records = {
        relative.as_posix(): _record_binding(root, relative)
        for relative in D43_RECORD_BINDINGS
    }
    receipt = _json(root / D43_RECEIPT, "D43 predecessor receipt")
    lifecycle = {
        "training_started": True, "completed_hmm_training_cells": 10,
        "total_hmm_training_cells": 30, "scientific_evaluations_started": False,
        "scientific_evaluations_inspected": False,
    }
    d43_archive = receipt.get("archives", {}).get("d43_measurement_successor")
    assurance = receipt.get("successor_assurance", {})
    if (receipt.get("schema") != D43_SCHEMA or receipt.get("status") != "PASS" or
            receipt.get("authorization") != "MEASUREMENT_AMENDMENT_GO" or
            not isinstance(d43_archive, dict) or d43_archive.get("sha256") != D43_PREDECESSOR_SHA256 or
            receipt.get("confirmatory_lifecycle") != lifecycle or
            receipt.get("declaration", {}).get("sha256") != records[D43_DECLARATION.as_posix()]["sha256"] or
            receipt.get("outcome_blind_atomic_continuation", {}).get(
                "atomic_continuation_state", {}).get("sha256") !=
            records[D43_CONTINUATION.as_posix()]["sha256"] or
            assurance.get("full_suite", {}).get("sha256") !=
            records[D43_TEST_RECEIPT.as_posix()]["sha256"] or
            assurance.get("independent_review", {}).get("sha256") !=
            records[D43_REVIEW_RECEIPT.as_posix()]["sha256"]):
        raise D44GateError("D43 predecessor receipt is stale, incomplete, or outcome-visible")

    source = _archive(archive_path)
    declaration = _json(root / D43_DECLARATION, "D43 predecessor declaration")
    decision = receipt.get("decisions", {}).get("d43_decision")
    required_source_hashes = {
        D43_DECISION_SOURCE_PATH: decision.get("sha256") if isinstance(decision, dict) else None,
        D43_GATE_SOURCE_PATH: declaration.get("exact_changed_files", {}).get(
            D43_GATE_SOURCE_PATH, {}).get("after_sha256"),
        D43_GATE_TEST_SOURCE_PATH: declaration.get("exact_changed_files", {}).get(
            D43_GATE_TEST_SOURCE_PATH, {}).get("after_sha256"),
    }
    for name, digest in required_source_hashes.items():
        if (not _SHA.fullmatch(str(digest)) or name not in source or
                hashlib.sha256(source[name]).hexdigest() != digest):
            raise D44GateError(f"D43 source archive/record binding is stale: {name}")
    return {
        "archive": {"path": str(archive_path.relative_to(root)),
                    "sha256": D43_PREDECESSOR_SHA256},
        "records": records,
    }


D43_DECISION_SOURCE_PATH = "docs/DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md"
D43_GATE_SOURCE_PATH = "scripts/d43_measurement_amendment_gate.py"
D43_GATE_TEST_SOURCE_PATH = "tests/test_d43_measurement_amendment_gate.py"


def _runtime_durability_methods(source: bytes) -> dict[str, str]:
    tree = ast.parse(source.decode("utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name == "RuntimeDurability"]
    if len(classes) != 1:
        raise D44GateError("operational controller lacks one RuntimeDurability class")
    methods: dict[str, str] = {}
    for node in classes[0].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[f"RuntimeDurability.{node.name}"] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
    return methods


def _allowed_controller_symbol(symbol: str) -> bool:
    return symbol in ALLOWED_CONTROLLER_SYMBOLS or symbol.startswith("RuntimeDurability._d44_")


def _sanitized_controller(source: bytes, changed_symbols: set[str]) -> str:
    """Return the whole controller AST after removing only the declared narrow repair nodes."""
    tree = ast.parse(source.decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeDurability":
            node.body = [
                child for child in node.body
                if not (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                        f"RuntimeDurability.{child.name}" in changed_symbols)
            ]
    return ast.dump(tree, include_attributes=False)


def _validate_controller_delta(before: bytes, after: bytes,
                               declared_symbols: object) -> list[str]:
    before_methods, after_methods = _runtime_durability_methods(before), _runtime_durability_methods(after)
    changed = {
        name for name in set(before_methods) | set(after_methods)
        if before_methods.get(name) != after_methods.get(name)
    }
    if (not isinstance(declared_symbols, list) or
            not all(isinstance(name, str) for name in declared_symbols) or
            declared_symbols != sorted(set(declared_symbols)) or
            set(declared_symbols) != changed or not changed):
        raise D44GateError("D44 controller declaration does not name its exact changed symbols")
    if not all(_allowed_controller_symbol(name) for name in changed):
        raise D44GateError("D44 controller change is not a permitted durability repair")
    # Repair methods may be added or modified but never deleted; a deletion can conceal an
    # operational guarantee removal while still satisfying a simple changed-symbol allowlist.
    if any(name not in after_methods for name in changed):
        raise D44GateError("D44 controller repair deletes a protected durability method")
    if _sanitized_controller(before, changed) != _sanitized_controller(after, changed):
        raise D44GateError("D44 controller contains a non-operational, training, or evaluation change")
    return sorted(changed)


def _validate_live_tree(root: pathlib.Path, successor_archive: pathlib.Path) -> dict[str, Any]:
    source = _archive(successor_archive)
    bindings: dict[str, str] = {}
    for name, payload in sorted(source.items()):
        pure = pathlib.PurePosixPath(name)
        live = root.joinpath(*pure.parts)
        if not live.is_file() or live.is_symlink() or live.read_bytes() != payload:
            raise D44GateError(f"D44 successor archive/live-tree mismatch: {name}")
        bindings[name] = hashlib.sha256(payload).hexdigest()
    return {"packaged_file_count": len(bindings),
            "packaged_files_sha256": canonical_sha256(bindings)}


def _validate_declaration(root: pathlib.Path, successor_sha: str) -> tuple[dict[str, Any], dict[str, str]]:
    declaration = _json(root / DECLARATION, "D44 operational durability declaration")
    expected_fields = {
        "schema", "status", "d43_predecessor_archive_sha256", "d43_predecessor_receipt",
        "successor_archive_sha256", "d44_decision", "allowed_changed_paths",
        "exact_changed_files", "operational_controller_changed_symbols",
    }
    if set(declaration) != expected_fields:
        raise D44GateError("D44 declaration field set mismatch")
    expected_d43_receipt = _record_binding(root, D43_RECEIPT)
    decision_path = root / D44_DECISION
    decision = {"path": D44_DECISION.as_posix(), "sha256": sha256_file(decision_path)} \
        if decision_path.is_file() else None
    if (declaration.get("schema") != DECLARATION_SCHEMA or declaration.get("status") != "FROZEN" or
            declaration.get("d43_predecessor_archive_sha256") != D43_PREDECESSOR_SHA256 or
            declaration.get("d43_predecessor_receipt") != expected_d43_receipt or
            declaration.get("successor_archive_sha256") != successor_sha or
            declaration.get("d44_decision") != decision or
            declaration.get("allowed_changed_paths") != sorted(ALLOWED_CHANGED_PATHS)):
        raise D44GateError("D44 declaration is stale, overbroad, or detached from D43")
    return declaration, decision


def _validate_delta(baseline_archive: pathlib.Path, successor_archive: pathlib.Path,
                    declaration: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(baseline_archive) != D43_PREDECESSOR_SHA256:
        raise D44GateError("D44 baseline is not the frozen D43 archive")
    successor_sha = sha256_file(successor_archive)
    if declaration.get("successor_archive_sha256") != successor_sha:
        raise D44GateError("D44 declaration has a stale successor archive")
    before, after = _archive(baseline_archive), _archive(successor_archive)
    changed = sorted(name for name in set(before) | set(after)
                     if before.get(name) != after.get(name))
    if set(changed) != ALLOWED_CHANGED_PATHS:
        unexpected = sorted(set(changed) - ALLOWED_CHANGED_PATHS)
        missing = sorted(ALLOWED_CHANGED_PATHS - set(changed))
        raise D44GateError("D44 archive delta is not exactly the declared operational repair" +
                           (f"; unexpected={unexpected}" if unexpected else "") +
                           (f"; missing={missing}" if missing else ""))
    records = declaration.get("exact_changed_files")
    if not isinstance(records, dict) or set(records) != set(changed):
        raise D44GateError("D44 declaration does not enumerate its exact archive delta")
    exact: dict[str, dict[str, str | None]] = {}
    for name in changed:
        expected = {
            "before_sha256": hashlib.sha256(before[name]).hexdigest() if name in before else None,
            "after_sha256": hashlib.sha256(after[name]).hexdigest() if name in after else None,
        }
        if records.get(name) != expected:
            raise D44GateError(f"D44 declared bytes are stale: {name}")
        exact[name] = expected

    # A path-only allowlist is insufficient because the controller owns training dispatch.  The
    # sanitized full-module AST forbids every non-durability change inside that mixed module.
    controller_symbols = _validate_controller_delta(
        before[CONTROLLER_PATH], after[CONTROLLER_PATH],
        declaration.get("operational_controller_changed_symbols"),
    )
    protected_prefixes = ("configs/", "data/", "manifests/", "src/", "upstream/")
    protected_exact = {
        "scripts/train_hmm.py", "scripts/run_hmm_matrix.py", "scripts/run_matrix.py",
        "scripts/evaluate_hmm_checkpoints.py", "scripts/evaluate_lurestar_checkpoints.py",
    }
    unchanged_protected = {
        name: hashlib.sha256(after[name]).hexdigest()
        for name in sorted(after)
        if name.startswith(protected_prefixes) or name in protected_exact
    }
    if not protected_exact <= set(unchanged_protected):
        raise D44GateError("D44 archive lacks protected scientific/training/evaluation authorities")
    return {
        "changed_paths": changed,
        "exact_changed_files": exact,
        "controller_changed_symbols": controller_symbols,
        "protected_surface_file_count": len(unchanged_protected),
        "protected_surface_sha256": canonical_sha256(unchanged_protected),
    }


def _validate_assurance(root: pathlib.Path, successor_sha: str) -> dict[str, Any]:
    test_path, review_path, incident_path = root / TEST_RECEIPT, root / REVIEW_RECEIPT, root / INCIDENT
    test, review = _json(test_path, "D44 full-suite receipt"), _json(review_path, "D44 review receipt")
    if (set(test) != {"schema", "source_sha256", "outcome", "exit_code", "tests_passed", "command"} or
            test.get("schema") != FULL_TEST_SCHEMA or test.get("source_sha256") != successor_sha or
            test.get("outcome") != "PASS" or test.get("exit_code") != 0 or
            isinstance(test.get("tests_passed"), bool) or not isinstance(test.get("tests_passed"), int) or
            test["tests_passed"] <= 0 or not isinstance(test.get("command"), list) or
            "pytest" not in " ".join(map(str, test["command"])) or "tests" not in test["command"]):
        raise D44GateError("D44 full-suite receipt is stale or nonpassing")
    report = root / str(review.get("report_path", ""))
    try:
        report.resolve().relative_to(root)
    except ValueError as exc:
        raise D44GateError("D44 review report escapes project root") from exc
    expected_review_fields = {
        "schema", "status", "source_sha256", "reviewer", "report_path", "report_sha256",
        "incident_attestation_sha256",
    }
    if (set(review) != expected_review_fields or review.get("schema") != REVIEW_SCHEMA or
            review.get("status") != "PASS" or review.get("source_sha256") != successor_sha or
            not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or
            not report.is_file() or review.get("report_sha256") != sha256_file(report) or
            not incident_path.is_file() or
            review.get("incident_attestation_sha256") != sha256_file(incident_path)):
        raise D44GateError("D44 independent operational review is stale or nonpassing")
    report_text = report.read_text(encoding="utf-8")
    for pattern in (
        r"(?i)D44", r"(?i)outcome[- ]blind[^\n]*PASS", r"(?i)P0\s*[=:]\s*0",
        r"(?i)P1\s*[=:]\s*0", r"(?i)training started\s*:\s*true",
        r"(?i)completed HMM training cells\s*:\s*10", r"(?i)pending HMM training cells\s*:\s*20",
        r"(?i)new HMM training cells\s*:\s*0", r"(?i)scientific evaluations started\s*:\s*false",
        r"(?i)evaluator invocations\s*:\s*0", r"(?i)scientific metrics inspected\s*:\s*false",
    ):
        if re.search(pattern, report_text) is None:
            raise D44GateError("D44 review lacks the required outcome-blind operational verdict")
    return {
        "full_suite": _binding(root, TEST_RECEIPT, FULL_TEST_SCHEMA),
        "independent_review": {
            "path": REVIEW_RECEIPT.as_posix(), "sha256": sha256_file(review_path),
            "schema": REVIEW_SCHEMA, "reviewer": review["reviewer"],
            "report_path": review["report_path"], "report_sha256": review["report_sha256"],
            "incident_attestation_sha256": review["incident_attestation_sha256"],
        },
    }


def _validate_incident(root: pathlib.Path, successor_sha: str,
                       review: Mapping[str, Any]) -> dict[str, str]:
    incident_path = root / INCIDENT
    incident = _json(incident_path, "D44 metadata-only incident attestation")
    review_binding = {
        "reviewer": review.get("reviewer"), "report_path": review.get("report_path"),
        "report_sha256": review.get("report_sha256"),
    }
    expected = {
        "schema": INCIDENT_SCHEMA, "status": "PASS",
        "d43_predecessor_archive_sha256": D43_PREDECESSOR_SHA256,
        "successor_archive_sha256": successor_sha,
        "training_started": True, "completed_hmm_training_cells": 10,
        "pending_hmm_training_cells": 20, "new_hmm_training_cells": 0,
        "scientific_evaluations_started": False, "evaluator_invocations": 0,
        "scientific_metrics_inspected": False,
        "statement": (
            "metadata-only operational incident; no new HMM cell, scientific evaluation, "
            "or scientific metric inspection occurred"
        ),
        "independent_review": review_binding,
        "attested_by": incident.get("attested_by"),
    }
    attestants = incident.get("attested_by")
    if (not isinstance(attestants, dict) or set(attestants) != {
            "research_controller", "independent_reviewer"} or
            not isinstance(attestants.get("research_controller"), str) or
            not attestants["research_controller"].strip() or
            attestants.get("independent_reviewer") != review_binding["reviewer"] or
            review.get("incident_attestation_sha256") != sha256_file(incident_path) or
            incident != expected):
        raise D44GateError("D44 metadata-only incident evidence is missing, forged, or outcome-visible")
    return _binding(root, INCIDENT, INCIDENT_SCHEMA)


def build_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                  successor_archive: pathlib.Path) -> dict[str, Any]:
    """Recompute the only valid D44 operational-continuation receipt."""
    root = root.resolve()
    predecessor_archive, successor_archive = predecessor_archive.resolve(), successor_archive.resolve()
    successor_sha = sha256_file(successor_archive)
    predecessor = _validate_d43_predecessor(root, predecessor_archive)
    declaration, decision = _validate_declaration(root, successor_sha)
    delta = _validate_delta(predecessor_archive, successor_archive, declaration)
    live_tree = _validate_live_tree(root, successor_archive)
    assurance = _validate_assurance(root, successor_sha)
    incident = _validate_incident(root, successor_sha, assurance["independent_review"])
    return {
        "schema": SCHEMA, "status": "PASS", "authorization": "OPERATIONAL_DURABILITY_GO",
        "archives": {
            "d43_predecessor": predecessor["archive"],
            "d44_operational_successor": {
                "path": str(successor_archive.relative_to(root)), "sha256": successor_sha,
            },
        },
        "d43_preserved_records": predecessor["records"],
        "declaration": _binding(root, DECLARATION, DECLARATION_SCHEMA),
        "decision": decision,
        "operational_delta": delta,
        "successor_live_tree_equivalence": live_tree,
        "successor_assurance": assurance,
        "metadata_only_incident": incident,
        "confirmatory_lifecycle": {
            "training_started": True, "completed_hmm_training_cells": 10,
            "pending_hmm_training_cells": 20, "new_hmm_training_cells": 0,
            "scientific_evaluations_started": False, "evaluator_invocations": 0,
            "scientific_metrics_inspected": False,
        },
        "claims": {
            "d43_source_and_records_preserved": True,
            "only_receipt_pinned_operational_durability_changed": True,
            "models_data_configs_training_and_evaluation_unchanged": True,
            "no_new_hmm_cells_or_outcome_access": True,
            "selector_and_clearance_integration_not_authorized": True,
        },
    }


def create_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                   successor_archive: pathlib.Path) -> pathlib.Path:
    document = build_receipt(root, predecessor_archive, successor_archive)
    target = root.resolve() / RECEIPT
    atomic_json(target, document)
    if _json(target, "new D44 receipt") != document:
        raise D44GateError("D44 receipt atomic write verification failed")
    return target


def validate_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                     successor_archive: pathlib.Path) -> dict[str, Any]:
    expected = build_receipt(root, predecessor_archive, successor_archive)
    actual = _json(root.resolve() / RECEIPT, "D44 receipt")
    if actual != expected:
        raise D44GateError("D44 receipt differs from recomputed evidence")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "validate"))
    parser.add_argument("--project-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--predecessor-archive", default=D43_PREDECESSOR_ARCHIVE.as_posix())
    parser.add_argument("--successor-archive", default=".agent_state/project.tar.gz")
    args = parser.parse_args()
    root = pathlib.Path(args.project_root).resolve()
    paths = [pathlib.Path(args.predecessor_archive), pathlib.Path(args.successor_archive)]
    paths = [path if path.is_absolute() else root / path for path in paths]
    if args.mode == "create":
        print(create_receipt(root, *paths))
    else:
        validate_receipt(root, *paths)
        print("D44_OPERATIONAL_DURABILITY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
