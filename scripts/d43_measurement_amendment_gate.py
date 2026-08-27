#!/usr/bin/env python
"""Fail-closed, outcome-blind D43 measurement-amendment continuation gate.

This gate does not read ``results/``, evaluation receipts, metrics, losses, or model outputs and
does not import an evaluator.  It proves a two-hop lineage instead:

1. the frozen pre-compute archive -> the already-reviewed D41 operational successor; and
2. that D41 successor -> an exact D43 archive whose changes are limited to the declared D42
   measurement implementation.

The ten recovered HMM checkpoint payloads remain predecessor-created, generation-pinned inputs.
They are never opened by this module; their prior host-deserialization proof and exact object
identities are revalidated from the D41 receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import tarfile
from typing import Any, Mapping


SCHEMA = "nextlat_forgetting/d43_measurement_amendment/1"
DECLARATION_SCHEMA = "nextlat_forgetting/d43_measurement_amendment_declaration/1"
NO_OUTCOME_SCHEMA = "nextlat_forgetting/d43_no_outcome_attestation/1"
CONTINUATION_SCHEMA = "nextlat_forgetting/d43_atomic_continuation_state/1"

ORIGINAL_PREDECESSOR_SHA256 = (
    "a962cdb94c865e16c2c7c86d5c18b9cc2d3bd301feeea12e42075751f52c9285"
)
D41_SUCCESSOR_SHA256 = (
    "ab4f4dd6a125a1c09e11e9f86a822c94358d84d92677bbc2e1671c653d4d6242"
)
D41_EQUIVALENCE_RECEIPT_SHA256 = (
    "4df54ebc0e6ce48aeffb85195768f550b084479790c99fc0eef7027824c2a668"
)
D41_RECOVERY_RECEIPT_SHA256 = (
    "d6147dbea8d84d4be535258c9240c875d935f98d4468d3db5e3962e9bbad1748"
)
D41_EXPECTED_CHANGED_PATHS = (
    "docs/DECISION_D41_RUNTIME_RECOVERY_AMENDMENT.md",
    "docs/RUNLOG.md",
    "scripts/colab_train_loop.py",
    "scripts/create_confirmatory_clearance.py",
    "scripts/d41_continuation_gate.py",
    "scripts/repair_hmm_exact_target_state.py",
    "scripts/restore_soft_deleted_objects.py",
    "scripts/run_hmm_matrix.py",
    "scripts/run_matrix.py",
    "source_snapshot/runtime_patch/runtime_patch.diff",
    "source_snapshot/runtime_patch/runtime_patch_receipt.json",
    "src/lurestar/durable_checkpoint.py",
    "tests/test_colab_train_loop.py",
    "tests/test_create_confirmatory_clearance.py",
    "tests/test_d41_continuation_gate.py",
    "tests/test_gcs_recovery_tools.py",
    "tests/test_resume.py",
    "tests/test_run_hmm_matrix.py",
    "tests/test_run_matrix.py",
    "tests/test_runtime_bootstrap.py",
)
INPUT_INVENTORY_SHA256 = (
    "33fbf4358b7c7def932fb96c1f4a5c04cb8713925dccbff5d385982e910a5c43"
)
INPUT_BUCKET = "nextlat-lurestar-project-flash-490419"
INPUT_PREFIX = f"lurestar/input_bundles/{INPUT_INVENTORY_SHA256}"
TARGET_STEP = 3_000
SEEDS = tuple(range(1234, 1239))
REGIMES = (
    "persistent_moderate", "fast_mixing_moderate", "persistent_high_aliasing",
)
ALL_HMM_JOB_IDS = tuple(
    f"{model}-seed{seed}-hmm-{regime}"
    for regime in REGIMES for model in ("gpt", "nextlat") for seed in SEEDS
)
EXACT_TEN_JOB_IDS = tuple(
    f"{model}-seed{seed}-hmm-persistent_moderate"
    for model in ("gpt", "nextlat") for seed in SEEDS
)

D42_DECISION = pathlib.Path("docs/DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md")
D43_DECISION = pathlib.Path("docs/DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md")
DECLARATION = pathlib.Path(".agent_state/d43-measurement-amendment-declaration.json")
NO_OUTCOME = pathlib.Path(".agent_state/d43-no-outcome-attestation.json")
CONTINUATION = pathlib.Path(".agent_state/d43-continuation-state.json")
RECEIPT = pathlib.Path(".agent_state/d43-measurement-amendment-receipt.json")
D41_RECEIPT = pathlib.Path(".agent_state/d41-exact-ten-recovery-receipt.json")
D41_EQUIVALENCE = pathlib.Path(".agent_state/d41-source-equivalence-receipt.json")
INPUT_RECEIPT = pathlib.Path(".agent_state/input-bundle-upload.json")
SEMANTIC_EVIDENCE = pathlib.Path(".agent_state/preregistration-evidence.json")
TEST_RECEIPT = pathlib.Path(".agent_state/confirmatory-test-receipt.json")
REVIEW_RECEIPT = pathlib.Path(".agent_state/confirmatory-review-receipt.json")

# Every byte that may differ on the D41-successor -> D43 hop is named here.  In particular,
# configs, corpora, manifests, model/training implementations, optimizer code, checkpoints and
# update-count ownership are absent.  A declaration must narrow this set to exact before/after
# hashes; membership in this set alone is never sufficient.
ALLOWED_MEASUREMENT_PATHS = frozenset({
    "docs/DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md",
    "docs/DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md",
    "docs/EXTRACTION.md",
    "scripts/build_preregistration_evidence.py",
    "scripts/d43_measurement_amendment_gate.py",
    "scripts/evaluate_hmm_checkpoints.py",
    "scripts/evaluate_lurestar_checkpoints.py",
    "scripts/extract_lurestar_evidence.py",
    "scripts/materialize_lurestar_evaluation.py",
    "scripts/run_hmm_matrix.py",
    "scripts/validate_preregistration.py",
    "src/hmm_geometry/aggregate.py",
    "src/lurestar/evaluate.py",
    "src/lurestar/representations.py",
    "tests/test_build_preregistration_evidence.py",
    "tests/test_d43_measurement_amendment_gate.py",
    "tests/test_hmm_family.py",
    "tests/test_lurestar_checkpoint_evaluator.py",
    "tests/test_lurestar_evidence_extractor.py",
    "tests/test_materialize_lurestar_evaluation.py",
    "tests/test_representations.py",
    "tests/test_run_hmm_matrix.py",
    "tests/test_validate_preregistration.py",
})

REQUIRED_MEASUREMENT_CHANGES = frozenset({
    "docs/DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md",
    "docs/DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md",
    "scripts/build_preregistration_evidence.py",
    "scripts/d43_measurement_amendment_gate.py",
    "scripts/evaluate_lurestar_checkpoints.py",
    "scripts/extract_lurestar_evidence.py",
    "scripts/materialize_lurestar_evaluation.py",
    "scripts/run_hmm_matrix.py",
    "scripts/validate_preregistration.py",
    "src/hmm_geometry/aggregate.py",
    "src/lurestar/evaluate.py",
    "src/lurestar/representations.py",
    "tests/test_d43_measurement_amendment_gate.py",
})

RUN_HMM_ALLOWED_CHANGED_SYMBOLS = frozenset({
    "_canonical_hmm_family_ids", "_verified_training_provenance",
    "preflight_hmm_evaluation_matrix", "run_hmm_evaluators",
})

EXPECTED_LURE_SCHEMAS = {
    "extraction_job": "nextlat_forgetting/lurestar_evidence_extraction_job/3",
    "extraction_progress": "nextlat_forgetting/lurestar_evidence_progress/1",
    "evidence_npz": "nextlat_forgetting/lurestar_evidence/4",
    "evidence_receipt": "nextlat_forgetting/lurestar_evidence/4",
    "evaluation_manifest": "nextlat_forgetting/lurestar_evaluation_manifest/4",
    "confirmatory_report": "nextlat_forgetting/lurestar_confirmatory_report/4",
    "evaluation_receipt": "nextlat_forgetting/lurestar_evaluation_receipt/4",
}
EXPECTED_HMM_AGGREGATE_SCHEMA = "nextlat_forgetting/hmm_cross_seed_aggregate/3"
SEMANTIC_EXECUTION_ROLE = "lurestar_schema_receipt"
SEMANTIC_EXECUTION_SCHEMA = "nextlat_forgetting/lurestar_schema_fixture/1"
SEMANTIC_EXECUTION_NAME = "g10-lurestar_schema_receipt.json"
REQUIRED_SEMANTIC_WITNESSES = frozenset({
    "npsi_formula_and_denominator", "paired_student_t_and_loso",
    "exact_sha_base_id_folds", "nested_h2_m0_delta_r2_identical_folds",
    "extractor_npsi_and_audit", "report_schema_and_required_statistics",
    "h1_four_state_classifier", "binary_h2_secondary_ceiling_status",
    "invalid_cells_terminal_schema", "terminal_required_fields_fail_closed",
    "non_equivalence_nulls_and_manipulation_failures",
})
_SHA = re.compile(r"[0-9a-f]{64}")


class D43GateError(RuntimeError):
    """The measurement amendment is stale, overbroad, outcome-visible, or non-atomic."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


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
        raise D43GateError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise D43GateError(f"{label} must be a JSON object")
    return value


def _binding(root: pathlib.Path, relative: pathlib.Path, schema: str | None = None) -> dict:
    path = root / relative
    value = _json(path, relative.as_posix())
    if schema is not None and value.get("schema") != schema:
        raise D43GateError(f"{relative} schema is stale")
    return {"path": relative.as_posix(), "sha256": sha256_file(path),
            "schema": value.get("schema")}


def _archive(path: pathlib.Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        archive = tarfile.open(path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise D43GateError(f"source archive is missing or invalid: {path}") from exc
    with archive:
        for member in archive.getmembers():
            pure = pathlib.PurePosixPath(member.name)
            if member.isdir():
                continue
            if (not member.isfile() or pure.is_absolute() or ".." in pure.parts or
                    member.name in files):
                raise D43GateError(f"unsafe, duplicate, or nonregular archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise D43GateError(f"unreadable archive member: {member.name}")
            files[member.name] = stream.read()
    return files


def _d41_module(root: pathlib.Path):
    path = root / "scripts/d41_continuation_gate.py"
    spec = importlib.util.spec_from_file_location("_d43_d41_validator", path)
    if spec is None or spec.loader is None:
        raise D43GateError("D41 validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_d41_hop(root: pathlib.Path, predecessor_path: pathlib.Path,
                      d41_successor_path: pathlib.Path) -> dict[str, Any]:
    if sha256_file(predecessor_path) != ORIGINAL_PREDECESSOR_SHA256:
        raise D43GateError("D43 original predecessor archive identity mismatch")
    if sha256_file(d41_successor_path) != D41_SUCCESSOR_SHA256:
        raise D43GateError("D43 D41-successor archive identity mismatch")
    d41 = _d41_module(root)
    receipt_path = root / D41_EQUIVALENCE
    receipt = _json(receipt_path, "D41 source-equivalence receipt")
    if (sha256_file(receipt_path) != D41_EQUIVALENCE_RECEIPT_SHA256 or
            receipt.get("schema") != d41.EQUIVALENCE_SCHEMA or receipt.get("status") != "PASS" or
            receipt.get("scientific_metrics_inspected") is not False or
            receipt.get("confirmatory_lifecycle") != {
                "compute_started": True, "scientific_evaluations_inspected": False,
            }):
        raise D43GateError("D41 source-equivalence receipt is stale or outcome-visible")
    before, after = _archive(predecessor_path), _archive(d41_successor_path)
    changed = sorted(set(before) | set(after))
    changed = [name for name in changed if before.get(name) != after.get(name)]
    if changed != list(D41_EXPECTED_CHANGED_PATHS) or receipt.get("changed_paths") != changed:
        raise D43GateError("D41 receipt differs from recomputed exact archive delta")
    if any(d41._is_scientific_path(name) for name in changed):
        raise D43GateError("D41 hop mutated its protected scientific surface")
    runner = "scripts/run_hmm_matrix.py"
    if d41._ast_projection(before[runner]) != d41._ast_projection(after[runner]):
        raise D43GateError("D41 hop mutated HMM scientific runner symbols")
    mixed = d41._mixed_module_projection(before, after)
    projection = {
        "exact_byte_files": {
            name: hashlib.sha256(after[name]).hexdigest()
            for name in sorted(after) if d41._is_scientific_path(name)
        },
        "hmm_runner_ast": d41._ast_projection(after[runner]),
        "mixed_module_ast": mixed,
    }
    if (receipt.get("predecessor_archive", {}).get("sha256") != ORIGINAL_PREDECESSOR_SHA256 or
            receipt.get("successor_archive", {}).get("sha256") != D41_SUCCESSOR_SHA256 or
            receipt.get("scientific_projection") != projection or
            receipt.get("scientific_projection_sha256") != canonical_sha256(projection) or
            not all(receipt.get("equivalence_claims", {}).values())):
        raise D43GateError("D41 source-equivalence semantic projection is stale")
    return {"path": D41_EQUIVALENCE.as_posix(), "sha256": sha256_file(receipt_path),
            "schema": receipt["schema"]}


def _symbol_projection(source: bytes) -> dict[str, str]:
    import ast
    tree = ast.parse(source.decode("utf-8"))
    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            values[node.name] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    values[f"{node.name}.{child.name}"] = hashlib.sha256(
                        ast.dump(child, include_attributes=False).encode()).hexdigest()
    return values


def _completion_summary_import_count(tree: Any) -> int:
    """Count the one reviewed import needed by the new completion preflight."""
    import ast
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "run_matrix"
        for alias in node.names
        if alias.name == "COMPLETION_SUMMARY" and alias.asname is None
    )


def _sanitized_hmm_module(source: bytes, *, remove_completion_import: bool) -> str:
    """Project the complete mixed orchestrator, excluding only reviewed evaluation nodes.

    Unlike the former symbol-only projection, this retains every import, assignment, class,
    top-level expression/conditional, main entrypoint, and training function.  The sole import
    exception is the explicitly reviewed ``COMPLETION_SUMMARY`` dependency used by the new
    evaluation preflight.
    """
    import ast
    tree = ast.parse(source.decode("utf-8"))
    body = []
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                node.name in RUN_HMM_ALLOWED_CHANGED_SYMBOLS):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "run_matrix":
            names = [
                alias for alias in node.names
                if not (remove_completion_import and alias.name == "COMPLETION_SUMMARY" and
                        alias.asname is None)
            ]
            if names:
                node = ast.ImportFrom(module=node.module, names=names, level=node.level)
            else:
                continue
        body.append(node)
    tree.body = body
    return ast.dump(tree, include_attributes=False)


def _validate_hmm_runner_delta(before: bytes, after: bytes) -> list[str]:
    """Require a full-module-identical runner outside four reviewed evaluation functions."""
    import ast
    before_tree, after_tree = ast.parse(before.decode("utf-8")), ast.parse(after.decode("utf-8"))
    before_symbols, after_symbols = _symbol_projection(before), _symbol_projection(after)
    changed_symbols = {
        name for name in set(before_symbols) | set(after_symbols)
        if before_symbols.get(name) != after_symbols.get(name)
    }
    if changed_symbols != RUN_HMM_ALLOWED_CHANGED_SYMBOLS:
        raise D43GateError("D43 HMM orchestrator delta is not evaluation-only: " +
                           ", ".join(sorted(changed_symbols)))
    if (_completion_summary_import_count(before_tree) != 0 or
            _completion_summary_import_count(after_tree) != 1):
        raise D43GateError("D43 HMM orchestrator completion import delta is not exact")
    if (_sanitized_hmm_module(before, remove_completion_import=False) !=
            _sanitized_hmm_module(after, remove_completion_import=True)):
        raise D43GateError(
            "D43 HMM orchestrator contains a non-evaluation top-level or training change")
    return sorted(changed_symbols)


def _validate_successor_live_tree(root: pathlib.Path, successor_path: pathlib.Path) -> dict[str, Any]:
    """Bind every packaged source byte used by live tests/witnesses to the successor archive."""
    packaged = _archive(successor_path)
    bindings: dict[str, str] = {}
    for name, payload in sorted(packaged.items()):
        pure = pathlib.PurePosixPath(name)
        live = root.joinpath(*pure.parts)
        if (not live.is_file() or live.is_symlink() or live.read_bytes() != payload):
            raise D43GateError(f"D43 successor archive/live-tree mismatch: {name}")
        bindings[name] = hashlib.sha256(payload).hexdigest()
    return {
        "packaged_file_count": len(bindings),
        "packaged_files_sha256": canonical_sha256(bindings),
    }


def _validate_measurement_delta(baseline_path: pathlib.Path, successor_path: pathlib.Path,
                                declaration: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(baseline_path) != D41_SUCCESSOR_SHA256:
        raise D43GateError("measurement baseline is not the exact D41 successor")
    successor_sha = sha256_file(successor_path)
    if declaration.get("successor_archive_sha256") != successor_sha:
        raise D43GateError("D43 declaration has a stale successor archive")
    before, after = _archive(baseline_path), _archive(successor_path)
    changed = sorted(name for name in set(before) | set(after)
                     if before.get(name) != after.get(name))
    if not set(changed) <= ALLOWED_MEASUREMENT_PATHS:
        raise D43GateError("D43 contains an unenumerated source change: " + ", ".join(
            sorted(set(changed) - ALLOWED_MEASUREMENT_PATHS)))
    if not REQUIRED_MEASUREMENT_CHANGES <= set(changed):
        raise D43GateError("D43 required measurement changes are missing: " + ", ".join(
            sorted(REQUIRED_MEASUREMENT_CHANGES - set(changed))))
    records = declaration.get("exact_changed_files")
    if not isinstance(records, dict) or set(records) != set(changed):
        raise D43GateError("D43 declaration does not enumerate the exact changed-path set")
    exact: dict[str, dict[str, Any]] = {}
    for name in changed:
        expected = {
            "before_sha256": hashlib.sha256(before[name]).hexdigest() if name in before else None,
            "after_sha256": hashlib.sha256(after[name]).hexdigest() if name in after else None,
        }
        if records.get(name) != expected:
            raise D43GateError(f"D43 declared bytes are stale for allowed path: {name}")
        exact[name] = expected

    # The HMM orchestrator owns both training and evaluation.  Only these four evaluation/preflight
    # nodes may differ; constants, matrix construction, launcher command and training phase remain
    # mechanically identical.
    runner = "scripts/run_hmm_matrix.py"
    changed_symbols = _validate_hmm_runner_delta(before[runner], after[runner])

    unchanged = {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(after.items())
                 if name not in ALLOWED_MEASUREMENT_PATHS}
    critical = {name: digest for name, digest in unchanged.items() if (
        name.startswith(("configs/", "data/", "manifests/", "upstream/")) or
        name in {"scripts/train_hmm.py", "scripts/run_matrix.py", "scripts/runtime_bootstrap.py",
                 "scripts/launch_train.sh", "src/lurestar/adaptation.py",
                 "src/lurestar/durable_checkpoint.py"}
    )}
    required_critical = {"scripts/train_hmm.py", "scripts/run_matrix.py",
                         "scripts/runtime_bootstrap.py", "src/lurestar/durable_checkpoint.py"}
    if not required_critical <= set(critical):
        raise D43GateError("D43 archive lacks required immutable training authorities")
    return {
        "changed_paths": changed,
        "exact_changed_files": exact,
        "immutable_nonmeasurement_files_sha256": canonical_sha256(unchanged),
        "immutable_nonmeasurement_file_count": len(unchanged),
        "immutable_training_surface_sha256": canonical_sha256(critical),
        "immutable_training_surface_file_count": len(critical),
        "hmm_orchestrator_changed_symbols": changed_symbols,
    }


def _validate_declaration(root: pathlib.Path, successor_sha: str) -> tuple[dict, dict]:
    declaration = _json(root / DECLARATION, "D43 amendment declaration")
    if set(declaration) != {
        "schema", "status", "original_predecessor_sha256", "d41_successor_sha256",
        "successor_archive_sha256", "d42_decision", "d43_decision",
        "allowed_changed_paths", "exact_changed_files", "confirmatory_lifecycle",
    }:
        raise D43GateError("D43 amendment declaration field set mismatch")
    lifecycle = {
        "training_started": True, "completed_hmm_training_cells": 10,
        "total_hmm_training_cells": 30, "scientific_evaluations_inspected": False,
    }
    if (declaration.get("schema") != DECLARATION_SCHEMA or declaration.get("status") != "FROZEN" or
            declaration.get("original_predecessor_sha256") != ORIGINAL_PREDECESSOR_SHA256 or
            declaration.get("d41_successor_sha256") != D41_SUCCESSOR_SHA256 or
            declaration.get("successor_archive_sha256") != successor_sha or
            declaration.get("allowed_changed_paths") != sorted(ALLOWED_MEASUREMENT_PATHS) or
            declaration.get("confirmatory_lifecycle") != lifecycle):
        raise D43GateError("D43 amendment declaration is stale, overbroad, or lifecycle-false")
    decisions = {}
    for key, relative in (("d42_decision", D42_DECISION), ("d43_decision", D43_DECISION)):
        path = root / relative
        expected = {"path": relative.as_posix(), "sha256": sha256_file(path)} if path.is_file() else None
        if declaration.get(key) != expected:
            raise D43GateError(f"declared amendment file hash mismatch: {relative}")
        decisions[key] = expected
    return declaration, decisions


def _validate_input_inventory(root: pathlib.Path) -> dict[str, Any]:
    inventory = root / "manifests/manifest_inventory.sha256"
    receipt_path = root / INPUT_RECEIPT
    if not inventory.is_file() or sha256_file(inventory) != INPUT_INVENTORY_SHA256:
        raise D43GateError("frozen input inventory identity changed")
    receipt = _json(receipt_path, "input-bundle upload receipt")
    if (set(receipt) != {"schema", "status", "bucket", "bundle_prefix",
                         "input_bundle_sha256", "object_count", "objects", "commit"} or
            receipt.get("schema") != "nextlat_forgetting/input_bundle_upload/1" or
            receipt.get("status") != "COMPLETE" or
            receipt.get("bucket") != INPUT_BUCKET or receipt.get("bundle_prefix") != INPUT_PREFIX or
            receipt.get("input_bundle_sha256") != INPUT_INVENTORY_SHA256):
        raise D43GateError("input-bundle receipt is stale")
    records: dict[str, tuple[str, int]] = {}
    for number, line in enumerate(inventory.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise D43GateError(f"malformed input inventory line {number}")
        digest, relative = match.groups()
        pure = pathlib.PurePosixPath(relative)
        path = root.joinpath(*pure.parts)
        if (pure.is_absolute() or ".." in pure.parts or relative in records or
                not path.is_file() or path.is_symlink() or sha256_file(path) != digest):
            raise D43GateError(f"frozen input path/hash mismatch: {relative}")
        records[relative] = (digest, path.stat().st_size)
    objects = receipt.get("objects")
    if (not isinstance(objects, list) or len(objects) != len(records) or
            receipt.get("object_count") != len(records)):
        raise D43GateError("input-bundle object inventory is incomplete")
    seen = set()
    for item in objects:
        if (not isinstance(item, dict) or set(item) != {
                "local_path", "name", "generation", "size_bytes", "sha256"} or
                item.get("local_path") not in records):
            raise D43GateError("input-bundle object is unrecognized")
        relative = item["local_path"]
        digest, size = records[relative]
        suffix = ("corpus/" + relative.removeprefix("data/")
                  if relative.startswith("data/") else relative)
        if (relative in seen or item.get("name") != f"{INPUT_PREFIX}/{suffix}" or
                item.get("sha256") != digest or item.get("size_bytes") != size or
                not str(item.get("generation", "")).isdigit()):
            raise D43GateError(f"input-bundle object provenance mismatch: {relative}")
        seen.add(relative)
    commit = receipt.get("commit")
    if (not isinstance(commit, dict) or set(commit) != {
            "local_path", "name", "generation", "size_bytes", "sha256"} or
            commit.get("sha256") != INPUT_INVENTORY_SHA256 or
            commit.get("local_path") != "manifests/manifest_inventory.sha256" or
            commit.get("name") != f"{INPUT_PREFIX}/manifests/manifest_inventory.sha256" or
            commit.get("size_bytes") != inventory.stat().st_size or
            not str(commit.get("generation", "")).isdigit()):
        raise D43GateError("input-bundle atomic commit provenance mismatch")
    return {"path": INPUT_RECEIPT.as_posix(), "sha256": sha256_file(receipt_path),
            "inventory_sha256": INPUT_INVENTORY_SHA256, "object_count": len(records),
            "commit_generation": str(commit["generation"])}


def _checkpoint_projection(root: pathlib.Path, successor_sha: str) -> dict[str, Any]:
    d41 = _d41_module(root)
    path = root / D41_RECEIPT
    receipt = _json(path, "D41 exact-ten recovery receipt")
    if sha256_file(path) != D41_RECOVERY_RECEIPT_SHA256:
        raise D43GateError("D41 exact-ten recovery receipt is not the frozen exact receipt")
    try:
        d41.validate_exact_ten_document(receipt)
    except Exception as exc:
        raise D43GateError(f"D41 exact-ten recovery receipt is invalid: {exc}") from exc
    jobs = []
    for item in receipt["jobs"]:
        jobs.append({
            "job_id": item["job_id"],
            "created_under_predecessor_source_sha256": item["predecessor_source_sha256"],
            "consumed_read_only_by_successor_source_sha256": successor_sha,
            "target_step": item["target_step"],
            "state_object": item["state_object"],
            "checkpoint_object": item["checkpoint_object"],
            "sidecar_object": item["sidecar_object"],
            "checkpoint_semantics": item["checkpoint_semantics"],
            "verification": item["verification"],
        })
    if [item["job_id"] for item in jobs] != list(EXACT_TEN_JOB_IDS):
        raise D43GateError("D43 checkpoint lineage is not exact-ten canonical")
    return {
        "receipt": {"path": D41_RECEIPT.as_posix(), "sha256": sha256_file(path),
                    "schema": receipt["schema"]},
        "predecessor_to_successor_provenance": jobs,
        "provenance_sha256": canonical_sha256(jobs),
        "reuse_policy": "read_only_exact_step_3000_no_retraining_no_evaluation",
    }


def _recompute_semantic_witnesses(root: pathlib.Path) -> dict[str, Any]:
    validator_path = root / "scripts/validate_preregistration.py"
    spec = importlib.util.spec_from_file_location("_d43_semantic_validator", validator_path)
    if spec is None or spec.loader is None:
        raise D43GateError("D43 semantic witness validator cannot be loaded")
    validator = importlib.util.module_from_spec(spec)
    try:
        sys.path.insert(0, str(root / "src"))
        spec.loader.exec_module(validator)
        result = validator.derive_lurestar_semantic_witnesses(root)
    except Exception as exc:
        raise D43GateError(f"D43 semantic witnesses cannot be recomputed: {exc}") from exc
    finally:
        if sys.path and sys.path[0] == str(root / "src"):
            sys.path.pop(0)
    if not isinstance(result, dict):
        raise D43GateError("D43 semantic witness validator returned a non-object")
    return result


def _validate_semantic_evidence(root: pathlib.Path, successor_sha: str) -> dict[str, Any]:
    path = root / SEMANTIC_EVIDENCE
    evidence = _json(path, "D43 semantic evidence")
    gate = evidence.get("gates", {}).get("10") if isinstance(evidence.get("gates"), dict) else None
    checks = gate.get("checks") if isinstance(gate, dict) else None
    if not isinstance(checks, dict):
        raise D43GateError("D43 gate-10 semantic evidence is missing")
    if checks.get("lurestar_schema_contract") != EXPECTED_LURE_SCHEMAS:
        raise D43GateError("D43 Lure-Star semantic schemas are stale")
    schemas = checks.get("schemas")
    required_schemas = set(EXPECTED_LURE_SCHEMAS.values()) | {EXPECTED_HMM_AGGREGATE_SCHEMA}
    if not isinstance(schemas, list) or not required_schemas <= set(schemas):
        raise D43GateError("D43 semantic evidence lacks current measurement schemas")
    for key in ("h1_h2_metrics_preserved", "missing_metrics_refused", "extra_metrics_refused",
                "invalid_cells_emitted", "nulls_emitted", "manipulation_failures_emitted",
                "multiplicity_fields_emitted"):
        if checks.get(key) is not True:
            raise D43GateError(f"D43 semantic evidence is nonpassing: {key}")
    artifacts = gate.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise D43GateError("D43 semantic evidence artifacts are missing")
    expected_semantic_path = (
        root / ".agent_state" / "preregistration" / successor_sha / SEMANTIC_EXECUTION_NAME
    ).resolve()
    semantic_artifacts: list[dict[str, Any]] = []
    for record in artifacts:
        if (not isinstance(record, dict) or set(record) != {"path", "role", "schema", "sha256"} or
                not isinstance(record.get("role"), str) or
                not isinstance(record.get("schema"), str) or
                not _SHA.fullmatch(str(record.get("sha256", "")))):
            raise D43GateError("D43 semantic artifact binding is malformed")
        artifact = pathlib.Path(str(record.get("path", "")))
        if not artifact.is_absolute():
            artifact = root / artifact
        try:
            artifact.resolve().relative_to(root)
        except ValueError as exc:
            raise D43GateError("D43 semantic artifact escapes project root") from exc
        artifact = artifact.resolve()
        value = _json(artifact, "D43 semantic artifact")
        if (sha256_file(artifact) != record["sha256"] or value.get("status") != "PASS" or
                value.get("source_archive_sha256") != successor_sha or
                value.get("schema") != record["schema"] or value.get("role") != record["role"]):
            raise D43GateError("D43 semantic artifact is stale or nonpassing")
        if (record["role"] == SEMANTIC_EXECUTION_ROLE or
                record["schema"] == SEMANTIC_EXECUTION_SCHEMA or
                artifact == expected_semantic_path):
            semantic_artifacts.append({"path": artifact, "record": record, "value": value})
    if (len(semantic_artifacts) != 1 or
            semantic_artifacts[0]["path"] != expected_semantic_path or
            semantic_artifacts[0]["record"]["role"] != SEMANTIC_EXECUTION_ROLE or
            semantic_artifacts[0]["record"]["schema"] != SEMANTIC_EXECUTION_SCHEMA):
        raise D43GateError("D43 exact semantic execution artifact is missing or ambiguous")
    semantic_value = semantic_artifacts[0]["value"]
    payload = semantic_value.get("payload")
    if (not isinstance(payload, dict) or
            semantic_value.get("payload_sha256") != canonical_sha256(payload)):
        raise D43GateError("D43 semantic execution payload binding is malformed")
    witnesses = payload.get("semantic_witnesses")
    recomputed = _recompute_semantic_witnesses(root)
    if (not isinstance(witnesses, dict) or
            not REQUIRED_SEMANTIC_WITNESSES <= set(witnesses) or
            set(witnesses) != set(recomputed)):
        raise D43GateError("D43 semantic execution witnesses are incomplete or noncanonical")
    if witnesses != recomputed:
        raise D43GateError("D43 semantic execution witnesses differ from source/test recomputation")
    return {"path": SEMANTIC_EVIDENCE.as_posix(), "sha256": sha256_file(path),
            "semantic_witnesses_sha256": canonical_sha256(witnesses),
            "schemas": sorted(required_schemas)}


def _validate_tests_and_review(root: pathlib.Path, successor_sha: str) -> dict[str, Any]:
    test_path, review_path = root / TEST_RECEIPT, root / REVIEW_RECEIPT
    no_outcome_path = root / NO_OUTCOME
    test, review = _json(test_path, "D43 full-suite receipt"), _json(
        review_path, "D43 independent review receipt")
    if (test.get("schema") != "nextlat_forgetting/full_test_suite_receipt/1" or
            test.get("source_sha256") != successor_sha or test.get("outcome") != "PASS" or
            test.get("exit_code") != 0 or isinstance(test.get("tests_passed"), bool) or
            not isinstance(test.get("tests_passed"), int) or test["tests_passed"] <= 0 or
            "pytest" not in " ".join(map(str, test.get("command", []))) or
            "tests" not in test.get("command", [])):
        raise D43GateError("D43 full-suite receipt is stale or nonpassing")
    report = root / str(review.get("report_path", ""))
    try:
        report.resolve().relative_to(root)
    except ValueError as exc:
        raise D43GateError("D43 review report escapes project root") from exc
    if (review.get("schema") != "nextlat_forgetting/independent_scientific_review/1" or
            review.get("source_sha256") != successor_sha or review.get("verdict") != "PASS" or
            not str(review.get("reviewer", "")).strip() or not report.is_file() or
            review.get("report_sha256") != sha256_file(report) or
            not no_outcome_path.is_file() or
            review.get("no_outcome_attestation_sha256") != sha256_file(no_outcome_path)):
        raise D43GateError("D43 independent review is stale or nonpassing")
    text = report.read_text(encoding="utf-8")
    for pattern in (
        r"(?i)D43", r"(?i)outcome[- ]blind[^\n]*PASS",
        r"(?i)P0\s*[=:]\s*0", r"(?i)P1\s*[=:]\s*0",
        r"(?i)training started\s*:\s*true",
        r"(?i)completed HMM training cells\s*:\s*10",
        r"(?i)scientific evaluations started\s*:\s*false",
        r"(?i)evaluator invocations\s*:\s*0",
        r"(?i)scientific metrics inspected\s*:\s*false",
    ):
        if re.search(pattern, text) is None:
            raise D43GateError("D43 independent review lacks required outcome-blind/P0/P1 verdict")
    return {
        "full_suite": {"path": TEST_RECEIPT.as_posix(), "sha256": sha256_file(test_path),
                       "schema": test["schema"], "tests_passed": test["tests_passed"]},
        "independent_review": {"path": REVIEW_RECEIPT.as_posix(),
                               "sha256": sha256_file(review_path), "schema": review["schema"],
                               "reviewer": review["reviewer"],
                               "report_path": str(review["report_path"]),
                               "report_sha256": review["report_sha256"],
                               "no_outcome_attestation_sha256":
                                   review["no_outcome_attestation_sha256"]},
    }


def _validate_no_outcome_and_continuation(root: pathlib.Path, successor_sha: str,
                                          independent_review: Mapping[str, Any]) -> dict[str, Any]:
    no_outcome = _json(root / NO_OUTCOME, "D43 no-outcome attestation")
    reviewer = independent_review.get("reviewer")
    review_binding = {
        "reviewer": reviewer,
        "report_path": independent_review.get("report_path"),
        "report_sha256": independent_review.get("report_sha256"),
    }
    expected_no_outcome = {
        "schema": NO_OUTCOME_SCHEMA, "status": "PASS",
        "original_predecessor_sha256": ORIGINAL_PREDECESSOR_SHA256,
        "successor_archive_sha256": successor_sha,
        "training_started": True, "completed_hmm_training_cells": 10,
        "scientific_evaluations_started": False, "scientific_metrics_inspected": False,
        "evaluator_invocations": 0,
        "statement": "ten HMM cells trained; no scientific evaluation was run, opened, or interpreted",
        "independent_review": review_binding,
        "attested_by": no_outcome.get("attested_by"),
    }
    attestants = no_outcome.get("attested_by")
    if (not isinstance(attestants, dict) or set(attestants) != {
            "research_controller", "independent_reviewer"} or
            not str(attestants.get("research_controller", "")).strip() or
            attestants.get("independent_reviewer") != reviewer or
            independent_review.get("no_outcome_attestation_sha256") !=
                sha256_file(root / NO_OUTCOME) or
            no_outcome != expected_no_outcome):
        raise D43GateError("D43 no-outcome evidence is missing, forged, or lifecycle-false")

    state = _json(root / CONTINUATION, "D43 atomic continuation state")
    pending = [job for job in ALL_HMM_JOB_IDS if job not in EXACT_TEN_JOB_IDS]
    expected_state = {
        "schema": CONTINUATION_SCHEMA, "status": "ATOMIC_READY_FOR_TRAINING_CONTINUATION",
        "source_archive_sha256": successor_sha,
        "original_predecessor_sha256": ORIGINAL_PREDECESSOR_SHA256,
        "training_started": True, "completed_job_ids": list(EXACT_TEN_JOB_IDS),
        "pending_job_ids": pending, "evaluated_job_ids": [],
        "scientific_evaluations_started": False, "scientific_metrics_inspected": False,
        "evaluator_invocations": 0, "total_hmm_training_cells": 30,
        "next_phase": "train_remaining_20_then_atomic_all_30_preflight_before_evaluation",
    }
    if state != expected_state:
        raise D43GateError("D43 continuation state is non-atomic, incomplete, or outcome-visible")
    if set(state["completed_job_ids"]) | set(state["pending_job_ids"]) != set(ALL_HMM_JOB_IDS):
        raise D43GateError("D43 continuation partition does not cover the frozen 30-cell matrix")
    return {
        "no_outcome_attestation": _binding(root, NO_OUTCOME, NO_OUTCOME_SCHEMA),
        "atomic_continuation_state": _binding(root, CONTINUATION, CONTINUATION_SCHEMA),
    }


def build_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                  d41_successor_archive: pathlib.Path, successor_archive: pathlib.Path) -> dict:
    """Validate all evidence and return the unique truthful D43 receipt document."""
    root = root.resolve()
    predecessor_archive, d41_successor_archive, successor_archive = (
        path.resolve() for path in (predecessor_archive, d41_successor_archive, successor_archive)
    )
    successor_sha = sha256_file(successor_archive)
    declaration, decisions = _validate_declaration(root, successor_sha)
    d41_hop = _validate_d41_hop(root, predecessor_archive, d41_successor_archive)
    delta = _validate_measurement_delta(d41_successor_archive, successor_archive, declaration)
    live_tree = _validate_successor_live_tree(root, successor_archive)
    inputs = _validate_input_inventory(root)
    checkpoints = _checkpoint_projection(root, successor_sha)
    semantics = _validate_semantic_evidence(root, successor_sha)
    assurance = _validate_tests_and_review(root, successor_sha)
    lifecycle = _validate_no_outcome_and_continuation(
        root, successor_sha, assurance["independent_review"])
    return {
        "schema": SCHEMA, "status": "PASS", "authorization": "MEASUREMENT_AMENDMENT_GO",
        "confirmatory_lifecycle": {
            "training_started": True, "completed_hmm_training_cells": 10,
            "total_hmm_training_cells": 30, "scientific_evaluations_started": False,
            "scientific_evaluations_inspected": False,
        },
        "archives": {
            "original_predecessor": {"path": str(predecessor_archive.relative_to(root)),
                                     "sha256": ORIGINAL_PREDECESSOR_SHA256},
            "d41_operational_successor": {"path": str(d41_successor_archive.relative_to(root)),
                                          "sha256": D41_SUCCESSOR_SHA256},
            "d43_measurement_successor": {"path": str(successor_archive.relative_to(root)),
                                          "sha256": successor_sha},
        },
        "decisions": decisions,
        "declaration": _binding(root, DECLARATION, DECLARATION_SCHEMA),
        "d41_source_equivalence": d41_hop,
        "measurement_delta": delta,
        "successor_live_tree_equivalence": live_tree,
        "frozen_input_inventory": inputs,
        "exact_ten_checkpoint_lineage": checkpoints,
        "semantic_evidence": semantics,
        "successor_assurance": assurance,
        "outcome_blind_atomic_continuation": lifecycle,
        "claims": {
            "training_objectives_unchanged": True, "model_code_unchanged": True,
            "optimizer_and_update_counts_unchanged": True, "configs_unchanged": True,
            "frozen_data_and_manifests_unchanged": True,
            "ten_recovered_checkpoint_payloads_unchanged": True,
            "only_declared_measurement_changes_permitted": True,
        },
    }


def create_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                   d41_successor_archive: pathlib.Path, successor_archive: pathlib.Path) -> pathlib.Path:
    document = build_receipt(root, predecessor_archive, d41_successor_archive, successor_archive)
    target = root.resolve() / RECEIPT
    atomic_json(target, document)
    if _json(target, "new D43 receipt") != document:
        target.unlink(missing_ok=True)
        raise D43GateError("D43 receipt atomic write verification failed")
    return target


def validate_receipt(root: pathlib.Path, predecessor_archive: pathlib.Path,
                     d41_successor_archive: pathlib.Path, successor_archive: pathlib.Path) -> dict:
    expected = build_receipt(root, predecessor_archive, d41_successor_archive, successor_archive)
    actual = _json(root.resolve() / RECEIPT, "D43 receipt")
    if actual != expected:
        raise D43GateError("D43 receipt differs from recomputed evidence")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "validate"))
    parser.add_argument("--project-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--predecessor-archive", default=".agent_state/project-predecessor-d41.tar.gz")
    parser.add_argument("--d41-successor-archive",
                        default=".agent_state/project-d41-operational-baseline.tar.gz")
    parser.add_argument("--successor-archive", default=".agent_state/project.tar.gz")
    args = parser.parse_args()
    root = pathlib.Path(args.project_root).resolve()
    paths = []
    for value in (args.predecessor_archive, args.d41_successor_archive, args.successor_archive):
        path = pathlib.Path(value)
        paths.append(path if path.is_absolute() else root / path)
    if args.mode == "create":
        print(create_receipt(root, *paths))
    else:
        validate_receipt(root, *paths)
        print("D43_MEASUREMENT_AMENDMENT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
