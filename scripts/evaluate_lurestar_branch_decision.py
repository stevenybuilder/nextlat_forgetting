#!/usr/bin/env python3
"""Fail-closed prospective H1-BD-1 evaluation at the first branch decision (h63).

This is intentionally a new analysis, not a changed legacy-H1 evaluator.  It consumes only
the source-bound h63 arrays already emitted by the frozen base-only extractor and refuses a
partial matrix.  It never calls the legacy H1 evaluator, so h62 results cannot influence this
new analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any, Mapping

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar import evaluate as E  # noqa: E402
from lurestar.durable_checkpoint import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


SCHEMA = "nextlat_forgetting/h1_branch_decision_report/1"
RECEIPT_SCHEMA = "nextlat_forgetting/h1_branch_decision_receipt/1"
DECLARATION_SCHEMA = "nextlat_forgetting/h1_branch_decision_analysis_declaration/1"
FREEZE_RECEIPT_SCHEMA = "nextlat_forgetting/h1_branch_decision_freeze_receipt/1"
ANALYSIS_ID = "H1-BD-1"
CANONICAL_SEEDS = (1234, 1235, 1236, 1237, 1238)
DECLARATION_PATH = _REPO / "manifests/h1_branch_decision/ANALYSIS_DECLARATION.json"
FREEZE_RECEIPT_PATH = _REPO / "manifests/h1_branch_decision/PRE_OUTCOME_FREEZE_RECEIPT.json"

LEGACY_EVALUATOR_PATH = _REPO / "scripts/evaluate_lurestar_checkpoints.py"
_legacy_spec = importlib.util.spec_from_file_location("_h1bd_legacy_evaluator", LEGACY_EVALUATOR_PATH)
if _legacy_spec is None or _legacy_spec.loader is None:  # pragma: no cover - installation error
    raise RuntimeError("cannot load the frozen Lure-Star evidence validator")
L = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(L)


class BranchDecisionRefused(RuntimeError):
    """A D46 source, design, or evidence binding is incomplete or stale."""


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchDecisionRefused(f"{label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise BranchDecisionRefused(f"{label} must be a JSON object")
    return value


def _record(root: pathlib.Path, relative: str) -> dict[str, str]:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise BranchDecisionRefused(f"freeze source is missing or unsafe: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def _validate_freeze() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Verify the D46 pre-outcome declaration and all source bytes it froze."""
    declaration = _json(DECLARATION_PATH, "H1-BD declaration")
    required_declaration = {
        "schema", "analysis_id", "status", "outcomes_inspected_before_freeze",
        "h1_legacy_control", "h1bd_role", "branch_decision_extraction_index",
        "population", "co_primary_metrics", "metric_name_mapping", "contrast", "decision_rule", "npsi_role",
    }
    if (set(declaration) != required_declaration or
            declaration.get("schema") != DECLARATION_SCHEMA or
            declaration.get("analysis_id") != ANALYSIS_ID or
            declaration.get("status") != "FROZEN_PRE_OUTCOME" or
            declaration.get("outcomes_inspected_before_freeze") is not False or
            declaration.get("branch_decision_extraction_index") != 63 or
            declaration.get("h1bd_role") != "new_prospective_mechanistic_confirmatory_analysis" or
            declaration.get("contrast") != "nextlat_minus_bst" or
            declaration.get("npsi_role") != "diagnostic_only_cannot_rescue_co_primary_metrics"):
        raise BranchDecisionRefused("H1-BD declaration semantics changed")
    legacy = declaration.get("h1_legacy_control")
    if legacy != {
        "classification_unmodified": True,
        "extraction_index": 62,
        "role": "originally_registered_delimiter_state_control",
    }:
        raise BranchDecisionRefused("legacy H1 h62 control was altered")
    if declaration.get("population") != {
        "calibration_base_count": 400,
        "calibration_condition_qualified_row_count": 2000,
        "scored_base_count": 1600,
        "source": "frozen_E_lure",
    }:
        raise BranchDecisionRefused("H1-BD population is not the frozen E_score split")
    expected_metrics = [
        {
            "critical_array": "secondary_index63_d_critical_centered_cosine",
            "metric": "centered_cosine",
            "safe_array": "secondary_index63_d_safe_centered_cosine",
            "whitener_prefix": "secondary_index63_whitener",
        },
        {
            "critical_array": "secondary_index63_d_critical_whitened",
            "metric": "whitened_euclidean",
            "safe_array": "secondary_index63_d_safe_whitened",
            "whitener_prefix": "secondary_index63_whitener",
        },
    ]
    if declaration.get("co_primary_metrics") != expected_metrics:
        raise BranchDecisionRefused("H1-BD h63 metric inventory changed")
    if declaration.get("metric_name_mapping") != {
        "centered_cosine": "centered cosine distance",
        "whitened_euclidean": (
            "held-out-whitened Mahalanobis distance (Euclidean norm after the held-out "
            "whitening transform)"
        ),
    }:
        raise BranchDecisionRefused("H1-BD metric name mapping changed")

    receipt = _json(FREEZE_RECEIPT_PATH, "H1-BD pre-outcome freeze receipt")
    required_receipt = {
        "schema", "status", "analysis_id", "outcomes_inspected_before_freeze",
        "legacy_h1_unchanged", "declaration", "sources",
    }
    if (set(receipt) != required_receipt or receipt.get("schema") != FREEZE_RECEIPT_SCHEMA or
            receipt.get("status") != "PASS" or receipt.get("analysis_id") != ANALYSIS_ID or
            receipt.get("outcomes_inspected_before_freeze") is not False or
            receipt.get("legacy_h1_unchanged") is not True):
        raise BranchDecisionRefused("H1-BD freeze receipt semantics changed")
    expected_sources = {
        "docs/DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md",
        "docs/DECISION_D46_H1_BRANCH_DECISION_ANALYSIS.md",
        "docs/PREREGISTRATION_AMENDMENT_2026-08-24.md",
        "manifests/h1_branch_decision/ANALYSIS_DECLARATION.json",
        "scripts/evaluate_lurestar_branch_decision.py",
        "scripts/evaluate_lurestar_checkpoints.py",
        "scripts/extract_lurestar_evidence.py",
        "src/lurestar/evaluate.py",
        "src/lurestar/representations.py",
    }
    sources = receipt.get("sources")
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise BranchDecisionRefused("H1-BD freeze receipt source set changed")
    observed = {relative: _record(_REPO, relative) for relative in sorted(expected_sources)}
    if receipt.get("sources") != observed:
        raise BranchDecisionRefused("H1-BD pre-outcome source receipt is stale")
    declaration_record = observed["manifests/h1_branch_decision/ANALYSIS_DECLARATION.json"]
    if receipt.get("declaration") != declaration_record:
        raise BranchDecisionRefused("H1-BD declaration receipt binding is stale")
    return declaration, observed


def _bind(path_record: Any, label: str) -> dict[str, str]:
    if not isinstance(path_record, dict) or set(path_record) != {"path", "sha256"}:
        raise BranchDecisionRefused(f"{label} binding is malformed")
    path = pathlib.Path(str(path_record["path"])).resolve()
    expected = str(path_record["sha256"])
    if not path.is_file() or path.is_symlink() or not L._is_sha(expected) or sha256_file(path) != expected:
        raise BranchDecisionRefused(f"{label} binding is stale")
    return {"path": str(path), "sha256": expected}


def _manifest(path: pathlib.Path, expected_seeds: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _json(path, "H1-BD evidence manifest")
    required = {
        "schema", "analysis_seed", "expected_arms", "expected_seeds", "n_boot",
        "identity_domains", "h3_permanent_block", "extractor", "local_measurement_sources",
        "frozen_inputs", "cells",
    }
    if set(manifest) != required or manifest.get("schema") != L.SCHEMA:
        raise BranchDecisionRefused("H1-BD manifest schema changed")
    if tuple(manifest.get("expected_arms", ())) != E.ARMS:
        raise BranchDecisionRefused("H1-BD manifest arms changed")
    if tuple(manifest.get("expected_seeds", ())) != expected_seeds:
        raise BranchDecisionRefused("H1-BD manifest seed set changed")
    if (not isinstance(manifest.get("analysis_seed"), int) or
            isinstance(manifest["analysis_seed"], bool) or manifest["analysis_seed"] < 0 or
            not isinstance(manifest.get("n_boot"), int) or manifest["n_boot"] < 100):
        raise BranchDecisionRefused("H1-BD manifest analysis seed/bootstrap count is invalid")
    domains = manifest.get("identity_domains")
    if (not isinstance(domains, dict) or set(domains) != {"h1_quartet"} or
            domains["h1_quartet"].get("count") != 1600 or
            not L._is_sha(str(domains["h1_quartet"].get("item_ids_sha256")))):
        raise BranchDecisionRefused("H1-BD manifest E_score identity domain changed")
    block = L._verified_h3_permanent_block(manifest.get("h3_permanent_block"))
    extractor = _bind(manifest.get("extractor"), "frozen GPU extractor")
    source_records = manifest.get("local_measurement_sources")
    if not isinstance(source_records, dict) or set(source_records) != set(L.LOCAL_MEASUREMENT_SOURCE_FIELDS):
        raise BranchDecisionRefused("H1-BD manifest local measurement source set changed")
    local_sources: dict[str, dict[str, str]] = {}
    for relative in L.LOCAL_MEASUREMENT_SOURCE_FIELDS:
        bound = _bind(source_records[relative], f"local measurement {relative}")
        if pathlib.Path(bound["path"]) != (_REPO / relative).resolve():
            raise BranchDecisionRefused("H1-BD local measurement source path changed")
        local_sources[relative] = bound
    frozen_inputs = manifest.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict) or set(frozen_inputs) != {"e_lure"}:
        raise BranchDecisionRefused("H1-BD manifest input set changed")
    _bind(frozen_inputs["e_lure"], "frozen E_lure")
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise BranchDecisionRefused("H1-BD manifest cells are not a list")
    wanted = {(arm, seed) for arm in E.ARMS for seed in expected_seeds}
    try:
        got = {(str(cell["arm"]), int(cell["seed"])) for cell in cells}
    except (KeyError, TypeError, ValueError) as exc:
        raise BranchDecisionRefused("H1-BD manifest cells are malformed") from exc
    if got != wanted or len(cells) != len(wanted):
        raise BranchDecisionRefused("H1-BD requires the complete frozen 15-cell base matrix")
    return manifest, block, local_sources


def _scalar(z: Any, key: str) -> str:
    value = np.asarray(z[key])
    if value.size != 1:
        raise BranchDecisionRefused(f"H1-BD evidence scalar {key} is malformed")
    return str(value.reshape(-1)[0])


def _cell(cell: Mapping[str, Any], domains: Mapping[str, Any], block: Mapping[str, str],
          local_sources: Mapping[str, Mapping[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = {"arm", "seed", "base_checkpoint", "evidence_npz", "evidence_sha256"}
    if not isinstance(cell, Mapping) or set(cell) != allowed:
        raise BranchDecisionRefused("H1-BD cell contains non-base-only fields")
    arm, seed = str(cell["arm"]), cell["seed"]
    if arm not in E.ARMS or not isinstance(seed, int) or isinstance(seed, bool):
        raise BranchDecisionRefused("H1-BD cell arm/seed is invalid")
    checkpoint = _bind(cell["base_checkpoint"], f"{arm}-s{seed} base checkpoint")
    evidence = _bind({"path": cell["evidence_npz"], "sha256": cell["evidence_sha256"]}, f"{arm}-s{seed} evidence")
    path = pathlib.Path(evidence["path"])
    with np.load(path, allow_pickle=False) as z:
        required = set(L.REQUIRED_ARRAYS) | set(L.BOUND_SCALARS)
        if arm == "bst":
            required |= set(L.BST_SECONDARY_ARRAYS) | set(L.BST_WHITENER_FIELDS)
        if set(z.files) != required:
            raise BranchDecisionRefused("H1-BD evidence schema is incomplete or expanded")
        scalar_expected = {
            "evidence_schema": "nextlat_forgetting/lurestar_evidence/4",
            "arm": arm, "seed": str(seed), "base_checkpoint_sha256": checkpoint["sha256"],
            "h3_permanent_block_sha256": block["sha256"],
            "h3_permanent_block_sidecar_sha256": block["sidecar_sha256"],
            **{field: local_sources[relative]["sha256"]
               for relative, field in L.LOCAL_MEASUREMENT_SOURCE_FIELDS.items()},
        }
        if any(_scalar(z, key) != value for key, value in scalar_expected.items()):
            raise BranchDecisionRefused("H1-BD evidence provenance binding changed")
        ids = np.asarray(z["h1_item_ids"]).ravel()
        observed_ids = L.item_ids_sha256(ids)
        if (ids.size != 1600 or observed_ids != domains["h1_quartet"]["item_ids_sha256"] or
                _scalar(z, "h1_item_ids_sha256") != observed_ids):
            raise BranchDecisionRefused("H1-BD E_score identities are stale")
        metrics: dict[str, dict[str, Any]] = {}
        for name, critical_key, safe_key, metric in (
            ("centered_cosine", "secondary_index63_d_critical_centered_cosine",
             "secondary_index63_d_safe_centered_cosine", "centered_cosine"),
            ("whitened_euclidean", "secondary_index63_d_critical_whitened",
             "secondary_index63_d_safe_whitened", "whitened_euclidean"),
        ):
            critical, safe = np.asarray(z[critical_key], dtype=np.float64).ravel(), np.asarray(z[safe_key], dtype=np.float64).ravel()
            if critical.shape != ids.shape or safe.shape != ids.shape or not np.all(np.isfinite(critical)) or not np.all(np.isfinite(safe)):
                raise BranchDecisionRefused(f"H1-BD {name} h63 arrays are invalid")
            metrics[name] = {"critical": critical, "safe": safe, "metric": metric}
        prefix = "secondary_index63_whitener"
        calibration_ids = np.asarray(z[f"{prefix}_calibration_ids"]).ravel()
        calibration_bases = np.asarray(z[f"{prefix}_calibration_base_ids"]).ravel()
        if (calibration_ids.size != 2000 or calibration_bases.size != 400 or
                len(set(map(str, calibration_ids.tolist()))) != 2000 or
                len(set(map(str, calibration_bases.tolist()))) != 400 or
                L.item_ids_sha256(calibration_ids) != _scalar(z, f"{prefix}_calibration_ids_sha256") or
                L.item_ids_sha256(calibration_bases) != _scalar(z, f"{prefix}_calibration_base_ids_sha256") or
                _scalar(z, f"{prefix}_n_pool") != "2000" or
                not L._is_sha(_scalar(z, f"{prefix}_fit_source_sha256"))):
            raise BranchDecisionRefused("H1-BD h63 whitener provenance is invalid")
        audit = {
            "calibration_ids_sha256": _scalar(z, f"{prefix}_calibration_ids_sha256"),
            "calibration_base_ids_sha256": _scalar(z, f"{prefix}_calibration_base_ids_sha256"),
            "fit_source_sha256": _scalar(z, f"{prefix}_fit_source_sha256"),
            "n_pool": 2000, "n_features": int(_scalar(z, f"{prefix}_n_features")),
        }
    identity = {
        "arm": arm, "seed": seed, "base_checkpoint": checkpoint,
        "evidence_npz": evidence,
        "h1_quartet": {"count": 1600, "item_ids_sha256": observed_ids},
    }
    return identity, {"metrics": metrics, "whitener_audit": audit}


def _classification(centered: E.ThreeArmReport, whitened: E.ThreeArmReport) -> dict[str, Any]:
    reports = {
        "centered_cosine": centered.by_name("nextlat_minus_bst"),
        "whitened_euclidean": whitened.by_name("nextlat_minus_bst"),
    }
    positive = {name: report.estimate > 0.0 for name, report in reports.items()}
    lower = {name: report.contrast.ci.ci_low > 0.0 for name, report in reports.items()}
    passed = {name: positive[name] and lower[name] for name in reports}
    if all(passed.values()):
        label = "metric-robust confirmatory support"
    elif all(positive.values()):
        label = "directionally consistent but unresolved evidence"
    elif any(positive.values()):
        label = "metric-dependent evidence"
    else:
        label = "no support"
    return {
        "contrast": "nextlat_minus_bst", "classification": label,
        "rule": "intersection-union over both prospective h63 co-primary metrics",
        "mean_positive": positive, "student_t_interval_lower_bound_above_zero": lower,
        "metric_pass": passed,
        "legacy_h1_h62_not_used": True,
    }


def evaluate_manifest(manifest_path: pathlib.Path, *, expected_seeds: tuple[int, ...] = CANONICAL_SEEDS) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration, freeze_sources = _validate_freeze()
    manifest, block, local_sources = _manifest(manifest_path, expected_seeds)
    cells: list[dict[str, Any]] = []
    by_metric: dict[str, dict[str, dict[int, float]]] = {
        "centered_cosine": {arm: {} for arm in E.ARMS},
        "whitened_euclidean": {arm: {} for arm in E.ARMS},
    }
    global_audit: dict[str, tuple[str, str]] = {}
    for cell in sorted(manifest["cells"], key=lambda item: (E.ARMS.index(str(item["arm"])), int(item["seed"]))):
        identity, data = _cell(cell, manifest["identity_domains"], block, local_sources)
        arm, seed = identity["arm"], identity["seed"]
        reports: dict[str, Any] = {}
        for offset, (name, values) in enumerate(data["metrics"].items()):
            rng = np.random.default_rng(int(manifest["analysis_seed"]) + 1009 * E.ARMS.index(arm) + 31 * seed + offset)
            item_report = E.bootstrap_psi_items(values["critical"], values["safe"], rng=rng, n_boot=int(manifest["n_boot"]), metric=values["metric"], extraction_index=63)
            reports[name] = item_report.as_dict()
            by_metric[name][arm][seed] = float(item_report.psi)
        audit = data["whitener_audit"]
        audit_key = (audit["calibration_ids_sha256"], audit["calibration_base_ids_sha256"])
        if global_audit and next(iter(global_audit.values())) != audit_key:
            raise BranchDecisionRefused("H1-BD h63 whiteners do not share the frozen calibration split")
        global_audit[f"{arm}-s{seed}"] = audit_key
        cells.append({"identity": identity, "h63_psi": reports, "h63_whitener_audit": audit})
    centered = E.three_arm_contrasts(by_metric["centered_cosine"], rng=np.random.default_rng(int(manifest["analysis_seed"])), n_boot=int(manifest["n_boot"]))
    whitened = E.three_arm_contrasts(by_metric["whitened_euclidean"], rng=np.random.default_rng(int(manifest["analysis_seed"]) + 1), n_boot=int(manifest["n_boot"]))
    classification = _classification(centered, whitened)
    report = {
        "schema": SCHEMA, "status": "COMPLETE", "analysis_id": ANALYSIS_ID,
        "analysis_seed": manifest["analysis_seed"], "n_boot": manifest["n_boot"],
        "h1_legacy_control": declaration["h1_legacy_control"],
        "h1bd_role": declaration["h1bd_role"], "extraction_index": 63,
        "metric_name_mapping": declaration["metric_name_mapping"],
        "cells": cells,
        "seed_level_contrasts": {
            "h63_centered_cosine": centered.as_dict(),
            "h63_whitened_euclidean": whitened.as_dict(),
        },
        "h1bd_confirmatory_classification": classification,
        "null_interpretation": "non-support is not evidence of equivalence",
    }
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "COMPLETE", "analysis_id": ANALYSIS_ID,
        "declaration": {"path": str(DECLARATION_PATH), "sha256": sha256_file(DECLARATION_PATH)},
        "pre_outcome_freeze": {"path": str(FREEZE_RECEIPT_PATH), "sha256": sha256_file(FREEZE_RECEIPT_PATH)},
        "freeze_sources": freeze_sources,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "expected_arms": list(E.ARMS), "expected_seeds": list(expected_seeds),
        "inputs": [cell["identity"] for cell in cells],
        "h1bd_classification": classification["classification"],
        "legacy_h1_h62_unmodified": True,
        "h63_branch_decision_index": 63,
    }
    return report, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report, receipt = evaluate_manifest(pathlib.Path(args.manifest).resolve())
        output = pathlib.Path(args.output).resolve()
        atomic_write_json(output, report)
        receipt["report"] = {"path": str(output), "sha256": sha256_file(output)}
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        atomic_write_json(receipt_path, receipt)
        atomic_write_text(receipt_path.with_suffix(receipt_path.suffix + ".sha256"), sha256_file(receipt_path) + "  " + receipt_path.name + "\n")
    except (BranchDecisionRefused, OSError, ValueError, RuntimeError) as exc:
        print(f"[evaluate_lurestar_branch_decision] REFUSED: {exc}", file=sys.stderr)
        return 2
    print("H1_BD_1=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
