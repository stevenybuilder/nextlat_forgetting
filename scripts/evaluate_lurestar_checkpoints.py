#!/usr/bin/env python
"""Fail-closed reduced H1/H2 Lure-Star analysis and provenance receipt.

The GPU extractor writes one analysis-ready NPZ per (arm, seed), embedding the exact base
checkpoint hash. H3 was permanently dropped by the prospectively frozen D40 stopping rule. This
command therefore accepts only H1/H2 evidence, rejects adaptation checkpoints and all H3 arrays or
analysis controls, and binds the exact canonical permanent-block document plus its sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar import evaluate as E  # noqa: E402
from lurestar.durable_checkpoint import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)

SCHEMA = "nextlat_forgetting/lurestar_evaluation_manifest/4"
REPORT_SCHEMA = "nextlat_forgetting/lurestar_confirmatory_report/4"
RECEIPT_SCHEMA = "nextlat_forgetting/lurestar_evaluation_receipt/4"
EXPECTED_H1_ITEMS = 1600
H3_BLOCK_PATH = (_REPO / "manifests" / "h3_selected" / "PERMANENT_H3_BLOCK.json").resolve()
H3_BLOCK_SHA256 = "82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778"
H3_BLOCK_SIDECAR_SHA256 = "24b47f2d49e084d4b09e39393938294d7ef1a7ba6e5bbf90d56b4e7145a65d0b"
H3_BLOCK_FORBIDDEN = (
    "candidate_expansion", "caliper_change", "weighting", "unmatched_restriction",
    "pilot_substitution", "matching_amendment",
)
H3_BLOCK_DOCUMENT = {
    "combined_loss_sha256": "814058a162e12fde36c7204dd30798b63bfbf02294fce768046070672e5afece",
    "expanded_manifest_sha256": "2effd4e13d384786546c71cc61b4138dc97f082e3992bf3cdf398e6bf93264f1",
    "forbidden": list(H3_BLOCK_FORBIDDEN),
    "no_further_amendments_permitted": True,
    "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
    "schema": "nextlat_forgetting/h3_mid_expansion/1",
    "status": "PERMANENT_H3_BLOCK",
    "unmatched_count": 4,
    "unmatched_identity_sha256":
        "ab4fb10a1e049912fb3e24046cf1498b1027e489864e076d91c10044cef82bf6",
}

H1_ARRAYS = {
    "d_critical",
    "d_safe",
    "d_repeat",
    "d_critical_whitened",
    "d_safe_whitened",
    "critical_margin",
    "base_margin",
    "secondary_raw_cosine_d_critical",
    "secondary_raw_cosine_d_safe",
    "secondary_uncentered_euclidean_d_critical",
    "secondary_uncentered_euclidean_d_safe",
    "secondary_index63_d_critical_centered_cosine",
    "secondary_index63_d_safe_centered_cosine",
    "secondary_index63_d_critical_whitened",
    "secondary_index63_d_safe_whitened",
}
BEHAVIOR_ARRAYS = {
    f"behavior_{condition}_{endpoint}"
    for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical")
    for endpoint in ("first_branch_accuracy", "exact_path_accuracy")
}
BEHAVIOR_PATH_ARRAYS = {
    f"behavior_{condition}_{kind}_path"
    for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical")
    for kind in ("generated", "true")
}
H1_ARRAYS |= BEHAVIOR_ARRAYS
NPSI_SCALARS = {"npsi", "npsi_whitened"}
SECONDARY_STATUS_SCALARS = {
    "secondary_bst_texthead_status", "secondary_intermediate_status",
    "secondary_exact_path_status",
}


def _whitener_fields(prefix: str) -> set[str]:
    return {
        f"{prefix}_shrinkage", f"{prefix}_shrinkage_rule",
        f"{prefix}_condition_number", f"{prefix}_calibration_ids",
        f"{prefix}_calibration_ids_sha256", f"{prefix}_fit_source_sha256",
        f"{prefix}_calibration_base_ids", f"{prefix}_calibration_base_ids_sha256",
        f"{prefix}_n_pool", f"{prefix}_n_features",
    }


COMMON_WHITENER_FIELDS = _whitener_fields("whitener") | _whitener_fields(
    "secondary_index63_whitener"
)
BST_SECONDARY_ARRAYS = {
    "secondary_bst_texthead_d_critical_centered_cosine",
    "secondary_bst_texthead_d_safe_centered_cosine",
    "secondary_bst_texthead_d_critical_whitened",
    "secondary_bst_texthead_d_safe_whitened",
}
BST_WHITENER_FIELDS = _whitener_fields("secondary_bst_texthead_whitener")
INTERMEDIATE_FIELDS = {
    "secondary_intermediate_blocks", "secondary_intermediate_positions",
    "secondary_intermediate_d_critical_centered_cosine",
    "secondary_intermediate_d_safe_centered_cosine",
    "secondary_intermediate_d_critical_whitened",
    "secondary_intermediate_d_safe_whitened",
    "secondary_intermediate_whitener_shrinkage",
    "secondary_intermediate_whitener_condition_number",
    "secondary_intermediate_whitener_fit_source_sha256",
    "secondary_intermediate_whitener_fit_ids",
    "secondary_intermediate_whitener_fit_ids_sha256",
    "secondary_intermediate_whitener_n_features",
    "secondary_intermediate_whitener_fit_dtype",
    "secondary_intermediate_whitener_fit_shape",
    "secondary_intermediate_whitener_shrinkage_rule",
    "secondary_intermediate_calibration_base_ids",
    "secondary_intermediate_calibration_base_ids_sha256",
}
REQUIRED_ARRAYS = {"h1_item_ids"} | H1_ARRAYS | BEHAVIOR_PATH_ARRAYS
BOUND_SCALARS = {
    "evidence_schema",
    "arm",
    "seed",
    "base_checkpoint_sha256",
    "h3_permanent_block_sha256",
    "h3_permanent_block_sidecar_sha256",
    "local_representations_sha256",
    "local_evaluate_sha256",
    "h1_item_ids_sha256",
} | NPSI_SCALARS | SECONDARY_STATUS_SCALARS | COMMON_WHITENER_FIELDS | INTERMEDIATE_FIELDS


class EvaluationRefused(RuntimeError):
    pass


LOCAL_MEASUREMENT_SOURCE_FIELDS = {
    "src/lurestar/representations.py": "local_representations_sha256",
    "src/lurestar/evaluate.py": "local_evaluate_sha256",
}


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


MANIPULATION_FAILURES = {
    "applicable": False,
    "reason": "H3_PERMANENTLY_DROPPED_AFTER_D40_FEASIBILITY_GATE",
    "interpretation": "no surviving Lure-Star manipulation endpoint; not an H1 outcome check",
}


def _null_contract(h1_classification: str | None = None) -> dict:
    return {
        "H1": {
            "null": "NextLat-minus-BST paired-seed PSI is not positive and resolved under both co-primary metrics",
            "classification": h1_classification,
            "non_support_interpretation": "not resolved at the detectable effect size; never evidence of equivalence",
            "metric_dependent_interpretation": "operational geometry dependence; cannot be promoted or called equivalence",
        },
        "H2": {
            "null": "base-to-critical distance adds no held-out prediction beyond base margin under the both-metric contract",
            "non_support_interpretation": "not resolved at the detectable effect size; never evidence of equivalence or no association",
            "causal_claim_permitted": False,
        },
    }


def _validate_terminal_contract(report: dict) -> None:
    for field in ("status", "primary_status", "invalid_cells", "nulls", "manipulation_failures"):
        if field not in report:
            raise EvaluationRefused(f"terminal report contract missing {field}")
    if report["manipulation_failures"] != MANIPULATION_FAILURES:
        raise EvaluationRefused("terminal report manipulation_failures contract changed")
    nulls = report["nulls"]
    if not isinstance(nulls, dict) or set(nulls) != {"H1", "H2"}:
        raise EvaluationRefused("terminal report null contract changed")
    for hypothesis in ("H1", "H2"):
        if "never evidence of equivalence" not in nulls[hypothesis][
            "non_support_interpretation"
        ]:
            raise EvaluationRefused(f"{hypothesis} null interpretation permits equivalence")
    invalid = report["invalid_cells"]
    if not isinstance(invalid, list) or any(
        not isinstance(cell, dict) or set(cell) != {
            "arm", "seed", "evidence_npz", "reason_code", "reason"
        } for cell in invalid
    ):
        raise EvaluationRefused("terminal report invalid_cells schema changed")


def _invalid_reason_code(reason: str) -> str:
    lowered = reason.lower()
    if "npsi" in lowered:
        return "NPSI_INVALID"
    if "constant within a training fold" in lowered or "primary h2" in lowered:
        return "H2_PRIMARY_INVALID"
    if "non-finite" in lowered or "finite" in lowered:
        return "NONFINITE_CELL"
    if "schema" in lowered or "missing=" in lowered or "extra=" in lowered:
        return "SCHEMA_INTEGRITY_INVALID"
    return "CELL_INTEGRITY_INVALID"


def item_ids_sha256(values: np.ndarray) -> str:
    """Canonical ordered identity hash, independent of numpy dtype/container encoding."""
    ids = np.asarray(values).ravel()
    normalized = [str(value) for value in ids.tolist()]
    if any("\n" in value or "\r" in value for value in normalized):
        raise EvaluationRefused("item identities may not contain newline characters")
    payload = ("".join(value + "\n" for value in normalized)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _domain_fields() -> dict[str, tuple[str, set[str]]]:
    return {"h1_quartet": ("h1_item_ids", set(H1_ARRAYS))}


def _scalar(z: Any, key: str) -> str:
    value = np.asarray(z[key])
    if value.size != 1:
        raise EvaluationRefused(f"NPZ field {key!r} must be scalar")
    return str(value.reshape(-1)[0])


def _record(path_record: dict, label: str) -> tuple[pathlib.Path, str]:
    if not isinstance(path_record, dict):
        raise EvaluationRefused(f"{label} checkpoint record is missing")
    path = pathlib.Path(str(path_record.get("path", ""))).resolve()
    digest = str(path_record.get("sha256", ""))
    if not path.is_file() or not _is_sha(digest) or sha256_file(path) != digest:
        raise EvaluationRefused(f"{label} checkpoint is absent or fails SHA-256 verification")
    return path, digest


def _bound_artifact(record: dict, label: str) -> dict:
    if not isinstance(record, dict):
        raise EvaluationRefused(f"{label} artifact record is missing")
    path = pathlib.Path(str(record.get("path", ""))).resolve()
    digest = str(record.get("sha256", ""))
    if not path.is_file() or not _is_sha(digest) or sha256_file(path) != digest:
        raise EvaluationRefused(f"{label} artifact is absent or fails SHA-256 verification")
    return {"path": str(path), "sha256": digest}


def _verified_h3_permanent_block(record: Any) -> dict[str, str]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "sidecar"}:
        raise EvaluationRefused("permanent H3 block record must bind path, SHA-256, and sidecar")
    if not isinstance(record["sidecar"], dict) or set(record["sidecar"]) != {"path", "sha256"}:
        raise EvaluationRefused("permanent H3 block sidecar must bind only path and SHA-256")
    bound = _bound_artifact(record, "canonical permanent H3 block")
    path = pathlib.Path(bound["path"])
    if path != H3_BLOCK_PATH or bound["sha256"] != H3_BLOCK_SHA256:
        raise EvaluationRefused("manifest must bind the exact canonical permanent H3 block")
    sidecar_record = record.get("sidecar") if isinstance(record, dict) else None
    sidecar = _bound_artifact(sidecar_record, "permanent H3 block sidecar")
    sidecar_path = pathlib.Path(sidecar["path"])
    if sidecar_path != pathlib.Path(f"{H3_BLOCK_PATH}.sha256"):
        raise EvaluationRefused("manifest must bind the canonical permanent H3 block sidecar")
    if sidecar["sha256"] != H3_BLOCK_SIDECAR_SHA256:
        raise EvaluationRefused("canonical permanent H3 block sidecar hash changed")
    if sidecar_path.read_text(encoding="utf-8").strip().split() != [
        H3_BLOCK_SHA256, H3_BLOCK_PATH.name,
    ]:
        raise EvaluationRefused("permanent H3 block sidecar content or filename binding changed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationRefused("permanent H3 block is invalid JSON") from exc
    if document != H3_BLOCK_DOCUMENT:
        raise EvaluationRefused("permanent H3 block semantics changed")
    return {
        "path": str(path), "sha256": bound["sha256"],
        "sidecar_path": str(sidecar_path), "sidecar_sha256": sidecar["sha256"],
    }




def _load_reduced_cell(
    cell: dict,
    expected_domains: dict,
    block: dict[str, str],
    local_measurement_sources: dict[str, dict[str, str]],
) -> tuple[dict, dict]:
    allowed = {"arm", "seed", "base_checkpoint", "evidence_npz", "evidence_sha256"}
    if not isinstance(cell, dict) or set(cell) != allowed:
        raise EvaluationRefused(
            "reduced H1/H2 cell must contain only arm, seed, base checkpoint, and evidence"
        )
    arm = str(cell.get("arm", ""))
    seed = cell.get("seed")
    if arm not in E.ARMS or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise EvaluationRefused("cell has an unknown arm or invalid seed")
    if not isinstance(cell["base_checkpoint"], dict) or set(cell["base_checkpoint"]) != {
        "path", "sha256",
    }:
        raise EvaluationRefused("base checkpoint record must bind only path and SHA-256")
    base_path, base_sha = _record(cell["base_checkpoint"], f"{arm}-s{seed} base")
    evidence = pathlib.Path(str(cell["evidence_npz"])).resolve()
    evidence_sha = str(cell["evidence_sha256"])
    if not evidence.is_file() or sha256_file(evidence) != evidence_sha:
        raise EvaluationRefused(f"{arm}-s{seed} evidence NPZ is absent or hash-mismatched")
    with np.load(evidence, allow_pickle=False) as z:
        expected_fields = REQUIRED_ARRAYS | BOUND_SCALARS
        if arm == "bst":
            expected_fields |= BST_SECONDARY_ARRAYS | BST_WHITENER_FIELDS
        if set(z.files) != expected_fields:
            missing = sorted(expected_fields - set(z.files))
            extra = sorted(set(z.files) - expected_fields)
            raise EvaluationRefused(
                f"{arm}-s{seed} reduced evidence schema differs; missing={missing}, extra={extra}"
            )
        scalar_bindings = {
            "evidence_schema": "nextlat_forgetting/lurestar_evidence/4",
            "arm": arm,
            "seed": str(seed),
            "base_checkpoint_sha256": base_sha,
            "h3_permanent_block_sha256": block["sha256"],
            "h3_permanent_block_sidecar_sha256": block["sidecar_sha256"],
            **{
                field: local_measurement_sources[relative]["sha256"]
                for relative, field in LOCAL_MEASUREMENT_SOURCE_FIELDS.items()
            },
        }
        for key, expected in scalar_bindings.items():
            if _scalar(z, key) != expected:
                raise EvaluationRefused(f"{arm}-s{seed} NPZ binding {key} disagrees")
        data = {key: np.asarray(z[key]) for key in expected_fields - {
            "evidence_schema", "arm", "seed", "base_checkpoint_sha256",
            "h3_permanent_block_sha256", "h3_permanent_block_sidecar_sha256",
            "local_representations_sha256", "local_evaluate_sha256",
        }}
        ids = np.asarray(data["h1_item_ids"]).ravel()
        domain = expected_domains.get("h1_quartet")
        observed_hash = item_ids_sha256(ids)
        if not isinstance(domain, dict):
            raise EvaluationRefused("manifest lacks H1 quartet identity domain")
        if ids.size != int(domain.get("count", -1)) or observed_hash != str(
            domain.get("item_ids_sha256", "")
        ):
            raise EvaluationRefused(f"{arm}-s{seed} H1 quartet identity differs from manifest")
        if _scalar(z, "h1_item_ids_sha256") != observed_hash:
            raise EvaluationRefused(f"{arm}-s{seed} embedded H1 identity hash disagrees")
        for key in H1_ARRAYS:
            values = np.asarray(data[key]).ravel()
            if values.size != ids.size:
                raise EvaluationRefused(
                    f"{arm}-s{seed} field-length group h1_quartet: {key} has {values.size}, "
                    f"identity has {ids.size}; truncation/padding is forbidden"
                )
            if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                raise EvaluationRefused(f"{arm}-s{seed} {key} contains a non-finite value")
        for name, expected in (
            ("npsi", E.normalized_psi(data["d_critical"], data["d_safe"])[0]),
            ("npsi_whitened", E.normalized_psi(
                data["d_critical_whitened"], data["d_safe_whitened"]
            )[0]),
        ):
            observed = np.asarray(data[name], dtype=np.float64)
            if observed.size != 1 or not np.isfinite(observed.item()) or not np.isclose(
                observed.item(), expected, rtol=0.0, atol=1e-12
            ):
                raise EvaluationRefused(f"{arm}-s{seed} mandatory {name} is absent or disagrees")
        if str(np.asarray(data["secondary_intermediate_status"]).item()) != (
            "AVAILABLE_ALL_BLOCKS_0_11_PRE_FINAL_NORM_POSITIONS_62_63"
        ):
            raise EvaluationRefused(f"{arm}-s{seed} intermediate-layer secondary status changed")
        if str(np.asarray(data["secondary_exact_path_status"]).item()) != (
            "AVAILABLE_EXPLICIT_ARGMAX_5_TOKENS"
        ):
            raise EvaluationRefused(f"{arm}-s{seed} exact-path secondary status changed")
        expected_texthead_status = "AVAILABLE_BST_ONLY" if arm == "bst" else "NOT_APPLICABLE_NON_BST"
        if str(np.asarray(data["secondary_bst_texthead_status"]).item()) != expected_texthead_status:
            raise EvaluationRefused(f"{arm}-s{seed} BST TextHead applicability is inconsistent")
        whitener_prefixes = ["whitener", "secondary_index63_whitener"]
        if arm == "bst":
            whitener_prefixes.append("secondary_bst_texthead_whitener")
        calibration_reference = None
        calibration_base_reference = None
        for prefix in whitener_prefixes:
            ids_key = f"{prefix}_calibration_ids"
            calibration_ids = [str(value) for value in np.asarray(data[ids_key]).ravel().tolist()]
            if len(calibration_ids) != 2000 or len(set(calibration_ids)) != len(calibration_ids):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} calibration IDs are incomplete")
            if any(
                len(value.split(":", 1)) != 2
                or not _is_sha(value.split(":", 1)[0])
                or value.split(":", 1)[1] not in {
                    "base", "repeat", "near_safe", "near_critical", "far_critical"
                }
                or value.split(":", 1)[0] in set(map(str, ids.tolist()))
                for value in calibration_ids
            ):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} calibration split is not held out")
            if item_ids_sha256(np.asarray(calibration_ids)) != _scalar(data, ids_key + "_sha256"):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} calibration ID hash disagrees")
            if calibration_reference is None:
                calibration_reference = calibration_ids
            elif calibration_ids != calibration_reference:
                raise EvaluationRefused(f"{arm}-s{seed} secondary whitener changed calibration IDs")
            calibration_base_ids = [
                str(value) for value in np.asarray(
                    data[f"{prefix}_calibration_base_ids"]
                ).ravel().tolist()
            ]
            if len(calibration_base_ids) != 400 or len(set(calibration_base_ids)) != 400:
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} calibration base IDs are incomplete")
            expected_condition_qualified_ids = [
                f"{base_id}:{condition}"
                for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical")
                for base_id in calibration_base_ids
            ]
            if calibration_ids != expected_condition_qualified_ids:
                raise EvaluationRefused(
                    f"{arm}-s{seed} {prefix} calibration IDs are not exact condition-qualified rows"
                )
            if item_ids_sha256(np.asarray(calibration_base_ids)) != _scalar(
                data, f"{prefix}_calibration_base_ids_sha256"
            ):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} calibration base hash disagrees")
            if calibration_base_reference is None:
                calibration_base_reference = calibration_base_ids
            elif calibration_base_ids != calibration_base_reference:
                raise EvaluationRefused(f"{arm}-s{seed} whitener calibration base order changed")
            shrinkage = float(np.asarray(data[f"{prefix}_shrinkage"]).item())
            condition = float(np.asarray(data[f"{prefix}_condition_number"]).item())
            n_pool = int(np.asarray(data[f"{prefix}_n_pool"]).item())
            n_features = int(np.asarray(data[f"{prefix}_n_features"]).item())
            if not (np.isfinite(shrinkage) and 1e-3 <= shrinkage <= 1.0):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} shrinkage is invalid")
            if str(np.asarray(data[f"{prefix}_shrinkage_rule"]).item()) != (
                "ledoit_wolf_with_1e-3_floor"
            ):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} shrinkage rule changed")
            if not np.isfinite(condition) or condition < 1.0 or n_pool != 2000 or n_features < 1:
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} audit dimensions are invalid")
            fit_source = str(np.asarray(data[f"{prefix}_fit_source_sha256"]).item())
            if len(fit_source) != 64 or any(ch not in "0123456789abcdef" for ch in fit_source):
                raise EvaluationRefused(f"{arm}-s{seed} {prefix} fit-source hash is invalid")
        for key in BST_SECONDARY_ARRAYS if arm == "bst" else ():
            values = np.asarray(data[key], dtype=np.float64).ravel()
            if values.size != ids.size or not np.all(np.isfinite(values)):
                raise EvaluationRefused(f"{arm}-s{seed} BST TextHead secondary is malformed")
        for key in BEHAVIOR_ARRAYS:
            values = np.asarray(data[key], dtype=np.float64).ravel()
            if not set(np.unique(values).tolist()).issubset({0.0, 1.0}):
                raise EvaluationRefused(f"{arm}-s{seed} {key} is not binary per-item behavior")
        for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical"):
            generated = np.asarray(data[f"behavior_{condition}_generated_path"])
            truth = np.asarray(data[f"behavior_{condition}_true_path"])
            indicator = np.asarray(
                data[f"behavior_{condition}_exact_path_accuracy"], dtype=np.float64
            )
            if generated.shape != (ids.size, 5) or truth.shape != (ids.size, 5):
                raise EvaluationRefused(f"{arm}-s{seed} {condition} paths must be E_score x 5")
            if generated.dtype.kind not in "iu" or truth.dtype.kind not in "iu":
                raise EvaluationRefused(f"{arm}-s{seed} {condition} paths must be integer tokens")
            recomputed = np.all(generated == truth, axis=1).astype(np.float64)
            if not np.array_equal(recomputed, indicator):
                raise EvaluationRefused(
                    f"{arm}-s{seed} {condition} exact-path indicator disagrees with tokens"
                )
        if np.asarray(data["secondary_intermediate_blocks"]).tolist() != list(range(12)):
            raise EvaluationRefused(f"{arm}-s{seed} intermediate block inventory changed")
        if np.asarray(data["secondary_intermediate_positions"]).tolist() != [62, 63]:
            raise EvaluationRefused(f"{arm}-s{seed} intermediate positions changed")
        for key in (
            "secondary_intermediate_d_critical_centered_cosine",
            "secondary_intermediate_d_safe_centered_cosine",
            "secondary_intermediate_d_critical_whitened",
            "secondary_intermediate_d_safe_whitened",
        ):
            values = np.asarray(data[key], dtype=np.float64)
            if values.shape != (12, 2, ids.size) or not np.all(np.isfinite(values)):
                raise EvaluationRefused(f"{arm}-s{seed} {key} is not fixed 12x2xE_score")
        intermediate_shrinkage = np.asarray(
            data["secondary_intermediate_whitener_shrinkage"], dtype=np.float64
        )
        if (
            intermediate_shrinkage.shape != (12, 2)
            or not np.all(np.isfinite(intermediate_shrinkage))
            or np.any(intermediate_shrinkage < 1e-3)
            or np.any(intermediate_shrinkage > 1.0)
        ):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener shrinkage is outside [1e-3, 1]"
            )
        intermediate_condition = np.asarray(
            data["secondary_intermediate_whitener_condition_number"], dtype=np.float64
        )
        if (
            intermediate_condition.shape != (12, 2)
            or not np.all(np.isfinite(intermediate_condition))
            or np.any(intermediate_condition < 1.0)
        ):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener condition number must be finite and >= 1"
            )
        intermediate_sources = np.asarray(
            data["secondary_intermediate_whitener_fit_source_sha256"]
        )
        if intermediate_sources.shape != (12, 2) or any(
            not _is_sha(str(value)) for value in intermediate_sources.ravel().tolist()
        ):
            raise EvaluationRefused(f"{arm}-s{seed} intermediate fit-source hashes malformed")
        intermediate_fit_ids = np.asarray(data["secondary_intermediate_whitener_fit_ids"])
        if intermediate_fit_ids.shape != (2000,):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate shared fit IDs must contain exactly 2000 rows"
            )
        intermediate_fit_id_hashes = np.asarray(
            data["secondary_intermediate_whitener_fit_ids_sha256"]
        )
        if intermediate_fit_id_hashes.shape != (12, 2):
            raise EvaluationRefused(f"{arm}-s{seed} intermediate fit-ID hashes malformed")
        for block_index in range(12):
            for position_offset in range(2):
                fit_ids = [str(value) for value in intermediate_fit_ids.tolist()]
                fit_id_hash = str(intermediate_fit_id_hashes[block_index, position_offset])
                if fit_ids != calibration_reference:
                    raise EvaluationRefused(
                        f"{arm}-s{seed} intermediate fit IDs changed condition-qualified order"
                    )
                if not _is_sha(fit_id_hash) or item_ids_sha256(np.asarray(fit_ids)) != fit_id_hash:
                    raise EvaluationRefused(
                        f"{arm}-s{seed} intermediate fit-ID hash disagrees"
                    )
        intermediate_features = np.asarray(
            data["secondary_intermediate_whitener_n_features"], dtype=np.int64
        )
        if intermediate_features.shape != (12, 2) or np.any(intermediate_features != 384):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener n_features must be exactly 384"
            )
        intermediate_dtypes = np.asarray(data["secondary_intermediate_whitener_fit_dtype"])
        if intermediate_dtypes.shape != (12, 2) or np.any(intermediate_dtypes != "float64-le"):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener canonical dtype must be float64-le"
            )
        intermediate_shapes = np.asarray(
            data["secondary_intermediate_whitener_fit_shape"], dtype=np.int64
        )
        if intermediate_shapes.shape != (12, 2, 2) or not np.all(
            intermediate_shapes == np.asarray([2000, 384], dtype=np.int64)
        ):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener fit shape must be exactly [2000, 384]"
            )
        intermediate_rules = np.asarray(
            data["secondary_intermediate_whitener_shrinkage_rule"]
        )
        if intermediate_rules.shape != (12, 2) or np.any(
            intermediate_rules != "ledoit_wolf_with_1e-3_floor"
        ):
            raise EvaluationRefused(
                f"{arm}-s{seed} intermediate whitener shrinkage rule changed"
            )
        intermediate_base_ids = np.asarray(
            data["secondary_intermediate_calibration_base_ids"]
        ).ravel()
        if intermediate_base_ids.tolist() != calibration_base_reference:
            raise EvaluationRefused(f"{arm}-s{seed} intermediate calibration bases changed")
        if item_ids_sha256(intermediate_base_ids) != _scalar(
            data, "secondary_intermediate_calibration_base_ids_sha256"
        ):
            raise EvaluationRefused(f"{arm}-s{seed} intermediate calibration hash disagrees")
    identity = {
        "arm": arm, "seed": seed,
        "base_checkpoint": {"path": str(base_path), "sha256": base_sha},
        "evidence_npz": {"path": str(evidence), "sha256": evidence_sha},
        "identity_domains": {
            "h1_quartet": {"count": int(ids.size), "item_ids_sha256": observed_hash},
        },
    }
    return identity, data


def _reduced_cell_metrics(identity: dict, data: dict, *, analysis_seed: int,
                          n_boot: int) -> dict:
    arm, seed = identity["arm"], identity["seed"]
    rng = np.random.default_rng(analysis_seed + 1009 * E.ARMS.index(arm) + int(seed))
    psi = E.bootstrap_psi_items(data["d_critical"], data["d_safe"], rng=rng, n_boot=n_boot)
    psi_whitened = E.bootstrap_psi_items(
        data["d_critical_whitened"], data["d_safe_whitened"], rng=rng, n_boot=n_boot,
        metric="whitened_euclidean",
    )
    safe = E.safe_lure_invariance(data["d_safe"], data["d_repeat"], rng=rng, n_boot=n_boot)
    folds = E.base_id_folds(data["h1_item_ids"])
    h2_centered = E.fit_h2(
        data["critical_margin"], data["d_critical"], data["base_margin"], folds=folds
    )["report"]
    h2_whitened = E.fit_h2(
        data["critical_margin"], data["d_critical_whitened"], data["base_margin"], folds=folds
    )["report"]

    def strip_inferential_pvalues(value):
        if isinstance(value, dict):
            return {
                key: strip_inferential_pvalues(item)
                for key, item in value.items()
                if key not in {"p", "spearman_p_pred_vs_actual"}
            }
        if isinstance(value, list):
            return [strip_inferential_pvalues(item) for item in value]
        return value

    def secondary_nested(outcome_key: str, baseline_key: str, distance_key: str) -> dict:
        outcome = np.asarray(data[outcome_key], dtype=np.float64)
        baseline = np.asarray(data[baseline_key], dtype=np.float64)
        distance = np.asarray(data[distance_key], dtype=np.float64)
        reasons = []
        if np.std(outcome) == 0:
            reasons.append("constant outcome due to ceiling/floor")
        for fold in (0, 1):
            training = folds != fold
            if np.std(baseline[training]) == 0:
                reasons.append(f"constant training-fold predictor in fold {fold}")
            if np.std(distance[training]) == 0:
                reasons.append(f"constant training-fold distance in fold {fold}")
        if reasons:
            return {
                "status": "not_estimable_due_to_ceiling/constant training-fold predictor",
                "reasons": reasons,
                "base_mean": float(baseline.mean()),
                "critical_mean": float(outcome.mean()),
                "promotion_prohibited": True,
            }
        fitted = E.fit_h2(
            outcome, distance, baseline, folds=folds,
            outcome_name=outcome_key, baseline_name=baseline_key,
        )["report"]
        return {
            "status": "estimable_nested_linear_probability_model",
            "report": strip_inferential_pvalues(fitted),
            "promotion_prohibited": True,
            "inferential_pvalues_reported": False,
        }

    h2_accuracy_centered = secondary_nested(
        "behavior_near_critical_first_branch_accuracy",
        "behavior_base_first_branch_accuracy", "d_critical"
    )
    h2_accuracy_whitened = secondary_nested(
        "behavior_near_critical_first_branch_accuracy",
        "behavior_base_first_branch_accuracy", "d_critical_whitened"
    )
    h2_exact_path_centered = secondary_nested(
        "behavior_near_critical_exact_path_accuracy",
        "behavior_base_exact_path_accuracy", "d_critical"
    )
    h2_exact_path_whitened = secondary_nested(
        "behavior_near_critical_exact_path_accuracy",
        "behavior_base_exact_path_accuracy", "d_critical_whitened"
    )

    def secondary(name: str, critical_key: str, safe_key: str, metric: str,
                  extraction_index: int) -> dict:
        result = E.bootstrap_psi_items(
            data[critical_key], data[safe_key], rng=rng, n_boot=n_boot,
            metric=metric, extraction_index=extraction_index,
        ).as_dict()
        result["name"] = name
        result["role"] = "mandatory labeled secondary; cannot alter H1 classification"
        result["promotion_prohibited"] = True
        return result

    secondaries = {
        "index63_centered_cosine": secondary(
            "index63_centered_cosine", "secondary_index63_d_critical_centered_cosine",
            "secondary_index63_d_safe_centered_cosine", "centered_cosine", 63,
        ),
        "index63_whitened_euclidean": secondary(
            "index63_whitened_euclidean", "secondary_index63_d_critical_whitened",
            "secondary_index63_d_safe_whitened", "whitened_euclidean", 63,
        ),
        "raw_cosine": secondary(
            "raw_cosine", "secondary_raw_cosine_d_critical",
            "secondary_raw_cosine_d_safe", "raw_cosine", 62,
        ),
        "uncentered_euclidean": secondary(
            "uncentered_euclidean", "secondary_uncentered_euclidean_d_critical",
            "secondary_uncentered_euclidean_d_safe", "uncentered_euclidean", 62,
        ),
    }
    intermediate_rows = []
    for block in range(12):
        for position_offset, position in enumerate((62, 63)):
            metrics = {}
            for metric, critical_key, safe_key in (
                ("centered_cosine", "secondary_intermediate_d_critical_centered_cosine",
                 "secondary_intermediate_d_safe_centered_cosine"),
                ("whitened_euclidean", "secondary_intermediate_d_critical_whitened",
                 "secondary_intermediate_d_safe_whitened"),
            ):
                critical = np.asarray(data[critical_key])[block, position_offset]
                safe_distance = np.asarray(data[safe_key])[block, position_offset]
                npsi_value, denominator = E.normalized_psi(critical, safe_distance)
                metrics[metric] = {
                    "psi": float(np.mean(critical - safe_distance)),
                    "npsi": npsi_value,
                    "npsi_denominator": denominator,
                    "d_critical_mean": float(np.mean(critical)),
                    "d_safe_mean": float(np.mean(safe_distance)),
                }
            intermediate_rows.append({
                "block": block, "position": position, "state": "pre_final_norm_block_output",
                "metrics": metrics, "promotion_prohibited": True,
                "whitener_audit": {
                    "shrinkage": float(np.asarray(
                        data["secondary_intermediate_whitener_shrinkage"]
                    )[block, position_offset]),
                    "condition_number": float(np.asarray(
                        data["secondary_intermediate_whitener_condition_number"]
                    )[block, position_offset]),
                    "fit_source_sha256": str(np.asarray(
                        data["secondary_intermediate_whitener_fit_source_sha256"]
                    )[block, position_offset]),
                    "fit_ids_sha256": str(np.asarray(
                        data["secondary_intermediate_whitener_fit_ids_sha256"]
                    )[block, position_offset]),
                    "fit_id_count": int(np.asarray(
                        data["secondary_intermediate_whitener_fit_ids"]
                    ).size),
                    "n_features": int(np.asarray(
                        data["secondary_intermediate_whitener_n_features"]
                    )[block, position_offset]),
                    "fit_dtype": str(np.asarray(
                        data["secondary_intermediate_whitener_fit_dtype"]
                    )[block, position_offset]),
                    "fit_shape": np.asarray(
                        data["secondary_intermediate_whitener_fit_shape"]
                    )[block, position_offset].astype(int).tolist(),
                    "shrinkage_rule": str(np.asarray(
                        data["secondary_intermediate_whitener_shrinkage_rule"]
                    )[block, position_offset]),
                    "calibration_base_ids_sha256": str(np.asarray(
                        data["secondary_intermediate_calibration_base_ids_sha256"]
                    ).item()),
                },
            })
    secondaries["intermediate_layers"] = {
        "status": str(np.asarray(data["secondary_intermediate_status"]).item()),
        "fixed_inventory": intermediate_rows,
        "inferential_pvalues_reported": False,
        "promotion_prohibited": True,
    }
    if arm == "bst":
        secondaries["bst_texthead_centered_cosine"] = secondary(
            "bst_texthead_centered_cosine",
            "secondary_bst_texthead_d_critical_centered_cosine",
            "secondary_bst_texthead_d_safe_centered_cosine", "centered_cosine", 62,
        )
        secondaries["bst_texthead_whitened_euclidean"] = secondary(
            "bst_texthead_whitened_euclidean",
            "secondary_bst_texthead_d_critical_whitened",
            "secondary_bst_texthead_d_safe_whitened", "whitened_euclidean", 62,
        )

    def whitener_report(prefix: str) -> dict:
        return {
            "shrinkage": float(np.asarray(data[f"{prefix}_shrinkage"]).item()),
            "shrinkage_rule": str(np.asarray(data[f"{prefix}_shrinkage_rule"]).item()),
            "condition_number": float(np.asarray(data[f"{prefix}_condition_number"]).item()),
            "calibration_ids": np.asarray(data[f"{prefix}_calibration_ids"]).ravel().tolist(),
            "calibration_ids_sha256": str(
                np.asarray(data[f"{prefix}_calibration_ids_sha256"]).item()
            ),
            "calibration_base_ids": np.asarray(
                data[f"{prefix}_calibration_base_ids"]
            ).ravel().tolist(),
            "calibration_base_ids_sha256": str(np.asarray(
                data[f"{prefix}_calibration_base_ids_sha256"]
            ).item()),
            "fit_source_sha256": str(np.asarray(data[f"{prefix}_fit_source_sha256"]).item()),
            "n_pool": int(np.asarray(data[f"{prefix}_n_pool"]).item()),
            "n_features": int(np.asarray(data[f"{prefix}_n_features"]).item()),
        }

    def h2_pass(report: dict) -> bool:
        return bool(
            report["delta_r2_heldout"] > 0.0
            and report["distance_coefficient_directions_standardized"]["signs"] == [1, 1]
        )

    h2_passes = {
        "centered_cosine": h2_pass(h2_centered),
        "whitened_euclidean": h2_pass(h2_whitened),
    }
    if all(h2_passes.values()):
        h2_classification = "metric-robust incremental predictive support"
    elif len(set(h2_passes.values())) > 1:
        h2_classification = "metric-dependent evidence"
    else:
        h2_classification = "inconclusive"
    return {
        "identity": identity,
        "h1_psi": psi.as_dict(),
        "h1_psi_whitened": psi_whitened.as_dict(),
        "safe_lure_invariance": safe.as_dict(),
        "whitener_audit": {
            "primary": whitener_report("whitener"),
            "index63": whitener_report("secondary_index63_whitener"),
            **({"bst_texthead": whitener_report("secondary_bst_texthead_whitener")}
               if arm == "bst" else {}),
        },
        "mandatory_secondaries": secondaries,
        "five_condition_behavior": {
            condition: {
                endpoint: {
                    "correct_count": int(np.sum(data[f"behavior_{condition}_{endpoint}"])),
                    "n": int(np.asarray(data[f"behavior_{condition}_{endpoint}"]).size),
                    "accuracy": float(np.mean(data[f"behavior_{condition}_{endpoint}"])),
                }
                for endpoint in ("first_branch_accuracy", "exact_path_accuracy")
            }
            for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical")
        },
        "h2": {
            "margin_primary": {
                "centered_cosine": h2_centered,
                "whitened_euclidean": h2_whitened,
                "both_metric_classification": h2_classification,
                "metric_pass": h2_passes,
            },
            "first_branch_accuracy_secondary": {
                "centered_cosine": h2_accuracy_centered,
                "whitened_euclidean": h2_accuracy_whitened,
                "role": "secondary ceiling-sensitive analysis; cannot alter H2 classification",
                "promotion_prohibited": True,
            },
            "exact_path_accuracy_secondary": {
                "status": str(np.asarray(data["secondary_exact_path_status"]).item()),
                "centered_cosine": h2_exact_path_centered,
                "whitened_euclidean": h2_exact_path_whitened,
                "role": "secondary ceiling-sensitive analysis; cannot alter H2 classification",
                "promotion_prohibited": True,
            },
        },
    }


def _h1_intersection_union(centered: E.ThreeArmReport,
                           whitened: E.ThreeArmReport) -> dict:
    """Frozen four-state H1 decision using only the NextLat-minus-BST seed contrast."""
    primary = {
        "centered_cosine": centered.by_name("nextlat_minus_bst"),
        "whitened_euclidean": whitened.by_name("nextlat_minus_bst"),
    }
    means_positive = {name: report.estimate > 0.0 for name, report in primary.items()}
    intervals_above_zero = {
        name: report.contrast.ci.ci_low > 0.0 for name, report in primary.items()
    }
    metric_pass = {
        name: means_positive[name] and intervals_above_zero[name] for name in primary
    }
    if all(metric_pass.values()):
        classification = "metric-robust confirmatory support"
    elif all(means_positive.values()):
        classification = "directionally consistent but unresolved evidence"
    elif any(means_positive.values()):
        classification = "metric-dependent evidence"
    else:
        classification = "no support"
    return {
        "contrast": "nextlat_minus_bst",
        "rule": "intersection-union over both frozen co-primary metrics",
        "classification": classification,
        "mean_positive": means_positive,
        "student_t_interval_lower_bound_above_zero": intervals_above_zero,
        "metric_pass": metric_pass,
        "inputs_used": [
            "centered_cosine NextLat-minus-BST paired-seed PSI",
            "whitened_euclidean NextLat-minus-BST paired-seed PSI",
        ],
        "inputs_forbidden_from_rescue": [
            "nPSI", "conditional item bootstrap", "index63", "BST TextHead",
            "raw cosine", "uncentered Euclidean", "intermediate layers",
        ],
    }


def evaluate_manifest(
    manifest_path: pathlib.Path, *, expected_seeds: list[int], n_boot: int
) -> tuple[dict, dict]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationRefused("evaluation manifest is unreadable") from exc
    allowed_manifest = {
        "schema", "analysis_seed", "expected_arms", "expected_seeds", "n_boot",
        "identity_domains", "h3_permanent_block", "extractor",
        "local_measurement_sources", "frozen_inputs", "cells",
    }
    if not isinstance(manifest, dict) or set(manifest) != allowed_manifest:
        raise EvaluationRefused(
            "reduced manifest has missing or unexpected fields; H3 analysis controls are forbidden"
        )
    if manifest.get("schema") != SCHEMA:
        raise EvaluationRefused(f"manifest schema must be {SCHEMA}")
    block = _verified_h3_permanent_block(manifest["h3_permanent_block"])
    analysis_seed = manifest.get("analysis_seed")
    if not isinstance(analysis_seed, int) or isinstance(analysis_seed, bool) or analysis_seed < 0:
        raise EvaluationRefused("analysis_seed must be a frozen nonnegative integer")
    if tuple(manifest.get("expected_arms", ())) != E.ARMS:
        raise EvaluationRefused(f"manifest expected_arms must be frozen to {list(E.ARMS)}")
    frozen_seeds = manifest.get("expected_seeds")
    if not isinstance(frozen_seeds, list) or sorted(map(int, frozen_seeds)) != expected_seeds:
        raise EvaluationRefused("requested seeds differ from the seed set frozen in the manifest")
    if int(manifest.get("n_boot", -1)) != n_boot:
        raise EvaluationRefused("requested bootstrap count differs from the frozen manifest")
    expected_domains = manifest.get("identity_domains")
    if not isinstance(expected_domains, dict) or set(expected_domains) != {"h1_quartet"}:
        raise EvaluationRefused("reduced manifest permits only the H1 quartet identity domain")
    if int(expected_domains["h1_quartet"].get("count", -1)) != EXPECTED_H1_ITEMS:
        raise EvaluationRefused(f"H1 scored quartet count must be exactly {EXPECTED_H1_ITEMS}")
    if not isinstance(manifest["extractor"], dict) or set(manifest["extractor"]) != {
        "path", "sha256",
    }:
        raise EvaluationRefused("GPU extractor record must bind only path and SHA-256")
    extractor = _bound_artifact(manifest["extractor"], "GPU extractor")
    local_source_records = manifest.get("local_measurement_sources")
    if not isinstance(local_source_records, dict) or set(local_source_records) != set(
        LOCAL_MEASUREMENT_SOURCE_FIELDS
    ):
        raise EvaluationRefused(
            "manifest must bind exactly local representations.py and evaluate.py"
        )
    local_measurement_sources = {}
    for relative in LOCAL_MEASUREMENT_SOURCE_FIELDS:
        bound = _bound_artifact(local_source_records[relative], f"local measurement {relative}")
        if pathlib.Path(bound["path"]) != (_REPO / relative).resolve():
            raise EvaluationRefused(f"manifest local measurement path differs for {relative}")
        local_measurement_sources[relative] = bound
    frozen_inputs = manifest.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict) or set(frozen_inputs) != {"e_lure"}:
        raise EvaluationRefused("reduced manifest permits only the frozen E_lure input")
    if not isinstance(frozen_inputs["e_lure"], dict) or set(frozen_inputs["e_lure"]) != {
        "path", "sha256",
    }:
        raise EvaluationRefused("frozen E_lure record must bind only path and SHA-256")
    bound_inputs = {"e_lure": _bound_artifact(frozen_inputs["e_lure"], "frozen E_lure input")}
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise EvaluationRefused("manifest cells must be a list")
    wanted = {(arm, seed) for arm in E.ARMS for seed in expected_seeds}
    try:
        got = {(str(cell.get("arm")), int(cell.get("seed", -1))) for cell in cells}
    except (AttributeError, TypeError, ValueError) as exc:
        raise EvaluationRefused("manifest cells contain an invalid arm/seed identity") from exc
    if got != wanted or len(cells) != len(wanted):
        raise EvaluationRefused(
            f"manifest must contain exactly the complete arm/seed matrix; missing={sorted(wanted-got)}, "
            f"extra={sorted(got-wanted)}"
        )
    exclusion = {
        "status": "PERMANENTLY_DROPPED_AFTER_D40_FEASIBILITY_GATE",
        "confirmatory_h3_included": False,
        "adaptation_checkpoints_included": False,
        "mechanism_probes_included": False,
        "h3_analysis_included": False,
        "unmatched_count": 4,
        "no_further_amendments_permitted": True,
        "block_sha256": block["sha256"],
        "block_sidecar_sha256": block["sidecar_sha256"],
    }
    analyzed_reports = []
    valid_cell_identities = []
    invalid_cells = []
    for cell in sorted(cells, key=lambda c: (E.ARMS.index(str(c["arm"])), int(c["seed"]))):
        try:
            identity, data = _load_reduced_cell(
                cell, expected_domains, block, local_measurement_sources
            )
            cell_report = _reduced_cell_metrics(
                identity, data, analysis_seed=analysis_seed, n_boot=n_boot
            )
        except Exception as exc:
            reason = str(exc)
            invalid_cells.append({
                "arm": str(cell.get("arm")),
                "seed": int(cell.get("seed", -1)),
                "evidence_npz": str(cell.get("evidence_npz", "")),
                "reason_code": _invalid_reason_code(reason),
                "reason": reason,
            })
            continue
        valid_cell_identities.append(identity)
        analyzed_reports.append(cell_report)
    receipt_base = {
        "schema": RECEIPT_SCHEMA,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "analyzers": {
            "script": {"path": str(pathlib.Path(__file__).resolve()), "sha256": sha256_file(__file__)},
            "evaluate_module": {
                "path": str((_REPO / "src/lurestar/evaluate.py").resolve()),
                "sha256": sha256_file(_REPO / "src/lurestar/evaluate.py"),
            },
        },
        "expected_arms": list(E.ARMS), "expected_seeds": expected_seeds, "n_boot": n_boot,
        "identity_domains": expected_domains, "h3_permanent_block": block,
        "h3_exclusion": exclusion, "gpu_extractor": extractor, "frozen_inputs": bound_inputs,
        "local_measurement_sources": local_measurement_sources,
        "manipulation_failures": dict(MANIPULATION_FAILURES),
    }
    if invalid_cells:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "INVALID_INCOMPLETE",
            "primary_status": "INVALID_INCOMPLETE",
            "analysis_seed": analysis_seed,
            "invalid_cells": invalid_cells,
            "valid_cell_identities": valid_cell_identities,
            "cells": [],
            "seed_level_contrasts": None,
            "h1_confirmatory_classification": None,
            "nulls": _null_contract(None),
            "manipulation_failures": dict(MANIPULATION_FAILURES),
            "h3_exclusion": exclusion,
        }
        _validate_terminal_contract(report)
        return report, {
            **receipt_base,
            "status": "INVALID_INCOMPLETE",
            "primary_status": "INVALID_INCOMPLETE",
            "invalid_cells": invalid_cells,
            "inputs": valid_cell_identities,
        }

    reports = analyzed_reports
    calibration_hashes = {
        cell["whitener_audit"]["primary"]["calibration_ids_sha256"] for cell in reports
    }
    if len(calibration_hashes) != 1:
        raise EvaluationRefused("cells do not reuse one global frozen whitening-calibration split")
    calibration_base_inventories = {
        tuple(cell["whitener_audit"]["primary"]["calibration_base_ids"])
        for cell in reports
    }
    if len(calibration_base_inventories) != 1:
        raise EvaluationRefused(
            "all cells must reuse the same 400 ordered calibration base identities"
        )
    calibration_base_ids = next(iter(calibration_base_inventories))
    psi_by_arm = {arm: {} for arm in E.ARMS}
    psi_whitened_by_arm = {arm: {} for arm in E.ARMS}
    for cell_report in reports:
        arm, seed = cell_report["identity"]["arm"], cell_report["identity"]["seed"]
        psi_by_arm[arm][seed] = cell_report["h1_psi"]["psi"]
        psi_whitened_by_arm[arm][seed] = cell_report["h1_psi_whitened"]["psi"]
    rng = np.random.default_rng(analysis_seed)
    centered_seed_report = E.three_arm_contrasts(psi_by_arm, rng=rng, n_boot=n_boot)
    whitened_seed_report = E.three_arm_contrasts(
        psi_whitened_by_arm, rng=rng, n_boot=n_boot
    )
    seed_level = {
        "h1_psi": centered_seed_report.as_dict(),
        "h1_psi_whitened": whitened_seed_report.as_dict(),
    }
    h1_classification = _h1_intersection_union(centered_seed_report, whitened_seed_report)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "primary_status": h1_classification["classification"],
        "invalid_cells": [],
        "analysis_seed": analysis_seed,
        "analysis_policy": {
            "outcomes_inspected_before_freeze": False,
            "primary_h1_contrast": "nextlat_minus_bst",
            "co_primary_metrics": ["centered_cosine", "whitened_euclidean"],
            "multiplicity": "both co-primary H1 metrics reported; no outcome-selected metric",
            "seed_interval": "two-sided paired Student-t 95% interval",
            "item_bootstrap_scope": "conditional within-checkpoint uncertainty only",
            "npsi_role": "mandatory diagnostic only; cannot rescue either co-primary metric",
        },
        "h3_exclusion": exclusion,
        "cells": reports,
        "seed_level_contrasts": seed_level,
        "h1_confirmatory_classification": h1_classification,
        "nulls": _null_contract(h1_classification["classification"]),
        "manipulation_failures": dict(MANIPULATION_FAILURES),
    }
    _validate_terminal_contract(report)
    receipt = {
        **receipt_base,
        "status": "COMPLETE",
        "primary_status": h1_classification["classification"],
        "invalid_cells": [],
        "whitener_calibration_ids_sha256": next(iter(calibration_hashes)),
        "whitener_calibration_base_ids": list(calibration_base_ids),
        "whitener_calibration_base_ids_sha256": item_ids_sha256(
            np.asarray(calibration_base_ids)
        ),
        "inputs": [r["identity"] for r in reports],
    }
    return report, receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236, 1237, 1238])
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.n_boot < 100:
        print("[evaluate_lurestar] REFUSED: unique seeds and n_boot >= 100 required", file=sys.stderr)
        return 2
    try:
        report, receipt = evaluate_manifest(
            pathlib.Path(args.manifest).resolve(), expected_seeds=sorted(args.seeds),
            n_boot=args.n_boot,
        )
        output = pathlib.Path(args.output).resolve()
        atomic_write_json(output, report)
        receipt["report"] = {"path": str(output), "sha256": sha256_file(output)}
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        atomic_write_json(receipt_path, receipt)
        atomic_write_text(
            receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
            sha256_file(receipt_path) + "  " + receipt_path.name + "\n",
        )
    except (EvaluationRefused, OSError, ValueError) as exc:
        print(f"[evaluate_lurestar] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(output), "receipt": str(receipt_path)}, sort_keys=True))
    return 2 if report.get("status") == "INVALID_INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
