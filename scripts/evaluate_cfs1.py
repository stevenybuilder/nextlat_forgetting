#!/usr/bin/env python3
"""Fail-closed evaluator for CFS-1, a fresh randomized causal-forgetting study.

The evaluator accepts no partial matrix.  It first verifies all eight independent NextLat
parents, both fixed episodes, and all four randomized update conditions (64 branches) against
the opaque evidence contract.  Only then does it calculate the primary per-parent
difference-in-differences.  A failed branch produces a terminal ``INVALID_INCOMPLETE`` report
with no partial effects, p-values, or rankings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
from scipy import stats

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cfs1 import evaluate as E  # noqa: E402
import extract_cfs1_evidence as X  # noqa: E402


SCHEMA = "nextlat_forgetting/cfs1_evaluation_manifest/1"
REPORT_SCHEMA = "nextlat_forgetting/cfs1_evaluation_report/1"
RECEIPT_SCHEMA = "nextlat_forgetting/cfs1_evaluation_receipt/1"


class EvaluationRefused(RuntimeError):
    """Raised only when an evaluation manifest itself cannot be safely interpreted."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    return X.sha256_file(path)


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _bound_record(value: Any, *, base: pathlib.Path, label: str) -> dict[str, str]:
    try:
        return X._record(value, base=base, label=label)  # type: ignore[attr-defined]
    except X.ExtractionRefused as exc:
        raise EvaluationRefused(str(exc)) from exc


def _invalid_report(*, analysis_seed: int | None, reason: str, invalid_branches: list[dict]) -> dict:
    """A terminal result, intentionally free of every outcome-derived statistic."""
    return {
        "schema": REPORT_SCHEMA,
        "status": "INVALID_INCOMPLETE",
        "primary_status": "INVALID_INCOMPLETE",
        "analysis_seed": analysis_seed,
        "required_branch_count": 64,
        "invalid_branches": invalid_branches,
        "primary": None,
        "secondary_endpoints": None,
        "geometry_moderation": None,
        "penultimate_state_patching": None,
        "nulls": {
            "CFS1_PRIMARY": {
                "null": "the high-overlap conflicting update has no excess correct-first-branch margin erosion under the prespecified difference-in-differences",
                "non_support_interpretation": "an invalid or unresolved result is never evidence of equivalence, absence of forgetting, or absence of a causal effect",
                "causal_claim_permitted": False,
            },
        },
        "reason": reason,
    }


def _validate_terminal_contract(report: Mapping[str, Any]) -> None:
    expected = {
        "schema", "status", "primary_status", "analysis_seed", "required_branch_count",
        "invalid_branches", "primary", "secondary_endpoints", "geometry_moderation",
        "penultimate_state_patching", "nulls", "reason",
    }
    if set(report) != expected or report["status"] != "INVALID_INCOMPLETE":
        raise EvaluationRefused("terminal invalid report contract was altered")
    if report["primary"] is not None or report["secondary_endpoints"] is not None:
        raise EvaluationRefused("invalid report may not contain partial outcome statistics")
    if report["nulls"]["CFS1_PRIMARY"]["causal_claim_permitted"]:
        raise EvaluationRefused("invalid report permits a causal claim")


def _read_manifest(path: pathlib.Path) -> tuple[dict, dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationRefused("CFS-1 evaluation manifest is unreadable") from exc
    required = {"schema", "analysis_seed", "n_boot", "extraction_job"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise EvaluationRefused(f"evaluation manifest must contain exactly {sorted(required)}")
    if payload["schema"] != SCHEMA:
        raise EvaluationRefused(f"evaluation manifest schema must be {SCHEMA}")
    if isinstance(payload["analysis_seed"], bool) or not isinstance(payload["analysis_seed"], int) or payload["analysis_seed"] < 0:
        raise EvaluationRefused("analysis_seed must be a nonnegative integer")
    if isinstance(payload["n_boot"], bool) or not isinstance(payload["n_boot"], int) or payload["n_boot"] < 100:
        raise EvaluationRefused("n_boot must be an integer >= 100")
    return payload, _bound_record(payload["extraction_job"], base=path.parent, label="extraction job")


def _endpoint_values(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        E.PRIMARY_ENDPOINT: E.margin_erosion(
            arrays["pre_correct_first_branch_margin"], arrays["post_correct_first_branch_margin"]
        ),
        "retention_cross_entropy_increase": arrays["post_retention_cross_entropy"] - arrays["pre_retention_cross_entropy"],
        "retention_exact_path_accuracy_loss": arrays["pre_retention_exact_path_accuracy"] - arrays["post_retention_exact_path_accuracy"],
        "adaptation_acquisition": arrays["adaptation_acquisition"],
        "global_control_margin_erosion": arrays["pre_global_control_margin"] - arrays["post_global_control_margin"],
        "penultimate_state_drift": arrays["penultimate_state_drift"],
    }


def _primary_classification(summary: Mapping[str, Any]) -> tuple[str, bool]:
    estimate = float(summary["estimate"])
    interval_positive = float(summary["ci"]["ci_low"]) > 0.0
    randomized_significance = float(summary["exact_two_sided_sign_flip_p"]) <= 0.05
    if estimate > 0.0 and interval_positive and randomized_significance:
        return "confirmatory causal support", True
    if estimate > 0.0:
        return "directionally positive but unresolved", False
    return "no confirmatory support", False


def _geometry_moderation(parent_geometry: Mapping[str, float], parent_primary: Mapping[str, float]) -> dict:
    """Prespecified observational moderation, deliberately not a mediation claim."""
    parents = sorted(parent_primary)
    geometry = np.asarray([parent_geometry[parent] for parent in parents], dtype=np.float64)
    effect = np.asarray([parent_primary[parent] for parent in parents], dtype=np.float64)
    if np.allclose(geometry, geometry[0], rtol=0.0, atol=1e-12) or np.allclose(
        effect, effect[0], rtol=0.0, atol=1e-12
    ):
        return {
            "status": "NOT_ESTIMABLE_CONSTANT_VALUE",
            "role": "prespecified noncausal moderation diagnostic",
            "causal_mediation_claim_permitted": False,
            "reason": "geometry or parent-level primary effect is constant",
        }
    correlation, p_value = stats.pearsonr(geometry, effect)
    slope, intercept, _, _, slope_se = stats.linregress(geometry, effect)
    return {
        "status": "COMPLETE_NONCAUSAL",
        "role": "prespecified noncausal moderation diagnostic",
        "parent_ids": parents,
        "predictor": "pre-adaptation predictive geometry",
        "outcome": "per-parent episode-mean primary margin-erosion DID",
        "pearson_r": float(correlation),
        "two_sided_p": float(p_value),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_slope_se": float(slope_se),
        "causal_mediation_claim_permitted": False,
        "interpretation": "association with a randomized-intervention susceptibility estimate; it does not identify mediation",
    }


def _patching_report(branches: list[dict]) -> dict:
    reports = [branch["state_patching"] for branch in branches]
    complete = [report for report in reports if report["status"] == "COMPLETE_WITH_NAMED_CONTROLS"]
    if not complete:
        status = "NOT_RUN_COMMITTED_SECONDARY"
    elif len(complete) != len(reports):
        status = "INCOMPLETE_COMMITTED_SECONDARY"
    else:
        status = "COMPLETE_WITH_NAMED_CONTROLS"
    out = {
        "status": status,
        "role": "committed secondary local readout intervention, not primary evidence",
        "completion_required_for": "full CFS-1 project completion, not validity of the separately locked primary inference",
        "required_named_controls": [
            "patch_parent_state_effect", "patch_unrelated_anchor_effect",
            "patch_norm_matched_random_subspace_effect",
        ],
        "causal_mediation_claim_permitted": False,
        "interpretation": "even a completed patching result can support a local readout intervention only; it cannot establish global causal mediation of forgetting",
    }
    if complete:
        out["completed_branch_count"] = len(complete)
        out["per_branch_named_controls"] = [report["named_controls"] for report in complete]
    return out


def evaluate_manifest(path: os.PathLike[str] | str) -> tuple[dict, dict]:
    """Evaluate a frozen CFS-1 job only after atomic 64-branch preflight succeeds."""
    manifest_path = pathlib.Path(path).resolve()
    manifest, job_record = _read_manifest(manifest_path)
    analysis_seed, n_boot = int(manifest["analysis_seed"]), int(manifest["n_boot"])
    receipt_base = {
        "schema": RECEIPT_SCHEMA,
        "evaluation_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "extraction_job": job_record,
        "analyzers": {
            "evaluator": {"path": str(pathlib.Path(__file__).resolve()), "sha256": sha256_file(__file__)},
            "evaluation_module": {"path": str((_REPO / "src/cfs1/evaluate.py").resolve()), "sha256": sha256_file(_REPO / "src/cfs1/evaluate.py")},
            "evidence_contract": {"path": str((_REPO / "scripts/extract_cfs1_evidence.py").resolve()), "sha256": sha256_file(_REPO / "scripts/extract_cfs1_evidence.py")},
        },
        "analysis_seed": analysis_seed, "n_boot": n_boot,
    }
    try:
        job = X.load_job(job_record["path"])
        if job.sha256 != job_record["sha256"]:
            raise X.ExtractionRefused("extraction job changed after evaluation manifest was bound")
        if job.payload["analysis_seed"] != analysis_seed:
            raise X.ExtractionRefused("evaluation and extraction jobs use different analysis seeds")
        preflight = X.preflight_job(job.path)
    except X.ExtractionRefused as exc:
        report = _invalid_report(
            analysis_seed=analysis_seed, reason=str(exc), invalid_branches=[{"identity": None, "reason": str(exc)}]
        )
        _validate_terminal_contract(report)
        return report, {**receipt_base, "status": "INVALID_INCOMPLETE", "primary_status": "INVALID_INCOMPLETE", "preflight": None}
    if preflight["status"] != "COMPLETE":
        report = _invalid_report(
            analysis_seed=analysis_seed, reason="atomic 64-branch preflight failed",
            invalid_branches=preflight["invalid_branches"],
        )
        _validate_terminal_contract(report)
        return report, {**receipt_base, "status": "INVALID_INCOMPLETE", "primary_status": "INVALID_INCOMPLETE", "preflight": preflight}

    # This second load is safe only because the all-64 preflight just succeeded.  It is kept
    # separate so no failed branch can leak a partial metric into a terminal report.
    loaded: list[dict] = []
    try:
        for key, branch in sorted(job.branch_by_key.items()):
            evidence = X.validate_branch_evidence(
                branch["evidence_npz"]["path"], branch=branch, parent=job.parent_by_id[key[0]]
            )
            if evidence["sha256"] != branch["evidence_npz"]["sha256"]:
                raise X.ExtractionRefused(f"branch {key} evidence SHA-256 changed after preflight")
            loaded.append({**evidence, "branch": branch})
    except X.ExtractionRefused as exc:  # a changed file between phases is still terminal invalid
        report = _invalid_report(
            analysis_seed=analysis_seed, reason=str(exc), invalid_branches=[{"identity": None, "reason": str(exc)}]
        )
        _validate_terminal_contract(report)
        return report, {**receipt_base, "status": "INVALID_INCOMPLETE", "primary_status": "INVALID_INCOMPLETE", "preflight": preflight}

    retention_hashes = {item["branch"]["retention_probe_manifest"]["sha256"] for item in loaded}
    global_control_hashes = {item["branch"]["global_control_manifest"]["sha256"] for item in loaded}
    item_hashes = {item["retention_probe_item_ids_sha256"] for item in loaded}
    generator_hashes = {item["branch"]["generator_manifest"]["sha256"] for item in loaded}
    if not (len(retention_hashes) == len(global_control_hashes) == len(item_hashes) == len(generator_hashes) == 1):
        report = _invalid_report(
            analysis_seed=analysis_seed,
            reason="CFS-1 branches do not share one frozen generator, retention probe, global control, and ordered probe identities",
            invalid_branches=[],
        )
        _validate_terminal_contract(report)
        return report, {**receipt_base, "status": "INVALID_INCOMPLETE", "primary_status": "INVALID_INCOMPLETE", "preflight": preflight}

    values: dict[str, dict[int, dict[str, np.ndarray]]] = {
        parent: {episode: {} for episode in E.EPISODES} for parent in job.parent_by_id
    }
    geometry_values: dict[str, list[float]] = {parent: [] for parent in job.parent_by_id}
    for item in loaded:
        identity = item["identity"]
        parent, episode = identity["parent_id"], identity["episode"]
        key = E.condition_key(identity["overlap"], identity["future_relation"])
        values[parent][episode][key] = _endpoint_values(item["arrays"])
        geometry_values[parent].append(float(item["pre_adaptation_predictive_geometry"]))
    parent_geometry: dict[str, float] = {}
    for parent, observations in geometry_values.items():
        if len(observations) != 8 or not np.allclose(observations, observations[0], rtol=0.0, atol=0.0):
            report = _invalid_report(
                analysis_seed=analysis_seed,
                reason=f"pre-adaptation predictive geometry differs across branches of parent {parent!r}",
                invalid_branches=[],
            )
            _validate_terminal_contract(report)
            return report, {**receipt_base, "status": "INVALID_INCOMPLETE", "primary_status": "INVALID_INCOMPLETE", "preflight": preflight}
        parent_geometry[parent] = observations[0]

    parent_endpoint_dids: dict[str, dict[str, float]] = {endpoint: {} for endpoint in (E.PRIMARY_ENDPOINT, *E.SECONDARY_ENDPOINTS)}
    parent_episode_dids: dict[str, dict[int, float]] = {}
    conditional_primary_bootstrap: dict[str, dict] = {}
    for parent in sorted(values):
        for endpoint in parent_endpoint_dids:
            by_episode = {
                episode: {
                    condition: float(values[parent][episode][condition][endpoint].mean())
                    for condition in sorted(values[parent][episode])
                }
                for episode in E.EPISODES
            }
            mean_did, per_episode = E.parent_episode_mean_did(by_episode)
            parent_endpoint_dids[endpoint][parent] = mean_did
            if endpoint == E.PRIMARY_ENDPOINT:
                parent_episode_dids[parent] = per_episode
                item_did = np.mean(np.stack([
                    values[parent][episode]["high_different"][endpoint]
                    - values[parent][episode]["high_same"][endpoint]
                    - values[parent][episode]["low_different"][endpoint]
                    + values[parent][episode]["low_same"][endpoint]
                    for episode in E.EPISODES
                ]), axis=0)
                conditional_primary_bootstrap[parent] = E.conditional_item_bootstrap(
                    item_did, rng=np.random.default_rng(analysis_seed + len(conditional_primary_bootstrap)), n_boot=n_boot
                )

    primary_summary = E.paired_parent_summary(parent_endpoint_dids[E.PRIMARY_ENDPOINT])
    primary_status, causal_claim_permitted = _primary_classification(primary_summary)
    secondary = {
        endpoint: E.paired_parent_summary(parent_endpoint_dids[endpoint])
        for endpoint in E.SECONDARY_ENDPOINTS
    }
    holm = E.holm_adjust({
        endpoint: summary["exact_two_sided_sign_flip_p"] for endpoint, summary in secondary.items()
    })
    for endpoint, summary in secondary.items():
        summary["holm_adjusted_exact_two_sided_sign_flip_p"] = holm[endpoint]
        summary["role"] = "prespecified secondary endpoint; Holm-adjusted across all five CFS-1 secondaries"
    report = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "primary_status": primary_status,
        "analysis_seed": analysis_seed,
        "required_branch_count": 64,
        "invalid_branches": [],
        "analysis_policy": {
            "study": "CFS-1 is a new experiment and does not revive retired Lure-Star H3",
            "intervention": "randomized construction-level overlap x future-relation update streams",
            "primary_endpoint": E.PRIMARY_ENDPOINT,
            "primary_contrast": "(high,different - high,same) - (low,different - low,same)",
            "episode_policy": "calculate each DID per fixed episode then average episodes within parent before eight-parent inference",
            "inferential_unit": "independently trained parent checkpoint (n=8)",
            "conditional_item_bootstrap": "reported per parent only; never substitutes for the parent-level interval/p-value",
            "secondary_multiplicity": "five prespecified endpoints adjusted with Holm",
            "geometry_moderation": "noncausal; cannot be called mediation",
            "state_patching": "optional local readout intervention with named controls; cannot establish global mediation",
        },
        "primary": {
            **primary_summary,
            "endpoint": E.PRIMARY_ENDPOINT,
            "per_parent_per_episode_did": parent_episode_dids,
            "conditional_item_bootstrap_by_parent": conditional_primary_bootstrap,
            "classification": primary_status,
            "causal_claim_permitted": causal_claim_permitted,
        },
        "secondary_endpoints": secondary,
        "geometry_moderation": _geometry_moderation(parent_geometry, parent_endpoint_dids[E.PRIMARY_ENDPOINT]),
        "penultimate_state_patching": _patching_report(loaded),
        "nulls": {
            "CFS1_PRIMARY": {
                "null": "the high-overlap conflicting update has no excess correct-first-branch margin erosion under the prespecified difference-in-differences",
                "non_support_interpretation": "a nonpositive or unresolved result is not evidence of equivalence, absence of forgetting, or absence of a causal effect",
                "causal_claim_permitted": causal_claim_permitted,
            },
        },
        "reason": None,
    }
    receipt = {
        **receipt_base,
        "status": "COMPLETE", "primary_status": primary_status, "preflight": preflight,
        "parent_ids": sorted(job.parent_by_id), "branch_count": len(loaded),
        "shared_generator_manifest_sha256": next(iter(generator_hashes)),
        "shared_retention_probe_manifest_sha256": next(iter(retention_hashes)),
        "shared_global_control_manifest_sha256": next(iter(global_control_hashes)),
        "shared_retention_probe_item_ids_sha256": next(iter(item_hashes)),
        "input_evidence": [
            {"identity": item["identity"], "path": item["path"], "sha256": item["sha256"]}
            for item in loaded
        ],
    }
    return report, receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    output = pathlib.Path(args.output).resolve()
    try:
        report, receipt = evaluate_manifest(args.manifest)
        atomic_write_json(output, report)
        receipt["report"] = {"path": str(output), "sha256": sha256_file(output)}
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        atomic_write_json(receipt_path, receipt)
        _atomic_bytes(
            receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
            f"{sha256_file(receipt_path)}  {receipt_path.name}\n".encode("utf-8"),
        )
    except (EvaluationRefused, OSError, ValueError) as exc:
        print(f"[evaluate_cfs1] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(output), "receipt": str(receipt_path), "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
