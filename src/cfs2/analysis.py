"""Fixed, parent-level scientific analysis for the complete CFS-2 matrix.

The module contains no model or file-system access.  It takes the 64 frozen
branch outcomes and their 64 mandatory activation-patching artifacts only after
an external integrity preflight has succeeded.  Every contrast is fixed by the
2 x 2 design; layers, controls, parents, episodes, and cells are never selected
from observed effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from cfs1 import evaluate as _stats
from cfs2.adaptation import CFS2_ARMS, CFS2_EPISODES


REPORT_SCHEMA = "nextlat_forgetting/cfs2_scientific_analysis/1"
EXPECTED_PARENT_COUNT = 8
PATCH_LAYERS = (3, 7, 10)
PATCH_CONTROLS = (
    "matching_parent",
    "unrelated_anchor",
    "norm_matched_random_subspace",
)
PRIMARY_ENDPOINT = "correct_first_branch_margin_erosion"
ENDPOINTS = (
    PRIMARY_ENDPOINT,
    "retention_cross_entropy_increase",
    "retention_exact_path_accuracy_loss",
    "adaptation_acquisition",
    "global_control_margin_erosion",
    "penultimate_state_drift",
)


class CFS2AnalysisError(ValueError):
    """A complete CFS-2 outcome matrix does not satisfy the fixed analysis."""


@dataclass(frozen=True)
class BranchOutcome:
    job_id: str
    parent_id: str
    episode: int
    overlap: str
    future_relation: str
    item_ids: np.ndarray
    arrays: Mapping[str, np.ndarray]
    pregeometry: float | None = None

    @property
    def identity(self) -> tuple[str, int, str, str]:
        return self.parent_id, self.episode, self.overlap, self.future_relation

    @property
    def condition(self) -> str:
        return f"{self.overlap}_{self.future_relation}"


@dataclass(frozen=True)
class PatchOutcome:
    job_id: str
    parent_id: str
    episode: int
    overlap: str
    future_relation: str
    probe_ids: np.ndarray
    effects: Mapping[int, Mapping[str, np.ndarray]]

    @property
    def identity(self) -> tuple[str, int, str, str]:
        return self.parent_id, self.episode, self.overlap, self.future_relation


_ENDPOINT_FIELDS: dict[str, tuple[str, ...]] = {
    PRIMARY_ENDPOINT: (
        "pre_correct_first_branch_margin",
        "post_correct_first_branch_margin",
    ),
    "retention_cross_entropy_increase": (
        "pre_retention_cross_entropy",
        "post_retention_cross_entropy",
    ),
    "retention_exact_path_accuracy_loss": (
        "pre_retention_exact_path_accuracy",
        "post_retention_exact_path_accuracy",
    ),
    "adaptation_acquisition": ("adaptation_acquisition",),
    "global_control_margin_erosion": (
        "pre_global_control_margin",
        "post_global_control_margin",
    ),
    "penultimate_state_drift": ("penultimate_state_drift",),
}


def _condition(overlap: str, future_relation: str) -> str:
    value = f"{overlap}_{future_relation}"
    if value not in CFS2_ARMS:
        raise CFS2AnalysisError(f"condition is outside CFS-2: {value}")
    return value


def _finite_1d(value: Any, *, label: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise CFS2AnalysisError(f"{label} must be a nonempty finite 1-D array")
    if size is not None and array.size != size:
        raise CFS2AnalysisError(f"{label} has {array.size} items; expected {size}")
    return array


def _expected(parent_ids: Sequence[str]) -> set[tuple[str, int, str, str]]:
    parents = tuple(str(value) for value in parent_ids)
    if len(parents) != EXPECTED_PARENT_COUNT or len(set(parents)) != len(parents):
        raise CFS2AnalysisError("CFS-2 requires exactly eight unique parent checkpoints")
    return {
        (parent, episode, *arm.split("_", 1))
        for parent in parents
        for episode in CFS2_EPISODES
        for arm in CFS2_ARMS
    }


def difference_in_differences(values: Mapping[str, float]) -> float:
    """Fixed CFS-2 overlap x future-relation interaction."""
    if set(values) != set(CFS2_ARMS):
        raise CFS2AnalysisError("DID requires exactly all four CFS-2 conditions")
    numbers = {name: float(values[name]) for name in CFS2_ARMS}
    if not all(np.isfinite(value) for value in numbers.values()):
        raise CFS2AnalysisError("DID condition means must be finite")
    return float(
        (numbers["high_different"] - numbers["high_same"])
        - (numbers["low_different"] - numbers["low_same"])
    )


def _parent_summary(values: Mapping[str, float], *, estimand: str) -> dict[str, Any]:
    parents = sorted(str(parent) for parent in values)
    if len(parents) != EXPECTED_PARENT_COUNT or len(set(parents)) != len(parents):
        raise CFS2AnalysisError(f"{estimand} requires all eight parents")
    vector = _finite_1d([values[parent] for parent in parents], label=estimand, size=8)
    ci = _stats.paired_student_t_ci(vector)
    ci["method"] = f"two-sided Student-t interval across independent parents: {estimand}"
    loso = []
    for index, omitted in enumerate(parents):
        retained = np.delete(vector, index)
        interval = _stats.paired_student_t_ci(retained)
        interval["method"] = f"leave-one-parent-out Student-t interval: {estimand}"
        loso.append({"omitted_parent_id": omitted, "estimate": float(retained.mean()), "ci": interval})
    sd = float(vector.std(ddof=1))
    return {
        "status": "COMPLETE",
        "estimand": estimand,
        "inferential_unit": "independently trained parent checkpoint",
        "parent_ids": parents,
        "per_parent_values": {parent: float(values[parent]) for parent in parents},
        "estimate": float(vector.mean()),
        "ci": ci,
        "exact_two_sided_sign_flip_p": _stats.exact_two_sided_sign_flip_p(vector),
        "minimum_detectable_effect": _stats.minimum_detectable_effect(vector),
        "paired_standardized_effect": None if sd == 0.0 else float(vector.mean() / sd),
        "leave_one_parent_out": loso,
    }


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm adjustment ordered by observed p-value, with stable name ties."""
    ordered = sorted(
        ((str(name), float(value)) for name, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    if not ordered or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for _, value in ordered):
        raise CFS2AnalysisError("Holm inputs must be finite p-values in [0,1]")
    result: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        result[name] = float(running)
    return result


def _endpoint_array(branch: BranchOutcome, endpoint: str) -> np.ndarray:
    arrays = branch.arrays
    if endpoint == PRIMARY_ENDPOINT:
        return _finite_1d(arrays[_ENDPOINT_FIELDS[endpoint][0]], label=endpoint) - _finite_1d(
            arrays[_ENDPOINT_FIELDS[endpoint][1]], label=endpoint
        )
    if endpoint == "retention_cross_entropy_increase":
        return _finite_1d(arrays["post_retention_cross_entropy"], label=endpoint) - _finite_1d(
            arrays["pre_retention_cross_entropy"], label=endpoint
        )
    if endpoint == "retention_exact_path_accuracy_loss":
        return _finite_1d(arrays["pre_retention_exact_path_accuracy"], label=endpoint) - _finite_1d(
            arrays["post_retention_exact_path_accuracy"], label=endpoint
        )
    if endpoint == "global_control_margin_erosion":
        return _finite_1d(arrays["pre_global_control_margin"], label=endpoint) - _finite_1d(
            arrays["post_global_control_margin"], label=endpoint
        )
    return _finite_1d(arrays[endpoint], label=endpoint)


def _endpoint_availability(branches: Sequence[BranchOutcome]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for endpoint, fields in _ENDPOINT_FIELDS.items():
        present = [all(field in branch.arrays for field in fields) for branch in branches]
        if any(present) and not all(present):
            raise CFS2AnalysisError(f"{endpoint} is present in only part of the 64-branch matrix")
        result[endpoint] = all(present)
    if not result[PRIMARY_ENDPOINT]:
        raise CFS2AnalysisError("the primary pre/post correct-branch margin is absent")
    return result


def _validate_inputs(
    branches: Sequence[BranchOutcome], patches: Sequence[PatchOutcome]
) -> tuple[list[str], dict[str, bool]]:
    if len(branches) != 64:
        raise CFS2AnalysisError(f"CFS-2 needs 64 branch outcomes, received {len(branches)}")
    if len(patches) != 64:
        raise CFS2AnalysisError(f"CFS-2 needs 64 patch outcomes, received {len(patches)}")
    parent_ids = sorted({branch.parent_id for branch in branches})
    expected = _expected(parent_ids)
    by_identity = {branch.identity: branch for branch in branches}
    if len(by_identity) != 64 or set(by_identity) != expected:
        raise CFS2AnalysisError("branch outcomes do not cover the exact 64-cell CFS-2 lattice")
    by_job = {branch.job_id: branch for branch in branches}
    if len(by_job) != 64:
        raise CFS2AnalysisError("branch job IDs are not unique")
    availability = _endpoint_availability(branches)
    reference_ids = np.asarray(branches[0].item_ids).astype(str)
    for branch in branches:
        _condition(branch.overlap, branch.future_relation)
        ids = np.asarray(branch.item_ids)
        if ids.ndim != 1 or ids.size == 0 or len(set(map(str, ids.tolist()))) != ids.size:
            raise CFS2AnalysisError(f"{branch.job_id} item IDs must be nonempty, ordered, and unique")
        if not np.array_equal(ids.astype(str), reference_ids):
            raise CFS2AnalysisError("all 64 branches must share one ordered retention-probe identity vector")
        for endpoint, available in availability.items():
            if available:
                value = _endpoint_array(branch, endpoint)
                if value.size != ids.size:
                    raise CFS2AnalysisError(f"{branch.job_id} {endpoint} does not align to probe IDs")

    patch_by_job = {patch.job_id: patch for patch in patches}
    if len(patch_by_job) != 64 or set(patch_by_job) != set(by_job):
        raise CFS2AnalysisError("patch outcomes do not cover the same exact 64 jobs")
    for job_id, patch in patch_by_job.items():
        branch = by_job[job_id]
        if patch.identity != branch.identity:
            raise CFS2AnalysisError(f"patch identity differs from branch identity for {job_id}")
        if set(patch.effects) != set(PATCH_LAYERS):
            raise CFS2AnalysisError(f"{job_id} must retain fixed patch layers 3, 7, and 10")
        if not np.array_equal(np.asarray(patch.probe_ids).astype(str), np.asarray(branch.item_ids).astype(str)):
            raise CFS2AnalysisError(f"{job_id} patch probes differ from ordered branch probes")
        for layer in PATCH_LAYERS:
            if set(patch.effects[layer]) != set(PATCH_CONTROLS):
                raise CFS2AnalysisError(f"{job_id} layer {layer} lacks all named patch controls")
            for control in PATCH_CONTROLS:
                _finite_1d(
                    patch.effects[layer][control],
                    label=f"{job_id} layer {layer} {control}",
                    size=np.asarray(branch.item_ids).size,
                )
    return parent_ids, availability


def _endpoint_report(
    branches: Sequence[BranchOutcome], endpoint: str, parent_ids: Sequence[str]
) -> dict[str, Any]:
    values = {branch.identity: _endpoint_array(branch, endpoint) for branch in branches}
    per_parent_episode: dict[str, dict[int, float]] = {}
    per_parent: dict[str, float] = {}
    for parent in parent_ids:
        per_parent_episode[parent] = {}
        for episode in CFS2_EPISODES:
            cells = {
                arm: float(values[(parent, episode, *arm.split("_", 1))].mean())
                for arm in CFS2_ARMS
            }
            per_parent_episode[parent][episode] = difference_in_differences(cells)
        per_parent[parent] = float(np.mean(list(per_parent_episode[parent].values())))
    episode_summaries = {
        str(episode): _parent_summary(
            {parent: per_parent_episode[parent][episode] for parent in parent_ids},
            estimand=f"episode-{episode} {endpoint} DID",
        )
        for episode in CFS2_EPISODES
    }
    episode_difference = _parent_summary(
        {
            parent: per_parent_episode[parent][1] - per_parent_episode[parent][0]
            for parent in parent_ids
        },
        estimand=f"episode-1 minus episode-0 {endpoint} DID",
    )
    return {
        "status": "COMPLETE",
        "contrast": "(high,different - high,same) - (low,different - low,same)",
        "parent_episode_mean": _parent_summary(
            per_parent, estimand=f"episode-mean {endpoint} DID"
        ),
        "per_parent_per_episode_did": per_parent_episode,
        "episode_robustness": {
            "separate_episode_summaries": episode_summaries,
            "episode_difference": episode_difference,
            "selection_policy": "both fixed episodes retained; no episode may be selected",
        },
    }


def _conditional_primary_bootstraps(
    branches: Sequence[BranchOutcome], parent_ids: Sequence[str], *, seed: int, n_boot: int
) -> dict[str, Any]:
    by_identity = {branch.identity: _endpoint_array(branch, PRIMARY_ENDPOINT) for branch in branches}
    output: dict[str, Any] = {}
    for number, parent in enumerate(parent_ids):
        episode_item_dids = []
        for episode in CFS2_EPISODES:
            cell = {
                arm: by_identity[(parent, episode, *arm.split("_", 1))]
                for arm in CFS2_ARMS
            }
            episode_item_dids.append(
                cell["high_different"] - cell["high_same"]
                - cell["low_different"] + cell["low_same"]
            )
        item_did = np.mean(np.stack(episode_item_dids), axis=0)
        output[parent] = _stats.conditional_item_bootstrap(
            item_did, rng=np.random.default_rng(seed + number), n_boot=n_boot
        )
    return output


def _patching_report(
    patches: Sequence[PatchOutcome], parent_ids: Sequence[str]
) -> dict[str, Any]:
    patch_by_identity = {patch.identity: patch for patch in patches}
    layers: dict[str, Any] = {}
    comparison_summaries: dict[str, dict[str, Any]] = {}
    for layer in PATCH_LAYERS:
        controls: dict[str, Any] = {}
        control_parent_means: dict[str, dict[str, float]] = {}
        control_parent_episode_dids: dict[str, dict[str, dict[int, float]]] = {}
        for control in PATCH_CONTROLS:
            parent_means: dict[str, float] = {}
            parent_episode_dids: dict[str, dict[int, float]] = {}
            for parent in parent_ids:
                all_cells = []
                parent_episode_dids[parent] = {}
                for episode in CFS2_EPISODES:
                    means = {}
                    for arm in CFS2_ARMS:
                        patch = patch_by_identity[(parent, episode, *arm.split("_", 1))]
                        value = _finite_1d(patch.effects[layer][control], label="patch effect")
                        means[arm] = float(value.mean())
                        all_cells.append(means[arm])
                    parent_episode_dids[parent][episode] = difference_in_differences(means)
                parent_means[parent] = float(np.mean(all_cells))
            control_parent_means[control] = parent_means
            control_parent_episode_dids[control] = parent_episode_dids
            controls[control] = {
                "all-condition_mean_effect": _parent_summary(
                    parent_means, estimand=f"layer-{layer} {control} patched-minus-unpatched margin"
                ),
                "condition_did": _parent_summary(
                    {
                        parent: float(np.mean(list(parent_episode_dids[parent].values())))
                        for parent in parent_ids
                    },
                    estimand=f"layer-{layer} {control} episode-mean patch-effect DID",
                ),
                "per_parent_per_episode_did": parent_episode_dids,
            }
        comparisons = {}
        for control in PATCH_CONTROLS[1:]:
            comparison = f"matching_parent_minus_{control}"
            comparisons[comparison] = _parent_summary(
                {
                    parent: control_parent_means["matching_parent"][parent]
                    - control_parent_means[control][parent]
                    for parent in parent_ids
                },
                estimand=f"layer-{layer} {comparison} all-condition patch effect",
            )
            comparison_summaries[f"layer_{layer}:{comparison}"] = comparisons[comparison]
        layers[str(layer)] = {
            "controls": controls,
            "matching_parent_control_comparisons": comparisons,
        }
    adjusted = _holm_adjust({
        name: summary["exact_two_sided_sign_flip_p"]
        for name, summary in comparison_summaries.items()
    })
    for name, summary in comparison_summaries.items():
        summary["holm_adjusted_exact_two_sided_sign_flip_p"] = adjusted[name]
        summary["holm_family"] = "six fixed matching-parent versus named-control comparisons"
    return {
        "status": "COMPLETE_ALL_64_BRANCHES_ALL_FIXED_LAYERS",
        "role": "local activation intervention; not evidence of global mediation",
        "patch_layers": list(PATCH_LAYERS),
        "layer_selection_permitted": False,
        "layers": layers,
    }


def _geometry_report(
    branches: Sequence[BranchOutcome], parent_primary: Mapping[str, float], parent_ids: Sequence[str]
) -> dict[str, Any]:
    presence = [branch.pregeometry is not None for branch in branches]
    if not any(presence):
        return {"status": "NOT_AVAILABLE", "estimate": None, "null": None}
    if not all(presence):
        raise CFS2AnalysisError("pregeometry is present in only part of the 64-branch matrix")
    parent_geometry: dict[str, float] = {}
    for parent in parent_ids:
        values = np.asarray([float(branch.pregeometry) for branch in branches if branch.parent_id == parent])
        if values.size != 8 or not np.all(np.isfinite(values)) or not np.allclose(values, values[0], atol=0.0, rtol=0.0):
            raise CFS2AnalysisError(f"pregeometry is not immutable within parent {parent}")
        parent_geometry[parent] = float(values[0])
    x = np.asarray([parent_geometry[parent] for parent in parent_ids])
    y = np.asarray([parent_primary[parent] for parent in parent_ids])
    if np.allclose(x, x[0], atol=1e-12, rtol=0.0) or np.allclose(y, y[0], atol=1e-12, rtol=0.0):
        return {
            "status": "NOT_ESTIMABLE_CONSTANT_VALUE",
            "parent_geometry": parent_geometry,
            "causal_mediation_claim_permitted": False,
        }
    correlation, p_value = stats.pearsonr(x, y)
    slope, intercept, _, _, slope_se = stats.linregress(x, y)
    return {
        "status": "COMPLETE_NONCAUSAL",
        "parent_geometry": parent_geometry,
        "pearson_r": float(correlation),
        "two_sided_p": float(p_value),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_slope_se": float(slope_se),
        "causal_mediation_claim_permitted": False,
    }


def analyze_complete_matrix(
    branches: Sequence[BranchOutcome],
    patches: Sequence[PatchOutcome],
    *,
    analysis_seed: int,
    n_boot: int,
) -> dict[str, Any]:
    """Analyze all fixed CFS-2 cells, or raise without emitting partial results."""
    if isinstance(analysis_seed, bool) or not isinstance(analysis_seed, int) or analysis_seed < 0:
        raise CFS2AnalysisError("analysis_seed must be a nonnegative integer")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
        raise CFS2AnalysisError("n_boot must be an integer >= 100")
    parent_ids, availability = _validate_inputs(branches, patches)
    endpoint_reports = {
        endpoint: (
            _endpoint_report(branches, endpoint, parent_ids)
            if availability[endpoint]
            else {"status": "NOT_AVAILABLE", "estimate": None, "null": None}
        )
        for endpoint in ENDPOINTS
    }
    primary_summary = endpoint_reports[PRIMARY_ENDPOINT]["parent_episode_mean"]
    primary_positive = (
        primary_summary["estimate"] > 0.0
        and primary_summary["ci"]["ci_low"] > 0.0
        and primary_summary["exact_two_sided_sign_flip_p"] <= 0.05
    )
    secondary_names = [name for name in ENDPOINTS if name != PRIMARY_ENDPOINT and availability[name]]
    if secondary_names:
        adjusted = _holm_adjust({
            name: endpoint_reports[name]["parent_episode_mean"]["exact_two_sided_sign_flip_p"]
            for name in secondary_names
        })
        for name in secondary_names:
            endpoint_reports[name]["parent_episode_mean"]["holm_adjusted_exact_two_sided_sign_flip_p"] = adjusted[name]

    branch_cells = []
    patch_by_job = {patch.job_id: patch for patch in patches}
    for branch in sorted(branches, key=lambda value: value.job_id):
        patch = patch_by_job[branch.job_id]
        branch_cells.append({
            "job_id": branch.job_id,
            "parent_id": branch.parent_id,
            "episode": branch.episode,
            "overlap": branch.overlap,
            "future_relation": branch.future_relation,
            "n_items": int(np.asarray(branch.item_ids).size),
            "endpoints": {
                endpoint: (float(_endpoint_array(branch, endpoint).mean()) if availability[endpoint] else None)
                for endpoint in ENDPOINTS
            },
            "patching": {
                str(layer): {
                    control: float(_finite_1d(patch.effects[layer][control], label="patch effect").mean())
                    for control in PATCH_CONTROLS
                }
                for layer in PATCH_LAYERS
            },
        })

    parent_primary = primary_summary["per_parent_values"]
    secondary_null_text = {
        "retention_cross_entropy_increase": "no overlap-by-future-relation DID in retention cross-entropy increase",
        "retention_exact_path_accuracy_loss": "no overlap-by-future-relation DID in exact-path accuracy loss",
        "adaptation_acquisition": "no overlap-by-future-relation DID in update-stream acquisition",
        "global_control_margin_erosion": "no overlap-by-future-relation DID on untouched global-control margin erosion",
        "penultimate_state_drift": "no overlap-by-future-relation DID in penultimate-state drift",
    }
    secondary_nulls = {}
    for endpoint, text in secondary_null_text.items():
        available = availability[endpoint]
        summary = endpoint_reports[endpoint].get("parent_episode_mean") if available else None
        secondary_nulls[endpoint] = {
            "null": text,
            "status": "TESTED" if available else "NOT_AVAILABLE",
            "holm_rejected": (
                bool(summary["holm_adjusted_exact_two_sided_sign_flip_p"] <= 0.05)
                if summary is not None else None
            ),
        }
    nulls = {
        "CFS2_PRIMARY": {
            "null": "no positive overlap-by-future-relation DID in correct-first-branch margin erosion",
            "resolved_in_prespecified_direction": bool(primary_positive),
            "non_support_interpretation": "an unresolved result is not evidence of equivalence; interpret with the reported parent-level MDE",
        },
        "CFS2_GLOBAL_CONTROL": {
            "null": "no overlap-by-future-relation DID on untouched global-control margin erosion",
            "status": "TESTED" if availability["global_control_margin_erosion"] else "NOT_AVAILABLE",
        },
        "CFS2_SECONDARY_ENDPOINTS": secondary_nulls,
        "CFS2_PATCH_CONTROLS": {
            str(layer): {
                "matching_parent_vs_unrelated_anchor": "matching-parent patch recovery does not exceed unrelated-anchor recovery",
                "matching_parent_vs_norm_matched_random_subspace": "matching-parent patch recovery does not exceed norm-matched random-subspace recovery",
            }
            for layer in PATCH_LAYERS
        },
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE",
        "analysis_seed": analysis_seed,
        "n_boot": n_boot,
        "branch_count": 64,
        "parent_count": 8,
        "analysis_policy": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "primary_contrast": "(high,different - high,same) - (low,different - low,same)",
            "episode_policy": "calculate within parent x episode, average both fixed episodes within parent, then infer across parents",
            "inferential_unit": "independently trained parent checkpoint",
            "missing_cell_policy": "no imputation; any missing branch or patch artifact blocks all output",
            "layer_policy": "emit blocks 3, 7, and 10 without selection",
        },
        "primary_classification": (
            "confirmatory causal support" if primary_positive
            else "directionally positive but unresolved" if primary_summary["estimate"] > 0.0
            else "no confirmatory support"
        ),
        "causal_claim_permitted": bool(primary_positive),
        "branch_cells": branch_cells,
        "endpoints": endpoint_reports,
        "conditional_item_bootstrap_by_parent": _conditional_primary_bootstraps(
            branches, parent_ids, seed=analysis_seed, n_boot=n_boot
        ),
        "pregeometry_moderation": _geometry_report(branches, parent_primary, parent_ids),
        "activation_patching": _patching_report(patches, parent_ids),
        "nulls": nulls,
    }
