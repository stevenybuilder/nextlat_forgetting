"""Pure, outcome-blind analysis primitives for CFS-1.

CFS-1 is a new causal forgetting experiment, not a revival of the retired Lure-Star
H3 branch.  It has a 2 x 2 randomized update intervention:

``overlap in {high, low}`` x ``future_relation in {different, same}``.

The primary estimand is calculated *within each independently trained NextLat parent*.
For each parent, the two fixed update-stream episodes are averaged within condition and
the difference in differences in correct-first-branch margin erosion is reported:

    (high,different - high,same) - (low,different - low,same).

The parent checkpoint—not an item, adaptation branch, or episode—is the inferential
unit.  Item bootstraps quantify conditional within-branch uncertainty only and may not
replace the eight-parent analysis.  This module intentionally has no torch dependency
and contains no file-system or GPU access, so its statistics can be tested without
inspecting experimental outcomes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "CONDITIONS",
    "EPISODES",
    "EXPECTED_PARENT_COUNT",
    "BASE_TRAINING_STEPS",
    "ADAPTATION_STEPS",
    "PRIMARY_ENDPOINT",
    "SECONDARY_ENDPOINTS",
    "CFS1EvaluationError",
    "condition_key",
    "expected_branch_keys",
    "margin_erosion",
    "paired_student_t_ci",
    "exact_two_sided_sign_flip_p",
    "minimum_detectable_effect",
    "paired_parent_summary",
    "conditional_item_bootstrap",
    "holm_adjust",
    "difference_in_differences",
    "parent_episode_mean_did",
]


CONDITIONS: tuple[tuple[str, str], ...] = (
    ("high", "different"),
    ("low", "different"),
    ("high", "same"),
    ("low", "same"),
)
EPISODES: tuple[int, int] = (0, 1)
EXPECTED_PARENT_COUNT = 8
BASE_TRAINING_STEPS = 20_000
ADAPTATION_STEPS = 500
PRIMARY_ENDPOINT = "correct_first_branch_margin_erosion"
SECONDARY_ENDPOINTS: tuple[str, ...] = (
    "retention_cross_entropy_increase",
    "retention_exact_path_accuracy_loss",
    "adaptation_acquisition",
    "global_control_margin_erosion",
    "penultimate_state_drift",
)


class CFS1EvaluationError(ValueError):
    """Raised when a CFS-1 statistical or identity contract is violated."""


def condition_key(overlap: str, future_relation: str) -> str:
    if (overlap, future_relation) not in CONDITIONS:
        raise CFS1EvaluationError(
            "condition must be one of high/low x different/same; "
            f"got overlap={overlap!r}, future_relation={future_relation!r}"
        )
    return f"{overlap}_{future_relation}"


def expected_branch_keys(parent_ids: Sequence[str]) -> set[tuple[str, int, str, str]]:
    """Return the complete 8 x 2 x 2 x 2 CFS-1 branch identity lattice.

    This deliberately returns full identity tuples rather than a branch count.  A count
    of 64 alone could hide a duplicate branch paired with a missing condition.
    """
    parents = tuple(str(parent) for parent in parent_ids)
    if len(parents) != EXPECTED_PARENT_COUNT or len(set(parents)) != len(parents):
        raise CFS1EvaluationError(
            f"CFS-1 primary analysis requires exactly {EXPECTED_PARENT_COUNT} unique parents"
        )
    return {
        (parent, episode, overlap, relation)
        for parent in parents
        for episode in EPISODES
        for overlap, relation in CONDITIONS
    }


def _finite_1d(values: Sequence[float] | np.ndarray, *, label: str, minimum: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum:
        raise CFS1EvaluationError(f"{label} must be a finite 1-D array with at least {minimum} units")
    if not np.all(np.isfinite(array)):
        raise CFS1EvaluationError(f"{label} contains non-finite values")
    return array


def margin_erosion(pre_adaptation_margin: Sequence[float] | np.ndarray,
                   post_adaptation_margin: Sequence[float] | np.ndarray) -> np.ndarray:
    """Correct-first-branch margin loss, with positive values denoting forgetting."""
    before = _finite_1d(pre_adaptation_margin, label="pre_adaptation_margin", minimum=1)
    after = _finite_1d(post_adaptation_margin, label="post_adaptation_margin", minimum=1)
    if before.shape != after.shape:
        raise CFS1EvaluationError(
            f"pre/post margin shapes differ: {before.shape} versus {after.shape}"
        )
    return before - after


def exact_two_sided_sign_flip_p(differences: Sequence[float] | np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value over all ``2**n`` assignments."""
    diffs = _finite_1d(differences, label="per-parent differences")
    if diffs.size > 20:
        raise CFS1EvaluationError("exact sign-flip enumeration is limited to at most 20 parents")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=diffs.size)), dtype=np.float64)
    randomized = np.abs(signs @ diffs) / diffs.size
    return float(np.mean(randomized >= abs(float(diffs.mean())) - 1e-12))


def paired_student_t_ci(differences: Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> dict:
    """Two-sided paired Student-t interval for independently trained parents."""
    diffs = _finite_1d(differences, label="per-parent differences")
    if not 0.0 < alpha < 1.0:
        raise CFS1EvaluationError("alpha must be in (0, 1)")
    estimate = float(diffs.mean())
    standard_error = float(diffs.std(ddof=1) / math.sqrt(diffs.size))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, diffs.size - 1))
    half_width = critical * standard_error
    return {
        "estimate": estimate,
        "ci_low": float(estimate - half_width),
        "ci_high": float(estimate + half_width),
        "se": standard_error,
        "n_units": int(diffs.size),
        "unit": "independently trained parent checkpoint",
        "alpha": float(alpha),
        "df": int(diffs.size - 1),
        "t_critical": critical,
        "method": "two-sided paired Student-t interval of per-parent episode-mean DIDs",
    }


def _solve_noncentrality(t_critical: float, df: int, power: float) -> float:
    def attained(ncp: float) -> float:
        return float(stats.nct.sf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp))

    low, high = 0.0, max(1.0, t_critical)
    for _ in range(100):
        if attained(high) >= power:
            break
        high *= 2.0
    else:  # pragma: no cover - a valid power target reaches this long before 100 rounds
        return float("inf")
    for _ in range(200):
        middle = (low + high) / 2.0
        if attained(middle) < power:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def minimum_detectable_effect(
    differences: Sequence[float] | np.ndarray, *, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """Retrospective paired-parent MDE, always reported next to the primary estimate."""
    diffs = _finite_1d(differences, label="per-parent differences")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise CFS1EvaluationError("alpha and power must each be in (0, 1)")
    n = int(diffs.size)
    sd = float(diffs.std(ddof=1))
    df = n - 1
    t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    noncentrality = _solve_noncentrality(t_critical, df, power)
    sign_flip_floor = float(2.0 ** (1 - n))
    return {
        "mde": float(noncentrality * sd / math.sqrt(n)),
        "sd_per_parent": sd,
        "n_parents": n,
        "alpha": float(alpha),
        "power": float(power),
        "sign_flip_p_floor": sign_flip_floor,
        "randomization_test_can_reject": bool(sign_flip_floor <= alpha),
        "method": "noncentral two-sided paired-t planning scale on observed per-parent DIDs",
        "caveat": "observed parent-level SD is retrospective; it is not a prospectively guaranteed power calculation",
    }


def paired_parent_summary(
    values_by_parent: Mapping[str, float], *, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """Report a single predeclared endpoint at the correct parent-level unit."""
    if not isinstance(values_by_parent, Mapping):
        raise TypeError("values_by_parent must map parent_id to one episode-mean DID")
    parents = tuple(sorted(str(parent) for parent in values_by_parent))
    if len(parents) != EXPECTED_PARENT_COUNT or len(set(parents)) != len(parents):
        raise CFS1EvaluationError(
            f"primary CFS-1 summary requires exactly {EXPECTED_PARENT_COUNT} parent IDs"
        )
    values = _finite_1d(
        [values_by_parent[parent] for parent in parents], label="per-parent episode-mean DIDs"
    )
    ci = paired_student_t_ci(values, alpha=alpha)
    loso = []
    for index, omitted_parent in enumerate(parents):
        retained = np.delete(values, index)
        loso.append({
            "omitted_parent_id": omitted_parent,
            "n_parents": int(retained.size),
            "estimate": float(retained.mean()),
            "ci": paired_student_t_ci(retained, alpha=alpha),
        })
    sd = float(values.std(ddof=1))
    return {
        "inferential_unit": "independently trained parent checkpoint",
        "parent_ids": list(parents),
        "per_parent_episode_mean_did": {
            parent: float(values_by_parent[parent]) for parent in parents
        },
        "estimate": float(values.mean()),
        "ci": ci,
        "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(values),
        "minimum_detectable_effect": minimum_detectable_effect(values, alpha=alpha, power=power),
        "paired_standardized_effect": None if sd == 0.0 else float(values.mean() / sd),
        "leave_one_parent_out": loso,
    }


def conditional_item_bootstrap(
    item_differences: Sequence[float] | np.ndarray,
    *, rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """Conditional item-level interval, explicitly barred from parent-level inference."""
    values = _finite_1d(item_differences, label="within-branch item differences")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    if n_boot < 100 or not 0.0 < alpha < 1.0:
        raise CFS1EvaluationError("n_boot must be >= 100 and alpha must be in (0, 1)")
    # Bound allocation without changing the bootstrap estimand.
    chunk = max(1, min(int(n_boot), 20_000_000 // values.size))
    draws = np.empty(int(n_boot), dtype=np.float64)
    offset = 0
    while offset < n_boot:
        count = min(chunk, n_boot - offset)
        indices = rng.integers(0, values.size, size=(count, values.size))
        draws[offset:offset + count] = values[indices].mean(axis=1)
        offset += count
    low, high = np.quantile(draws, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "se": float(draws.std(ddof=1)),
        "n_items": int(values.size),
        "n_boot": int(n_boot),
        "unit": "items conditional on one fixed parent/checkpoint/episode/condition",
        "scope": "conditional uncertainty only; cannot replace eight-parent inference or primary p-value",
        "method": "paired percentile bootstrap of within-item differences",
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust a predeclared family of secondary exact p-values."""
    if not isinstance(p_values, Mapping) or not p_values:
        raise CFS1EvaluationError("p_values must be a nonempty mapping")
    ordered = sorted((str(name), float(value)) for name, value in p_values.items())
    if len({name for name, _ in ordered}) != len(ordered) or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0 for _, value in ordered
    ):
        raise CFS1EvaluationError("Holm p-values must be unique finite values in [0, 1]")
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[name] = float(running)
    return adjusted


def difference_in_differences(condition_means: Mapping[str, float]) -> float:
    """The frozen causal contrast of a single parent/episode or episode-mean cell."""
    expected = {condition_key(*condition) for condition in CONDITIONS}
    if not isinstance(condition_means, Mapping) or set(condition_means) != expected:
        raise CFS1EvaluationError(
            "condition means must contain exactly high_different, low_different, high_same, low_same"
        )
    values = {name: float(value) for name, value in condition_means.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise CFS1EvaluationError("condition means contain non-finite values")
    return float(
        (values["high_different"] - values["high_same"])
        - (values["low_different"] - values["low_same"])
    )


def parent_episode_mean_did(
    values_by_episode: Mapping[int, Mapping[str, float]]
) -> tuple[float, dict[int, float]]:
    """Average two fixed episodes within a parent before cross-parent inference."""
    if set(values_by_episode) != set(EPISODES):
        raise CFS1EvaluationError("each parent must contain exactly fixed episodes 0 and 1")
    per_episode = {
        int(episode): difference_in_differences(values_by_episode[episode])
        for episode in EPISODES
    }
    return float(np.mean(list(per_episode.values()))), per_episode
