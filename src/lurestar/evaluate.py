"""H1/H2 (and the shared H3 margin math) for Lure-Star — pure numpy, no torch.

Everything here consumes the numpy arrays produced by
``lurestar.representations.extract_positions`` and produces the preregistered numbers of
spec §6 and §10.  It is deliberately import-clean on a CPU-only host so that every
estimator is unit-testable without a GPU.

The three things this module refuses to let a caller get wrong:

1. **The centering pool** for the primary distance is an explicit argument, threaded
   through from :func:`lurestar.representations.centered_cosine_distance`.
2. **The whitening pool** for the robustness check is held out, enforced by item id.
3. **The inferential unit.**  Items are the unit for item-level intervals.  SEEDS are the
   unit for every cross-model contrast, and items never substitute for seeds
   (spec §6/H3: "Items do not substitute for independent training seeds"; the same rule
   governs the H1 model contrast).  The two live in separately named functions with
   different argument types — item functions take arrays, the seed function takes a
   mapping keyed by seed id — so an array of 20,000 items cannot be passed where five
   seeds belong.

**The design is three-armed** (spec §8, and `docs/DECISION_D20_competence_gate.md`
"Superseded in part"): NextLat, BST and GPT, width/depth-matched at 12L/6H/384 on
G(5,5). BST is competence-matched but adds a second transformer, more parameters, and
O(T²) prefix-suffix supervision. The paper's
Figure 6 puts it at ~99.9% where GPT sits at ~18.6%, which is 1/d chance — so
:data:`PREREGISTERED_CONTRASTS` fixes the priority order *before* any number exists:

1. NextLat - BST   primary, competence-matched;
2. NextLat - GPT   secondary, competence-confounded, reported with the confound attached;
3. BST - GPT       reference, shows how much of any effect is competence alone.

Every contrast is reported through :func:`contrast_with_mde`, which carries the estimate,
the seed-level interval, the exact sign-flip p, **and** the smallest effect that this many
seeds could have detected. The last one is not decoration: with five seeds the exact
randomization test still cannot reach p <= 0.05 at any effect size (its floor is 0.0625).
A writeup that reports the estimate without that number is claiming a null it never had
the resolution to see.
"""

from __future__ import annotations

import itertools
import hashlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from .representations import (
    BRANCH_MARGIN_INDEX,
    PSI_EXTRACTION_INDEX,
    CENTERING_POOL_POLICY,
    CenteringPool,
    LeakageError,
    Whitener,
    branch_margin,
    centered_cosine_distance,
    centering_mean,
)

__all__ = [
    "ARMS",
    "ContrastSpec",
    "PREREGISTERED_CONTRASTS",
    "MDEResult",
    "ContrastReport",
    "ArmPSI",
    "ThreeArmReport",
    "minimum_detectable_effect",
    "contrast_with_mde",
    "psi_per_arm",
    "three_arm_contrasts",
    "BootstrapCI",
    "StudentTCI",
    "PSIResult",
    "SeedContrast",
    "CrossFitResult",
    "MIN_SEEDS_FOR_INTERVAL",
    "psi_items",
    "normalized_psi",
    "psi_distances_centered_cosine",
    "psi_distances_whitened",
    "paired_bootstrap_mean",
    "bootstrap_psi_items",
    "model_contrast_seed_level",
    "crossfit_linear",
    "fit_h2",
    "base_id_folds",
    "first_branch_accuracy",
    "exact_path_accuracy",
    "safe_lure_invariance",
    "margin_erosion",
    "similarity_dependent_interference",
    "EXACT_ITEM_LOSS",
    "EXACT_ADAPTATION_LOSS",
    "DISTANCE_DIAGNOSTIC_QUANTILES",
    "gradient_alignment_baselines",
    "jacobian_ntk_overlap",
    "grouped_folds",
    "fit_h3_incremental",
    "fit_h3_model_distance_interaction",
    "h3_distance_diagnostic",
]

#: Below this many seeds a seed-level interval is reported but is not evidence on its own.
#: The confirmatory design has three (spec §8), which is enough to show replication and
#: not enough for a tight interval — so we report the interval AND the exact sign-flip
#: p-value AND every per-seed value, and never quote the interval alone.
MIN_SEEDS_FOR_INTERVAL = 5

_MAX_BOOTSTRAP_CELLS = 20_000_000  # chunking budget so n_boot * n never blows up RAM

# These strings are part of the analysis contract, not prose supplied by a caller.  A gradient
# of a convenient surrogate is not an admissible control for the confirmatory H3 analysis.
EXACT_ITEM_LOSS = "cross_entropy(correct_first_branch_token@index63)"
EXACT_ADAPTATION_LOSS = (
    "mean_autoregressive_next_token_cross_entropy(first_effective_adaptation_batch)"
)

# Frozen before outcomes.  Every bin is reported in order; no bin may be selected because its
# outcome looks interesting.  The inferential nonlinear check below is continuous (d + d^2).
DISTANCE_DIAGNOSTIC_QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# =====================================================================================
# Distances and PSI
# =====================================================================================


def psi_distances_centered_cosine(
    h_base: np.ndarray,
    h_near_critical: np.ndarray,
    h_near_safe: np.ndarray,
    *,
    centering_pool: CenteringPool,
) -> dict:
    """Both PSI distances under the PRIMARY metric, from one CHECKED centering pool.

    ``centering_pool`` must be a
    :class:`lurestar.representations.CenteringPool`, built by
    :meth:`~lurestar.representations.CenteringPool.from_conditions`.  A bare ndarray is
    refused, for the same reason :func:`model_contrast_seed_level` refuses one: naming
    the argument does not stop a caller passing the scored pair, and centering over only
    the scored conditions is the silent way to manufacture (or erase) a PSI effect.  Three
    things are enforced here rather than documented:

    1. the pool object could only be built by accounting for every condition of
       :data:`~lurestar.representations.CENTERING_POOL_CONDITIONS`, so a dropped
       condition is a recorded ``declared_missing`` entry, never an omission;
    2. ``base``, ``near_safe`` and ``near_critical`` must actually be among the pooled
       conditions;
    3. every scored state must be a row of the pool, so a pool from another
       (model, seed, extraction_index) cell — or noise — is rejected outright.

    The returned dict carries the pool's full report, so what was centred on travels with
    the number.
    """
    if not isinstance(centering_pool, CenteringPool):
        raise TypeError(
            "centering_pool must be a CenteringPool built with "
            "CenteringPool.from_conditions(base=..., repeat=..., near_safe=..., "
            "near_critical=..., far_critical=...). A raw array is refused because the "
            "wrong pool is the easiest way to manufacture a PSI effect and it cannot be "
            "detected after the fact; see CENTERING_POOL_POLICY."
        )
    centering_pool.require_conditions("base", "near_safe", "near_critical")
    centering_pool.require_contains("h_base", h_base)
    centering_pool.require_contains("h_near_critical", h_near_critical)
    centering_pool.require_contains("h_near_safe", h_near_safe)
    mean = centering_pool.mean
    return {
        "metric": "centered_cosine",
        "role": "primary (spec §6/H1)",
        "centering_pool_policy": CENTERING_POOL_POLICY,
        "centering_pool_n": int(centering_pool.n),
        "centering_pool": centering_pool.report(),
        "d_critical": centered_cosine_distance(h_base, h_near_critical, mean=mean),
        "d_safe": centered_cosine_distance(h_base, h_near_safe, mean=mean),
    }


def psi_distances_whitened(
    h_base: np.ndarray,
    h_near_critical: np.ndarray,
    h_near_safe: np.ndarray,
    *,
    whitener: Whitener,
    item_ids: Optional[Sequence] = None,
    group_ids: Optional[Sequence] = None,
) -> dict:
    """Both PSI distances under the DECLARED ROBUSTNESS CHECK (whitened Euclidean).

    Both id sets are mandatory here even though :meth:`Whitener.distance` tolerates a
    whitener fit without ids: this is a *reported* metric, and "the covariance came from
    a held-out pool" is a claim that has to be checkable.
    """
    if not whitener.fit_item_ids:
        raise LeakageError(
            "the whitener was fit without item_ids, so its held-out claim cannot be "
            "checked; refit with Whitener.fit(pool, item_ids=...)"
        )
    if item_ids is None:
        raise LeakageError("item_ids is required for a reported whitened-distance metric")
    if whitener.fit_group_ids and group_ids is None:
        raise LeakageError("group_ids is required for a group-held-out whitened metric")
    return {
        "metric": "whitened_euclidean",
        "role": "declared robustness check (spec §6/H1)",
        "whitener": whitener.report(),
        "d_critical": whitener.distance(
            h_base, h_near_critical, item_ids=item_ids, group_ids=group_ids
        ),
        "d_safe": whitener.distance(
            h_base, h_near_safe, item_ids=item_ids, group_ids=group_ids
        ),
    }


def psi_items(d_critical: np.ndarray, d_safe: np.ndarray) -> np.ndarray:
    """Per-item PSI = ``d(h_base, h_near_critical) - d(h_base, h_near_safe)``.

    Differencing *within item* is what makes the later bootstrap paired: resampling this
    array resamples quartets, never a critical distance without its matched safe one.
    """
    c = np.asarray(d_critical, dtype=np.float64)
    s = np.asarray(d_safe, dtype=np.float64)
    if c.shape != s.shape:
        raise ValueError(f"shape mismatch {c.shape} vs {s.shape}")
    if c.ndim != 1:
        raise ValueError("distances must be 1-D, one entry per quartet")
    return c - s


# =====================================================================================
# Bootstrap
# =====================================================================================


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_units: int
    n_boot: int
    unit: str
    alpha: float
    method: str = "paired percentile bootstrap of the within-unit difference"
    replicates: Optional[np.ndarray] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "se": self.se,
            "n_units": self.n_units,
            "n_boot": self.n_boot,
            "unit": self.unit,
            "alpha": self.alpha,
            "method": self.method,
        }


def _bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = values.shape[0]
    block = max(1, min(n_boot, _MAX_BOOTSTRAP_CELLS // max(n, 1)))
    out = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        out[done : done + b] = values[idx].mean(axis=1)
        done += b
    return out


def paired_bootstrap_mean(
    differences: np.ndarray,
    *,
    unit: str,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    keep_replicates: bool = False,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of an array of within-unit differences.

    ``unit`` is mandatory and free text that is carried into the result and into every
    serialized metric, so a reader can always see what was resampled.  Pairing is honoured
    by resampling the *difference*, not the two conditions independently.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError("differences must be 1-D, one entry per resampling unit")
    n = d.shape[0]
    if n < 2:
        raise ValueError("need at least 2 units to bootstrap")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator (determinism is threaded)")
    reps = _bootstrap_means(d, int(n_boot), rng)
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=float(d.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        se=float(reps.std(ddof=1)),
        n_units=int(n),
        n_boot=int(n_boot),
        unit=unit,
        alpha=float(alpha),
        replicates=reps if keep_replicates else None,
    )


@dataclass(frozen=True)
class PSIResult:
    psi: float
    ci: BootstrapCI
    per_item: np.ndarray = field(repr=False)
    d_critical_mean: float = 0.0
    d_safe_mean: float = 0.0
    metric: str = "centered_cosine"
    extraction_index: int = PSI_EXTRACTION_INDEX
    npsi: float = 0.0
    npsi_denominator: float = 0.0

    def as_dict(self) -> dict:
        return {
            "psi": self.psi,
            "metric": self.metric,
            "extraction_index": self.extraction_index,
            "d_critical_mean": self.d_critical_mean,
            "d_safe_mean": self.d_safe_mean,
            "npsi": self.npsi,
            "npsi_denominator": self.npsi_denominator,
            "npsi_role": "mandatory scale diagnostic; cannot rescue a co-primary result",
            "ci": self.ci.as_dict(),
        }


def normalized_psi(d_critical: np.ndarray, d_safe: np.ndarray) -> tuple[float, float]:
    """Return the frozen dimensionless nPSI diagnostic and its denominator.

    The numerator and denominator use the same aligned scored items.  A non-finite or
    non-positive denominator invalidates the cell; silently substituting an epsilon would
    turn this diagnostic into a tunable statistic.
    """
    c = np.asarray(d_critical, dtype=np.float64)
    s = np.asarray(d_safe, dtype=np.float64)
    if c.shape != s.shape or c.ndim != 1:
        raise ValueError("nPSI distances must be aligned 1-D arrays")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("nPSI distances must be finite")
    denominator = float(np.sum(c + s, dtype=np.float64))
    numerator = float(2.0 * np.sum(c - s, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("nPSI denominator must be strictly positive and finite")
    if not np.isfinite(numerator):
        raise ValueError("nPSI numerator must be finite")
    value = numerator / denominator
    if not np.isfinite(value):
        raise ValueError("nPSI must be finite")
    return float(value), denominator


def bootstrap_psi_items(
    d_critical: np.ndarray,
    d_safe: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    metric: str = "centered_cosine",
    extraction_index: int = PSI_EXTRACTION_INDEX,
) -> PSIResult:
    """ITEM-LEVEL PSI interval for one (model, seed) cell.  Unit of resampling: quartet.

    This interval describes the precision of PSI *within one trained model*.  It says
    nothing about whether the effect replicates across training seeds, and must never be
    used for the GPT-vs-NextLat contrast — that is
    :func:`model_contrast_seed_level`.
    """
    per_item = psi_items(d_critical, d_safe)
    npsi, npsi_denominator = normalized_psi(d_critical, d_safe)
    ci = paired_bootstrap_mean(
        per_item, unit="item (quartet)", rng=rng, n_boot=n_boot, alpha=alpha
    )
    return PSIResult(
        psi=float(per_item.mean()),
        ci=ci,
        per_item=per_item,
        d_critical_mean=float(np.mean(d_critical)),
        d_safe_mean=float(np.mean(d_safe)),
        metric=metric,
        extraction_index=int(extraction_index),
        npsi=npsi,
        npsi_denominator=npsi_denominator,
    )


# =====================================================================================
# The model contrast — SEEDS are the unit
# =====================================================================================


@dataclass(frozen=True)
class StudentTCI:
    """Two-sided paired Student-t interval over independent training-seed differences."""

    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_units: int
    unit: str
    alpha: float
    df: int
    t_critical: float
    method: str = "two-sided paired Student-t interval over training-seed differences"

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "se": self.se,
            "n_units": self.n_units,
            "unit": self.unit,
            "alpha": self.alpha,
            "df": self.df,
            "t_critical": self.t_critical,
            "method": self.method,
        }


def _paired_student_t_interval(differences: np.ndarray, *, alpha: float) -> StudentTCI:
    d = np.asarray(differences, dtype=np.float64).ravel()
    if d.size < 2 or not np.all(np.isfinite(d)):
        raise ValueError("paired Student-t interval needs at least two finite seed differences")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    estimate = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, d.size - 1))
    half_width = critical * se
    return StudentTCI(
        estimate=estimate,
        ci_low=float(estimate - half_width),
        ci_high=float(estimate + half_width),
        se=se,
        n_units=int(d.size),
        unit="training seed",
        alpha=float(alpha),
        df=int(d.size - 1),
        t_critical=critical,
    )


@dataclass(frozen=True)
class SeedContrast:
    label_a: str
    label_b: str
    seeds: tuple
    value_a: tuple
    value_b: tuple
    per_seed_difference: tuple
    estimate: float
    ci: StudentTCI
    sign_flip_p: float
    min_attainable_p: float
    n_seeds: int
    underpowered: bool
    paired_standardized_effect: Optional[float]
    leave_one_seed_out: tuple

    def as_dict(self) -> dict:
        return {
            "contrast": f"{self.label_a} - {self.label_b}",
            "unit": "training seed",
            "seeds": list(self.seeds),
            f"per_seed_{self.label_a}": list(self.value_a),
            f"per_seed_{self.label_b}": list(self.value_b),
            "per_seed_difference": list(self.per_seed_difference),
            "estimate": self.estimate,
            "ci": self.ci.as_dict(),
            "sign_flip_p": self.sign_flip_p,
            "min_attainable_p": self.min_attainable_p,
            "n_seeds": self.n_seeds,
            "underpowered": self.underpowered,
            "paired_standardized_effect": self.paired_standardized_effect,
            "paired_standardized_effect_definition": "mean(per-seed difference) / SD(per-seed difference)",
            "leave_one_seed_out": list(self.leave_one_seed_out),
        }


def _exact_sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided randomization p over all 2^n sign assignments of paired diffs."""
    n = diffs.shape[0]
    if n > 20:
        raise ValueError("exact sign-flip enumeration is only for small seed counts")
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    means = np.abs(signs @ diffs) / n
    return float(np.mean(means >= abs(diffs.mean()) - 1e-12))


def model_contrast_seed_level(
    value_by_seed_a: Mapping,
    value_by_seed_b: Mapping,
    *,
    label_a: str,
    label_b: str,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> SeedContrast:
    """GPT-vs-NextLat contrast with the TRAINING SEED as the unit of inference.

    Both arguments are mappings ``{seed_id: statistic}`` — one scalar per trained model,
    typically the item-mean PSI of that (model, seed) cell.  A mapping is demanded, not an
    array, precisely so that an item-level array cannot be passed here by accident; with
    five confirmatory seeds the correct n is 5, and an interval computed over 20,000
    items would be roughly 80x too narrow while describing a different estimand.

    Seeds are paired (the same seed trains both models, spec §8), so the estimator is the
    mean of the per-seed differences.  Reported alongside the interval:

    * every per-seed value and difference (spec §6/H3 "report every seed");
    * the exact two-sided sign-flip randomization p-value over all 2^n assignments, plus
      the smallest p that n seeds could possibly attain (2^{1-n}) — with n=5 that floor is
      0.0625, still above the conventional 0.05 threshold.
    """
    for name, m in (("value_by_seed_a", value_by_seed_a), ("value_by_seed_b", value_by_seed_b)):
        if not isinstance(m, Mapping):
            raise TypeError(
                f"{name} must be a Mapping {{seed_id: statistic}} — seeds are the "
                "inferential unit for this contrast and items never substitute for them"
            )
    seeds = tuple(sorted(value_by_seed_a))
    if seeds != tuple(sorted(value_by_seed_b)):
        raise ValueError(
            f"seed sets differ: {sorted(value_by_seed_a)} vs {sorted(value_by_seed_b)}; "
            "the contrast is paired within seed"
        )
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds for a seed-level contrast")
    a = np.array([float(value_by_seed_a[s]) for s in seeds])
    b = np.array([float(value_by_seed_b[s]) for s in seeds])
    diffs = a - b
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    # ``rng``/``n_boot`` remain accepted for API compatibility with item-level callers, but
    # the confirmatory seed interval is analytically fixed to paired Student-t.
    ci = _paired_student_t_interval(diffs, alpha=alpha)
    n = len(seeds)
    if n < MIN_SEEDS_FOR_INTERVAL:
        warnings.warn(
            f"seed-level interval computed from {n} seeds (< {MIN_SEEDS_FOR_INTERVAL}); "
            "report it with the per-seed values and the sign-flip p, never alone",
            stacklevel=2,
        )
    sd = float(diffs.std(ddof=1))
    standardized = None if sd == 0.0 else float(diffs.mean() / sd)
    loso = []
    for omitted_index, omitted_seed in enumerate(seeds):
        retained = np.delete(diffs, omitted_index)
        retained_ci = (
            _paired_student_t_interval(retained, alpha=alpha).as_dict()
            if retained.size >= 2 else None
        )
        loso.append({
            "omitted_seed": omitted_seed,
            "n_seeds": int(retained.size),
            "estimate": float(retained.mean()),
            "ci": retained_ci,
        })
    return SeedContrast(
        label_a=label_a,
        label_b=label_b,
        seeds=seeds,
        value_a=tuple(a.tolist()),
        value_b=tuple(b.tolist()),
        per_seed_difference=tuple(diffs.tolist()),
        estimate=float(diffs.mean()),
        ci=ci,
        sign_flip_p=_exact_sign_flip_p(diffs),
        min_attainable_p=float(2.0 ** (1 - n)),
        n_seeds=n,
        underpowered=bool(n < MIN_SEEDS_FOR_INTERVAL),
        paired_standardized_effect=standardized,
        leave_one_seed_out=tuple(loso),
    )


# =====================================================================================
# THE THREE-ARM DESIGN — arms, preregistered contrasts, and what the design could not see
# =====================================================================================

#: The three architecture-matched arms of spec §8, in preregistered priority order.
ARMS = ("nextlat", "bst", "gpt")


@dataclass(frozen=True)
class ContrastSpec:
    """One preregistered cross-model contrast, with its interpretation fixed in advance."""

    label_a: str
    label_b: str
    priority: int
    role: str
    reading: str

    @property
    def name(self) -> str:
        return f"{self.label_a}_minus_{self.label_b}"

    def as_dict(self) -> dict:
        return {
            "contrast": f"{self.label_a} - {self.label_b}",
            "priority": self.priority,
            "role": self.role,
            "reading": self.reading,
        }


#: Frozen before any model exists.  The order is the claim; re-ordering it after seeing
#: the numbers would convert a preregistered comparison into a selected one.
PREREGISTERED_CONTRASTS = (
    ContrastSpec(
        label_a="nextlat",
        label_b="bst",
        priority=1,
        role="primary (competence-matched)",
        reading=(
            "Both arms solve G(5,5) (paper Fig. 6: NextLat ~99.8%, BST ~99.9%), so this "
            "contrast reduces the task-competence confound. It is not an objective-only "
            "causal comparison: BST adds a backward transformer, more parameters, and O(T^2) "
            "prefix-suffix supervision. This controlled but non-isolating contrast is the "
            "contrast the project's claim rests on."
        ),
    ),
    ContrastSpec(
        label_a="nextlat",
        label_b="gpt",
        priority=2,
        role="secondary (competence-confounded)",
        reading=(
            "GPT is at 1/d chance on G(5,5) (paper Fig. 6, ~18.6%), so any PSI gap admits "
            "the trivial reading that NextLat organises the space because NextLat solved "
            "the task. Report the number and the confound in the same breath; it cannot "
            "carry the argument on its own."
        ),
    ),
    ContrastSpec(
        label_a="bst",
        label_b="gpt",
        priority=3,
        role="reference (competence only)",
        reading=(
            "Neither arm has a latent-transition objective and only one solves the task, "
            "so this is the size of the PSI effect that competence alone buys. It is the "
            "yardstick the NextLat-vs-BST gap is read against, not a hypothesis test."
        ),
    ),
)


# ------------------------------------------------------------ minimum detectable effect ---


def _solve_noncentrality(t_crit: float, df: int, power: float) -> float:
    """Smallest noncentrality whose two-sided t-test power reaches ``power``."""

    def attained(ncp: float) -> float:
        return float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))

    hi = max(1.0, t_crit)
    for _ in range(80):
        if attained(hi) >= power:
            break
        hi *= 2.0
    else:  # pragma: no cover - power < 1 always solves long before this
        return float("inf")
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if attained(mid) < power:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class MDEResult:
    """The smallest effect this many seeds could have detected, and the caveats."""

    mde: float
    sd_per_seed: float
    n_seeds: int
    alpha: float
    power: float
    mde_two_t_approximation: float
    sign_flip_p_floor: float
    randomization_test_can_reject: bool
    method: str = (
        "two-sided paired t on the per-seed differences; noncentral-t power solved exactly"
    )

    def as_dict(self) -> dict:
        return {
            "mde": self.mde,
            "sd_per_seed": self.sd_per_seed,
            "n_seeds": self.n_seeds,
            "alpha": self.alpha,
            "power": self.power,
            "mde_two_t_approximation": self.mde_two_t_approximation,
            "sign_flip_p_floor": self.sign_flip_p_floor,
            "randomization_test_can_reject": self.randomization_test_can_reject,
            "method": self.method,
        }


def minimum_detectable_effect(
    sd_per_seed: float,
    n_seeds: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> MDEResult:
    """Smallest |effect| a paired ``n_seeds``-seed contrast could have detected.

    ``sd_per_seed`` is the standard deviation of the *per-seed differences*, not of the
    items. With five confirmatory seeds this remains the honest answer to "what could this
    design not have seen".

    Two facts travel with it, both of which a reader needs and neither of which the point
    estimate shows:

    * ``sign_flip_p_floor`` = ``2^{1-n}``.  The exact randomization test used by
      :func:`model_contrast_seed_level` cannot return a p below this at ANY effect size,
      so with ``n = 3`` (floor 0.25) or ``n = 5`` (floor 0.0625) it can never reject at
      ``alpha = 0.05``.  ``randomization_test_can_reject`` says so directly.
    * with ``n`` this small the observed ``sd_per_seed`` is itself estimated from ``n-1``
      degrees of freedom, so the MDE is a rough scale, not a precise threshold.
    """
    sd = float(sd_per_seed)
    n = int(n_seeds)
    if not np.isfinite(sd) or sd < 0.0:
        raise ValueError(f"sd_per_seed must be finite and non-negative; got {sd_per_seed!r}")
    if n < 2:
        raise ValueError("need at least 2 seeds for a paired contrast")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ncp = _solve_noncentrality(t_crit, df, power)
    scale = sd / math.sqrt(n)
    floor = 2.0 ** (1 - n)
    return MDEResult(
        mde=float(ncp * scale),
        sd_per_seed=sd,
        n_seeds=n,
        alpha=float(alpha),
        power=float(power),
        mde_two_t_approximation=float((t_crit + float(stats.t.ppf(power, df))) * scale),
        sign_flip_p_floor=float(floor),
        randomization_test_can_reject=bool(floor <= alpha),
    )


# ------------------------------------------------------------------ a reported contrast ---


@dataclass(frozen=True)
class ContrastReport:
    """A seed-level contrast reported the only way it is allowed to be reported."""

    spec: ContrastSpec
    contrast: SeedContrast
    mde: MDEResult
    exceeds_mde: bool

    @property
    def estimate(self) -> float:
        return self.contrast.estimate

    def as_dict(self) -> dict:
        return {
            **self.spec.as_dict(),
            **self.contrast.as_dict(),
            "minimum_detectable_effect": self.mde.as_dict(),
            "exceeds_mde": self.exceeds_mde,
        }


def contrast_with_mde(
    value_by_seed_a: Mapping,
    value_by_seed_b: Mapping,
    *,
    spec: Optional[ContrastSpec] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    power: float = 0.80,
) -> ContrastReport:
    """Effect size, interval, sign-flip p, **and** what this many seeds could have seen.

    Wraps :func:`model_contrast_seed_level` so that the three numbers a reader needs
    always arrive together.  ``sd_per_seed`` for the MDE is the observed SD of the
    per-seed differences (``ddof=1``) — a retrospective MDE, which is the right one for
    "state what the design could not have detected" and the wrong one for planning a new
    study.
    """
    if spec is None:
        if label_a is None or label_b is None:
            raise ValueError("pass either spec= or both label_a= and label_b=")
        spec = ContrastSpec(
            label_a=label_a,
            label_b=label_b,
            priority=0,
            role="ad hoc (not preregistered)",
            reading="not one of PREREGISTERED_CONTRASTS",
        )
    elif label_a is not None or label_b is not None:
        raise ValueError("pass spec= or labels, not both")
    contrast = model_contrast_seed_level(
        value_by_seed_a,
        value_by_seed_b,
        label_a=spec.label_a,
        label_b=spec.label_b,
        rng=rng,
        n_boot=n_boot,
        alpha=alpha,
    )
    diffs = np.asarray(contrast.per_seed_difference, dtype=np.float64)
    mde = minimum_detectable_effect(
        float(diffs.std(ddof=1)), contrast.n_seeds, alpha=alpha, power=power
    )
    return ContrastReport(
        spec=spec,
        contrast=contrast,
        mde=mde,
        exceeds_mde=bool(abs(contrast.estimate) >= mde.mde),
    )


# ---------------------------------------------------------------------- PSI per arm ---


@dataclass(frozen=True)
class ArmPSI:
    """Item-level PSI for one arm, one cell per seed, plus the seed-level summary."""

    arm: str
    seeds: tuple
    per_seed: dict = field(repr=False)
    psi_by_seed: dict = field(default_factory=dict)
    seed_mean: float = 0.0

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "seeds": list(self.seeds),
            "psi_by_seed": {s: self.psi_by_seed[s] for s in self.seeds},
            "seed_mean": self.seed_mean,
            "per_seed_item_level": {s: self.per_seed[s].as_dict() for s in self.seeds},
        }


def psi_per_arm(
    distances_by_arm: Mapping[str, Mapping],
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    metric: str = "centered_cosine",
    extraction_index: int = PSI_EXTRACTION_INDEX,
    arms: Sequence[str] = ARMS,
) -> dict:
    """PSI for every (arm, seed) cell, with an ITEM-level paired bootstrap interval each.

    ``distances_by_arm`` is ``{arm: {seed: (d_critical, d_safe)}}``, the two per-item
    distance arrays of one cell as returned by
    :func:`psi_distances_centered_cosine`.  Every arm must carry the same seed set,
    because every cross-model contrast is paired within seed (spec §8: the same seed
    trains all three arms).

    The intervals here are **item-level**.  They describe precision inside one trained
    model and say nothing about replication across seeds; the seed-level statement is
    :func:`three_arm_contrasts`.  Both are returned so neither can be quoted as the other.
    """
    if not isinstance(distances_by_arm, Mapping):
        raise TypeError("distances_by_arm must be a Mapping {arm: {seed: (d_crit, d_safe)}}")
    wanted = tuple(arms)
    missing = [a for a in wanted if a not in distances_by_arm]
    if missing:
        raise ValueError(
            f"missing arm(s) {missing}; the design is three-armed (spec §8) and a dropped "
            "arm must be a declared deviation, not an omission"
        )
    extra = [a for a in distances_by_arm if a not in wanted]
    if extra:
        raise ValueError(f"unknown arm(s) {extra}; expected {list(wanted)}")

    seed_sets = {}
    for arm in wanted:
        cells = distances_by_arm[arm]
        if not isinstance(cells, Mapping):
            raise TypeError(
                f"distances_by_arm[{arm!r}] must be a Mapping {{seed: (d_crit, d_safe)}} — "
                "seeds are the inferential unit for every cross-model contrast and an "
                "item-level array must not be passable here"
            )
        seed_sets[arm] = tuple(sorted(cells))
    reference = seed_sets[wanted[0]]
    for arm in wanted[1:]:
        if seed_sets[arm] != reference:
            raise ValueError(
                f"seed sets differ between {wanted[0]!r} {list(reference)} and {arm!r} "
                f"{list(seed_sets[arm])}; contrasts are paired within seed"
            )

    out = {}
    for arm in wanted:
        per_seed, psi_by_seed = {}, {}
        for seed in reference:
            pair = distances_by_arm[arm][seed]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(
                    f"distances_by_arm[{arm!r}][{seed!r}] must be (d_critical, d_safe)"
                )
            res = bootstrap_psi_items(
                pair[0],
                pair[1],
                rng=rng,
                n_boot=n_boot,
                alpha=alpha,
                metric=metric,
                extraction_index=extraction_index,
            )
            per_seed[seed] = res
            psi_by_seed[seed] = res.psi
        out[arm] = ArmPSI(
            arm=arm,
            seeds=reference,
            per_seed=per_seed,
            psi_by_seed=psi_by_seed,
            seed_mean=float(np.mean([psi_by_seed[s] for s in reference])),
        )
    return out


# -------------------------------------------------------------- the three-way report ---


@dataclass(frozen=True)
class ThreeArmReport:
    psi_by_seed: dict = field(repr=False)
    contrasts: tuple = ()
    arms: tuple = ARMS

    def by_name(self, name: str) -> ContrastReport:
        for c in self.contrasts:
            if c.spec.name == name or f"{c.spec.label_a}-{c.spec.label_b}" == name:
                return c
        raise KeyError(f"no such contrast: {name!r}")

    @property
    def primary(self) -> ContrastReport:
        return self.contrasts[0]

    def as_dict(self) -> dict:
        return {
            "arms": list(self.arms),
            "psi_by_seed": {a: dict(v) for a, v in self.psi_by_seed.items()},
            "contrasts_in_preregistered_order": [c.as_dict() for c in self.contrasts],
        }


def _seed_values(arm_result) -> Mapping:
    """Accept either an :class:`ArmPSI` or a plain ``{seed: statistic}`` mapping."""
    if isinstance(arm_result, ArmPSI):
        return arm_result.psi_by_seed
    if isinstance(arm_result, Mapping):
        return arm_result
    raise TypeError(
        "each arm must map {seed: statistic} (or be an ArmPSI); seeds are the "
        "inferential unit and an item-level array is not a seed mapping"
    )


def three_arm_contrasts(
    statistic_by_arm: Mapping[str, Mapping],
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    power: float = 0.80,
    contrasts: Sequence[ContrastSpec] = PREREGISTERED_CONTRASTS,
) -> ThreeArmReport:
    """The three preregistered cross-model contrasts, in priority order, with MDEs.

    ``statistic_by_arm`` is ``{arm: {seed: statistic}}`` — typically the item-mean PSI of
    each (arm, seed) cell, i.e. the ``psi_by_seed`` field of :func:`psi_per_arm`'s output,
    which is accepted directly.  A raw array is refused at every level: the unit of
    inference for a cross-model statement is the training seed, and 20,000 quartets are
    not 20,000 independent trainings.

    Returned in the order of :data:`PREREGISTERED_CONTRASTS` and never re-sorted by
    effect size or by p-value.
    """
    if not isinstance(statistic_by_arm, Mapping):
        raise TypeError("statistic_by_arm must be a Mapping {arm: {seed: statistic}}")
    needed = sorted({c.label_a for c in contrasts} | {c.label_b for c in contrasts})
    missing = [a for a in needed if a not in statistic_by_arm]
    if missing:
        raise ValueError(
            f"missing arm(s) {missing}; the preregistered contrasts need {needed}"
        )
    values = {arm: _seed_values(statistic_by_arm[arm]) for arm in needed}
    reference = tuple(sorted(values[needed[0]]))
    for arm in needed[1:]:
        if tuple(sorted(values[arm])) != reference:
            raise ValueError(
                f"seed sets differ between {needed[0]!r} and {arm!r}; contrasts are "
                "paired within seed (spec §8: the same seed trains all three arms)"
            )
    reports = tuple(
        contrast_with_mde(
            values[spec.label_a],
            values[spec.label_b],
            spec=spec,
            rng=rng,
            n_boot=n_boot,
            alpha=alpha,
            power=power,
        )
        for spec in sorted(contrasts, key=lambda c: c.priority)
    )
    return ThreeArmReport(
        psi_by_seed={arm: dict(values[arm]) for arm in needed},
        contrasts=reports,
        arms=tuple(needed),
    )


# =====================================================================================
# H2 — two-fold cross-fitted linear model
# =====================================================================================


@dataclass(frozen=True)
class CrossFitResult:
    feature_names: tuple
    y: np.ndarray = field(repr=False)
    y_pred_heldout: np.ndarray = field(repr=False)
    fold_index: np.ndarray = field(repr=False)
    train_indices: tuple = field(repr=False, default=())
    test_indices: tuple = field(repr=False, default=())
    coefficients: np.ndarray = field(repr=False, default=None)   # (n_folds, n_features), standardized
    intercepts: np.ndarray = field(repr=False, default=None)
    r2_heldout: float = float("nan")
    r2_heldout_per_fold: tuple = ()
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")
    n: int = 0
    n_folds: int = 0

    def coefficient_directions(self) -> dict:
        """Sign of each standardized coefficient in every fold, and whether they agree."""
        C = np.asarray(self.coefficients)
        return {
            name: {
                "per_fold": [float(v) for v in C[:, j]],
                "signs": [int(np.sign(v)) for v in C[:, j]],
                "sign_consistent": bool(len(set(np.sign(C[:, j]).tolist())) == 1),
            }
            for j, name in enumerate(self.feature_names)
        }

    def as_dict(self) -> dict:
        return {
            "model": f"y ~ {' + '.join(self.feature_names)}",
            "cross_fitting": f"{self.n_folds}-fold, out-of-fold predictions only",
            "n": self.n,
            "r2_heldout": self.r2_heldout,
            "r2_heldout_per_fold": list(self.r2_heldout_per_fold),
            "spearman_rho_pred_vs_actual": self.spearman_rho,
            "spearman_p_pred_vs_actual": self.spearman_p,
            "coefficient_directions_standardized": self.coefficient_directions(),
        }


def _make_folds(n: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least 2 folds")
    if n < 2 * n_folds:
        raise ValueError(f"need at least {2 * n_folds} items for {n_folds}-fold cross-fitting")
    perm = rng.permutation(n)
    fold = np.empty(n, dtype=np.int64)
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        fold[chunk] = k
    return fold


def _ols(X: np.ndarray, y: np.ndarray):
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1:]


def crossfit_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 2,
    rng: Optional[np.random.Generator] = None,
    folds: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    standardize: bool = True,
) -> CrossFitResult:
    """K-fold cross-FITTED OLS: every prediction is made by a model that never saw it.

    For each fold k the model is fit on the complement of k and used only to predict k.
    Standardization constants are also fit on the training complement, because scaling by
    a full-sample mean/sd is itself a leak of held-out information.

    ``r2_heldout`` is ``1 - SS_res/SS_tot`` over the pooled out-of-fold predictions with
    the full-sample mean of ``y`` as the baseline, so it is genuinely bounded above by 1
    and can go negative when the fit does not generalize.  Per-fold R² is reported too.

    ``folds`` may be supplied explicitly (an integer array of length n) for a
    deterministic or grouped split; otherwise a permutation is drawn from ``rng``.
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).ravel()
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D; got {Xa.shape}")
    n, p = Xa.shape
    if ya.shape[0] != n:
        raise ValueError(f"X has {n} rows but y has {ya.shape[0]}")
    if not np.all(np.isfinite(Xa)) or not np.all(np.isfinite(ya)):
        raise ValueError("X and y must be finite")
    if feature_names is None:
        feature_names = tuple(f"x{j}" for j in range(p))
    feature_names = tuple(feature_names)
    if len(feature_names) != p:
        raise ValueError(f"{len(feature_names)} names for {p} features")

    if folds is None:
        if rng is None:
            raise ValueError("pass an explicit rng or an explicit `folds` array")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        fold = _make_folds(n, n_folds, rng)
    else:
        fold = np.asarray(folds, dtype=np.int64).ravel()
        if fold.shape[0] != n:
            raise ValueError("folds must have one entry per item")
        n_folds = int(fold.max()) + 1
        if set(np.unique(fold).tolist()) != set(range(n_folds)):
            raise ValueError("fold labels must be 0..n_folds-1 with none empty")

    y_pred = np.full(n, np.nan)
    coefs = np.zeros((n_folds, p))
    intercepts = np.zeros(n_folds)
    train_idx, test_idx, r2_fold = [], [], []
    all_idx = np.arange(n)

    for k in range(n_folds):
        te = all_idx[fold == k]
        tr = all_idx[fold != k]
        if te.size == 0 or tr.size <= p:
            raise ValueError(f"fold {k} is degenerate (train {tr.size}, test {te.size})")
        # --- the leakage guard, asserted rather than assumed -------------------------
        if np.intersect1d(tr, te).size:
            raise RuntimeError("cross-fitting bug: train and test folds overlap")
        Xtr, Xte = Xa[tr], Xa[te]
        ytr = ya[tr]
        if standardize:
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0, ddof=0)
            if np.any(sd == 0):
                raise ValueError("a feature is constant within a training fold")
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
        b0, b = _ols(Xtr, ytr)
        y_pred[te] = b0 + Xte @ b
        coefs[k] = b
        intercepts[k] = b0
        train_idx.append(tr)
        test_idx.append(te)
        ss_res_k = float(np.sum((ya[te] - y_pred[te]) ** 2))
        ss_tot_k = float(np.sum((ya[te] - ya.mean()) ** 2))
        r2_fold.append(1.0 - ss_res_k / ss_tot_k if ss_tot_k > 0 else float("nan"))

    if np.any(np.isnan(y_pred)):
        raise RuntimeError("cross-fitting bug: some item never received a held-out prediction")

    ss_res = float(np.sum((ya - y_pred) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if np.std(y_pred) == 0 or np.std(ya) == 0:
        rho, pval = float("nan"), float("nan")
    else:
        rho, pval = stats.spearmanr(y_pred, ya)

    return CrossFitResult(
        feature_names=feature_names,
        y=ya,
        y_pred_heldout=y_pred,
        fold_index=fold,
        train_indices=tuple(train_idx),
        test_indices=tuple(test_idx),
        coefficients=coefs,
        intercepts=intercepts,
        r2_heldout=float(r2),
        r2_heldout_per_fold=tuple(r2_fold),
        spearman_rho=float(rho),
        spearman_p=float(pval),
        n=int(n),
        n_folds=int(n_folds),
    )


def fit_h2(
    critical_correct_branch_margin: np.ndarray,
    base_critical_distance: np.ndarray,
    base_correct_branch_margin: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    folds: Optional[np.ndarray] = None,
    n_folds: int = 2,
    outcome_name: str = "critical_correct_branch_margin",
    baseline_name: str = "base_correct_branch_margin",
    extraction_index: int = BRANCH_MARGIN_INDEX,
) -> dict:
    """The preregistered nested H2 models, with fixed two-fold cross-fitting.

        M0: critical_correct_branch_margin ~ base_correct_branch_margin
        M1: critical_correct_branch_margin ~ base_correct_branch_margin + distance

    Margin is primary because accuracy may be at ceiling (spec §6/H2).  Every margin here
    is computed from the logits at :data:`BRANCH_MARGIN_INDEX` = 63, not at the ``=``
    delimiter — see the frozen correction in ``lurestar.representations``.

    Returns the cross-fit report plus the marginal Spearman between the distance
    predictor and the outcome, which is what the "coefficient direction" claim rests on.
    """
    y = np.asarray(critical_correct_branch_margin, dtype=np.float64).ravel()
    d = np.asarray(base_critical_distance, dtype=np.float64).ravel()
    m = np.asarray(base_correct_branch_margin, dtype=np.float64).ravel()
    if not (y.shape == d.shape == m.shape):
        raise ValueError(f"shape mismatch: y {y.shape}, distance {d.shape}, margin {m.shape}")
    if folds is None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("pass an explicit rng or the frozen base-id folds")
        folds = _make_folds(y.size, n_folds, rng)
    base = crossfit_linear(
        m[:, None], y,
        n_folds=n_folds,
        folds=folds,
        feature_names=(baseline_name,),
    )
    full = crossfit_linear(
        np.column_stack([m, d]), y,
        n_folds=n_folds,
        folds=folds,
        feature_names=(baseline_name, "base_critical_distance"),
    )
    if not np.array_equal(base.fold_index, full.fold_index):
        raise RuntimeError("H2 nested models did not reuse identical folds")
    incremental_prediction = full.y_pred_heldout - base.y_pred_heldout
    baseline_residual = y - base.y_pred_heldout
    if np.std(incremental_prediction) == 0 or np.std(baseline_residual) == 0:
        rho_incremental, p_incremental = float("nan"), float("nan")
    else:
        rho_incremental, p_incremental = stats.spearmanr(
            incremental_prediction, baseline_residual
        )
    out = {
        "outcome": outcome_name,
        "extraction_index": int(extraction_index),
        "estimand": "incremental out-of-fold predictive value of base-to-critical distance",
        "M0": base.as_dict(),
        "M1": full.as_dict(),
        "r2_heldout_M0": base.r2_heldout,
        "r2_heldout_M1": full.r2_heldout,
        "delta_r2_heldout": float(full.r2_heldout - base.r2_heldout),
        "heldout_spearman_incremental": {
            "rho": float(rho_incremental),
            "p": float(p_incremental),
            "definition": "OOF (M1-M0) prediction versus OOF M0 residual",
        },
        "distance_coefficient_directions_standardized": full.coefficient_directions()[
            "base_critical_distance"
        ],
        "folds_reused_exactly": True,
        "fold_assignment_sha256": hashlib.sha256(
            np.asarray(full.fold_index, dtype="<i8").tobytes()
        ).hexdigest(),
        "standardization": "training complement only",
    }
    return {"report": out, "baseline_result": base, "result": full}


def base_id_folds(base_ids: Sequence) -> np.ndarray:
    """Frozen H2 folds: ``int(SHA256(base_id UTF-8), 16) % 2``.

    The evidence layer carries each canonical base serialization by its lowercase SHA-256
    identity.  That 64-character identity is nevertheless the *input text* to this second
    SHA-256 operation; its hexadecimal numeric parity is not the fold.  Thus no analysis seed,
    sorting, balancing, or outcome-dependent operation enters the split.
    """
    ids = np.asarray(base_ids).ravel()
    if ids.size < 4:
        raise ValueError("need at least four base identities for two-fold cross-fitting")
    normalized = [str(value) for value in ids.tolist()]
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        for value in normalized
    ):
        raise ValueError("every base identity must be a lowercase SHA-256 digest")
    folds = np.asarray([
        int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 2
        for value in normalized
    ], dtype=np.int64)
    if set(folds.tolist()) != {0, 1}:
        raise ValueError("SHA-256 parity produced an empty H2 fold")
    return folds


# =====================================================================================
# Behaviour (spec §10) and the shared H3 margin arithmetic
# =====================================================================================


def first_branch_accuracy(logits: np.ndarray, correct_ids: np.ndarray) -> np.ndarray:
    """Per-item 0/1: does the argmax at index 63 equal the goal arm's first node?"""
    L = np.asarray(logits, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("logits must be (n_items, vocab)")
    c = np.asarray(correct_ids, dtype=np.int64).ravel()
    if c.shape[0] != L.shape[0]:
        raise ValueError("one correct id per item")
    return (L.argmax(axis=1) == c).astype(np.float64)


def exact_path_accuracy(predicted_paths: np.ndarray, true_paths: np.ndarray) -> np.ndarray:
    """Per-item 0/1: the whole emitted path equals the solver's answer."""
    P = np.asarray(predicted_paths)
    T = np.asarray(true_paths)
    if P.shape != T.shape or P.ndim != 2:
        raise ValueError(f"paths must be equal-shaped (n_items, path_len); {P.shape} vs {T.shape}")
    return np.all(P == T, axis=1).astype(np.float64)


def safe_lure_invariance(
    d_safe: np.ndarray,
    d_repeat: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> BootstrapCI:
    """``d(base, near_safe) - d(base, repeat)`` — the future-irrelevant control.

    Near-safe changes two endpoint tokens without changing the correct future; repeat
    changes only serialization order.  A model that separates by surface form alone will
    show a large positive value here, which is the confound PSI needs ruled out.
    """
    return paired_bootstrap_mean(
        np.asarray(d_safe, dtype=np.float64) - np.asarray(d_repeat, dtype=np.float64),
        unit="item (quartet)",
        rng=rng,
        n_boot=n_boot,
        alpha=alpha,
    )


def margin_erosion(margin_before: np.ndarray, margin_after: np.ndarray) -> np.ndarray:
    """H3 primary outcome component: ``before - after`` correct-branch margin.

    Positive = the margin got worse after adaptation (spec §6/H3 sign convention).
    """
    b = np.asarray(margin_before, dtype=np.float64).ravel()
    a = np.asarray(margin_after, dtype=np.float64).ravel()
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch {b.shape} vs {a.shape}")
    return b - a


def similarity_dependent_interference(
    margin_before: np.ndarray,
    margin_after_near: np.ndarray,
    margin_after_far: np.ndarray,
    *,
    item_ids_near: Optional[Sequence] = None,
    item_ids_far: Optional[Sequence] = None,
    parent_sha256_near: Optional[str] = None,
    parent_sha256_far: Optional[str] = None,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> dict:
    """``erosion_near - erosion_far``, paired within A_pair item and parent checkpoint.

    Both branches start from the same frozen parent, so ``margin_before`` is shared and
    cancels: the difference is exactly ``margin_after_far - margin_after_near``.  We still
    take it through the erosion definitions so the reported components match the spec's
    wording, and we assert the cancellation holds.
    """
    if (item_ids_near is None) != (item_ids_far is None):
        raise ValueError("pass both item-id arrays or neither")
    if item_ids_near is not None:
        ids_n = np.asarray(item_ids_near).ravel()
        ids_f = np.asarray(item_ids_far).ravel()
        if ids_n.shape != ids_f.shape or not np.array_equal(ids_n, ids_f):
            raise ValueError("near and far evaluations must contain identical A_pair ids in order")
        if ids_n.shape[0] != np.asarray(margin_before).size:
            raise ValueError("A_pair item-id count does not match the margin arrays")
    if (parent_sha256_near is None) != (parent_sha256_far is None):
        raise ValueError("pass both parent checkpoint hashes or neither")
    if parent_sha256_near is not None:
        for value in (parent_sha256_near, parent_sha256_far):
            if not isinstance(value, str) or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value
            ):
                raise ValueError("parent checkpoint hashes must be lowercase SHA-256")
        if parent_sha256_near != parent_sha256_far:
            raise ValueError("near and far branches do not share the same parent checkpoint")

    e_near = margin_erosion(margin_before, margin_after_near)
    e_far = margin_erosion(margin_before, margin_after_far)
    diff = e_near - e_far
    if not np.allclose(
        diff,
        np.asarray(margin_after_far, dtype=np.float64).ravel()
        - np.asarray(margin_after_near, dtype=np.float64).ravel(),
    ):
        raise RuntimeError("erosion difference does not reduce to the shared-parent form")
    return {
        "erosion_near_mean": float(e_near.mean()),
        "erosion_far_mean": float(e_far.mean()),
        "similarity_dependent_interference": float(diff.mean()),
        "shared_parent_verified": bool(parent_sha256_near is not None),
        "paired_item_order_verified": bool(item_ids_near is not None),
        "ci": paired_bootstrap_mean(
            diff, unit="item (A_pair, within parent checkpoint)", rng=rng,
            n_boot=n_boot, alpha=alpha,
        ),
    }


# =====================================================================================
# Reviewer-grade H3 controls — exact gradient/NTK baselines and incremental prediction
# =====================================================================================


def _rows(x: np.ndarray, name: str) -> np.ndarray:
    """Flatten every observation after axis zero without changing its values."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError(f"{name} must have shape (n_items, ...); got {a.shape}")
    a = a.reshape(a.shape[0], -1)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _reference_rows(x: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = np.broadcast_to(a.reshape(1, -1), (n, a.size))
    else:
        a = _rows(a, name)
        if a.shape[0] == 1:
            a = np.broadcast_to(a, (n, a.shape[1]))
    if a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} rows; expected 1 or {n}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def gradient_alignment_baselines(
    item_gradients: np.ndarray,
    adaptation_gradients: np.ndarray,
    *,
    item_loss: str,
    adaptation_loss: str,
    epsilon: float = 1e-30,
) -> dict:
    """Exact per-item gradient dot product and cosine against the adaptation update.

    The inputs are gradients with respect to the same ordered parameter vector.  The
    adaptation input may be one bank-mean gradient, or one matched gradient per item.  We
    deliberately require the frozen loss identifiers: silently substituting a logit margin,
    hidden-state proxy, or validation loss would make the control easier to compute but would
    no longer test the first-order interference explanation.
    """
    if item_loss != EXACT_ITEM_LOSS or adaptation_loss != EXACT_ADAPTATION_LOSS:
        raise ValueError(
            "gradient controls must use the frozen exact item and adaptation losses: "
            f"{EXACT_ITEM_LOSS!r} and {EXACT_ADAPTATION_LOSS!r}"
        )
    g = _rows(item_gradients, "item_gradients")
    a = _reference_rows(adaptation_gradients, g.shape[0], "adaptation_gradients")
    if g.shape[1] != a.shape[1]:
        raise ValueError("item and adaptation gradients use different parameter vectors")
    dot = np.einsum("ij,ij->i", g, a)
    item_norm = np.linalg.norm(g, axis=1)
    adaptation_norm = np.linalg.norm(a, axis=1)
    denom = item_norm * adaptation_norm
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > epsilon)
    return {
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "item_gradient_norm": item_norm,
        "adaptation_gradient_norm": adaptation_norm,
        "item_loss": item_loss,
        "adaptation_loss": adaptation_loss,
        "zero_norm_policy": "cosine=0",
        "first_order_sign": (
            "positive dot predicts margin/loss movement in the direction implied by the "
            "explicit outcome sign; do not take an absolute value"
        ),
    }


def jacobian_ntk_overlap(
    item_jacobians: np.ndarray,
    adaptation_jacobians: np.ndarray,
    *,
    item_target: str = "correct_first_branch_logit_margin@index63",
    adaptation_target: str = "next_token_logits_on_adaptation_bank",
    epsilon: float = 1e-30,
) -> dict:
    """Frobenius inner product/cosine of exact output Jacobians (an empirical NTK control).

    All non-item axes — output coordinates and the ordered parameter vector — are flattened.
    This is intentionally a diagnostic alongside the loss-gradient baseline, not a replacement:
    a Jacobian overlap omits residual-dependent weighting present in cross-entropy gradients.
    """
    j = _rows(item_jacobians, "item_jacobians")
    a = _reference_rows(adaptation_jacobians, j.shape[0], "adaptation_jacobians")
    if j.shape[1] != a.shape[1]:
        raise ValueError("item and adaptation Jacobians use different output/parameter axes")
    dot = np.einsum("ij,ij->i", j, a)
    jn, an = np.linalg.norm(j, axis=1), np.linalg.norm(a, axis=1)
    denom = jn * an
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > epsilon)
    return {
        "ntk_frobenius_dot": dot,
        "ntk_frobenius_cosine": cosine,
        "item_jacobian_norm": jn,
        "adaptation_jacobian_norm": an,
        "item_target": item_target,
        "adaptation_target": adaptation_target,
        "role": "preregistered diagnostic; gradient alignment is the primary mechanistic control",
        "zero_norm_policy": "cosine=0",
    }


def grouped_folds(group_ids: Sequence, *, n_folds: int = 2, seed: int = 73021) -> np.ndarray:
    """Outcome-independent deterministic folds keeping repeated item/parent groups together."""
    ids = np.asarray(group_ids).ravel()
    if ids.size < 2 * n_folds:
        raise ValueError(f"need at least {2 * n_folds} rows for {n_folds}-fold cross-fitting")
    groups = sorted(set(str(x) for x in ids.tolist()))
    if len(groups) < n_folds:
        raise ValueError("fewer unique groups than folds")
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(),
    )
    mapping = {value: i % n_folds for i, value in enumerate(ordered)}
    folds = np.asarray([mapping[str(x)] for x in ids.tolist()], dtype=np.int64)
    if min(np.sum(folds == k) for k in range(n_folds)) == 0:
        raise RuntimeError("group-fold construction produced an empty fold")
    return folds


def fit_h3_incremental(
    erosion: np.ndarray,
    hidden_distance: np.ndarray,
    gradient_dot: np.ndarray,
    gradient_cosine: np.ndarray,
    initial_margin: np.ndarray,
    initial_lure_loss: np.ndarray,
    realized_acquisition: np.ndarray,
    *,
    folds: np.ndarray,
) -> dict:
    """Does frozen hidden distance add held-out prediction beyond mechanistic controls?

    Baseline and augmented models use *identical, caller-frozen folds*.  The confirmatory
    incremental estimand is ``OOF R2(full) - OOF R2(controls)``; neither model is selected or
    tuned after outcomes.  Realized acquisition is included because unequal branch learning can
    otherwise masquerade as similarity-dependent interference.
    """
    arrays = [
        np.asarray(v, dtype=np.float64).ravel()
        for v in (erosion, hidden_distance, gradient_dot, gradient_cosine,
                  initial_margin, initial_lure_loss, realized_acquisition)
    ]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("all H3 arrays must have one value per item")
    y, distance, gdot, gcos, margin, lure, acquisition = arrays
    baseline_names = (
        "gradient_dot", "gradient_cosine", "initial_margin",
        "initial_lure_loss", "realized_acquisition",
    )
    X0 = np.column_stack([gdot, gcos, margin, lure, acquisition])
    base = crossfit_linear(X0, y, folds=folds, feature_names=baseline_names)
    full = crossfit_linear(
        np.column_stack([X0, distance]), y, folds=folds,
        feature_names=baseline_names + ("frozen_hidden_distance",),
    )
    mse0 = float(np.mean((y - base.y_pred_heldout) ** 2))
    mse1 = float(np.mean((y - full.y_pred_heldout) ** 2))
    return {
        "estimand": "incremental out-of-fold predictive value of frozen hidden distance",
        "outcome": "item-level margin erosion (before - after; positive is worse)",
        "baseline": base.as_dict(),
        "augmented": full.as_dict(),
        "delta_r2_heldout": float(full.r2_heldout - base.r2_heldout),
        "delta_mse_heldout": float(mse1 - mse0),
        "folds_reused_exactly": bool(np.array_equal(base.fold_index, full.fold_index)),
        "fold_assignment_sha256": hashlib.sha256(
            np.asarray(folds, dtype="<i8").tobytes()
        ).hexdigest(),
        "baseline_result": base,
        "augmented_result": full,
    }


def fit_h3_model_distance_interaction(
    erosion: np.ndarray,
    hidden_distance: np.ndarray,
    model_labels: Sequence[str],
    controls: np.ndarray,
    *,
    folds: np.ndarray,
    control_names: Sequence[str],
    model_order: Sequence[str] = ARMS,
    reference_model: str = "bst",
) -> dict:
    """Fixed pooled model-by-distance interaction with BST as the declared reference arm."""
    y = np.asarray(erosion, dtype=np.float64).ravel()
    d = np.asarray(hidden_distance, dtype=np.float64).ravel()
    labels = np.asarray(model_labels).astype(str).ravel()
    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if y.shape != d.shape or labels.shape != y.shape or C.shape[0] != y.size:
        raise ValueError("interaction inputs must have one aligned row per observation")
    order = tuple(model_order)
    if reference_model not in order:
        raise ValueError("reference_model must be in model_order")
    unknown = sorted(set(labels) - set(order))
    if unknown or set(labels) != set(order):
        raise ValueError(f"model labels must contain every frozen arm exactly as levels; got {unknown}")
    other = tuple(m for m in order if m != reference_model)
    indicators = np.column_stack([(labels == model).astype(float) for model in other])
    interactions = indicators * d[:, None]
    names = tuple(control_names) + ("frozen_hidden_distance",) + tuple(
        f"model[{m}]" for m in other
    ) + tuple(f"distance:model[{m}]" for m in other)
    if len(control_names) != C.shape[1]:
        raise ValueError("control_names length does not match controls")
    result = crossfit_linear(
        np.column_stack([C, d, indicators, interactions]), y,
        folds=folds, feature_names=names,
    )
    return {
        "reference_model": reference_model,
        "model_order_frozen": list(order),
        "interaction_terms": [f"distance:model[{m}]" for m in other],
        "report": result.as_dict(),
        "result": result,
    }


def h3_distance_diagnostic(
    erosion: np.ndarray,
    hidden_distance: np.ndarray,
    controls: np.ndarray,
    *,
    folds: np.ndarray,
    control_names: Sequence[str],
    quantiles: Sequence[float] = DISTANCE_DIAGNOSTIC_QUANTILES,
) -> dict:
    """Frozen continuous quadratic check plus exhaustive distance-quantile descriptives.

    The quadratic term is the only nonlinear inferential diagnostic.  Quantile bins depend only
    on the pre-adaptation distance and all bins are reported; they are descriptive and may not be
    searched, merged, or promoted into the primary result after seeing erosion.
    """
    y = np.asarray(erosion, dtype=np.float64).ravel()
    d = np.asarray(hidden_distance, dtype=np.float64).ravel()
    C = np.asarray(controls, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if y.shape != d.shape or C.shape[0] != y.size:
        raise ValueError("diagnostic inputs must have one aligned row per observation")
    if len(control_names) != C.shape[1]:
        raise ValueError("control_names length does not match controls")
    q = tuple(float(x) for x in quantiles)
    if q != DISTANCE_DIAGNOSTIC_QUANTILES:
        raise ValueError("confirmatory diagnostic quantiles are frozen and may not be changed")
    linear = crossfit_linear(
        np.column_stack([C, d]), y, folds=folds,
        feature_names=tuple(control_names) + ("frozen_hidden_distance",),
    )
    quadratic = crossfit_linear(
        np.column_stack([C, d, d * d]), y, folds=folds,
        feature_names=tuple(control_names) + ("frozen_hidden_distance", "distance_squared"),
    )
    edges = np.quantile(d, q)
    # searchsorted with interior edges gives stable left-to-right bins and keeps the maximum.
    bins = np.searchsorted(edges[1:-1], d, side="right")
    summaries = []
    for k in range(len(q) - 1):
        mask = bins == k
        summaries.append({
            "bin": k + 1,
            "quantile_interval": [q[k], q[k + 1]],
            "distance_edge": [float(edges[k]), float(edges[k + 1])],
            "n": int(mask.sum()),
            "mean_distance": float(d[mask].mean()) if mask.any() else None,
            "mean_erosion": float(y[mask].mean()) if mask.any() else None,
        })
    return {
        "status": "preregistered diagnostic; not multiplicity-free confirmatory evidence",
        "linear": linear.as_dict(),
        "quadratic": quadratic.as_dict(),
        "delta_r2_quadratic_over_linear": float(quadratic.r2_heldout - linear.r2_heldout),
        "distance_quantiles": summaries,
        "bin_policy": "distance-only fixed quintiles; report all; no outcome-selected bins",
        "linear_result": linear,
        "quadratic_result": quadratic,
    }
