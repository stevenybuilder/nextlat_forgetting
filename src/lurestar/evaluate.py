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
   unit for the GPT-vs-NextLat contrast, and items never substitute for seeds
   (spec §6/H3: "Items do not substitute for independent training seeds"; the same rule
   governs the H1 model contrast).  The two live in separately named functions with
   different argument types — item functions take arrays, the seed function takes a
   mapping keyed by seed id — so an array of 20,000 items cannot be passed where three
   seeds belong.
"""

from __future__ import annotations

import itertools
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
    "BootstrapCI",
    "PSIResult",
    "SeedContrast",
    "CrossFitResult",
    "MIN_SEEDS_FOR_INTERVAL",
    "psi_items",
    "psi_distances_centered_cosine",
    "psi_distances_whitened",
    "paired_bootstrap_mean",
    "bootstrap_psi_items",
    "model_contrast_seed_level",
    "crossfit_linear",
    "fit_h2",
    "first_branch_accuracy",
    "exact_path_accuracy",
    "safe_lure_invariance",
    "margin_erosion",
    "similarity_dependent_interference",
]

#: Below this many seeds a seed-level interval is reported but is not evidence on its own.
#: The confirmatory design has three (spec §8), which is enough to show replication and
#: not enough for a tight interval — so we report the interval AND the exact sign-flip
#: p-value AND every per-seed value, and never quote the interval alone.
MIN_SEEDS_FOR_INTERVAL = 5

_MAX_BOOTSTRAP_CELLS = 20_000_000  # chunking budget so n_boot * n never blows up RAM


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
    return {
        "metric": "whitened_euclidean",
        "role": "declared robustness check (spec §6/H1)",
        "whitener": whitener.report(),
        "d_critical": whitener.distance(h_base, h_near_critical, item_ids=item_ids),
        "d_safe": whitener.distance(h_base, h_near_safe, item_ids=item_ids),
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

    def as_dict(self) -> dict:
        return {
            "psi": self.psi,
            "metric": self.metric,
            "extraction_index": self.extraction_index,
            "d_critical_mean": self.d_critical_mean,
            "d_safe_mean": self.d_safe_mean,
            "ci": self.ci.as_dict(),
        }


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
    )


# =====================================================================================
# The model contrast — SEEDS are the unit
# =====================================================================================


@dataclass(frozen=True)
class SeedContrast:
    label_a: str
    label_b: str
    seeds: tuple
    value_a: tuple
    value_b: tuple
    per_seed_difference: tuple
    estimate: float
    ci: BootstrapCI
    sign_flip_p: float
    min_attainable_p: float
    n_seeds: int
    underpowered: bool

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
    three confirmatory seeds the correct n is 3, and an interval computed over 20,000
    items would be roughly 80x too narrow while describing a different estimand.

    Seeds are paired (the same seed trains both models, spec §8), so the estimator is the
    mean of the per-seed differences.  Reported alongside the interval:

    * every per-seed value and difference (spec §6/H3 "report every seed");
    * the exact two-sided sign-flip randomization p-value over all 2^n assignments, plus
      the smallest p that n seeds could possibly attain (2^{1-n}) — with n=3 that floor is
      0.25, which is the honest statement of what three seeds can show.
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
    ci = paired_bootstrap_mean(
        diffs, unit="training seed", rng=rng, n_boot=n_boot, alpha=alpha
    )
    n = len(seeds)
    if n < MIN_SEEDS_FOR_INTERVAL:
        warnings.warn(
            f"seed-level interval computed from {n} seeds (< {MIN_SEEDS_FOR_INTERVAL}); "
            "report it with the per-seed values and the sign-flip p, never alone",
            stacklevel=2,
        )
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
) -> dict:
    """The preregistered H2 held-out model (spec §6/H2), with two-fold cross-fitting.

        critical_correct_branch_margin
            ~ base_critical_distance + base_correct_branch_margin

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
    X = np.column_stack([d, m])
    res = crossfit_linear(
        X,
        y,
        n_folds=n_folds,
        rng=rng,
        folds=folds,
        feature_names=("base_critical_distance", "base_correct_branch_margin"),
    )
    rho_d, p_d = stats.spearmanr(d, y)
    rho_m, p_m = stats.spearmanr(m, y)
    out = res.as_dict()
    out.update(
        {
            "outcome": "critical_correct_branch_margin",
            "margin_extraction_index": int(BRANCH_MARGIN_INDEX),
            "marginal_spearman": {
                "base_critical_distance": {"rho": float(rho_d), "p": float(p_d)},
                "base_correct_branch_margin": {"rho": float(rho_m), "p": float(p_m)},
            },
        }
    )
    return {"report": out, "result": res}


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
        "ci": paired_bootstrap_mean(
            diff, unit="item (A_pair, within parent checkpoint)", rng=rng,
            n_boot=n_boot, alpha=alpha,
        ),
    }
