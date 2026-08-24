"""HMM-H1/H2/H3 estimators (spec section 12).

This module is deliberately model-agnostic: it takes hidden states as plain arrays and knows
nothing about GPT, NextLat, checkpoints or torch. That keeps the analysis testable on this host,
which has no GPU and no torch, and it keeps the estimator identical for both conditions -- the
comparison the whole experiment rests on is only meaningful if the two models are measured by the
same code path.

The three preregistered tests:

* **HMM-H1, predictive equivalence.** Do histories with nearly identical exact posteriors sit
  closer together in model space than histories at the *same observation-history edit distance*
  whose posteriors differ? The matched control is what turns this from "similar things are
  similar" into a claim about predictive rather than surface similarity.
* **HMM-H2, relative belief geometry.** Does model-state distance track posterior JS divergence?
  Reported as a Spearman correlation, as a Spearman correlation partialled on history edit
  distance and prefix length, and as held-out neighbourhood retrieval.
* **HMM-H3, posterior decodability.** A held-out linear probe from `h_t` to `b_t`, scored by
  reconstruction error, KL and JS to the exact posterior, calibration against the realised hidden
  states, and length-64 generalisation. The exact next-observation distribution is decoded as
  well, because it is invariant to the coordinate freedom the spec warns about: a sufficient
  predictive state is only defined up to an invertible transformation, so literal alignment to
  the belief simplex is not the target.

Primary distance is centered cosine, matching spec section 6's choice for Lure-Star. Whitened
Euclidean is computed alongside as the declared robustness check, never as an alternative primary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

try:  # pragma: no cover - import shim
    from .forward import js_divergence
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hmm_geometry.forward import js_divergence

__all__ = [
    "centered_cosine_distance",
    "fit_whitener",
    "whitened_euclidean_distance",
    "bootstrap_ci",
    "h1_predictive_equivalence",
    "h2_relative_geometry",
    "neighborhood_retrieval",
    "fit_linear_probe",
    "project_to_simplex",
    "h3_posterior_decodability",
]


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------


def centered_cosine_distance(
    xa: np.ndarray, xb: np.ndarray, center: np.ndarray | None = None
) -> np.ndarray:
    """Cosine distance after subtracting a shared center.

    The center matters. Raw cosine on transformer hidden states is dominated by the population
    mean direction, which is shared by every history and therefore carries no information about
    which history this is; centering removes it. The center must be estimated once, on the
    evaluation population, and reused for every pair and both models -- passing a per-condition
    center would let a model look better simply by having a larger mean offset.
    """
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    if center is None:
        center = np.concatenate([xa, xb], axis=0).mean(axis=0)
    a = xa - center
    b = xb - center
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-30)
    return 1.0 - (a * b).sum(axis=1) / denom


def fit_whitener(x: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Mean and whitening matrix from the evaluation population.

    ``ridge`` is a fraction of the mean eigenvalue, so the regularisation does not depend on the
    arbitrary scale of the hidden states. Returns ``(mean, W)`` with ``W`` such that
    ``(x - mean) @ W`` has identity covariance up to the ridge.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None) + ridge * max(evals.mean(), 1e-12)
    w = evecs @ np.diag(evals**-0.5) @ evecs.T
    return mean, w


def whitened_euclidean_distance(
    xa: np.ndarray, xb: np.ndarray, mean: np.ndarray, whitener: np.ndarray
) -> np.ndarray:
    """Declared robustness check for the primary centered-cosine distance."""
    a = (np.asarray(xa, dtype=np.float64) - mean) @ whitener
    b = (np.asarray(xb, dtype=np.float64) - mean) @ whitener
    return np.linalg.norm(a - b, axis=1)


# --------------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap over the *inferential unit the caller passes in*.

    The unit is the caller's responsibility and it is the thing that is easy to get wrong: pairs
    are the unit within a seed, seeds are the unit across seeds, and resampling pairs to make a
    claim about seeds would manufacture precision that three seeds cannot support.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = np.array([statistic(values[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def _rank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(np.asarray(x, dtype=np.float64))


def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]
) -> tuple[float, float]:
    """Spearman correlation between ``x`` and ``y`` after linearly removing ranked controls.

    Rank-transform everything, regress out the control ranks by least squares, and correlate the
    residuals. With no controls this reduces exactly to Spearman's rho.
    """
    rx, ry = _rank(x), _rank(y)
    if controls:
        c = np.column_stack([_rank(c) for c in controls])
        c = np.column_stack([np.ones(len(c)), c])
        rx = rx - c @ np.linalg.lstsq(c, rx, rcond=None)[0]
        ry = ry - c @ np.linalg.lstsq(c, ry, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


# --------------------------------------------------------------------------------------------
# HMM-H1
# --------------------------------------------------------------------------------------------


@dataclass
class PairStates:
    """Hidden states for one pair set, aligned row-by-row with the pair list."""

    a: np.ndarray  # (P, D)
    b: np.ndarray  # (P, D)
    edit_distance: np.ndarray  # (P,)
    prefix_len: np.ndarray  # (P,)
    js_bits: np.ndarray  # (P,)

    def __post_init__(self) -> None:
        n = len(self.a)
        for name in ("b", "edit_distance", "prefix_len", "js_bits"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"PairStates.{name} has length {len(getattr(self, name))}, not {n}")


def h1_predictive_equivalence(
    equivalent: PairStates,
    control: PairStates,
    center: np.ndarray | None = None,
    whitener: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Are predictively equivalent histories closer than history-distance-matched controls?

    The controls are matched pair-for-pair on prefix length and edit distance by
    `pair_bank.build_bank`, so the difference is taken as a *paired* statistic and the matching is
    re-verified here rather than trusted.
    """
    if len(equivalent.a) != len(control.a):
        raise ValueError("H1 requires one matched control per equivalent pair")
    if not np.array_equal(equivalent.edit_distance, control.edit_distance):
        raise ValueError("control set is not matched on observation-history edit distance")
    if not np.array_equal(equivalent.prefix_len, control.prefix_len):
        raise ValueError("control set is not matched on prefix length")

    if center is None:
        center = np.concatenate(
            [equivalent.a, equivalent.b, control.a, control.b], axis=0
        ).mean(axis=0)
    d_eq = centered_cosine_distance(equivalent.a, equivalent.b, center)
    d_ct = centered_cosine_distance(control.a, control.b, center)
    delta = d_eq - d_ct  # negative means equivalence is respected

    out = {
        "n_pairs": int(len(d_eq)),
        "distance": "centered_cosine",
        "equivalent_distance_mean": float(d_eq.mean()),
        "control_distance_mean": float(d_ct.mean()),
        "paired_delta_mean": float(delta.mean()),
        "equivalent_js_bits_mean": float(equivalent.js_bits.mean()),
        "control_js_bits_mean": float(control.js_bits.mean()),
        "edit_distance_mean": float(equivalent.edit_distance.mean()),
    }
    est, lo, hi = bootstrap_ci(delta, rng=rng)
    out.update({"paired_delta_ci_lo": lo, "paired_delta_ci_hi": hi})
    # Standardised effect size on the paired differences; with three seeds this is the honest
    # summary, not a p-value from thousands of pairs that share a single training run.
    sd = delta.std(ddof=1)
    out["paired_delta_cohens_dz"] = float(delta.mean() / sd) if sd > 0 else float("nan")

    if whitener is not None:
        mean, w = whitener
        rd_eq = whitened_euclidean_distance(equivalent.a, equivalent.b, mean, w)
        rd_ct = whitened_euclidean_distance(control.a, control.b, mean, w)
        out["robustness_whitened_euclidean"] = {
            "equivalent_distance_mean": float(rd_eq.mean()),
            "control_distance_mean": float(rd_ct.mean()),
            "paired_delta_mean": float((rd_eq - rd_ct).mean()),
        }
    return out


def h1_near_lure_separation(
    near_lures: PairStates,
    control: PairStates | None = None,
    center: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """The complementary direction: near-lures should be *far apart* despite tiny edit distance.

    Reported next to H1 because a model can trivially satisfy predictive equivalence by collapsing
    everything. If equivalent pairs are close *and* near-lures are far, the geometry is selective
    rather than merely compressed -- the same argument spec section 6 makes for Lure-Star's PSI.
    """
    if center is None:
        center = np.concatenate([near_lures.a, near_lures.b], axis=0).mean(axis=0)
    d = centered_cosine_distance(near_lures.a, near_lures.b, center)
    est, lo, hi = bootstrap_ci(d, rng=rng)
    out = {
        "n_pairs": int(len(d)),
        "near_lure_distance_mean": est,
        "near_lure_distance_ci_lo": lo,
        "near_lure_distance_ci_hi": hi,
        "near_lure_js_bits_mean": float(near_lures.js_bits.mean()),
        "near_lure_edit_mean": float(near_lures.edit_distance.mean()),
    }
    if control is not None:
        d_ct = centered_cosine_distance(control.a, control.b, center)
        out["control_distance_mean"] = float(d_ct.mean())
        out["separation_index"] = float(d.mean() - d_ct.mean())
    return out


# --------------------------------------------------------------------------------------------
# HMM-H2
# --------------------------------------------------------------------------------------------


def h2_relative_geometry(pairs: PairStates, center: np.ndarray | None = None) -> dict:
    """Does model-state distance track exact posterior JS divergence?

    Reports the raw Spearman correlation and the correlation partialled on observation-history
    edit distance and prefix length, which are the two confounds the spec names. A model that
    merely encodes "how many symbols differ" would score well on the raw correlation and poorly on
    the partial one.
    """
    d = centered_cosine_distance(pairs.a, pairs.b, center)
    rho, p = stats.spearmanr(d, pairs.js_bits)
    rho_partial, p_partial = partial_spearman(
        d, pairs.js_bits, [pairs.edit_distance, pairs.prefix_len]
    )
    rho_edit, _ = stats.spearmanr(d, pairs.edit_distance)
    return {
        "n_pairs": int(len(d)),
        "spearman_distance_vs_js": float(rho),
        "spearman_p": float(p),
        "partial_spearman_given_edit_and_length": rho_partial,
        "partial_spearman_p": p_partial,
        "spearman_distance_vs_edit_distance": float(rho_edit),
        "mean_distance": float(d.mean()),
    }


def neighborhood_retrieval(
    states: np.ndarray, beliefs: np.ndarray, k: int = 10, center: np.ndarray | None = None
) -> dict:
    """Overlap between belief-space and state-space k-nearest-neighbour sets.

    For every item, take the ``k`` nearest items in exact-posterior JS divergence and the ``k``
    nearest in centered cosine distance on the model state, and report the mean overlap. Chance is
    ``k / (n - 1)``, which is reported alongside so the number is never read without its baseline.
    """
    states = np.asarray(states, dtype=np.float64)
    beliefs = np.asarray(beliefs, dtype=np.float64)
    n = len(states)
    if n <= k + 1:
        raise ValueError("need more items than neighbours")
    if center is None:
        center = states.mean(axis=0)
    x = states - center
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    cos_dist = 1.0 - x @ x.T
    np.fill_diagonal(cos_dist, np.inf)

    js = js_divergence(beliefs[:, None, :], beliefs[None, :, :])
    np.fill_diagonal(js, np.inf)

    knn_state = np.argsort(cos_dist, axis=1, kind="stable")[:, :k]
    knn_belief = np.argsort(js, axis=1, kind="stable")[:, :k]
    overlap = np.array(
        [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(knn_state, knn_belief)]
    )
    chance = k * (k / (n - 1))
    return {
        "k": int(k),
        "n_items": int(n),
        "mean_overlap": float(overlap.mean()),
        "precision_at_k": float(overlap.mean() / k),
        "chance_precision_at_k": float(chance / k),
        "lift_over_chance": float(overlap.mean() / chance) if chance > 0 else float("nan"),
    }


# --------------------------------------------------------------------------------------------
# HMM-H3
# --------------------------------------------------------------------------------------------


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex (Duchi et al., 2008).

    A linear probe's raw output is not a distribution, and clipping-then-renormalising is not a
    projection -- it can move a point further than necessary and it is not idempotent. This is the
    exact projection, so the reported KL and JS are to the closest distribution the linear map can
    be read as, not to an arbitrary rescaling of it.
    """
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    n, d = v.shape
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, d + 1)
    cond = u - css / idx > 0
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.clip(v - theta[:, None], 0.0, None)


@dataclass
class LinearProbe:
    """Ridge regression from hidden states to a target, with the intercept handled explicitly."""

    weight: np.ndarray
    bias: np.ndarray
    ridge: float

    def predict(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) @ self.weight + self.bias


def fit_linear_probe(h: np.ndarray, target: np.ndarray, ridge: float = 1e-2) -> LinearProbe:
    """Closed-form ridge fit. ``ridge`` is scaled by the mean feature variance, so it is
    invariant to the arbitrary scale of a model's hidden states -- the same nominal ridge means
    the same thing for GPT and for NextLat, which it would not if it were an absolute constant."""
    h = np.asarray(h, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    hm = h.mean(axis=0)
    ym = y.mean(axis=0)
    hc = h - hm
    scale = max(float((hc**2).mean()), 1e-12)
    gram = hc.T @ hc + ridge * scale * len(hc) * np.eye(h.shape[1])
    weight = np.linalg.solve(gram, hc.T @ (y - ym))
    return LinearProbe(weight=weight, bias=ym - hm @ weight, ridge=ridge)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    q = np.clip(q, eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0).sum(axis=-1) / np.log(2.0)


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def calibration_curve(
    predicted: np.ndarray, hidden_states: np.ndarray, n_bins: int = 10
) -> dict:
    """Reliability of the decoded posterior against the *realised* hidden states.

    For every (item, state) the probe emits a probability; bin those probabilities and compare
    each bin's mean prediction against the fraction of items in that bin whose true hidden state
    was that state. The exact posterior is perfectly calibrated by construction, so this measures
    how much of that property survives the probe.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    n, s = predicted.shape
    onehot = np.zeros((n, s))
    onehot[np.arange(n), np.asarray(hidden_states, dtype=int)] = 1.0
    p = predicted.ravel()
    y = onehot.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        rows.append({"bin": b, "n": int(m.sum()), "confidence": conf, "frequency": acc})
        ece += m.mean() * abs(conf - acc)
    return {"bins": rows, "expected_calibration_error": float(ece)}


def h3_posterior_decodability(
    h_train: np.ndarray,
    b_train: np.ndarray,
    h_test: np.ndarray,
    b_test: np.ndarray,
    next_obs_train: np.ndarray | None = None,
    next_obs_test: np.ndarray | None = None,
    hidden_states_test: np.ndarray | None = None,
    h_lengen: np.ndarray | None = None,
    b_lengen: np.ndarray | None = None,
    next_obs_lengen: np.ndarray | None = None,
    ridge: float = 1e-2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Fit the held-out linear probe and score it.

    The probe is fit on one pool and evaluated on a disjoint one; the length-64 arrays, when
    supplied, are scored with the probe fit at length 32 and never refit, which is what makes the
    length-generalisation number a generalisation number.
    """
    probe = fit_linear_probe(h_train, b_train, ridge=ridge)

    def _score(h: np.ndarray, b: np.ndarray) -> dict:
        raw = probe.predict(h)
        proj = project_to_simplex(raw)
        return {
            "n": int(len(h)),
            "r2_raw": _r2(raw, b),
            "r2_projected": _r2(proj, b),
            "mae": float(np.abs(proj - b).mean()),
            "kl_true_to_pred_bits": float(_kl(b, proj).mean()),
            "js_bits": float(js_divergence(b, proj).mean()),
            "argmax_agreement": float((proj.argmax(1) == b.argmax(1)).mean()),
        }

    out = {"probe_ridge": ridge, "posterior": {"test": _score(h_test, b_test)}}
    est, lo, hi = bootstrap_ci(
        js_divergence(b_test, project_to_simplex(probe.predict(h_test))), rng=rng
    )
    out["posterior"]["test"]["js_bits_ci"] = [lo, hi]

    if hidden_states_test is not None:
        out["posterior"]["calibration"] = calibration_curve(
            project_to_simplex(probe.predict(h_test)), hidden_states_test
        )
        out["posterior"]["calibration_of_exact_posterior"] = calibration_curve(
            b_test, hidden_states_test
        )

    if h_lengen is not None and b_lengen is not None:
        out["posterior"]["lengen64"] = _score(h_lengen, b_lengen)

    if next_obs_train is not None and next_obs_test is not None:
        # The coordinate-free target: a sufficient predictive state is defined only up to an
        # invertible map, so P(X_{t+1} | X_1:t) is the quantity that must be decodable regardless
        # of how the model chose to parameterise its beliefs.
        obs_probe = fit_linear_probe(h_train, next_obs_train, ridge=ridge)

        def _score_obs(h: np.ndarray, y: np.ndarray) -> dict:
            raw = obs_probe.predict(h)
            proj = project_to_simplex(raw)
            return {
                "n": int(len(h)),
                "r2_raw": _r2(raw, y),
                "kl_true_to_pred_bits": float(_kl(y, proj).mean()),
                "js_bits": float(js_divergence(y, proj).mean()),
                "argmax_agreement": float((proj.argmax(1) == y.argmax(1)).mean()),
            }

        out["next_obs"] = {"test": _score_obs(h_test, next_obs_test)}
        if h_lengen is not None and next_obs_lengen is not None:
            out["next_obs"]["lengen64"] = _score_obs(h_lengen, next_obs_lengen)

    return out


def baseline_probe_scores(b_train: np.ndarray, b_test: np.ndarray) -> dict:
    """What a probe with no information gets: predict the training-pool mean posterior.

    Every decodability number above is meaningless without this. A constant predictor already
    achieves a low mean absolute error on a 4-simplex, and reporting `mae` alone would look like
    success.
    """
    const = np.repeat(np.asarray(b_train).mean(axis=0)[None, :], len(b_test), axis=0)
    return {
        "r2": _r2(const, b_test),
        "mae": float(np.abs(const - b_test).mean()),
        "kl_true_to_pred_bits": float(_kl(b_test, const).mean()),
        "js_bits": float(js_divergence(b_test, const).mean()),
        "argmax_agreement": float((const.argmax(1) == np.asarray(b_test).argmax(1)).mean()),
    }
