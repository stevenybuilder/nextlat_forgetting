"""Tests for the HMM-H1/H2/H3 estimators.

Model states are synthesised here with a known relationship to the exact posteriors, so each
estimator can be checked in both directions: it must find the effect when the effect is present,
and it must report nothing when the states are noise. An estimator that only passes the first kind
of test is an estimator that will report a positive result on any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_geometry import evaluate as ev  # noqa: E402
from hmm_geometry.forward import HMM, forward_batch, js_divergence, sample_sequences  # noqa: E402


def _hmm() -> HMM:
    transition = np.array(
        [
            [0.70, 0.21, 0.06, 0.03],
            [0.03, 0.70, 0.21, 0.06],
            [0.06, 0.03, 0.70, 0.21],
            [0.21, 0.06, 0.03, 0.70],
        ]
    )
    emission = np.full((4, 4), 0.10)
    np.fill_diagonal(emission, 0.70)
    base = HMM(transition, emission, np.full(4, 0.25))
    return HMM(transition, emission, base.stationary())


@pytest.fixture(scope="module")
def belief_data():
    hmm = _hmm()
    obs, states = sample_sequences(hmm, 600, 24, np.random.default_rng(0))
    res = forward_batch(hmm, obs.astype(np.int64))
    b = res.beliefs[:, 8:, :].reshape(-1, 4)
    nxt = res.next_obs[:, 9:, :].reshape(-1, 4)
    s = states[:, 8:].reshape(-1)
    return hmm, b, nxt, s


def _states_from_beliefs(b: np.ndarray, dim: int, noise: float, seed: int) -> np.ndarray:
    """A model that has genuinely encoded the belief: a random linear embedding plus noise.

    Deliberately *not* an identity embedding. The spec's caveat is that a sufficient predictive
    state is defined only up to an invertible transformation, so every estimator must be invariant
    to the choice of coordinates -- testing against an identity embedding would hide a metric that
    secretly depends on the belief simplex's own axes.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(b.shape[1], dim))
    return b @ a + rng.normal(scale=noise, size=(len(b), dim)) + rng.normal(size=dim) * 3.0


# --------------------------------------------------------------------------------------------
# Distances and the simplex projection
# --------------------------------------------------------------------------------------------


def test_centered_cosine_matches_scipy_after_centering() -> None:
    from scipy.spatial.distance import cosine

    rng = np.random.default_rng(1)
    xa, xb = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
    center = np.concatenate([xa, xb]).mean(axis=0)
    got = ev.centered_cosine_distance(xa, xb, center)
    want = np.array([cosine(a - center, b - center) for a, b in zip(xa, xb)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_centered_cosine_is_invariant_to_a_shared_translation() -> None:
    rng = np.random.default_rng(2)
    xa, xb = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    shift = rng.normal(size=6) * 10
    c = np.concatenate([xa, xb]).mean(axis=0)
    d0 = ev.centered_cosine_distance(xa, xb, c)
    d1 = ev.centered_cosine_distance(xa + shift, xb + shift, c + shift)
    np.testing.assert_allclose(d0, d1, atol=1e-12)


def test_simplex_projection_agrees_with_a_constrained_optimiser() -> None:
    rng = np.random.default_rng(3)
    v = rng.normal(size=(12, 4)) * 2
    got = ev.project_to_simplex(v)
    for row, g in zip(v, got):
        res = optimize.minimize(
            lambda x: ((x - row) ** 2).sum(),
            x0=np.full(4, 0.25),
            bounds=[(0, 1)] * 4,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 500},
        )
        np.testing.assert_allclose(g, res.x, atol=1e-6)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, atol=1e-12)
    assert (got >= 0).all()


def test_simplex_projection_is_idempotent_and_fixes_distributions() -> None:
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(4), size=20)
    np.testing.assert_allclose(ev.project_to_simplex(p), p, atol=1e-12)
    v = rng.normal(size=(20, 4))
    once = ev.project_to_simplex(v)
    np.testing.assert_allclose(ev.project_to_simplex(once), once, atol=1e-12)


def test_whitener_produces_identity_covariance() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 6))
    x = rng.normal(size=(4000, 6)) @ a + 5.0
    mean, w = ev.fit_whitener(x, ridge=1e-6)
    z = (x - mean) @ w
    np.testing.assert_allclose(np.cov(z.T), np.eye(6), atol=0.1)


# --------------------------------------------------------------------------------------------
# HMM-H1
# --------------------------------------------------------------------------------------------


def _pair_sets(b: np.ndarray, states: np.ndarray, rng: np.random.Generator, n: int = 400):
    """Equivalent pairs (matched beliefs) and controls (mismatched beliefs) at matched edit
    distance -- the edit distances are copied across so the matching is exact by construction."""
    order = np.argsort(b[:, 0])
    # "Equivalent": nearest neighbours in belief space. "Control": arbitrary other items.
    ia = order[: 2 * n : 2]
    ib = order[1 : 2 * n : 2]
    ca = rng.integers(0, len(b), size=n)
    cb = rng.integers(0, len(b), size=n)
    edit = rng.integers(8, 16, size=n)
    plen = rng.integers(10, 20, size=n)
    equiv = ev.PairStates(
        a=states[ia],
        b=states[ib],
        edit_distance=edit,
        prefix_len=plen,
        js_bits=js_divergence(b[ia], b[ib]),
    )
    ctrl = ev.PairStates(
        a=states[ca],
        b=states[cb],
        edit_distance=edit,
        prefix_len=plen,
        js_bits=js_divergence(b[ca], b[cb]),
    )
    return equiv, ctrl


def test_h1_finds_predictive_equivalence_when_the_state_encodes_the_belief(belief_data) -> None:
    _, b, _, _ = belief_data
    rng = np.random.default_rng(10)
    states = _states_from_beliefs(b, dim=32, noise=0.02, seed=1)
    equiv, ctrl = _pair_sets(b, states, rng)
    out = ev.h1_predictive_equivalence(equiv, ctrl, rng=np.random.default_rng(0))
    assert out["equivalent_js_bits_mean"] < out["control_js_bits_mean"]
    assert out["paired_delta_mean"] < 0
    assert out["paired_delta_ci_hi"] < 0  # the interval excludes "no effect"


def test_h1_reports_no_effect_when_the_state_is_noise(belief_data) -> None:
    _, b, _, _ = belief_data
    rng = np.random.default_rng(11)
    states = np.random.default_rng(2).normal(size=(len(b), 32))
    equiv, ctrl = _pair_sets(b, states, rng)
    out = ev.h1_predictive_equivalence(equiv, ctrl, rng=np.random.default_rng(0))
    assert out["paired_delta_ci_lo"] < 0 < out["paired_delta_ci_hi"]


def test_h1_refuses_an_unmatched_control(belief_data) -> None:
    _, b, _, _ = belief_data
    rng = np.random.default_rng(12)
    states = _states_from_beliefs(b, dim=16, noise=0.05, seed=3)
    equiv, ctrl = _pair_sets(b, states, rng)
    broken = ev.PairStates(
        a=ctrl.a,
        b=ctrl.b,
        edit_distance=ctrl.edit_distance + 1,
        prefix_len=ctrl.prefix_len,
        js_bits=ctrl.js_bits,
    )
    with pytest.raises(ValueError, match="edit distance"):
        ev.h1_predictive_equivalence(equiv, broken)


# --------------------------------------------------------------------------------------------
# HMM-H2
# --------------------------------------------------------------------------------------------


def test_h2_correlation_is_strong_for_a_belief_encoding_and_absent_for_noise(belief_data) -> None:
    _, b, _, _ = belief_data
    rng = np.random.default_rng(20)
    n = 3000
    i = rng.integers(0, len(b), size=n)
    j = rng.integers(0, len(b), size=n)
    js = js_divergence(b[i], b[j])
    edit = rng.integers(4, 20, size=n)
    plen = rng.integers(10, 24, size=n)

    good = _states_from_beliefs(b, dim=32, noise=0.02, seed=4)
    out = ev.h2_relative_geometry(
        ev.PairStates(good[i], good[j], edit, plen, js)
    )
    assert out["spearman_distance_vs_js"] > 0.6
    assert out["partial_spearman_given_edit_and_length"] > 0.6

    noise = np.random.default_rng(5).normal(size=(len(b), 32))
    null = ev.h2_relative_geometry(ev.PairStates(noise[i], noise[j], edit, plen, js))
    assert abs(null["spearman_distance_vs_js"]) < 0.06
    assert abs(null["partial_spearman_given_edit_and_length"]) < 0.06


def test_partial_spearman_removes_a_planted_confound() -> None:
    rng = np.random.default_rng(21)
    c = rng.normal(size=2000)
    x = c + rng.normal(scale=0.3, size=2000)
    y = c + rng.normal(scale=0.3, size=2000)
    raw, _ = ev.partial_spearman(x, y, [])
    partial, _ = ev.partial_spearman(x, y, [c])
    assert raw > 0.8
    assert abs(partial) < 0.1
    # With no controls it must reduce exactly to Spearman's rho.
    assert raw == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-10)


def test_neighborhood_retrieval_beats_chance_only_for_a_belief_encoding(belief_data) -> None:
    _, b, _, _ = belief_data
    idx = np.random.default_rng(22).choice(len(b), size=800, replace=False)
    good = _states_from_beliefs(b, dim=32, noise=0.01, seed=6)[idx]
    out = ev.neighborhood_retrieval(good, b[idx], k=10)
    assert out["precision_at_k"] > 0.5
    assert out["lift_over_chance"] > 20

    noise = np.random.default_rng(7).normal(size=(800, 32))
    null = ev.neighborhood_retrieval(noise, b[idx], k=10)
    assert null["precision_at_k"] < 3 * null["chance_precision_at_k"]


# --------------------------------------------------------------------------------------------
# HMM-H3
# --------------------------------------------------------------------------------------------


def test_probe_recovers_the_posterior_from_a_linear_encoding(belief_data) -> None:
    _, b, nxt, s = belief_data
    states = _states_from_beliefs(b, dim=40, noise=0.02, seed=8)
    n = len(b) // 2
    out = ev.h3_posterior_decodability(
        states[:n],
        b[:n],
        states[n:],
        b[n:],
        next_obs_train=nxt[:n],
        next_obs_test=nxt[n:],
        hidden_states_test=s[n:],
        rng=np.random.default_rng(0),
    )
    assert out["posterior"]["test"]["r2_projected"] > 0.95
    assert out["posterior"]["test"]["js_bits"] < 0.01
    assert out["next_obs"]["test"]["js_bits"] < 0.01
    # The exact posterior is calibrated by construction, so its ECE is the floor the probe is
    # measured against; a probe that beat it would mean the test is measuring something else.
    exact_ece = out["posterior"]["calibration_of_exact_posterior"]["expected_calibration_error"]
    assert exact_ece < 0.03
    assert out["posterior"]["calibration"]["expected_calibration_error"] < 0.05


def test_probe_fails_on_states_that_carry_no_belief_information(belief_data) -> None:
    _, b, _, _ = belief_data
    states = np.random.default_rng(9).normal(size=(len(b), 40))
    n = len(b) // 2
    out = ev.h3_posterior_decodability(
        states[:n], b[:n], states[n:], b[n:], rng=np.random.default_rng(0)
    )
    assert out["posterior"]["test"]["r2_projected"] < 0.05
    baseline = ev.baseline_probe_scores(b[:n], b[n:])
    # No better than predicting the mean posterior, which is the only honest reading of a probe
    # fitted on noise.
    assert out["posterior"]["test"]["js_bits"] >= baseline["js_bits"] * 0.9


def test_baseline_constant_probe_scores_zero_r2(belief_data) -> None:
    _, b, _, _ = belief_data
    n = len(b) // 2
    base = ev.baseline_probe_scores(b[:n], b[n:])
    assert abs(base["r2"]) < 0.02
    assert base["js_bits"] > 0.05  # a constant is genuinely wrong, despite a small MAE
    assert base["mae"] < 0.25  # ... which is why MAE alone must never be reported


def test_probe_ridge_is_scale_invariant(belief_data) -> None:
    """A model whose hidden states are 100x larger must get the same probe, or the GPT-vs-NextLat
    comparison would be confounded by activation scale."""
    _, b, _, _ = belief_data
    states = _states_from_beliefs(b, dim=24, noise=0.05, seed=11)
    n = len(b) // 2
    a = ev.h3_posterior_decodability(states[:n], b[:n], states[n:], b[n:])
    c = ev.h3_posterior_decodability(
        states[:n] * 100.0, b[:n], states[n:] * 100.0, b[n:]
    )
    assert a["posterior"]["test"]["r2_projected"] == pytest.approx(
        c["posterior"]["test"]["r2_projected"], abs=1e-6
    )


def test_length_generalisation_uses_the_probe_fitted_at_length_32(belief_data) -> None:
    """The length-64 score must come from the length-32 probe, never a refit."""
    hmm, b, _, _ = belief_data
    obs64, _ = sample_sequences(hmm, 300, 64, np.random.default_rng(30))
    res64 = forward_batch(hmm, obs64.astype(np.int64))
    b64 = res64.beliefs[:, 40:, :].reshape(-1, 4)

    embed = np.random.default_rng(12).normal(size=(4, 32))
    states = b @ embed
    states64 = b64 @ embed
    n = len(b) // 2
    out = ev.h3_posterior_decodability(
        states[:n], b[:n], states[n:], b[n:], h_lengen=states64, b_lengen=b64
    )
    assert out["posterior"]["lengen64"]["r2_projected"] > 0.95

    # A probe that had been refit on the length-64 pool would be unaffected by corrupting the
    # length-32 training pool; this one must degrade.
    broken = ev.h3_posterior_decodability(
        np.random.default_rng(13).normal(size=(n, 32)),
        b[:n],
        states[n:],
        b[n:],
        h_lengen=states64,
        b_lengen=b64,
    )
    assert broken["posterior"]["lengen64"]["r2_projected"] < 0.2


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(40)
    x = rng.normal(loc=0.5, scale=1.0, size=500)
    a = ev.bootstrap_ci(x, rng=np.random.default_rng(1))
    b = ev.bootstrap_ci(x, rng=np.random.default_rng(1), n_boot=10_000)
    assert a == b
    est, lo, hi = a
    assert lo < est < hi
    assert est == pytest.approx(x.mean())
    # The interval must be wider for less data.
    _, lo2, hi2 = ev.bootstrap_ci(x[:50], rng=np.random.default_rng(1))
    assert (hi2 - lo2) > (hi - lo)
