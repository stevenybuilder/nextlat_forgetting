"""The forward algorithm is the ground truth of experiment B, so it is tested against the
definition rather than against itself.

The load-bearing test is `test_brute_force_agreement`: every posterior, every predictive state
prior, every next-observation distribution and every conditional log-likelihood is compared
against explicit enumeration over all hidden-state paths (spec section 12's explicit
requirement). With 4 states and length <= 6 that is at most 4096 paths per sequence, so the
comparison is exhaustive rather than sampled.

Every test here is constructed so that it fails on a wrong implementation: the brute-force
reference shares no code with the algorithm under test, the deterministic-HMM test pins an exact
one-hot answer, and the negative controls check that the assertions would have caught a
transposed matrix or a mismatched observation stream.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmm_geometry.forward import (  # noqa: E402
    HMM,
    brute_force_posteriors,
    forward_batch,
    js_divergence,
    log_forward_batch,
    sample_sequences,
)


def _reference_hmm() -> HMM:
    """A deliberately awkward 4-state / 4-observation HMM.

    Nothing here is symmetric: the transition matrix is not doubly stochastic, the initial
    distribution is not uniform, and the emission rows overlap heavily. A symmetric test HMM
    would hide index-transposition bugs, which is exactly the class of bug this file exists to
    catch.
    """
    transition = np.array(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.05, 0.55, 0.30, 0.10],
            [0.15, 0.05, 0.50, 0.30],
            [0.30, 0.10, 0.05, 0.55],
        ]
    )
    emission = np.array(
        [
            [0.50, 0.30, 0.15, 0.05],
            [0.10, 0.45, 0.35, 0.10],
            [0.20, 0.10, 0.40, 0.30],
            [0.35, 0.15, 0.10, 0.40],
        ]
    )
    initial = np.array([0.4, 0.3, 0.2, 0.1])
    return HMM(transition=transition, emission=emission, initial=initial)


def _deterministic_hmm() -> HMM:
    """Emissions are a permutation of the states, so the observation names the state exactly."""
    transition = np.array(
        [
            [0.10, 0.60, 0.20, 0.10],
            [0.25, 0.10, 0.50, 0.15],
            [0.40, 0.20, 0.10, 0.30],
            [0.30, 0.30, 0.30, 0.10],
        ]
    )
    emission = np.eye(4)[[2, 0, 3, 1]]  # state 0 -> obs 2, state 1 -> obs 0, ...
    initial = np.array([0.25, 0.25, 0.25, 0.25])
    return HMM(transition=transition, emission=emission, initial=initial)


# --------------------------------------------------------------------------------------------
# The required test: brute-force enumeration over all hidden-state paths.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6])
def test_brute_force_agreement(length: int) -> None:
    """Every prefix quantity matches enumeration over all 4**length hidden-state paths."""
    hmm = _reference_hmm()
    rng = np.random.default_rng(20260823)
    # Fixed hand-picked sequences plus random ones, so the test covers both the "all one symbol"
    # corner (which makes the posterior drift to a single mode) and generic mixtures.
    sequences = [
        [0] * length,
        [3] * length,
        list(np.arange(length) % 4),
        list(rng.integers(0, 4, size=length)),
        list(rng.integers(0, 4, size=length)),
    ]
    obs = np.array(sequences, dtype=np.int64)

    fast = forward_batch(hmm, obs)
    for i, seq in enumerate(sequences):
        ref = brute_force_posteriors(hmm, seq)
        np.testing.assert_allclose(fast.beliefs[i], ref.beliefs[0], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(fast.predictive[i], ref.predictive[0], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(fast.next_obs[i], ref.next_obs[0], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(fast.cond_logp[i], ref.cond_logp[0], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(
            fast.log_evidence[i], ref.log_evidence[0], atol=1e-12, rtol=1e-10
        )


def test_brute_force_agreement_deterministic_hmm() -> None:
    """The same comparison on the degenerate HMM, where zeros in the emission matrix make the
    joint table sparse -- the case most likely to expose a division-by-zero or a wrong mask."""
    hmm = _deterministic_hmm()
    for seq in itertools.product(range(4), repeat=4):
        fast = forward_batch(hmm, np.array([seq]))
        ref = brute_force_posteriors(hmm, seq)
        np.testing.assert_allclose(fast.beliefs[0], ref.beliefs[0], atol=1e-12)
        np.testing.assert_allclose(fast.next_obs[0], ref.next_obs[0], atol=1e-12)


def test_brute_force_reference_would_catch_a_transposed_transition() -> None:
    """Negative control for the test itself.

    If the comparison above passed for any HMM, it would be worthless. Transposing the
    transition matrix produces a different (still valid) chain, and enumeration under the
    original must then disagree with the forward pass under the transpose.
    """
    hmm = _reference_hmm()
    t = hmm.transition.T.copy()
    t = t / t.sum(axis=1, keepdims=True)
    wrong = HMM(transition=t, emission=hmm.emission, initial=hmm.initial)
    seq = [0, 3, 1, 2, 2, 0]
    ref = brute_force_posteriors(hmm, seq)
    got = forward_batch(wrong, np.array([seq]))
    assert not np.allclose(got.beliefs[0], ref.beliefs[0], atol=1e-6)


def test_brute_force_reference_would_catch_a_shifted_observation_stream() -> None:
    """Second negative control: comparing against the wrong observation stream must fail."""
    hmm = _reference_hmm()
    seq = [0, 3, 1, 2, 2, 0]
    shifted = [3, 1, 2, 2, 0, 0]
    ref = brute_force_posteriors(hmm, seq)
    got = forward_batch(hmm, np.array([shifted]))
    assert not np.allclose(got.beliefs[0], ref.beliefs[0], atol=1e-6)


# --------------------------------------------------------------------------------------------
# Normalisation, determinism, and the degenerate HMM.
# --------------------------------------------------------------------------------------------


def test_posteriors_sum_to_one_and_stay_in_the_simplex() -> None:
    hmm = _reference_hmm()
    rng = np.random.default_rng(7)
    obs, _ = sample_sequences(hmm, 512, 32, rng)
    res = forward_batch(hmm, obs.astype(np.int64))

    for arr in (res.beliefs, res.predictive, res.next_obs):
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)
        np.testing.assert_allclose(arr.sum(axis=-1), 1.0, atol=1e-12)
    # A posterior that is uniformly 0.25 everywhere would also sum to 1, so check the pass is
    # actually tracking the data: some posteriors must be far from uniform.
    assert res.beliefs.max() > 0.6
    assert res.beliefs.min() < 0.05


def test_deterministic_hmm_gives_one_hot_posteriors() -> None:
    """When emissions are a bijection, the posterior is a point mass on the emitting state."""
    hmm = _deterministic_hmm()
    rng = np.random.default_rng(11)
    obs, states = sample_sequences(hmm, 256, 24, rng)
    res = forward_batch(hmm, obs.astype(np.int64))

    onehot = np.zeros_like(res.beliefs)
    np.put_along_axis(onehot, states.astype(np.int64)[:, :, None], 1.0, axis=-1)
    np.testing.assert_allclose(res.beliefs, onehot, atol=1e-12)
    # And the entropy is exactly zero, not merely small.
    assert res.belief_entropy().max() < 1e-12


def test_partially_deterministic_emission_pins_a_state_only_when_it_should() -> None:
    """A single unambiguous symbol pins the state; the ambiguous symbols must not.

    This is the test that fails if the Bayes update multiplies by the wrong column: symbol 0 is
    emitted only by state 0, so any prefix ending in 0 has a one-hot posterior, while a prefix
    ending in the shared symbol 3 must have entropy strictly above zero.
    """
    emission = np.array(
        [
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.6, 0.0, 0.4],
            [0.0, 0.0, 0.7, 0.3],
            [0.0, 0.2, 0.2, 0.6],
        ]
    )
    transition = np.full((4, 4), 0.25)
    hmm = HMM(transition=transition, emission=emission, initial=np.full(4, 0.25))
    rng = np.random.default_rng(3)
    obs, _ = sample_sequences(hmm, 400, 12, rng)
    res = forward_batch(hmm, obs.astype(np.int64))
    ent = res.belief_entropy()

    pinned = obs == 0
    assert pinned.sum() > 20
    assert ent[pinned].max() < 1e-12
    np.testing.assert_allclose(res.beliefs[pinned][:, 0], 1.0, atol=1e-12)

    ambiguous = obs == 3
    assert ambiguous.sum() > 20
    assert ent[ambiguous].min() > 0.5


# --------------------------------------------------------------------------------------------
# Numerical stability at the lengths the experiment actually uses.
# --------------------------------------------------------------------------------------------


def test_length_64_is_stable_and_matches_the_log_space_implementation() -> None:
    """Length 64 is the length-generalisation split, so it is the length that must not drift."""
    hmm = _reference_hmm()
    rng = np.random.default_rng(64)
    obs, _ = sample_sequences(hmm, 256, 64, rng)
    obs = obs.astype(np.int64)

    fast = forward_batch(hmm, obs)
    slow = log_forward_batch(hmm, obs)

    assert np.all(np.isfinite(fast.beliefs))
    assert np.all(np.isfinite(fast.log_evidence))
    np.testing.assert_allclose(fast.beliefs.sum(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(fast.beliefs, slow.beliefs, atol=1e-11)
    np.testing.assert_allclose(fast.next_obs, slow.next_obs, atol=1e-11)
    np.testing.assert_allclose(fast.log_evidence, slow.log_evidence, atol=1e-9)

    # The evidence at length 64 is far below 1 but nowhere near an underflow; the point of the
    # scaled recursion is that this number never has to exist as a float.
    assert fast.log_evidence.max() < -50.0


def test_unnormalised_recursion_underflows_where_the_scaled_one_does_not() -> None:
    """Demonstrates the failure mode the scaling exists to prevent.

    The naive recursion on raw alphas is run here in float64 at length 600. It collapses to
    exactly zero, at which point a posterior would be 0/0. The scaled implementation returns
    valid, normalised posteriors at the same length and still agrees with log space.
    """
    hmm = _reference_hmm()
    rng = np.random.default_rng(600)
    obs, _ = sample_sequences(hmm, 4, 600, rng)
    obs = obs.astype(np.int64)

    alpha = np.broadcast_to(hmm.initial, (4, 4)).copy()
    emis_t = hmm.emission.T
    for t in range(obs.shape[1]):
        alpha = alpha * emis_t[obs[:, t]]
        if t + 1 < obs.shape[1]:
            alpha = alpha @ hmm.transition
    assert np.all(alpha == 0.0), "expected the unnormalised recursion to underflow to zero"

    res = forward_batch(hmm, obs)
    assert np.all(np.isfinite(res.beliefs))
    np.testing.assert_allclose(res.beliefs.sum(axis=-1), 1.0, atol=1e-12)
    slow = log_forward_batch(hmm, obs)
    np.testing.assert_allclose(res.beliefs, slow.beliefs, atol=1e-10)


def test_next_obs_is_the_predictive_distribution_of_the_actual_data() -> None:
    """A calibration check that would fail on an off-by-one in the prefix indexing.

    `next_obs[:, t]` claims to be `P(X_{t+1} | X_1:t)`. Averaging the probability assigned to the
    symbol that actually occurred must reproduce the empirical frequency, and the mean
    conditional log-likelihood must match `cond_logp`. Shifting the index by one breaks both.
    """
    hmm = _reference_hmm()
    rng = np.random.default_rng(99)
    obs, _ = sample_sequences(hmm, 4000, 32, rng)
    obs = obs.astype(np.int64)
    res = forward_batch(hmm, obs)

    p_actual = np.take_along_axis(res.next_obs[:, :-1, :], obs[:, :, None], axis=2)[:, :, 0]
    np.testing.assert_allclose(np.log(p_actual), res.cond_logp, atol=1e-12)

    # Empirical negative log-likelihood per symbol, versus the model's own claim.
    assert -res.cond_logp.mean() == pytest.approx(-np.log(p_actual).mean(), abs=1e-12)

    # And the marginal at t=0 matches the empirical first-symbol frequencies.
    emp = np.bincount(obs[:, 0], minlength=4) / obs.shape[0]
    np.testing.assert_allclose(res.next_obs[0, 0], hmm.initial @ hmm.emission, atol=1e-12)
    assert np.abs(emp - res.next_obs[0, 0]).max() < 0.02


def test_sampling_is_deterministic_in_the_generator_only() -> None:
    hmm = _reference_hmm()
    a, sa = sample_sequences(hmm, 200, 16, np.random.default_rng(5))
    b, sb = sample_sequences(hmm, 200, 16, np.random.default_rng(5))
    c, _ = sample_sequences(hmm, 200, 16, np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sa, sb)
    assert not np.array_equal(a, c)

    # The sampler is a whole-batch draw, so sequence i is not independent of the batch size.
    # What must hold regardless is that the marginal statistics are the model's.
    big, _ = sample_sequences(hmm, 20000, 16, np.random.default_rng(1))
    freq = np.bincount(big.ravel(), minlength=4) / big.size
    np.testing.assert_allclose(freq, hmm.obs_marginal(), atol=0.01)


def test_sampled_states_follow_the_transition_matrix() -> None:
    """Guards the sampler itself: an inverse-CDF bug would show up as a wrong transition count."""
    hmm = _reference_hmm()
    _, states = sample_sequences(hmm, 5000, 64, np.random.default_rng(2))
    counts = np.zeros((4, 4))
    np.add.at(counts, (states[:, :-1].ravel(), states[:, 1:].ravel()), 1)
    emp = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(emp - hmm.transition).max() < 0.02


# --------------------------------------------------------------------------------------------
# Small utilities the pair bank depends on.
# --------------------------------------------------------------------------------------------


def test_js_divergence_bounds_and_symmetry() -> None:
    p = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]])
    q = np.array([[0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.2, 0.0]])
    d = js_divergence(p, q)
    np.testing.assert_allclose(d, js_divergence(q, p), atol=1e-15)
    assert d[0] == pytest.approx(1.0, abs=1e-12)  # disjoint support -> exactly 1 bit
    assert d[1] == pytest.approx(0.0, abs=1e-15)
    assert 0.0 < d[2] < 1.0


def test_hmm_rejects_invalid_matrices() -> None:
    with pytest.raises(ValueError):
        HMM(np.array([[0.5, 0.4], [0.5, 0.5]]), np.eye(2), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        HMM(np.eye(2), np.eye(2), np.array([0.6, 0.6]))
    with pytest.raises(ValueError):
        HMM(np.array([[1.5, -0.5], [0.5, 0.5]]), np.eye(2), np.array([0.5, 0.5]))


def test_hmm_matrices_are_immutable_and_hash_stable() -> None:
    hmm = _reference_hmm()
    with pytest.raises(ValueError):
        hmm.transition[0, 0] = 0.9
    assert hmm.sha256() == _reference_hmm().sha256()
    other = HMM(hmm.transition, hmm.emission, np.array([0.25, 0.25, 0.25, 0.25]))
    assert other.sha256() != hmm.sha256()


def test_stationary_distribution_is_a_fixed_point() -> None:
    hmm = _reference_hmm()
    pi = hmm.stationary()
    np.testing.assert_allclose(pi @ hmm.transition, pi, atol=1e-12)
    assert pi.sum() == pytest.approx(1.0)
    assert hmm.mean_dwell_time() > 1.0
