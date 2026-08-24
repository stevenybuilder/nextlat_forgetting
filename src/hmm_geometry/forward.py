"""Exact Bayesian belief states for a discrete HMM: the scaled (normalised) forward algorithm.

Spec section 12 asks for `b_t(s) = P(S_t = s | X_1:t)` for every prefix of every sequence, plus
the exact next-observation distribution `P(X_{t+1} | X_1:t)`.

## Why explicit renormalisation rather than log space

Two implementations were on the table: run the recursion on log-alphas with a `logsumexp` at
every step, or run it on normalised beliefs and carry the per-step scaling factor separately.
This module does the second, for three reasons.

1. **The scaling factor is the quantity we want anyway.** In the scaled recursion the
   normaliser at step `t` is exactly `c_t = P(X_t = x_t | X_1:t-1)`, the one-step predictive
   likelihood. So the conditional log-likelihood, the sequence evidence, and the
   next-observation distribution all fall out of the same pass with no extra work, whereas the
   log-space version has to reconstruct them.
2. **Underflow is structurally impossible, not merely unlikely.** After every step the state
   vector sums to 1 by construction, so the dynamic range of the carried quantity is bounded by
   `[0, 1]` no matter how long the sequence is. Log space also avoids underflow, but it does so
   by an argument about the exponent range; here there is nothing to underflow. The only failure
   mode left is `c_t == 0`, which happens only if the observation is impossible under the
   posterior, and that is asserted rather than silently divided by.
3. **It is the faster shape for a batch.** Each step is two dense mat-muls over the batch
   (`(N,S) @ (S,S)` and `(N,S) @ (S,O)`) plus an elementwise product. `logsumexp` per step would
   replace those with an exp/log pair over the same arrays. With `L = 32..64` python-level steps
   and `N` in the tens of thousands, the batch dimension carries the vectorisation and the loop
   over `t` is unavoidable in either version (the recursion is sequential).

The log-space variant is still implemented, as `log_forward_batch`, and the tests hold the two
against each other on a length-64 batch. It exists as an independent second opinion, not as a
fallback.

## Conventions

Indices follow the spec's 1-based maths with 0-based storage:

* `obs[n, t]` is `x_{t+1}` for `t = 0 .. L-1`.
* `beliefs[n, t]` is `b_{t+1}`, i.e. `P(S_{t+1} = s | X_1:t+1)`.
* `predictive[n, t]` is `P(S_{t+1} = s | X_1:t)` for `t = 0 .. L`, so `predictive[n, 0]` is the
  initial distribution `pi` and `predictive[n, L]` is the belief about the state that would emit
  the observation after the end of the sequence.
* `next_obs[n, t]` is `P(X_{t+1} = o | X_1:t)` for `t = 0 .. L`, so `next_obs[n, 0]` is the
  marginal `P(X_1)` and `next_obs[n, t]` for `t >= 1` is the prediction made from the prefix of
  length `t`. `next_obs[n, t] = predictive[n, t] @ E`.

`S_1` is drawn from `pi` and emits `X_1`; there is no burn-in step in which a state is drawn and
discarded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "HMM",
    "ForwardResult",
    "forward_batch",
    "log_forward_batch",
    "brute_force_posteriors",
    "sample_sequences",
]

_PROB_TOL = 1e-12


@dataclass(frozen=True)
class HMM:
    """A discrete, stationary, fully-observed-emission HMM.

    Attributes
    ----------
    transition:
        ``(S, S)`` row-stochastic. ``transition[i, j] = P(S_{t+1} = j | S_t = i)``.
    emission:
        ``(S, O)`` row-stochastic. ``emission[s, o] = P(X_t = o | S_t = s)``.
    initial:
        ``(S,)`` distribution of ``S_1``.
    """

    transition: np.ndarray
    emission: np.ndarray
    initial: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.transition, dtype=np.float64)
        e = np.asarray(self.emission, dtype=np.float64)
        p = np.asarray(self.initial, dtype=np.float64)
        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise ValueError(f"transition must be square, got {t.shape}")
        if e.ndim != 2 or e.shape[0] != t.shape[0]:
            raise ValueError(f"emission must be (S, O) with S={t.shape[0]}, got {e.shape}")
        if p.shape != (t.shape[0],):
            raise ValueError(f"initial must be (S,) with S={t.shape[0]}, got {p.shape}")
        for name, m in (("transition", t), ("emission", e), ("initial", p[None, :])):
            if np.any(m < -_PROB_TOL):
                raise ValueError(f"{name} has negative entries")
            sums = m.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-9):
                raise ValueError(f"{name} rows must sum to 1, got {sums}")
        # Freeze normalised, read-only copies so no caller can mutate a "frozen" HMM in place.
        for name, m in (("transition", t), ("emission", e), ("initial", p)):
            m = np.clip(m, 0.0, None)
            m = m / m.sum(axis=-1, keepdims=True)
            m.flags.writeable = False
            object.__setattr__(self, name, m)

    @property
    def n_states(self) -> int:
        return self.transition.shape[0]

    @property
    def n_obs(self) -> int:
        return self.emission.shape[1]

    def stationary(self) -> np.ndarray:
        """Stationary distribution over hidden states (left eigenvector for eigenvalue 1).

        Uses the eigen-decomposition rather than power iteration so the answer does not depend
        on an iteration count.
        """
        vals, vecs = np.linalg.eig(self.transition.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        if abs(vals[idx] - 1.0) > 1e-8:
            raise ValueError("transition matrix has no eigenvalue 1; not a stochastic matrix")
        v = np.real(vecs[:, idx])
        if v.sum() < 0:
            v = -v
        if np.any(v < -1e-9):
            raise ValueError("stationary vector is not non-negative; chain may be reducible")
        v = np.clip(v, 0.0, None)
        return v / v.sum()

    def obs_marginal(self) -> np.ndarray:
        """Stationary marginal over observations, ``P(X = o)``."""
        return self.stationary() @ self.emission

    def mean_dwell_time(self) -> float:
        """Stationary-weighted mean number of consecutive steps spent in a state.

        For a state with self-transition ``q`` the dwell length is geometric with mean
        ``1 / (1 - q)``. Weighting by the stationary distribution gives the dwell time an
        observer actually experiences. A uniform-random 4-state chain gives 4/3.
        """
        q = np.diag(self.transition)
        if np.any(q >= 1.0 - 1e-12):
            return float("inf")
        return float(self.stationary() @ (1.0 / (1.0 - q)))

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "n_obs": self.n_obs,
            "transition": self.transition.tolist(),
            "emission": self.emission.tolist(),
            "initial": self.initial.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HMM":
        return cls(
            transition=np.asarray(d["transition"], dtype=np.float64),
            emission=np.asarray(d["emission"], dtype=np.float64),
            initial=np.asarray(d["initial"], dtype=np.float64),
        )

    def sha256(self) -> str:
        """Hash of the canonical JSON of the matrices.

        This is the identity used everywhere downstream: a manifest, a pair bank or a dataset
        that was built under a different HMM must not silently be usable with this one.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ForwardResult:
    """Everything the forward pass produces for a batch of sequences.

    Shapes, for a batch of ``N`` sequences of length ``L`` over ``S`` states and ``O`` symbols:

    ``beliefs`` ``(N, L, S)``; ``predictive`` ``(N, L+1, S)``; ``next_obs`` ``(N, L+1, O)``;
    ``cond_logp`` ``(N, L)``; ``log_evidence`` ``(N,)``.
    """

    beliefs: np.ndarray
    predictive: np.ndarray
    next_obs: np.ndarray
    cond_logp: np.ndarray
    log_evidence: np.ndarray

    def belief_entropy(self, base: float = 2.0) -> np.ndarray:
        """Entropy of every posterior, shape ``(N, L)``."""
        b = self.beliefs
        with np.errstate(divide="ignore", invalid="ignore"):
            logb = np.where(b > 0, np.log(b), 0.0)
        return -(b * logb).sum(axis=-1) / np.log(base)


def forward_batch(hmm: HMM, obs: np.ndarray) -> ForwardResult:
    """Scaled forward algorithm over a batch of equal-length observation sequences.

    Parameters
    ----------
    hmm:
        The generative model. Not fitted, not estimated -- the frozen ground truth.
    obs:
        ``(N, L)`` integer array of observation symbols in ``[0, O)``.

    Returns
    -------
    ForwardResult
        Posteriors for every prefix, the one-step-ahead state prior for every prefix, the exact
        next-observation distribution for every prefix, and the conditional log-likelihoods.
    """
    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2:
        raise ValueError(f"obs must be (N, L), got shape {obs.shape}")
    if not np.issubdtype(obs.dtype, np.integer):
        raise TypeError(f"obs must be an integer array, got dtype {obs.dtype}")
    if obs.size and (obs.min() < 0 or obs.max() >= hmm.n_obs):
        raise ValueError(f"obs values must lie in [0, {hmm.n_obs})")

    n, length = obs.shape
    s = hmm.n_states
    trans = hmm.transition
    emis = hmm.emission
    emis_t = np.ascontiguousarray(emis.T)  # (O, S): emis_t[o] = P(x=o | s) over s

    beliefs = np.empty((n, length, s), dtype=np.float64)
    predictive = np.empty((n, length + 1, s), dtype=np.float64)
    next_obs = np.empty((n, length + 1, hmm.n_obs), dtype=np.float64)
    cond_logp = np.empty((n, length), dtype=np.float64)

    pred = np.broadcast_to(hmm.initial, (n, s)).copy()
    for t in range(length):
        predictive[:, t, :] = pred
        next_obs[:, t, :] = pred @ emis
        # Bayes update: multiply by the likelihood of the observation that actually occurred.
        weighted = pred * emis_t[obs[:, t]]
        c = weighted.sum(axis=1)
        if np.any(c <= 0.0):
            bad = int(np.argmin(c))
            raise FloatingPointError(
                f"observation {int(obs[bad, t])} at step {t} of sequence {bad} has zero "
                "probability under the posterior; the sequence is impossible under this HMM"
            )
        cond_logp[:, t] = np.log(c)
        b = weighted / c[:, None]
        beliefs[:, t, :] = b
        pred = b @ trans

    predictive[:, length, :] = pred
    next_obs[:, length, :] = pred @ emis

    return ForwardResult(
        beliefs=beliefs,
        predictive=predictive,
        next_obs=next_obs,
        cond_logp=cond_logp,
        log_evidence=cond_logp.sum(axis=1),
    )


def log_forward_batch(hmm: HMM, obs: np.ndarray) -> ForwardResult:
    """Independent log-space implementation, used to cross-check :func:`forward_batch`.

    Carries unnormalised ``log alpha_t(s) = log P(S_t = s, X_1:t)`` and reduces with
    ``logsumexp``. Slower and it has to exponentiate to report a posterior, but it shares no
    arithmetic with the scaled version, so agreement between the two is evidence rather than a
    tautology.
    """
    from scipy.special import logsumexp

    obs = np.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :]
    n, length = obs.shape
    s = hmm.n_states

    with np.errstate(divide="ignore"):
        log_t = np.log(hmm.transition)
        log_e = np.log(hmm.emission)
        log_pi = np.log(hmm.initial)
    log_e_t = np.ascontiguousarray(log_e.T)  # (O, S)

    beliefs = np.empty((n, length, s))
    predictive = np.empty((n, length + 1, s))
    next_obs = np.empty((n, length + 1, hmm.n_obs))
    cond_logp = np.empty((n, length))

    log_pred = np.broadcast_to(log_pi, (n, s)).copy()  # log P(S_t=s, X_1:t-1)
    prev_evidence = np.zeros(n)
    for t in range(length):
        pred = np.exp(log_pred - logsumexp(log_pred, axis=1, keepdims=True))
        predictive[:, t, :] = pred
        next_obs[:, t, :] = pred @ hmm.emission
        log_alpha = log_pred + log_e_t[obs[:, t]]
        evidence = logsumexp(log_alpha, axis=1)
        cond_logp[:, t] = evidence - prev_evidence
        prev_evidence = evidence
        beliefs[:, t, :] = np.exp(log_alpha - evidence[:, None])
        log_pred = logsumexp(log_alpha[:, :, None] + log_t[None, :, :], axis=1)

    pred = np.exp(log_pred - logsumexp(log_pred, axis=1, keepdims=True))
    predictive[:, length, :] = pred
    next_obs[:, length, :] = pred @ hmm.emission

    return ForwardResult(
        beliefs=beliefs,
        predictive=predictive,
        next_obs=next_obs,
        cond_logp=cond_logp,
        log_evidence=cond_logp.sum(axis=1),
    )


def brute_force_posteriors(hmm: HMM, obs: Sequence[int]) -> ForwardResult:
    """Reference implementation by explicit enumeration over all hidden-state paths.

    For a single sequence of length ``L``, enumerate all ``S**L`` state paths, score each one
    with the joint ``P(s_1:L, x_1:L)``, and read every quantity off the joint table by summation.
    This is the definition, not an algorithm: it is exponential and only usable for ``L <= 6``
    with ``S = 4`` (4096 paths), which is exactly the regime spec section 12 requires the
    forward algorithm to be validated in.

    The posteriors it returns are per-prefix, so a length-6 sequence produces the same six
    posteriors that :func:`forward_batch` reports -- each computed independently from its own
    enumeration over paths of that prefix length, with no recursion between prefixes.
    """
    obs = list(int(o) for o in obs)
    length = len(obs)
    s = hmm.n_states
    if length == 0:
        raise ValueError("brute force needs at least one observation")
    if s**length > 2_000_000:
        raise ValueError(
            f"refusing to enumerate {s}^{length} paths; brute force is for short sequences"
        )

    beliefs = np.zeros((1, length, s))
    predictive = np.zeros((1, length + 1, s))
    next_obs = np.zeros((1, length + 1, hmm.n_obs))
    cond_logp = np.zeros((1, length))

    # P(X_1 = o) with no conditioning.
    predictive[0, 0, :] = hmm.initial
    next_obs[0, 0, :] = hmm.initial @ hmm.emission

    prefix_evidence = 1.0
    for t in range(1, length + 1):
        joint = np.zeros(s)  # joint[s_t] = P(S_t = s_t, X_1:t = x_1:t)
        joint_next = np.zeros(s)  # joint_next[s_{t+1}] = P(S_{t+1}, X_1:t)
        for path in itertools.product(range(s), repeat=t):
            p = hmm.initial[path[0]] * hmm.emission[path[0], obs[0]]
            for k in range(1, t):
                p *= hmm.transition[path[k - 1], path[k]] * hmm.emission[path[k], obs[k]]
            joint[path[-1]] += p
            joint_next += p * hmm.transition[path[-1]]
        evidence = joint.sum()
        beliefs[0, t - 1, :] = joint / evidence
        predictive[0, t, :] = joint_next / evidence
        next_obs[0, t, :] = (joint_next / evidence) @ hmm.emission
        cond_logp[0, t - 1] = np.log(evidence / prefix_evidence)
        prefix_evidence = evidence

    return ForwardResult(
        beliefs=beliefs,
        predictive=predictive,
        next_obs=next_obs,
        cond_logp=cond_logp,
        log_evidence=cond_logp.sum(axis=1),
    )


def sample_sequences(
    hmm: HMM, n_sequences: int, length: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw observation sequences (and their true hidden states) from the HMM.

    Determinism is threaded through the explicit ``rng`` argument and nothing else: the result
    depends on the generator's state and the shape of the draws, never on worker count or
    iteration order. Sampling is vectorised across the batch with one inverse-CDF draw per step,
    so a 100,000 x 32 corpus costs 32 python iterations.
    """
    if n_sequences <= 0 or length <= 0:
        raise ValueError("n_sequences and length must be positive")
    s = hmm.n_states
    states = np.empty((n_sequences, length), dtype=np.int8)
    obs = np.empty((n_sequences, length), dtype=np.int8)

    trans_cdf = np.cumsum(hmm.transition, axis=1)
    emis_cdf = np.cumsum(hmm.emission, axis=1)
    init_cdf = np.cumsum(hmm.initial)

    u_state = rng.random((n_sequences, length))
    u_obs = rng.random((n_sequences, length))

    cur = np.searchsorted(init_cdf, u_state[:, 0], side="right").clip(0, s - 1)
    for t in range(length):
        if t > 0:
            # Row-wise inverse CDF: compare against each sequence's own transition row.
            cur = (u_state[:, t, None] >= trans_cdf[cur]).sum(axis=1).clip(0, s - 1)
        states[:, t] = cur
        obs[:, t] = (u_obs[:, t, None] >= emis_cdf[cur]).sum(axis=1).clip(0, hmm.n_obs - 1)
    return obs, states


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Jensen-Shannon divergence between rows of ``p`` and rows of ``q``.

    In bits by default, so the value lies in ``[0, 1]`` and thresholds read as fractions of a
    bit. Broadcasting follows numpy rules; the distribution axis is the last one.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(axis=-1)
        kl_qm = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(axis=-1)
    out = 0.5 * (kl_pm + kl_qm) / np.log(base)
    return np.clip(out, 0.0, None)


def entropy(p: np.ndarray, base: float = 2.0) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return h / np.log(base)


def iter_prefix_items(obs: np.ndarray, min_t: int, max_t: int) -> Iterable[tuple[int, int]]:
    """Yield ``(sequence_index, prefix_length)`` for every prefix in the requested band."""
    n, length = obs.shape
    hi = min(max_t, length)
    for i in range(n):
        for t in range(min_t, hi + 1):
            yield i, t
