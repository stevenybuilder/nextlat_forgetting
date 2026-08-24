"""Representation extraction and the pure-numpy geometry layer for Lure-Star H1/H2/H3.

Two strictly separated layers live in this file.

**Layer A — pure numpy (everything below `# --- LAYER A ---`).**  Distances, centering,
whitening, shrinkage, and logit margins.  Imports nothing but numpy/scipy, runs on a
CPU-only host, and is fully unit-tested by ``tests/test_representations.py``.

**Layer B — the tensor-touching shim (everything below `# --- LAYER B ---`).**  The only
code that imports torch.  It is a thin adapter over the pinned upstream forwards; it does
no arithmetic beyond `lm_head` and an index-select, and it hands numpy arrays back to
Layer A immediately.  ``import torch`` happens *inside* the functions, so this module
imports cleanly on a machine with no torch.

------------------------------------------------------------------------------------
WHERE THE STATE COMES FROM  (verified against upstream/NextLat @ 3770be6)
------------------------------------------------------------------------------------
The "final post-normalization hidden state" of spec §7 is

    GPT      x         = self.transformer.norm(x)      models/model_gpt.py:276
    NextLat  text_embd = self.transformer.norm(x)      models/model_nextlat.py:197

Both are shape ``(B, T, n_embd)`` with ``n_embd = 384`` for G(5,5).

The two forwards are ASYMMETRIC under ``return_hidden_states=True``:

    GPT      returns (output, x)             model_gpt.py:290-291
             `output` is logits when targets is None (model_gpt.py:279-280)
    NextLat  returns (token_embeds, text_embd)   model_nextlat.py:199-200
             and EARLY-RETURNS *before* lm_head is ever applied (model_nextlat.py:203-204)

So for NextLat the caller must apply ``model.lm_head(h)`` itself (the head is defined at
models/model_nextlat.py:121).  `_forward_gpt` / `_forward_nextlat` below encode exactly
this difference and nothing else.

**BST is a third case and it does not have a single clean analogue.**  Its encoder is two
stacks, and each has its own final norm:

    BST forward   fwd = self.transformer_f.norm(fwd)   models/model_bst.py:287
    BST backward  bwd = self.transformer_b.norm(bwd)   models/model_bst.py:313

and neither feeds ``lm_head`` directly.  The readout is a two-input ``TextHead`` that
concatenates them, adds a SwiGLU MLP residual, applies a THIRD norm over 2*n_embd, splits
the result and only then projects (models/model_bst.py:91-104).  In GPT and NextLat the
"final post-norm state" and the "immediate pre-logit state" are the same tensor; in BST
they are three different tensors.  The choice we make, and the reasons, are in
docs/EXTRACTION.md §3.  In one line: the PRIMARY BST state is the forward encoder's
post-norm output (model_bst.py:287) because it is the only one that is architecturally the
same object as the other two arms, is causal over the prompt alone, and is the only
item-varying input to BST's own inference path.  The backward encoder is excluded because
under the reverse-causal mask (model_bst.py:213-217) ``bwd[62]`` attends tokens 62..68 and
therefore CONTAINS THE ANSWER.

``transformer.norm`` is ``LayerNorm(n_embd, bias=config.bias)`` and the shipped 5_5 configs
set ``bias: false``.  ``LayerNorm.forward`` (models/model_base.py:823-830) dispatches to
``F.rms_norm`` when ``bias is None``.  **The extracted state is therefore RMS-normalized,
not mean-centered**: its direction is preserved and its per-vector scale is normalized, but
the coordinate mean across the feature axis is *not* removed.  This is why the primary
distance below re-centers explicitly over an item pool: RMSNorm leaves a large shared mean
component that would otherwise dominate raw cosine similarity.

The preferred capture uses **no hook at all** — call the inner transformer directly with
``return_hidden_states=True``, which is a public keyword on both forwards.  A forward hook
on ``model.model.transformer.norm`` is provided as a fallback for capture inside an
unmodified ``compute_loss`` call; see :func:`hidden_state_hook`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "EQ_TOKEN_ID",
    "EOS_TOKEN_ID",
    "ARCHITECTURES",
    "BST_STATE_POLICY",
    "BST_STATE_ROLES",
    "HIDDEN_STATE_MODULE_PATH",
    "STATE_SOURCE",
    "PSI_EXTRACTION_INDEX",
    "BRANCH_MARGIN_INDEX",
    "EXTRACTION_INDICES",
    "CENTERING_POOL_POLICY",
    "CENTERING_POOL_CONDITIONS",
    "CenteringPool",
    "LeakageError",
    "locate_prompt_delimiter",
    "resolve_extraction_indices",
    "next_token_targets",
    "centering_mean",
    "centered_cosine_distance",
    "cosine_distance_raw",
    "Whitener",
    "ledoit_wolf_shrinkage",
    "whitened_euclidean_distance",
    "branch_margin",
    "full_vocab_margin",
    "TORCH_AVAILABLE",
    "hidden_state_hook",
    "forward_states_and_logits",
    "forward_all_states",
    "bst_backward_eos_state",
    "bst_texthead_prelogit",
    "extract_positions",
]


# =====================================================================================
# FROZEN EXTRACTION CONSTANTS
# =====================================================================================
#
# Tokenizer (upstream data/stargraph.py:9-30) with maxNodes=100:
#     node ids 0..99 | '|'=100 | '='=101 | '/'=102 | '$'=103 | EOS=104, vocab_size=106.
EQ_TOKEN_ID = 101

#: EOS is ``maxNodes + 4`` (data/stargraph.py:32).  Stargraph has no semantic EOS; the id
#: exists "for compatibility with the bst training code" (data/stargraph.py:29-31) and is
#: appended to every serialized line by ``Tokenizer.tokenize`` (data/stargraph.py:52-58),
#: so index 68 of a G(5,5) sequence is an EOS.  BST needs it: its inference-time backward
#: embedding is the encoding of a lone EOS (model_bst.py:806-810).
EOS_TOKEN_ID = 104

#: The three architecture-matched arms of spec §8.  Order is the preregistered priority
#: order of the cross-model contrasts, not alphabetical.
ARCHITECTURES = ("nextlat", "bst", "gpt")

# ------------------------------------------------------------------------------------
# THE CORRECTION (frozen 2026-08-23, before any model exists; see docs/EXTRACTION.md)
# ------------------------------------------------------------------------------------
# Spec §6/H1 says: "Extract the state at the final prompt delimiter `=`, before the first
# answer token."  For G(5,5) that is sequence index 62 — verified in
# docs/UPSTREAM_REPORT.md §1.6 (20 edges*2 + 19 '|' + 1 '/' + 2 + 1 '=' = 62 prompt
# tokens, then 5 path tokens and EOS for total_len 69), and it is the same 62 that
# upstream writes into `config.model.context_length` at data/stargraph.py:251.
#
# But the token GENERATED from index 62 is `path[0]`, which for a Path-Star answer is the
# SOURCE node.  The source is already printed verbatim in the prompt's `/source,goal`
# field, so predicting it is a copy, and it is identical across every member of a quartet
# by construction (spec §5 fixes source and goal across base/repeat/near-safe/
# near-critical).  A logit margin measured at index 62 therefore CANNOT differ between
# near-safe and near-critical: the quantity being scored is the same token in both.
#
# The first genuine branch decision is `path[1]` — the first node of the goal arm —
# generated from the hidden state at index 63 (the source token, now inside the answer
# region).  Worked example from docs/UPSTREAM_REPORT.md §2.4: prompt `…53,5/49,33=`,
# answer `49,97,53,5,33`; h[62] scores 49 (the source), h[63] scores 97 (the branch).
#
# RESOLUTION, frozen now:
#   * index 62 REMAINS the preregistered primary extraction point for PSI (H1).  We do
#     not move the preregistered primary after the fact.
#   * index 63 is extracted alongside it, and EVERY correct-branch logit margin used by
#     H2 and H3 is computed from the logits at index 63.
#   * both indices are reported for every metric, so the choice is auditable rather than
#     load-bearing.
PSI_EXTRACTION_INDEX = 62
BRANCH_MARGIN_INDEX = 63

#: Named roles -> sequence index.  Both are always extracted together.
EXTRACTION_INDICES = {
    "psi_primary": PSI_EXTRACTION_INDEX,      # H1 PSI, preregistered
    "branch_margin": BRANCH_MARGIN_INDEX,     # H2/H3 correct-branch logit margin
}

#: The centering pool is a *stated* policy, not a default that can drift.  See
#: :func:`centered_cosine_distance` — the mean is always an explicit argument.
CENTERING_POOL_POLICY = (
    "The centering mean for centered cosine distance is computed over the FULL E_lure "
    "evaluation pool for one (model, seed, extraction_index) cell: all base, repeat, "
    "near-safe, near-critical and far-critical states pooled together, one mean vector "
    "per cell. It is NEVER computed per condition, and never over only the pair being "
    "scored. Centering per condition would subtract exactly the between-condition shift "
    "that PSI is trying to measure, which is a silent way to manufacture (or erase) an "
    "effect. The pool is an explicit required argument everywhere in this module."
)


#: The five E_lure conditions of spec §5 that together make up the declared centering
#: pool.  :meth:`CenteringPool.from_conditions` refuses to build a pool unless every one
#: of these is either supplied or *explicitly* declared missing, so dropping a condition
#: becomes a recorded statement rather than a silent substitution.
CENTERING_POOL_CONDITIONS = (
    "base",
    "repeat",
    "near_safe",
    "near_critical",
    "far_critical",
)


#: What "the final post-normalization hidden state" resolves to in each arm, with the
#: upstream line that produces it.  Written down here because for BST the answer is a
#: CHOICE among three candidates rather than a lookup, and a choice that is not recorded
#: next to the code is a choice that gets quietly revised.  Full argument: docs/EXTRACTION.md §3.
STATE_SOURCE = {
    "gpt": "models/model_gpt.py:276  x = self.transformer.norm(x)",
    "nextlat": "models/model_nextlat.py:197  text_embd = self.transformer.norm(x)",
    "bst": "models/model_bst.py:287  fwd = self.transformer_f.norm(fwd)",
}

#: Module path of that state's producing submodule, relative to the ``ModelBase`` wrapper.
#: Used only by :func:`hidden_state_hook`; the preferred capture needs no hook.
HIDDEN_STATE_MODULE_PATH = {
    "gpt": "model.transformer.norm",
    "nextlat": "model.transformer.norm",
    "bst": "encoder.transformer_f.norm",
}

#: BST returns more than one candidate state, so every one that we extract is named and
#: given a role.  Nothing is extracted "and then we decide".
BST_STATE_ROLES = {
    "hidden": (
        "PRIMARY. Forward-encoder final post-norm state, model_bst.py:287. The analogue "
        "of the GPT/NextLat state: same Block class (imported at model_bst.py:28), same "
        "12L/6H/384, same LayerNorm(bias=False) -> F.rms_norm dispatch, same causal mask, "
        "same (B,T,384) shape, same next-token index convention (fwd[t] predicts seq[t+1], "
        "model_bst.py:581)."
    ),
    "hidden_texthead": (
        "DECLARED SECONDARY. TextHead pre-logit state x_next, model_bst.py:91-100, computed "
        "with the inference-time backward embedding. This is the state lm_head actually "
        "consumes (model_bst.py:103), so it is the functional analogue; it is secondary "
        "because it is 384-d only after a chunk of a 768-d norm and because, with the "
        "backward input held at its inference-time constant, it is a fixed nonlinear "
        "reparameterization of `hidden` and carries no extra information."
    ),
    "excluded_backward": (
        "EXCLUDED. Backward-encoder post-norm state, model_bst.py:313. Under the "
        "reverse-causal mask (model_bst.py:213-217) bwd[t] attends tokens t..T, so bwd[62] "
        "encodes the answer path. Using it would make PSI a measurement of the target, not "
        "of the history representation that spec §7 defines."
    ),
}

#: How BST's logits are produced, and why they cannot be ``lm_head(hidden)``.
BST_STATE_POLICY = (
    "BST has no single tensor that is both the final post-normalization hidden state and "
    "the immediate pre-logit state; in GPT and NextLat those coincide and in BST they do "
    "not. We take the FORWARD encoder's post-norm state (model_bst.py:287) as the primary "
    "analogue and report the TextHead pre-logit state (model_bst.py:91-100) as a declared "
    "secondary. BST logits are NEVER lm_head(hidden): the head is a two-input TextHead "
    "(model_bst.py:83-110) whose lm_head is trained on the post-MLP, post-norm, chunked "
    "768->384 half, so lm_head(forward_state) is type-compatible and semantically wrong. "
    "The backward input is the encoding of a lone EOS token, which is exactly what "
    "BST.generate does at model_bst.py:806-810 and therefore what produced the paper's "
    "Figure 6 accuracy. It equals bwd[68] of the full sequence (the terminal EOS attends "
    "only itself under the reverse-causal document mask and EOS always takes position 0, "
    "model_base.py:571-583), so it is a state the model was trained on rather than an "
    "off-distribution stand-in."
)


class LeakageError(RuntimeError):
    """Raised when a statistic would be computed on data it was estimated from."""


# =====================================================================================
# --- LAYER A: pure numpy ---------------------------------------------------------
# =====================================================================================


def _as2d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_items, n_features); got shape {a.shape}")
    return a


# ---------------------------------------------------------------- index resolution ---


def locate_prompt_delimiter(tokens: np.ndarray, eq_token_id: int = EQ_TOKEN_ID) -> np.ndarray:
    """Index of the final ``=`` delimiter in each row of ``tokens`` (n_items, seq_len).

    Uses the LAST occurrence, because a node id can never equal 101 for maxNodes=100 but
    we do not want the function to silently depend on that.
    """
    tok = np.asarray(tokens)
    if tok.ndim != 2:
        raise ValueError(f"tokens must be 2-D (n_items, seq_len); got {tok.shape}")
    hit = tok == eq_token_id
    if not np.all(hit.any(axis=1)):
        bad = int(np.flatnonzero(~hit.any(axis=1))[0])
        raise ValueError(f"row {bad} contains no '=' token (id {eq_token_id})")
    # last occurrence per row
    idx = tok.shape[1] - 1 - np.argmax(hit[:, ::-1], axis=1)
    return idx.astype(np.int64)


def resolve_extraction_indices(
    tokens: np.ndarray,
    *,
    eq_token_id: int = EQ_TOKEN_ID,
    expect_delimiter_index: Optional[int] = PSI_EXTRACTION_INDEX,
) -> dict:
    """Verify the frozen indices against a real token batch and return them.

    Returns ``{"psi_primary": 62, "branch_margin": 63, "delimiter_index": 62}``.

    Raises if the batch is ragged with respect to the delimiter, if the delimiter is not
    at ``expect_delimiter_index``, or if index 63 would fall outside the sequence.  This
    is the guard that would catch a G(d,l) change or a tokenizer change before any
    geometry is computed.
    """
    delim = locate_prompt_delimiter(tokens, eq_token_id)
    if delim.size == 0:
        raise ValueError("empty token batch")
    if not np.all(delim == delim[0]):
        raise ValueError(
            "delimiter index is not constant across the batch "
            f"(min {int(delim.min())}, max {int(delim.max())}); upstream "
            "data/stargraph.py:237-247 measures the layout from data[0] only, so a "
            "ragged file would silently mis-slice every batch"
        )
    d = int(delim[0])
    if expect_delimiter_index is not None and d != expect_delimiter_index:
        raise ValueError(
            f"'=' is at index {d}, not the frozen preregistered index "
            f"{expect_delimiter_index}. Extraction indices are frozen; do not move them."
        )
    seq_len = int(np.asarray(tokens).shape[1])
    if d + 1 >= seq_len:
        raise ValueError(f"branch-margin index {d + 1} is outside seq_len {seq_len}")
    return {"psi_primary": d, "branch_margin": d + 1, "delimiter_index": d}


def next_token_targets(tokens: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """The token *generated from* each position, i.e. ``tokens[:, p + 1]``.

    Shape ``(n_items, len(positions))``.  This is the function that makes the correction
    above checkable: ``next_token_targets(t, [62])`` is the source and is constant within
    a quartet; ``next_token_targets(t, [63])`` is the first branch node and is not.
    """
    tok = np.asarray(tokens)
    pos = np.asarray(positions, dtype=np.int64)
    if np.any(pos + 1 >= tok.shape[1]):
        raise ValueError("position+1 out of range for the given sequence length")
    return tok[:, pos + 1]


# --------------------------------------------------------------- centered cosine -----


def centering_mean(pool_states: np.ndarray) -> np.ndarray:
    """Mean vector of a raw stacked pool.  See :data:`CENTERING_POOL_POLICY`.

    LOW-LEVEL.  This helper cannot tell whether the array it was handed is the declared
    E_lure pool or the scored pair, so it must not be used to build the pool for a
    reported PSI.  Use :meth:`CenteringPool.from_conditions`, which names the conditions
    and checks them, and which :func:`lurestar.evaluate.psi_distances_centered_cosine`
    now requires.
    """
    pool = _as2d(pool_states, "pool_states")
    if pool.shape[0] < 2:
        raise ValueError("centering pool needs at least 2 states")
    return pool.mean(axis=0)


def cosine_distance_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Uncentered cosine distance ``1 - cos(a, b)``, rowwise.  Diagnostic only."""
    A = _as2d(a, "a")
    B = _as2d(b, "b")
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("zero-norm vector in cosine distance")
    return 1.0 - np.einsum("ij,ij->i", A, B) / (na * nb)


def centered_cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mean: np.ndarray,
) -> np.ndarray:
    """PRIMARY distance (spec §6/H1).  ``1 - cos(a - mean, b - mean)`` rowwise.

    ``mean`` is REQUIRED and keyword-only.  There is deliberately no default and no
    "center over whatever you passed in" fallback: the centering pool is the single
    easiest place to manufacture a PSI effect, so it must be named by the caller and
    recorded.  Build it with :func:`centering_mean` over the pool described by
    :data:`CENTERING_POOL_POLICY`.

    Range is [0, 2].  Invariant to a translation of the *whole* pool (because the mean
    translates with it) and, by design, NOT invariant to translating one condition only.
    """
    A = _as2d(a, "a")
    B = _as2d(b, "b")
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
    m = np.asarray(mean, dtype=np.float64)
    if m.shape != (A.shape[1],):
        raise ValueError(f"mean must have shape ({A.shape[1]},); got {m.shape}")
    Ac = A - m
    Bc = B - m
    na = np.linalg.norm(Ac, axis=1)
    nb = np.linalg.norm(Bc, axis=1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError(
            "a state coincides with the centering mean; centered cosine is undefined"
        )
    return 1.0 - np.einsum("ij,ij->i", Ac, Bc) / (na * nb)


def _row_keys(x: np.ndarray) -> frozenset:
    """Exact byte keys for the rows of a float64 view of ``x``.

    Used only for *containment*: the pool is built by stacking the very arrays that are
    later scored, so the rows are bit-identical and an exact key is the right test.  It
    is O(n) and never used for anything numeric.
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return frozenset(a[i].tobytes() for i in range(a.shape[0]))


@dataclass(frozen=True)
class CenteringPool:
    """The centering pool for the PRIMARY distance, as a checked object.

    Rationale.  ``centered_cosine_distance`` takes a bare mean vector, which makes the
    pool the single easiest place to manufacture (or erase) a PSI effect: a caller who
    passes the scored pair instead of the declared E_lure pool gets a plausible number
    and no complaint.  Naming the argument is not enough — the *caller* is the mutant
    that survives.  So the pool is constructed here, from named conditions, and

    * every one of :data:`CENTERING_POOL_CONDITIONS` must be either supplied or listed
      in ``declared_missing`` — dropping ``far_critical`` silently is impossible;
    * :meth:`require_contains` checks that the states being scored are actually in the
      pool, so a pool from a different cell (or pure noise) is rejected outright;
    * :meth:`report` carries the condition list, the per-condition row counts and the
      declared omissions into every serialized metric, so what was pooled is auditable
      after the fact rather than taken on trust.
    """

    mean: np.ndarray
    conditions: tuple
    counts: tuple
    declared_missing: tuple
    n: int
    n_features: int
    row_keys: frozenset = field(default=frozenset(), repr=False)

    @staticmethod
    def from_conditions(
        *,
        declared_missing: Sequence[str] = (),
        **states: np.ndarray,
    ) -> "CenteringPool":
        missing = tuple(declared_missing)
        unknown = [c for c in list(states) + list(missing) if c not in CENTERING_POOL_CONDITIONS]
        if unknown:
            raise ValueError(
                f"unknown centering-pool condition(s) {unknown}; the declared pool is "
                f"{list(CENTERING_POOL_CONDITIONS)} (spec §5)"
            )
        both = set(states) & set(missing)
        if both:
            raise ValueError(f"condition(s) {sorted(both)} both supplied and declared missing")
        unaccounted = [c for c in CENTERING_POOL_CONDITIONS if c not in states and c not in missing]
        if unaccounted:
            raise ValueError(
                f"centering-pool condition(s) {unaccounted} were neither supplied nor "
                "listed in declared_missing. The pool policy is a preregistered claim "
                "about WHICH states were pooled; omitting one has to be stated, not "
                "assumed. Pass the array, or pass "
                f"declared_missing={tuple(unaccounted)!r} and say why in the run record."
            )
        if not states:
            raise ValueError("a centering pool needs at least one condition")
        order = [c for c in CENTERING_POOL_CONDITIONS if c in states]
        arrays = [_as2d(states[c], c) for c in order]
        widths = {a.shape[1] for a in arrays}
        if len(widths) != 1:
            raise ValueError(f"conditions disagree on n_features: {sorted(widths)}")
        stacked = np.vstack(arrays)
        if stacked.shape[0] < 2:
            raise ValueError("centering pool needs at least 2 states")
        return CenteringPool(
            mean=stacked.mean(axis=0),
            conditions=tuple(order),
            counts=tuple(int(a.shape[0]) for a in arrays),
            declared_missing=tuple(sorted(missing)),
            n=int(stacked.shape[0]),
            n_features=int(stacked.shape[1]),
            row_keys=_row_keys(stacked),
        )

    def require_contains(self, name: str, states: np.ndarray) -> None:
        """Raise unless every row of ``states`` is a row of the pool."""
        keys = _row_keys(_as2d(states, name))
        outside = keys - self.row_keys
        if outside:
            raise ValueError(
                f"{len(outside)} distinct row(s) of `{name}` are not in the centering "
                f"pool (pool n={self.n}, conditions={list(self.conditions)}). The "
                "primary distance must be centred on the pool the scored states belong "
                "to; a pool from another cell silently rescales PSI."
            )

    def require_conditions(self, *names: str) -> None:
        absent = [c for c in names if c not in self.conditions]
        if absent:
            raise ValueError(
                f"condition(s) {absent} are being scored but were not pooled "
                f"(pool conditions {list(self.conditions)}, declared missing "
                f"{list(self.declared_missing)})"
            )

    def report(self) -> dict:
        return {
            "policy": CENTERING_POOL_POLICY,
            "conditions": list(self.conditions),
            "counts": list(self.counts),
            "declared_missing": list(self.declared_missing),
            "n": self.n,
            "n_features": self.n_features,
            "complete": not self.declared_missing,
        }


# ------------------------------------------------------------------- whitening -------


def ledoit_wolf_shrinkage(centered: np.ndarray) -> float:
    """Ledoit-Wolf (2004) optimal shrinkage intensity toward ``(tr(S)/p) I``.

    ``centered`` is (n, p) already mean-subtracted.  Returned intensity is in [0, 1].
    """
    X = _as2d(centered, "centered")
    n, p = X.shape
    S = (X.T @ X) / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    if d2 <= 0:
        return 1.0
    sq_norms = np.einsum("ij,ij->i", X, X)          # ||x_k||^2
    # sum_k ||x_k x_k^T - S||_F^2 = sum_k ||x_k||^4 - n ||S||_F^2
    bbar2 = (np.sum(sq_norms**2) - n * np.sum(S**2)) / (n * n * p)
    b2 = min(max(bbar2, 0.0), d2)
    return float(b2 / d2)


@dataclass(frozen=True)
class Whitener:
    """Whitening transform for the DECLARED ROBUSTNESS CHECK (whitened Euclidean).

    Contract, enforced rather than documented:

    * the covariance is estimated on a **held-out pool** supplied to :meth:`fit`;
    * shrinkage toward ``(tr(S)/p) I`` is always applied and always reported
      (:attr:`shrinkage`), with a floor so it can never silently be exactly zero;
    * :meth:`distance` raises :class:`LeakageError` if any item id being scored was in
      the fitting pool.  Item ids are how "held-out" stops being an honour system.
    """

    mean: np.ndarray
    transform: np.ndarray          # W, symmetric, with W @ W == inv(Sigma)
    covariance: np.ndarray         # Sigma actually used (post-shrinkage)
    shrinkage: float
    n_pool: int
    n_features: int
    condition_number: float
    fit_item_ids: frozenset = field(default_factory=frozenset)

    @staticmethod
    def fit(
        pool_states: np.ndarray,
        *,
        item_ids: Optional[Iterable] = None,
        shrinkage: object = "ledoit_wolf",
        min_shrinkage: float = 1e-3,
    ) -> "Whitener":
        X = _as2d(pool_states, "pool_states")
        n, p = X.shape
        if n < 2:
            raise ValueError("whitening pool needs at least 2 states")
        mean = X.mean(axis=0)
        Xc = X - mean
        S = (Xc.T @ Xc) / n
        if isinstance(shrinkage, str):
            if shrinkage != "ledoit_wolf":
                raise ValueError(f"unknown shrinkage rule {shrinkage!r}")
            alpha = ledoit_wolf_shrinkage(Xc)
        else:
            alpha = float(shrinkage)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1]; got {alpha}")
        alpha = max(alpha, float(min_shrinkage))
        mu = np.trace(S) / p
        Sig = (1.0 - alpha) * S + alpha * mu * np.eye(p)
        evals, evecs = np.linalg.eigh(Sig)
        if np.any(evals <= 0):
            raise np.linalg.LinAlgError(
                "shrunk covariance is not positive definite; raise `shrinkage`"
            )
        W = (evecs * evals**-0.5) @ evecs.T
        # NOTE: no numpy coercion here.  `frozenset(np.asarray(list(ids)).tolist())`
        # silently casts a heterogeneous id list to strings, so a genuinely leaked
        # integer id stops matching its own entry and the guard reports "clean".
        ids = frozenset() if item_ids is None else frozenset(item_ids)
        if item_ids is not None and len(ids) != n:
            raise ValueError(
                f"item_ids has {len(ids)} unique entries but the pool has {n} rows"
            )
        return Whitener(
            mean=mean,
            transform=W,
            covariance=Sig,
            shrinkage=float(alpha),
            n_pool=int(n),
            n_features=int(p),
            condition_number=float(evals.max() / evals.min()),
            fit_item_ids=ids,
        )

    def _check_ids(self, item_ids: Optional[Iterable]) -> None:
        if not self.fit_item_ids:
            # The whitener was fit without ids, so "held out" cannot be checked at all.
            # `report()["pool_is_heldout"]` is False; the caller is responsible.
            return
        if item_ids is None:
            raise LeakageError(
                "this Whitener was fit with item ids, so every scored batch must supply "
                "item_ids too; omitting them turns the held-out guarantee back into an "
                "honour system"
            )
        overlap = self.fit_item_ids & frozenset(item_ids)
        if overlap:
            raise LeakageError(
                f"{len(overlap)} scored item(s) were in the whitening pool "
                f"(e.g. {sorted(map(str, overlap))[:3]}); the covariance must be "
                "estimated on a disjoint held-out pool"
            )

    def apply(self, states: np.ndarray) -> np.ndarray:
        X = _as2d(states, "states")
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return (X - self.mean) @ self.transform

    def distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        item_ids: Optional[Iterable] = None,
    ) -> np.ndarray:
        """Whitened Euclidean distance, rowwise.

        Equals the Mahalanobis distance under :attr:`covariance`:
        ``sqrt((a-b)^T Sigma^{-1} (a-b))``.  The shared ``mean`` cancels in ``a - b``,
        so only the metric (not the location) of the held-out pool enters.
        """
        self._check_ids(item_ids)
        A = _as2d(a, "a")
        B = _as2d(b, "b")
        if A.shape != B.shape:
            raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
        D = (A - B) @ self.transform
        return np.sqrt(np.einsum("ij,ij->i", D, D))

    def report(self) -> dict:
        return {
            "metric": "whitened_euclidean",
            "role": "declared robustness check (spec §6/H1)",
            "shrinkage": self.shrinkage,
            "shrinkage_target": "(trace(S)/p) * I",
            "n_pool": self.n_pool,
            "n_features": self.n_features,
            "condition_number": self.condition_number,
            "pool_is_heldout": bool(self.fit_item_ids),
        }


def whitened_euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    whitener: Whitener,
    item_ids: Optional[Iterable] = None,
) -> np.ndarray:
    """Functional wrapper around :meth:`Whitener.distance` (keyword-only whitener)."""
    return whitener.distance(a, b, item_ids=item_ids)


# ------------------------------------------------------------------- margins ---------


def branch_margin(
    logits: np.ndarray,
    correct_ids: np.ndarray,
    competitor_ids: np.ndarray,
    competitor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Correct-branch logit margin: ``logit[correct] - max_j logit[competitor_j]``.

    ``logits``        (n_items, vocab)  — logits at :data:`BRANCH_MARGIN_INDEX`.
    ``correct_ids``   (n_items,)        — the goal arm's first node, i.e. ``path[1]``.
    ``competitor_ids``(n_items, k)      — the OTHER arms' first nodes (for G(5,5), k=4).
    ``competitor_mask``(n_items, k)     — optional bool, False marks a padded slot.

    Restricting the competitor set to the sibling branch heads is what makes this a
    *branch* margin rather than a generic confidence score: it is the quantity that a
    near-critical suffix swap is designed to move.  Positive means the model prefers the
    correct branch.
    """
    L = _as2d(logits, "logits")
    n, v = L.shape
    c = np.asarray(correct_ids, dtype=np.int64)
    K = np.asarray(competitor_ids, dtype=np.int64)
    if c.shape != (n,):
        raise ValueError(f"correct_ids must have shape ({n},); got {c.shape}")
    if K.ndim != 2 or K.shape[0] != n:
        raise ValueError(f"competitor_ids must have shape ({n}, k); got {K.shape}")
    if np.any(c < 0) or np.any(c >= v) or np.any(K < 0) or np.any(K >= v):
        raise ValueError("token id out of vocabulary range")
    if np.any(K == c[:, None]):
        raise ValueError("the correct id appears in its own competitor set")
    rows = np.arange(n)
    correct = L[rows, c]
    comp = L[rows[:, None], K]
    if competitor_mask is not None:
        m = np.asarray(competitor_mask, dtype=bool)
        if m.shape != K.shape:
            raise ValueError("competitor_mask must match competitor_ids shape")
        if not np.all(m.any(axis=1)):
            raise ValueError("some item has an empty competitor set")
        comp = np.where(m, comp, -np.inf)
    return correct - comp.max(axis=1)


def full_vocab_margin(logits: np.ndarray, correct_ids: np.ndarray) -> np.ndarray:
    """``logit[correct] - max over all other vocabulary entries``.  Secondary report.

    Note the G(5,5) vocab has an unused slack row (id 105, docs/UPSTREAM_REPORT.md §1.5);
    it participates here only as a never-argmax competitor, which is harmless.
    """
    L = _as2d(logits, "logits")
    n, v = L.shape
    c = np.asarray(correct_ids, dtype=np.int64)
    if c.shape != (n,):
        raise ValueError(f"correct_ids must have shape ({n},); got {c.shape}")
    rows = np.arange(n)
    correct = L[rows, c].copy()
    masked = L.copy()
    masked[rows, c] = -np.inf
    return correct - masked.max(axis=1)


# =====================================================================================
# --- LAYER B: the only torch-touching code -------------------------------------------
# =====================================================================================


def _torch():
    try:
        import torch  # noqa: PLC0415  (deliberately lazy: this host has no torch)
    except ImportError as exc:  # pragma: no cover - exercised only on a GPU host
        raise RuntimeError(
            "Layer B needs torch. All geometry math lives in Layer A and runs without it."
        ) from exc
    return torch


try:  # pragma: no cover - depends on the host
    import importlib.util as _ilu

    TORCH_AVAILABLE = _ilu.find_spec("torch") is not None
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


def hidden_state_hook(store: dict, key: str = "h"):
    """Fallback capture: a forward hook on the state-producing norm module.

    Prefer :func:`forward_states_and_logits`, which needs no hook at all.  Use this only
    when the state must be captured inside an unmodified ``compute_loss`` call.  The
    module path is ``model.transformer.norm`` for GPT and NextLat and
    ``encoder.transformer_f.norm`` for BST; see :data:`HIDDEN_STATE_MODULE_PATH`.
    Requires ``trainer.compile: false`` (which the spec already requires) or the submodule
    path gains a ``_orig_mod`` level — and for BST specifically, ``compile()`` rebinds
    ``raw_f_enc`` itself (model_bst.py:317-325), so a compiled BST is a harder case than a
    compiled GPT and is simply out of scope.
    """

    def _capture(module, inputs, output):  # pragma: no cover - needs torch
        store[key] = output.detach()

    return _capture


def _check_architecture(architecture: str) -> str:
    arch = str(architecture).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"architecture must be 'gpt', 'nextlat' or 'bst'; got {architecture!r}"
        )
    return arch


def bst_backward_eos_state(encoder, tokens, *, eos_token_id: int = EOS_TOKEN_ID):
    """BST's inference-time backward embedding: the encoding of a lone EOS token.

    This is not our invention.  ``BST.generate`` computes exactly this
    (model_bst.py:806-810) and it is the only backward input BST ever uses at inference,
    so it is the one behind the paper's Figure 6 accuracy.  It is also not off
    distribution: under the reverse-causal document mask the terminal EOS of a serialized
    sequence attends only itself (model_bst.py:205-217) and EOS always takes document
    position 0 (model_base.py:571-583), so this vector equals ``bwd[68]`` of a real
    G(5,5) sequence — an endpoint the model was trained against.

    Returns a ``(B, 1, n_embd)`` tensor.
    """
    eos = tokens.new_full((tokens.shape[0], 1), int(eos_token_id))
    _, bwd = encoder(eos, compute_forward=False, compute_backward=True)
    return bwd


def bst_texthead_prelogit(text_head, fwd, bwd):
    """Re-derive ``TextHead``'s pre-logit split ``(x_next, x_prev)``.

    Line-for-line model_bst.py:91-100.  It is duplicated rather than hooked because the
    upstream head returns only stacked logits when targets are None (model_bst.py:102-110)
    and we want the state, not the projection.  The duplication is made self-checking by
    :func:`forward_all_states`, which asserts ``lm_head(x_next)`` reproduces the head's own
    returned next-token logits; if upstream's head ever changes, that assertion fails
    rather than this function silently returning a stale state.
    """
    torch = _torch()
    x = torch.cat([fwd, bwd], dim=-1)      # model_bst.py:91
    x = x + text_head.mlp(x)               # model_bst.py:94
    x = text_head.norm(x)                  # model_bst.py:95
    x_next, x_prev = x.chunk(2, dim=-1)    # model_bst.py:100
    return x_next, x_prev


def _forward_gpt(inner_model, tokens, kwargs):
    # model_gpt.py:279-291 — targets is None, so the first return IS the logits.
    logits, hidden = inner_model(tokens, **kwargs)
    return {"hidden": hidden, "logits": logits}


def _forward_nextlat(inner_model, tokens, kwargs):
    # model_nextlat.py:199-200 early-returns (token_embeds, text_embd); the head at
    # model_nextlat.py:121 is never applied, so apply it ourselves.
    _token_embeds, hidden = inner_model(tokens, **kwargs)
    return {"hidden": hidden, "logits": inner_model.lm_head(hidden)}


def _forward_bst(model, tokens, *, eos_token_id: int, verify_texthead: bool = True):
    """BST: forward-encoder state as `hidden`, TextHead next-token logits as `logits`.

    ``model`` here is the ``BST`` wrapper itself (model_bst.py:327), NOT ``wrapper.model``
    — BST has no ``.model``, and the logit path needs both ``.encoder`` (model_bst.py:338)
    and ``.text_head`` (model_bst.py:339).  That asymmetry in the argument is the price of
    BST having two submodules where the other two arms have one; it is checked below and
    raises a named error rather than an ``AttributeError`` forty lines deep.
    """
    torch = _torch()
    encoder = getattr(model, "encoder", None)
    text_head = getattr(model, "text_head", None)
    if encoder is None or text_head is None:
        raise ValueError(
            "for architecture='bst' pass the BST wrapper itself (it has .encoder and "
            ".text_head, model_bst.py:338-339), not wrapper.model as for gpt/nextlat"
        )
    # Forward encoder only.  It is causal (model_bst.py:210-211), so fwd[62] depends on
    # tokens 0..62 exactly as the GPT/NextLat state does — feeding the full serialized
    # sequence leaks nothing into the extraction indices.
    fwd, _ = encoder(tokens, compute_forward=True, compute_backward=False)  # :245-270
    bwd_eos = bst_backward_eos_state(encoder, tokens, eos_token_id=eos_token_id)
    bwd = bwd_eos.expand(fwd.shape[0], fwd.shape[1], fwd.shape[2])
    # model_bst.py:102-110 stacks [next, prev] at dim=1, so index 0 is the next-token half.
    logits = text_head(fwd, bwd)[:, 0]
    x_next, _x_prev = bst_texthead_prelogit(text_head, fwd, bwd)
    if verify_texthead:
        check = text_head.lm_head(x_next)
        if not torch.allclose(check, logits, atol=1e-4, rtol=1e-4):
            raise RuntimeError(
                "bst_texthead_prelogit no longer reproduces TextHead's own next-token "
                "logits; upstream models/model_bst.py:91-104 must have changed"
            )
    return {"hidden": fwd, "logits": logits, "hidden_texthead": x_next}


def forward_all_states(
    model,
    tokens,
    *,
    architecture: str,
    mask=None,
    eos_token_id: int = EOS_TOKEN_ID,
    verify_texthead: bool = True,
) -> dict:
    """One forward pass -> ``{"hidden": ..., "logits": ..., ["hidden_texthead": ...]}``.

    ``model`` is the *inner* ``Transformer`` / ``NextLatTransformer`` (``wrapper.model``)
    for ``"gpt"`` and ``"nextlat"``, and the ``BST`` wrapper for ``"bst"``.  ``tokens`` is
    a ``(B, T)`` LongTensor of the FULL serialized sequence.

    This function exists solely to absorb the three-way asymmetry documented at the top of
    this module and in :data:`BST_STATE_POLICY`.  It applies no normalization, no centering
    and no scaling.  ``hidden`` is exactly the tensor named in :data:`STATE_SOURCE`, which
    is RMS-normalized in all three arms because ``bias: false``.  For BST, and only for
    BST, ``hidden_texthead`` is returned as well — the declared secondary state.
    """
    arch = _check_architecture(architecture)
    torch = _torch()
    if arch == "bst":
        if mask is not None:
            raise ValueError(
                "BST builds its own forward/backward document masks inside the encoder "
                "(model_bst.py:186-219); an external mask has nowhere to go"
            )
        with torch.no_grad():
            return _forward_bst(
                model, tokens, eos_token_id=eos_token_id, verify_texthead=verify_texthead
            )
    kwargs = {"return_hidden_states": True}
    if mask is not None:
        kwargs["mask"] = mask
    with torch.no_grad():
        if arch == "gpt":
            return _forward_gpt(model, tokens, kwargs)
        return _forward_nextlat(model, tokens, kwargs)


def forward_states_and_logits(inner_model, tokens, *, architecture: str, mask=None, **kw):
    """``(hidden_states, logits)`` for one forward pass.  See :func:`forward_all_states`.

    Kept as the two-tuple entry point because that is what the GPT/NextLat call sites
    want.  For BST it returns the PRIMARY state and drops the declared secondary; use
    :func:`forward_all_states` when you need both.
    """
    out = forward_all_states(
        inner_model, tokens, architecture=architecture, mask=mask, **kw
    )
    return out["hidden"], out["logits"]


def extract_positions(
    inner_model,
    tokens,
    *,
    architecture: str,
    positions: Sequence[int] = (PSI_EXTRACTION_INDEX, BRANCH_MARGIN_INDEX),
    batch_size: int = 256,
    device: Optional[str] = None,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict:
    """Batched extraction -> plain numpy, ready for Layer A.

    Returns ``{"positions": (P,) int, "hidden": (N, P, D) float32,
    "logits": (N, P, V) float32, "architecture": str, "state_source": str}``, plus
    ``"hidden_texthead": (N, P, D) float32`` for BST only (the declared secondary state,
    :data:`BST_STATE_ROLES`).  Nothing else crosses the torch boundary.
    """
    _check_architecture(architecture)
    torch = _torch()
    tok = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
    if tok.ndim != 2:
        raise ValueError(f"tokens must be (N, T); got {tuple(tok.shape)}")
    pos = np.asarray(positions, dtype=np.int64)
    if np.any(pos < 0) or np.any(pos >= tok.shape[1]):
        raise ValueError("extraction position outside the sequence")
    if device is not None:
        tok = tok.to(device)
    pos_t = torch.as_tensor(pos, dtype=torch.long, device=tok.device)

    was_training = getattr(inner_model, "training", False)
    if hasattr(inner_model, "eval"):
        inner_model.eval()
    parts: dict = {}
    try:
        for start in range(0, tok.shape[0], batch_size):
            chunk = tok[start : start + batch_size]
            out = forward_all_states(
                inner_model, chunk, architecture=architecture, eos_token_id=eos_token_id
            )
            for name, tensor in out.items():
                parts.setdefault(name, []).append(
                    tensor.index_select(1, pos_t).float().cpu().numpy()
                )
    finally:
        if was_training and hasattr(inner_model, "train"):
            inner_model.train()
    arch = str(architecture).lower()
    result = {
        "positions": pos,
        "architecture": arch,
        "state_source": STATE_SOURCE[arch],
    }
    for name, chunks in parts.items():
        result[name] = np.concatenate(chunks, axis=0)
    return result
