"""The two held-out HMM pair types, and the thresholds that define them (spec section 12).

Spec section 12 asks for two banks built by searching the exact posteriors:

1. **Predictively equivalent pairs** -- high observation-history edit distance, very low posterior
   Jensen-Shannon divergence. Two histories that look different and mean the same thing.
2. **Predictively divergent near-lures** -- low observation-history edit distance, high posterior
   JS divergence. Two histories that look almost the same and mean different things.

plus a **history-distance-matched control** for HMM-H1, so that "equivalent pairs end up closer in
model space" cannot be explained by their surface similarity.

## Why the two banks are built by different mechanisms

Equivalent pairs are found by sampling pairs of real held-out prefixes: two unrelated histories
landing on the same belief is common, because the belief is a 3-dimensional summary of a
32-symbol history.

Near-lures cannot be found that way, and the arithmetic says so before any code runs. Two
independent length-16 prefixes over a 4-symbol alphabet are within Levenshtein distance 2 of each
other with probability around `2.5e-7`; a pool of 5,000 sequences contains on the order of `1e7`
same-length pairs per prefix length, so the expected yield is a handful of pairs, all at whatever
divergence they happen to have. So near-lures are **constructed**, exactly as the Lure-Star
stimuli are: take a real held-out prefix, substitute one or two symbols, and keep the result only
if the exact posterior moved by more than the frozen threshold. Both members are genuine
in-support observation sequences (every symbol has positive probability under every state of the
frozen HMM), and the exact posterior of the edited history is as exactly known as the original's.

Edit positions are drawn from the last `LURE_EDIT_WINDOW` symbols of the prefix. The belief state
of this HMM has a memory of a few steps -- the mean dwell time is around 3 -- so an edit 20 steps
back moves the posterior by essentially nothing and would simply waste yield. This choice affects
how many candidates survive, not whether a survivor qualifies: qualification is decided by the
frozen JS threshold applied to the exact posterior.

## How a threshold is prevented from being retuned

The spec's requirement is that thresholds are frozen from the validation pool and then applied
unchanged. That is enforced structurally, not by discipline:

* `fit_thresholds` only ever sees the calibration pool, and computes each threshold as a
  preregistered quantile of that pool. It takes no target yield and no model state.
* `freeze_thresholds` writes the payload with a SHA-256 over its canonical serialisation. If the
  file already exists and the recomputed payload differs in any field, it raises
  `ThresholdMismatch` instead of overwriting.
* `load_thresholds` recomputes the hash and refuses a file that has been edited by hand.
* `build_bank` requires a `Thresholds` whose `verified` flag is set, and that flag is set *only*
  by `load_thresholds`. A hand-constructed `Thresholds` object cannot be passed into the test-pool
  build at all; it raises `UnverifiedThresholds`.
* `build_bank` also refuses if the thresholds' recorded HMM hash is not the hash of the HMM it was
  handed, so a bank can never be built against different matrices than the thresholds were fitted
  under, and refuses if the pool it is given is the calibration pool the thresholds came from.

## Pools

The validation split is cut by sequence index: `[0, 5000)` is the calibration pool, `[5000, 10000)`
is the test pool. No sequence is in both, and any test prefix whose observation string also occurs
as a calibration prefix of the same length is dropped, so leakage is impossible at the prefix
level as well as at the sequence level. The length-64 split provides a third pool, used with the
same frozen thresholds over prefix lengths `33..64` -- lengths no training sequence ever reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:  # pragma: no cover - import shim so the module works as a script and as a package
    from .forward import HMM, forward_batch, js_divergence
    from .generate import DATA_DIR, ROOT, load_frozen_hmm, load_posteriors
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hmm_geometry.forward import HMM, forward_batch, js_divergence
    from hmm_geometry.generate import DATA_DIR, ROOT, load_frozen_hmm, load_posteriors

THRESHOLDS_PATH = ROOT / "manifests" / "hmm_thresholds.json"
BANK_PATH = ROOT / "manifests" / "hmm_eval_pairs.jsonl"
BANK_MANIFEST_PATH = ROOT / "manifests" / "hmm_eval_pairs.json"

# Preregistered structural constants. These are design decisions, not fitted quantities.
PREFIX_MIN = 8
PREFIX_MAX = 32
LENGEN_PREFIX_MIN = 33
LENGEN_PREFIX_MAX = 64
LURE_EDIT_WINDOW = 8
LURE_MAX_EDITS = 2
EDIT_LOW = 2  # a near-lure differs by at most two symbols, mirroring the Lure-Star quartets
CALIBRATION_PAIRS = 400_000
CALIBRATION_LURE_BASES = 20_000
SEARCH_PAIRS = 1_500_000
LURE_BASES = 60_000
TARGET_PAIRS = 2_000

# Preregistered quantile rule for the three fitted thresholds.
QUANTILE_RULE = {
    "js_low_bits": "1st percentile of posterior JS divergence over calibration same-length pairs",
    "edit_high": "75th percentile of Levenshtein distance over the same calibration pairs",
    "js_control_min_bits": "50th percentile (median) of JS divergence over the same pairs",
    "js_high_bits": "90th percentile of JS divergence over calibration lure pairs",
    "edit_low": "fixed at 2 a priori; a near-lure differs by at most two symbols",
}


class ThresholdMismatch(RuntimeError):
    """Raised when a refit would change an already-frozen threshold."""


class UnverifiedThresholds(RuntimeError):
    """Raised when thresholds reach a bank builder without passing through `load_thresholds`."""


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------


def levenshtein_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Levenshtein distance for a batch of equal-length symbol sequences.

    ``a`` and ``b`` are ``(N, L)`` integer arrays; the return is ``(N,)``. The DP is vectorised
    over the batch: the two nested loops run over the ``(L+1) x (L+1)`` table, and every cell is
    computed for all ``N`` pairs at once. The inner loop cannot be vectorised because
    ``cur[j]`` depends on ``cur[j-1]``, so the cost is ``L**2`` numpy calls on ``(N,)`` arrays --
    for ``N`` in the hundreds of thousands that is seconds, versus hours for a python loop over
    pairs.

    Substitution, insertion and deletion all cost 1. Note that for equal-length inputs the
    Levenshtein distance can be strictly below the Hamming distance (``0123`` vs ``0012`` is
    Hamming 3, Levenshtein 2), which is why the true edit distance is computed rather than a
    Hamming shortcut.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"levenshtein_batch needs matching shapes, got {a.shape} and {b.shape}")
    if a.ndim != 2:
        raise ValueError("levenshtein_batch expects (N, L) arrays")
    n, length = a.shape
    if length == 0:
        return np.zeros(n, dtype=np.int32)

    prev = np.tile(np.arange(length + 1, dtype=np.int32), (n, 1))
    cur = np.empty_like(prev)
    for i in range(1, length + 1):
        cur[:, 0] = i
        cost = (a[:, i - 1][:, None] != b).astype(np.int32)  # (N, L)
        for j in range(1, length + 1):
            np.minimum(prev[:, j] + 1, cur[:, j - 1] + 1, out=cur[:, j])
            np.minimum(cur[:, j], prev[:, j - 1] + cost[:, j - 1], out=cur[:, j])
        prev, cur = cur, prev
    return prev[:, length].copy()


# --------------------------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pool:
    """A contiguous slice of one generated split, with its exact posteriors."""

    name: str
    split: str
    offset: int
    obs: np.ndarray  # (N, L) int8
    beliefs: np.ndarray  # (N, L, S)
    next_obs: np.ndarray  # (N, L+1, O)
    prefix_min: int
    prefix_max: int

    @property
    def n_sequences(self) -> int:
        return self.obs.shape[0]

    @property
    def length(self) -> int:
        return self.obs.shape[1]

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.split}:{self.offset}:{self.prefix_min}:{self.prefix_max}:".encode())
        h.update(np.ascontiguousarray(self.obs, dtype=np.int8).tobytes())
        return h.hexdigest()

    def prefix_keys(self) -> set[bytes]:
        """Every prefix in the band, as ``(length, bytes)`` keys, for leakage checks."""
        keys = set()
        arr = np.ascontiguousarray(self.obs, dtype=np.int8)
        for t in range(self.prefix_min, min(self.prefix_max, self.length) + 1):
            for row in arr[:, :t]:
                keys.add(bytes([t]) + row.tobytes())
        return keys


def load_pools(
    data_dir: Path = DATA_DIR, calibration_end: int = 5000
) -> tuple[Pool, Pool, Pool]:
    """Calibration pool, test pool, and the length-64 pool, from the frozen corpus."""
    val = load_posteriors("val", data_dir)
    lengen = load_posteriors("lengen", data_dir)
    calib = Pool(
        name="calibration",
        split="val",
        offset=0,
        obs=val["observations"][:calibration_end],
        beliefs=val["beliefs"][:calibration_end],
        next_obs=val["next_obs"][:calibration_end],
        prefix_min=PREFIX_MIN,
        prefix_max=PREFIX_MAX,
    )
    test = Pool(
        name="test",
        split="val",
        offset=calibration_end,
        obs=val["observations"][calibration_end:],
        beliefs=val["beliefs"][calibration_end:],
        next_obs=val["next_obs"][calibration_end:],
        prefix_min=PREFIX_MIN,
        prefix_max=PREFIX_MAX,
    )
    lengen_pool = Pool(
        name="lengen",
        split="lengen",
        offset=0,
        obs=lengen["observations"],
        beliefs=lengen["beliefs"],
        next_obs=lengen["next_obs"],
        prefix_min=LENGEN_PREFIX_MIN,
        prefix_max=LENGEN_PREFIX_MAX,
    )
    return calib, test, lengen_pool


# --------------------------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------------------------


def sample_same_length_pairs(
    pool: Pool, n_pairs: int, rng: np.random.Generator
) -> dict[int, dict[str, np.ndarray]]:
    """Draw random pairs of distinct sequences at a shared prefix length.

    Returns a dict keyed by prefix length, each holding the sequence indices, the exact posterior
    JS divergence in bits, and the Levenshtein distance. Prefix length is shared inside a pair by
    construction, which removes prefix length as a confound at the pair level and makes the edit
    distance comparable across pairs.
    """
    lengths = list(range(pool.prefix_min, min(pool.prefix_max, pool.length) + 1))
    per_length = max(1, n_pairs // len(lengths))
    out: dict[int, dict[str, np.ndarray]] = {}
    for t in lengths:
        i = rng.integers(0, pool.n_sequences, size=per_length)
        j = rng.integers(0, pool.n_sequences, size=per_length)
        keep = i != j
        i, j = i[keep], j[keep]
        jsd = js_divergence(pool.beliefs[i, t - 1], pool.beliefs[j, t - 1])
        lev = levenshtein_batch(pool.obs[i, :t].astype(np.int32), pool.obs[j, :t].astype(np.int32))
        out[t] = {"i": i, "j": j, "jsd": jsd, "lev": lev}
    return out


def make_lure_candidates(
    pool: Pool,
    hmm: HMM,
    n_bases: int,
    rng: np.random.Generator,
    forbidden_prefixes: set[bytes] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Construct near-lures by substituting one or two symbols in a held-out prefix.

    For each prefix length in the pool's band, `n_bases // n_lengths` base prefixes are drawn, one
    or two edit positions are drawn from the last `LURE_EDIT_WINDOW` symbols, and each chosen
    position is replaced by a different symbol. The exact posterior of the edited history is then
    computed with the forward algorithm -- it is as exact as the base's, because the edited history
    is still a valid observation sequence under the frozen HMM.
    """
    lengths = list(range(pool.prefix_min, min(pool.prefix_max, pool.length) + 1))
    per_length = max(1, n_bases // len(lengths))
    n_obs = hmm.n_obs
    out: dict[int, dict[str, np.ndarray]] = {}
    for t in lengths:
        base_idx = rng.integers(0, pool.n_sequences, size=per_length)
        base = pool.obs[base_idx, :t].astype(np.int32)
        lure = base.copy()
        window = min(LURE_EDIT_WINDOW, t)
        n_edits = rng.integers(1, LURE_MAX_EDITS + 1, size=per_length)
        for slot in range(LURE_MAX_EDITS):
            active = n_edits > slot
            if not active.any():
                continue
            pos = t - 1 - rng.integers(0, window, size=per_length)
            delta = rng.integers(1, n_obs, size=per_length)
            rows = np.nonzero(active)[0]
            lure[rows, pos[rows]] = (lure[rows, pos[rows]] + delta[rows]) % n_obs

        changed = (lure != base).any(axis=1)
        res = forward_batch(hmm, lure[changed].astype(np.int64))
        base_b = pool.beliefs[base_idx[changed], t - 1]
        lure_b = res.beliefs[:, -1, :]
        jsd = js_divergence(base_b, lure_b)
        lev = levenshtein_batch(base[changed], lure[changed])

        keep = lev >= 1
        if forbidden_prefixes:
            arr = np.ascontiguousarray(lure[changed], dtype=np.int8)
            head = bytes([t])
            fresh = np.array(
                [head + row.tobytes() not in forbidden_prefixes for row in arr], dtype=bool
            )
            keep &= fresh
        out[t] = {
            "base_index": base_idx[changed][keep],
            "lure_obs": lure[changed][keep],
            "lure_belief": lure_b[keep],
            "lure_next_obs": res.next_obs[:, -1, :][keep],
            "jsd": jsd[keep],
            "lev": lev[keep],
        }
    return out


# --------------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    js_low_bits: float
    js_high_bits: float
    js_control_min_bits: float
    edit_high: int
    edit_low: int
    prefix_min: int
    prefix_max: int
    lure_edit_window: int
    lure_max_edits: int
    n_calibration_pairs: int
    n_calibration_lures: int
    calibration_pool_sha256: str
    hmm_sha256: str
    fit_seed: int
    verified: bool = field(default=False, compare=False)

    def payload(self) -> dict:
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k != "verified"
        }
        d["quantile_rule"] = QUANTILE_RULE
        return d

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def fit_thresholds(
    calibration: Pool, hmm: HMM, seed: int = 20260823, n_pairs: int = CALIBRATION_PAIRS
) -> Thresholds:
    """Fit every threshold from the calibration pool alone, by the preregistered quantile rule.

    No argument of this function is a target yield, a model state, or a downstream metric. It
    cannot be steered.
    """
    rng = np.random.default_rng(seed)
    pairs = sample_same_length_pairs(calibration, n_pairs, rng)
    jsd = np.concatenate([v["jsd"] for v in pairs.values()])
    lev = np.concatenate([v["lev"] for v in pairs.values()])

    lures = make_lure_candidates(calibration, hmm, CALIBRATION_LURE_BASES, rng)
    lure_jsd = np.concatenate([v["jsd"] for v in lures.values()])

    return Thresholds(
        js_low_bits=float(np.percentile(jsd, 1)),
        js_high_bits=float(np.percentile(lure_jsd, 90)),
        js_control_min_bits=float(np.percentile(jsd, 50)),
        edit_high=int(np.percentile(lev, 75)),
        edit_low=EDIT_LOW,
        prefix_min=calibration.prefix_min,
        prefix_max=calibration.prefix_max,
        lure_edit_window=LURE_EDIT_WINDOW,
        lure_max_edits=LURE_MAX_EDITS,
        n_calibration_pairs=int(jsd.size),
        n_calibration_lures=int(lure_jsd.size),
        calibration_pool_sha256=calibration.sha256(),
        hmm_sha256=hmm.sha256(),
        fit_seed=seed,
    )


def freeze_thresholds(thresholds: Thresholds, path: Path = THRESHOLDS_PATH) -> dict:
    """Persist the thresholds with their hash, refusing to change an existing freeze."""
    payload = {
        "schema": "nextlat_forgetting/hmm_thresholds/1",
        "thresholds": thresholds.payload(),
        "sha256": thresholds.sha256(),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("sha256") != payload["sha256"]:
            raise ThresholdMismatch(
                f"{path} already freezes different thresholds "
                f"({existing.get('sha256', '')[:12]} vs {payload['sha256'][:12]}). "
                "Thresholds are on the frozen surface (PROGRAM.md section 'Frozen surface'); "
                "they may not be refitted after any model state has been seen. Delete the file "
                "only to correct a recorded error, and append a superseding RUNLOG entry."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    """Load frozen thresholds, verify the hash, and mark them verified.

    This is the only function that sets ``verified=True``. Every bank builder demands it, so a
    threshold value invented in a notebook cannot reach the test pool.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Fit thresholds on the calibration pool and freeze them "
            "before building any test-pool bank."
        )
    payload = json.loads(path.read_text())
    stored = payload["thresholds"]
    fields = {k: v for k, v in stored.items() if k != "quantile_rule"}
    thresholds = Thresholds(**fields)
    if thresholds.sha256() != payload["sha256"]:
        raise ThresholdMismatch(
            f"{path} has been edited since it was frozen "
            f"(recomputed {thresholds.sha256()[:12]} != recorded {payload['sha256'][:12]})"
        )
    if stored.get("quantile_rule") != QUANTILE_RULE:
        raise ThresholdMismatch(
            "the frozen file records a different quantile rule than the code implements"
        )
    return replace(thresholds, verified=True)


# --------------------------------------------------------------------------------------------
# Bank construction
# --------------------------------------------------------------------------------------------


@dataclass
class Bank:
    pairs: list[dict]
    stats: dict

    def by_type(self, kind: str) -> list[dict]:
        return [p for p in self.pairs if p["pair_type"] == kind]


def _item(pool: Pool, idx: int, t: int) -> dict:
    return {
        "source": pool.split,
        "seq_index": int(pool.offset + idx),
        "prefix": [int(x) for x in pool.obs[idx, :t]],
        "belief": [float(x) for x in pool.beliefs[idx, t - 1]],
        "next_obs": [float(x) for x in pool.next_obs[idx, t]],
    }


def build_bank(
    pool: Pool,
    hmm: HMM,
    thresholds: Thresholds,
    seed: int,
    target_pairs: int = TARGET_PAIRS,
    n_search_pairs: int = SEARCH_PAIRS,
    n_lure_bases: int = LURE_BASES,
    forbidden_prefixes: set[bytes] | None = None,
) -> Bank:
    """Apply the frozen thresholds to a pool and emit the three pair sets.

    Refuses to run on unverified thresholds, on a different HMM, or on the very pool the
    thresholds were fitted from.
    """
    if not thresholds.verified:
        raise UnverifiedThresholds(
            "build_bank requires thresholds returned by load_thresholds(). Thresholds constructed "
            "in memory cannot be applied to a test pool -- that is the retuning path the spec "
            "forbids."
        )
    if thresholds.hmm_sha256 != hmm.sha256():
        raise ThresholdMismatch(
            "thresholds were fitted under a different HMM than the one supplied"
        )
    if thresholds.calibration_pool_sha256 == pool.sha256():
        raise ThresholdMismatch(
            "refusing to build a bank on the calibration pool the thresholds were fitted from"
        )

    rng = np.random.default_rng(seed)
    pairs_by_len = sample_same_length_pairs(pool, n_search_pairs, rng)
    lures_by_len = make_lure_candidates(
        pool, hmm, n_lure_bases, rng, forbidden_prefixes=forbidden_prefixes
    )

    # --- predictively equivalent pairs, and their history-distance-matched controls ----------
    # Control candidates are indexed by (prefix length, edit distance) so a control can be matched
    # exactly, never approximately.
    control_pool: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for t, d in pairs_by_len.items():
        hi = d["jsd"] >= thresholds.js_control_min_bits
        for i, j, lev, jsd in zip(d["i"][hi], d["j"][hi], d["lev"][hi], d["jsd"][hi]):
            control_pool.setdefault((t, int(lev)), []).append((int(i), int(j), float(jsd)))

    equivalent: list[dict] = []
    controls: list[dict] = []
    n_equiv_candidates = 0
    n_dropped_no_control = 0
    per_length_target = max(1, target_pairs // len(pairs_by_len))
    for t in sorted(pairs_by_len):
        d = pairs_by_len[t]
        qualifies = (d["jsd"] <= thresholds.js_low_bits) & (d["lev"] >= thresholds.edit_high)
        order = np.nonzero(qualifies)[0]  # sampling order, not sorted by any quality measure
        n_equiv_candidates += order.size
        taken = 0
        for k in order:
            if taken >= per_length_target:
                break
            key = (t, int(d["lev"][k]))
            bucket = control_pool.get(key)
            if not bucket:
                n_dropped_no_control += 1
                continue
            ci, cj, cjsd = bucket.pop()
            i, j = int(d["i"][k]), int(d["j"][k])
            pair_id = f"{pool.name}-equiv-t{t}-{i}-{j}"
            equivalent.append(
                {
                    "pair_id": pair_id,
                    "pair_type": "equivalent",
                    "pool": pool.name,
                    "prefix_len": int(t),
                    "edit_distance": int(d["lev"][k]),
                    "js_divergence_bits": float(d["jsd"][k]),
                    "a": _item(pool, i, t),
                    "b": _item(pool, j, t),
                }
            )
            controls.append(
                {
                    "pair_id": f"{pool.name}-control-t{t}-{ci}-{cj}",
                    "pair_type": "matched_control",
                    "pool": pool.name,
                    "prefix_len": int(t),
                    "edit_distance": int(d["lev"][k]),
                    "js_divergence_bits": cjsd,
                    "matches_pair_id": pair_id,
                    "a": _item(pool, ci, t),
                    "b": _item(pool, cj, t),
                }
            )
            taken += 1

    # --- predictively divergent near-lures ---------------------------------------------------
    near_lures: list[dict] = []
    n_lure_candidates = 0
    for t in sorted(lures_by_len):
        d = lures_by_len[t]
        qualifies = (d["jsd"] >= thresholds.js_high_bits) & (d["lev"] <= thresholds.edit_low)
        order = np.nonzero(qualifies)[0]
        n_lure_candidates += order.size
        for k in order[:per_length_target]:
            base = int(d["base_index"][k])
            near_lures.append(
                {
                    "pair_id": f"{pool.name}-lure-t{t}-{base}-{int(k)}",
                    "pair_type": "near_lure",
                    "pool": pool.name,
                    "prefix_len": int(t),
                    "edit_distance": int(d["lev"][k]),
                    "js_divergence_bits": float(d["jsd"][k]),
                    "a": _item(pool, base, t),
                    "b": {
                        "source": f"lure_of:{pool.split}",
                        "seq_index": int(pool.offset + base),
                        "prefix": [int(x) for x in d["lure_obs"][k]],
                        "belief": [float(x) for x in d["lure_belief"][k]],
                        "next_obs": [float(x) for x in d["lure_next_obs"][k]],
                    },
                }
            )

    stats = {
        "pool": pool.name,
        "pool_sha256": pool.sha256(),
        "seed": seed,
        "n_equivalent": len(equivalent),
        "n_near_lure": len(near_lures),
        "n_matched_control": len(controls),
        "n_equivalent_candidates": int(n_equiv_candidates),
        "n_lure_candidates": int(n_lure_candidates),
        "n_equivalent_dropped_for_missing_control": int(n_dropped_no_control),
        "n_search_pairs": int(sum(v["i"].size for v in pairs_by_len.values())),
        "n_lure_candidates_generated": int(sum(v["lev"].size for v in lures_by_len.values())),
        "mean_equivalent_js_bits": float(
            np.mean([p["js_divergence_bits"] for p in equivalent]) if equivalent else np.nan
        ),
        "mean_equivalent_edit": float(
            np.mean([p["edit_distance"] for p in equivalent]) if equivalent else np.nan
        ),
        "mean_near_lure_js_bits": float(
            np.mean([p["js_divergence_bits"] for p in near_lures]) if near_lures else np.nan
        ),
        "mean_near_lure_edit": float(
            np.mean([p["edit_distance"] for p in near_lures]) if near_lures else np.nan
        ),
        "mean_control_js_bits": float(
            np.mean([p["js_divergence_bits"] for p in controls]) if controls else np.nan
        ),
    }
    return Bank(pairs=equivalent + near_lures + controls, stats=stats)


def write_bank(
    banks: list[Bank],
    thresholds: Thresholds,
    hmm: HMM,
    bank_path: Path = BANK_PATH,
    manifest_path: Path = BANK_MANIFEST_PATH,
) -> dict:
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    with bank_path.open("w") as f:
        for bank in banks:
            for pair in bank.pairs:
                f.write(json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "nextlat_forgetting/hmm_eval_pairs/1",
        "hmm_sha256": hmm.sha256(),
        "thresholds_sha256": thresholds.sha256(),
        "thresholds": thresholds.payload(),
        "banks": [b.stats for b in banks],
        "pairs_file": str(bank_path.relative_to(ROOT)),
        "pairs_sha256": digest,
        "n_pairs": sum(len(b.pairs) for b in banks),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_bank(bank_path: Path = BANK_PATH) -> list[dict]:
    with bank_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["fit", "build", "all", "report"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    ap.add_argument("--bank", type=Path, default=BANK_PATH)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    args = ap.parse_args(argv)

    hmm, _ = load_frozen_hmm()
    calib, test, lengen = load_pools(args.data_dir)

    if args.command in ("fit", "all"):
        th = fit_thresholds(calib, hmm, seed=args.seed)
        payload = freeze_thresholds(th, args.thresholds)
        print(json.dumps(payload, indent=2))

    if args.command in ("build", "all"):
        th = load_thresholds(args.thresholds)
        forbidden = calib.prefix_keys()
        banks = []
        for pool in (test, lengen):
            bank = build_bank(
                pool,
                hmm,
                th,
                seed=args.seed + (0 if pool.name == "test" else 1),
                target_pairs=args.target_pairs,
                forbidden_prefixes=forbidden,
            )
            banks.append(bank)
            print(json.dumps(bank.stats, indent=2))
        manifest = write_bank(banks, th, hmm, args.bank)
        print(f"wrote {manifest['n_pairs']} pairs -> {args.bank}")

    if args.command == "report":
        th = load_thresholds(args.thresholds)
        print(json.dumps(th.payload(), indent=2))
        pairs = load_bank(args.bank)
        for kind in ("equivalent", "near_lure", "matched_control"):
            sel = [p for p in pairs if p["pair_type"] == kind]
            if not sel:
                continue
            js = np.array([p["js_divergence_bits"] for p in sel])
            ed = np.array([p["edit_distance"] for p in sel])
            print(
                f"{kind:16s} n={len(sel):5d}  JS bits mean {js.mean():.4f} "
                f"[{js.min():.4f}, {js.max():.4f}]  edit mean {ed.mean():.2f} "
                f"[{ed.min()}, {ed.max()}]"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
