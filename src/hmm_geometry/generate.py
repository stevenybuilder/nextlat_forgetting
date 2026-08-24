"""Choose, freeze, and sample from the preregistered 4-state / 4-observation HMM (spec section 12).

The spec asks for *one* HMM chosen in a **model-blind generator test**. Model-blind here has a
precise operational meaning: every quantity used to accept or reject a candidate is a property of
the generative process and its exact posteriors. No trained network, no hidden state, and no
downstream geometry metric is involved, and the matrices are written to
`manifests/hmm_matrices.json` before any model exists. Once that file is on disk this module
refuses to change it.

## The four acceptance criteria, stated before the search

Spec section 12 lists four properties. Each is turned into an interval, and the intervals are
fixed in `ACCEPTANCE` below rather than being adjusted to whatever the grid happened to produce:

1. *State persistence neither trivial nor nearly random.* A uniform-random 4-state chain has a
   stationary-weighted mean dwell time of 4/3. "Trivial" persistence would be a chain that sits
   in one state for most of a length-32 sequence. Accept mean dwell in `[1.8, 4.0]`, with every
   individual self-transition in `[0.35, 0.80]`.
2. *Every observation has nonzero probability under at least two hidden states.* Taken in its
   useful sense, not its vacuous one: a `1e-9` leak satisfies the literal wording while producing
   an effectively identifiable state. Accept only if at least two states put probability `>= 0.10`
   on each symbol.
3. *Posterior entropy spans a broad range.* The available range is `[0, 2]` bits. Accept if the
   5th percentile is below `0.5` bits (some prefixes nearly resolve the state), the 95th
   percentile is above `1.3` bits (some prefixes leave it genuinely ambiguous), and the two are at
   least `0.9` bits apart.
4. *Next-observation accuracy meaningfully above chance but below determinism.* Chance is the best
   constant predictor -- the frequency of the most common symbol -- not `1/4`, because a skewed
   marginal would otherwise be scored as skill. Accept if the Bayes-optimal accuracy exceeds that
   baseline by `>= 0.10` and stays in `[0.35, 0.80]`.

Two structural guards are added so the chosen process is not degenerate in a way the four criteria
do not cover: no hidden state may carry stationary mass below `0.15`, and no observation symbol may
have stationary marginal below `0.10`.

## Selection among the passers

Passing candidates are scored by their *worst* normalised slack: how far the tightest diagnostic
sits from the edge of its acceptance interval, in units of half the interval width. The maximiser
is the candidate least likely to fall out of the box under resampling. Ties break on the
lexicographic order of the parameter tuple, so the choice is a deterministic function of the grid
and the acceptance intervals alone.

## Splits

`numpy.random.SeedSequence(DATA_SEED).spawn(3)` gives the three splits independent streams, so the
validation corpus does not shift if the training corpus size ever changes. Exact posteriors and
exact next-observation distributions are stored for the two evaluation splits; the training split's
posteriors are recomputable exactly from the frozen matrices and are not written, because at
100,000 x 32 x 4 they are a 100 MB artifact with no evaluation role.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

try:  # pragma: no cover - import shim so the module works as a script and as a package
    from .forward import HMM, forward_batch, sample_sequences
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hmm_geometry.forward import HMM, forward_batch, sample_sequences

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "manifests" / "hmm_matrices.json"
DATASET_MANIFEST_PATH = ROOT / "manifests" / "hmm_dataset.json"
DATA_DIR = ROOT / "data" / "hmm"

DATA_SEED = 5963  # arXiv:2511.05963, recorded so the corpus is reproducible from the manifest
PILOT_SEED = 11_05963
PILOT_SEQUENCES = 4000
PILOT_LENGTH = 32

TRAIN_SEQUENCES = 100_000
VAL_SEQUENCES = 10_000
LENGEN_SEQUENCES = 10_000
SEQUENCE_LENGTH = 32
LENGEN_LENGTH = 64

N_STATES = 4
N_OBS = 4

# (low, high) inclusive acceptance intervals. Frozen before the search; see the module docstring.
ACCEPTANCE: dict[str, tuple[float, float]] = {
    "mean_dwell_time": (1.8, 4.0),
    "min_self_transition": (0.35, 0.80),
    "max_self_transition": (0.35, 0.80),
    "min_states_per_obs_at_0.10": (2.0, 4.0),
    "belief_entropy_p05_bits": (0.0, 0.5),
    "belief_entropy_p95_bits": (1.3, 2.0),
    "belief_entropy_spread_bits": (0.9, 2.0),
    "bayes_next_obs_accuracy": (0.35, 0.80),
    "bayes_minus_chance": (0.10, 0.60),
    "min_stationary_state": (0.15, 0.40),
    "min_stationary_obs": (0.10, 0.40),
}

# The preregistered candidate grid.
GRID_SELF_TRANSITION = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
GRID_PERSISTENCE_TILT = (0.00, 0.04, 0.08)
GRID_OFFDIAG_SKEW = (
    (1 / 3, 1 / 3, 1 / 3),
    (0.50, 0.30, 0.20),
    (0.60, 0.25, 0.15),
    (0.70, 0.20, 0.10),
)
GRID_EMISSION_SELF = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
GRID_EMISSION_NEIGHBOUR = (0.10, 0.15, 0.20, 0.25)


@dataclass(frozen=True)
class Candidate:
    """One point of the preregistered grid, plus the HMM it induces."""

    self_transition: float
    persistence_tilt: float
    offdiag_skew: tuple[float, float, float]
    emission_self: float
    emission_neighbour: float

    def key(self) -> tuple:
        return (
            self.self_transition,
            self.persistence_tilt,
            self.offdiag_skew,
            self.emission_self,
            self.emission_neighbour,
        )

    def to_dict(self) -> dict:
        return {
            "self_transition": self.self_transition,
            "persistence_tilt": self.persistence_tilt,
            "offdiag_skew": list(self.offdiag_skew),
            "emission_self": self.emission_self,
            "emission_neighbour": self.emission_neighbour,
        }

    def build(self) -> HMM:
        """Materialise the transition and emission matrices.

        Transition: state `s` keeps probability `p_s = self_transition + tilt * (s - 1.5) / 1.5`,
        so persistence varies across states rather than being a single shared constant, and the
        remaining mass is split over the three cyclic successors by `offdiag_skew`.

        Emission: state `s` puts `emission_self` on symbol `s`, `emission_neighbour` on symbol
        `(s+1) % 4`, and splits the rest equally over the other two symbols. Every symbol is
        therefore emitted by all four states with strictly positive probability, and by at least
        two states with substantial probability -- overlapping emissions in the sense the spec
        asks for.
        """
        s = N_STATES
        transition = np.zeros((s, s))
        for i in range(s):
            p = self.self_transition + self.persistence_tilt * (i - 1.5) / 1.5
            transition[i, i] = p
            for k, w in enumerate(self.offdiag_skew, start=1):
                transition[i, (i + k) % s] = (1.0 - p) * w

        emission = np.zeros((s, N_OBS))
        rest = (1.0 - self.emission_self - self.emission_neighbour) / (N_OBS - 2)
        if rest <= 0:
            raise ValueError("emission parameters leave no mass for the remaining symbols")
        for i in range(s):
            for o in range(N_OBS):
                if o == i:
                    emission[i, o] = self.emission_self
                elif o == (i + 1) % N_OBS:
                    emission[i, o] = self.emission_neighbour
                else:
                    emission[i, o] = rest

        transition = transition / transition.sum(axis=1, keepdims=True)
        emission = emission / emission.sum(axis=1, keepdims=True)
        stationary = HMM(transition, emission, np.full(s, 1.0 / s)).stationary()
        # A stationary HMM: S_1 is drawn from the chain's stationary distribution, so every
        # prefix length is a sample from the same process rather than from a transient.
        return HMM(transition=transition, emission=emission, initial=stationary)


@dataclass
class Diagnostics:
    values: dict[str, float]
    entropy_histogram: dict[str, list] = field(default_factory=dict)
    chance_accuracy: float = 0.0
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    slack: float = -np.inf


def _grid() -> Iterator[Candidate]:
    for p, tilt, skew, e_self, e_nb in itertools.product(
        GRID_SELF_TRANSITION,
        GRID_PERSISTENCE_TILT,
        GRID_OFFDIAG_SKEW,
        GRID_EMISSION_SELF,
        GRID_EMISSION_NEIGHBOUR,
    ):
        if e_self + e_nb >= 0.95:
            continue  # would starve the two remaining symbols
        if e_nb >= e_self:
            continue  # the "own" symbol must be the dominant one, else the labelling is a lie
        yield Candidate(p, tilt, skew, e_self, e_nb)


def diagnose(hmm: HMM, *, seed: int = PILOT_SEED, n_pilot: int = PILOT_SEQUENCES) -> Diagnostics:
    """Compute every acceptance diagnostic for one HMM. Model-blind by construction."""
    stationary = hmm.stationary()
    obs_marginal = hmm.obs_marginal()
    self_trans = np.diag(hmm.transition)
    states_per_obs = (hmm.emission >= 0.10).sum(axis=0)

    rng = np.random.default_rng(seed)
    obs, _ = sample_sequences(hmm, n_pilot, PILOT_LENGTH, rng)
    res = forward_batch(hmm, obs.astype(np.int64))

    ent = res.belief_entropy(base=2.0).ravel()
    p05, p25, p50, p75, p95 = np.percentile(ent, [5, 25, 50, 75, 95])

    # Realised Bayes-optimal next-observation accuracy: argmax of the exact predictive
    # distribution against the symbol that actually occurred, over every prefix including the
    # empty one.
    pred = res.next_obs[:, :-1, :]
    bayes_acc = float((pred.argmax(axis=-1) == obs).mean())
    # Chance is the best constant predictor available to a model with no memory at all.
    counts = np.bincount(obs.ravel(), minlength=N_OBS)
    chance = float(counts.max() / counts.sum())

    values = {
        "mean_dwell_time": hmm.mean_dwell_time(),
        "min_self_transition": float(self_trans.min()),
        "max_self_transition": float(self_trans.max()),
        "min_states_per_obs_at_0.10": float(states_per_obs.min()),
        "belief_entropy_p05_bits": float(p05),
        "belief_entropy_p25_bits": float(p25),
        "belief_entropy_p50_bits": float(p50),
        "belief_entropy_p75_bits": float(p75),
        "belief_entropy_p95_bits": float(p95),
        "belief_entropy_mean_bits": float(ent.mean()),
        "belief_entropy_spread_bits": float(p95 - p05),
        "bayes_next_obs_accuracy": bayes_acc,
        "chance_next_obs_accuracy": chance,
        "bayes_minus_chance": bayes_acc - chance,
        "uniform_chance_accuracy": 1.0 / N_OBS,
        "min_stationary_state": float(stationary.min()),
        "min_stationary_obs": float(obs_marginal.min()),
        "min_emission_probability": float(hmm.emission.min()),
        "mean_conditional_nll_nats": float(-res.cond_logp.mean()),
    }

    failures = []
    slacks = []
    for name, (lo, hi) in ACCEPTANCE.items():
        v = values[name]
        if not (lo - 1e-12 <= v <= hi + 1e-12):
            failures.append(f"{name}={v:.4f} outside [{lo}, {hi}]")
        half = (hi - lo) / 2.0
        centre = (hi + lo) / 2.0
        slacks.append((half - abs(v - centre)) / half if half > 0 else 0.0)

    counts_hist, edges = np.histogram(ent, bins=20, range=(0.0, 2.0))
    return Diagnostics(
        values=values,
        entropy_histogram={"bin_edges_bits": edges.tolist(), "counts": counts_hist.tolist()},
        chance_accuracy=chance,
        passed=not failures,
        failures=failures,
        slack=float(min(slacks)) if not failures else -np.inf,
    )


def search(n_pilot: int = PILOT_SEQUENCES, verbose: bool = True) -> tuple[Candidate, Diagnostics, dict]:
    """Run the full model-blind generator test over the preregistered grid."""
    t0 = time.time()
    candidates = list(_grid())
    best: tuple[float, tuple, Candidate, Diagnostics] | None = None
    n_pass = 0
    failure_counts: dict[str, int] = {}

    for cand in candidates:
        hmm = cand.build()
        # Cheap analytic guards first; the pilot forward pass is the expensive part.
        if not (
            ACCEPTANCE["min_self_transition"][0]
            <= np.diag(hmm.transition).min()
            <= ACCEPTANCE["max_self_transition"][1]
        ):
            failure_counts["self_transition"] = failure_counts.get("self_transition", 0) + 1
            continue
        dwell = hmm.mean_dwell_time()
        lo, hi = ACCEPTANCE["mean_dwell_time"]
        if not (lo <= dwell <= hi):
            failure_counts["mean_dwell_time"] = failure_counts.get("mean_dwell_time", 0) + 1
            continue

        diag = diagnose(hmm, n_pilot=n_pilot)
        if not diag.passed:
            for f in diag.failures:
                k = f.split("=")[0]
                failure_counts[k] = failure_counts.get(k, 0) + 1
            continue
        n_pass += 1
        rank = (diag.slack, tuple(-x for x in _key_as_floats(cand)))
        if best is None or rank > (best[0], best[1]):
            best = (diag.slack, tuple(-x for x in _key_as_floats(cand)), cand, diag)

    if best is None:
        raise RuntimeError(
            "no candidate in the preregistered grid satisfied the acceptance criteria; "
            f"failure counts: {failure_counts}"
        )

    provenance = {
        "grid_size": len(candidates),
        "n_passing": n_pass,
        "failure_counts": failure_counts,
        "acceptance": {k: list(v) for k, v in ACCEPTANCE.items()},
        "selection_rule": (
            "maximise the minimum normalised slack across all acceptance diagnostics; "
            "ties broken by lexicographic order of the parameter tuple"
        ),
        "pilot_sequences": n_pilot,
        "pilot_length": PILOT_LENGTH,
        "pilot_seed": PILOT_SEED,
        "search_seconds": round(time.time() - t0, 2),
    }
    if verbose:
        print(
            f"[search] {len(candidates)} candidates, {n_pass} passed, "
            f"{provenance['search_seconds']}s"
        )
    return best[2], best[3], provenance


def _key_as_floats(c: Candidate) -> tuple[float, ...]:
    return (
        c.self_transition,
        c.persistence_tilt,
        *c.offdiag_skew,
        c.emission_self,
        c.emission_neighbour,
    )


# --------------------------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------------------------


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def freeze_matrices(
    candidate: Candidate,
    diag: Diagnostics,
    provenance: dict,
    path: Path = MANIFEST_PATH,
    force: bool = False,
) -> dict:
    """Write `manifests/hmm_matrices.json`, refusing to silently change an existing freeze."""
    hmm = candidate.build()
    payload = {
        "schema": "nextlat_forgetting/hmm_matrices/1",
        "n_states": N_STATES,
        "n_obs": N_OBS,
        "hmm": hmm.to_dict(),
        "hmm_sha256": hmm.sha256(),
        "candidate": candidate.to_dict(),
        "diagnostics": diag.values,
        "entropy_histogram": diag.entropy_histogram,
        "search": provenance,
        "generated_at_unix": int(time.time()),
    }
    body = {k: v for k, v in payload.items() if k != "generated_at_unix"}
    payload["payload_sha256"] = hashlib.sha256(_canonical(body).encode()).hexdigest()

    if path.exists():
        existing = json.loads(path.read_text())
        if existing["hmm_sha256"] != payload["hmm_sha256"]:
            if not force:
                raise RuntimeError(
                    f"{path} already freezes a different HMM "
                    f"({existing['hmm_sha256'][:12]} vs {payload['hmm_sha256'][:12]}). "
                    "The matrices are on the frozen surface (PROGRAM.md); refusing to overwrite. "
                    "Pass force=True only to correct a recorded error, and append a superseding "
                    "entry to docs/RUNLOG.md."
                )
        else:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_frozen_hmm(path: Path = MANIFEST_PATH) -> tuple[HMM, dict]:
    """Load the frozen HMM and verify both hashes before returning it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python src/hmm_geometry/generate.py freeze` first; "
            "nothing downstream may invent its own matrices."
        )
    payload = json.loads(path.read_text())
    body = {k: v for k, v in payload.items() if k not in ("generated_at_unix", "payload_sha256")}
    digest = hashlib.sha256(_canonical(body).encode()).hexdigest()
    if digest != payload["payload_sha256"]:
        raise RuntimeError(
            f"{path} has been edited since it was frozen "
            f"(payload sha256 {digest[:12]} != recorded {payload['payload_sha256'][:12]})"
        )
    hmm = HMM.from_dict(payload["hmm"])
    if hmm.sha256() != payload["hmm_sha256"]:
        raise RuntimeError("frozen matrices do not match their recorded hash")
    return hmm, payload


# --------------------------------------------------------------------------------------------
# Corpus generation
# --------------------------------------------------------------------------------------------


SPLITS = (
    ("train", TRAIN_SEQUENCES, SEQUENCE_LENGTH, 0),
    ("val", VAL_SEQUENCES, SEQUENCE_LENGTH, 1),
    ("lengen", LENGEN_SEQUENCES, LENGEN_LENGTH, 2),
)


def split_path(name: str, n: int, length: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"hmm4x4_{name}_len{length}_{n}.npy"


def posterior_path(name: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"hmm4x4_{name}_posteriors.npz"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_corpus(
    hmm: HMM,
    data_dir: Path = DATA_DIR,
    manifest_path: Path = DATASET_MANIFEST_PATH,
    hmm_manifest: dict | None = None,
    store_posteriors_for: tuple[str, ...] = ("val", "lengen"),
    verbose: bool = True,
) -> dict:
    """Sample the three splits, store exact posteriors for the evaluation splits, hash everything."""
    data_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(DATA_SEED).spawn(len(SPLITS))
    entry: dict[str, dict] = {}

    for (name, n, length, idx), seq in zip(SPLITS, seeds):
        t0 = time.time()
        rng = np.random.default_rng(seq)
        obs, states = sample_sequences(hmm, n, length, rng)
        p_obs = split_path(name, n, length, data_dir)
        np.save(p_obs, obs)

        rec = {
            "n_sequences": int(n),
            "length": int(length),
            "spawn_index": int(idx),
            "observations_file": str(p_obs.relative_to(ROOT)),
            "observations_sha256": _sha256_file(p_obs),
            "observations_dtype": str(obs.dtype),
        }

        if name in store_posteriors_for:
            res = forward_batch(hmm, obs.astype(np.int64))
            p_post = posterior_path(name, data_dir)
            np.savez(
                p_post,
                observations=obs,
                hidden_states=states,
                beliefs=res.beliefs,
                predictive=res.predictive,
                next_obs=res.next_obs,
                cond_logp=res.cond_logp,
            )
            rec.update(
                {
                    "posteriors_file": str(p_post.relative_to(ROOT)),
                    "posteriors_sha256": _sha256_file(p_post),
                    "mean_belief_entropy_bits": float(res.belief_entropy().mean()),
                    "bayes_next_obs_accuracy": float(
                        (res.next_obs[:, :-1, :].argmax(-1) == obs).mean()
                    ),
                    "mean_conditional_nll_nats": float(-res.cond_logp.mean()),
                }
            )
        rec["seconds"] = round(time.time() - t0, 2)
        entry[name] = rec
        if verbose:
            print(f"[corpus] {name}: {n} x {length} -> {p_obs.name} ({rec['seconds']}s)")

    manifest = {
        "schema": "nextlat_forgetting/hmm_dataset/1",
        "hmm_sha256": hmm.sha256(),
        "hmm_manifest_sha256": (hmm_manifest or {}).get("payload_sha256"),
        "data_seed": DATA_SEED,
        "seed_scheme": "numpy.random.SeedSequence(DATA_SEED).spawn(3) -> train, val, lengen",
        "splits": entry,
        "pair_bank_split_rule": (
            "both evaluation splits are cut in half by sequence index. val[0:5000] and "
            "lengen[0:5000] are the pair-bank calibration pools that thresholds are frozen from; "
            "val[5000:10000] and lengen[5000:10000] are the test pools those frozen thresholds "
            "are applied to, unchanged. The length-64 band is calibrated from its own half "
            "because the high-edit-distance cut is a per-length quantile and no length in "
            "33..64 appears in the length-32 band."
        ),
        "train_posteriors": (
            "not stored; recomputable exactly with forward_batch(load_frozen_hmm()[0], obs)"
        ),
        "generated_at_unix": int(time.time()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    for split_name, n, length, _ in SPLITS:
        if split_name == name:
            return np.load(split_path(name, n, length, data_dir))
    raise KeyError(name)


def load_posteriors(name: str, data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    with np.load(posterior_path(name, data_dir)) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _report(diag: Diagnostics, hmm: HMM) -> str:
    v = diag.values
    lines = [
        "Chosen HMM (4 states, 4 observations, stationary initial distribution)",
        "  transition = " + np.array2string(hmm.transition, precision=4, suppress_small=True),
        "  emission   = " + np.array2string(hmm.emission, precision=4, suppress_small=True),
        "  stationary = " + np.array2string(hmm.stationary(), precision=4),
        "  obs marginal = " + np.array2string(hmm.obs_marginal(), precision=4),
        "",
        "Model-blind diagnostics",
        f"  mean state dwell time            {v['mean_dwell_time']:.3f}   "
        f"(random 4-state chain = 1.333)",
        f"  self-transitions                 [{v['min_self_transition']:.3f}, "
        f"{v['max_self_transition']:.3f}]",
        f"  states with P(o|s) >= 0.10       min {int(v['min_states_per_obs_at_0.10'])} per symbol",
        f"  min emission probability         {v['min_emission_probability']:.4f}",
        f"  posterior entropy (bits)         p05 {v['belief_entropy_p05_bits']:.3f}  "
        f"p25 {v['belief_entropy_p25_bits']:.3f}  p50 {v['belief_entropy_p50_bits']:.3f}  "
        f"p75 {v['belief_entropy_p75_bits']:.3f}  p95 {v['belief_entropy_p95_bits']:.3f}",
        f"  posterior entropy mean           {v['belief_entropy_mean_bits']:.3f} bits of 2.000",
        f"  Bayes next-obs accuracy          {v['bayes_next_obs_accuracy']:.4f}",
        f"  best-constant-predictor chance   {v['chance_next_obs_accuracy']:.4f}",
        f"  uniform chance                   {v['uniform_chance_accuracy']:.4f}",
        f"  Bayes - chance                   {v['bayes_minus_chance']:.4f}",
        f"  mean conditional NLL             {v['mean_conditional_nll_nats']:.4f} nats",
        f"  min stationary state / obs mass  {v['min_stationary_state']:.4f} / "
        f"{v['min_stationary_obs']:.4f}",
        f"  worst normalised slack           {diag.slack:.4f}",
        "",
        "Posterior entropy histogram (bits, 20 bins over [0, 2])",
    ]
    counts = diag.entropy_histogram["counts"]
    edges = diag.entropy_histogram["bin_edges_bits"]
    peak = max(counts) or 1
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * c / peak))
        lines.append(f"  [{lo:4.2f},{hi:4.2f})  {c:7d}  {bar}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "command",
        choices=["search", "freeze", "corpus", "all", "report"],
        help=(
            "search: run the generator test and print the winner without writing anything. "
            "freeze: search and write manifests/hmm_matrices.json (refuses to change an existing "
            "freeze). corpus: sample the three splits from the frozen HMM. all: freeze then "
            "corpus. report: print the diagnostics of the already-frozen HMM."
        ),
    )
    ap.add_argument("--pilot-sequences", type=int, default=PILOT_SEQUENCES)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument(
        "--force-refreeze",
        action="store_true",
        help="overwrite an existing, different frozen HMM (requires a RUNLOG entry)",
    )
    args = ap.parse_args(argv)

    if args.command in ("search", "freeze", "all"):
        cand, diag, prov = search(n_pilot=args.pilot_sequences)
        hmm = cand.build()
        print(_report(diag, hmm))
        print(f"\nchosen candidate: {cand.to_dict()}")
        print(f"hmm sha256: {hmm.sha256()}")
        if args.command == "search":
            return 0
        payload = freeze_matrices(cand, diag, prov, path=args.manifest, force=args.force_refreeze)
        print(f"\nfrozen -> {args.manifest} (payload sha256 {payload['payload_sha256'][:16]})")

    if args.command in ("corpus", "all"):
        hmm, payload = load_frozen_hmm(args.manifest)
        manifest = generate_corpus(hmm, data_dir=args.data_dir, hmm_manifest=payload)
        print(json.dumps(manifest["splits"], indent=2))

    if args.command == "report":
        hmm, payload = load_frozen_hmm(args.manifest)
        diag = Diagnostics(
            values=payload["diagnostics"],
            entropy_histogram=payload["entropy_histogram"],
            slack=payload["diagnostics"].get("worst_slack", float("nan")),
        )
        print(_report(diag, hmm))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
