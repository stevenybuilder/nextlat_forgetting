#!/usr/bin/env python
"""Extract and score all preregistered HMM geometry outcomes for one trained checkpoint.

The expensive forward pass is a durable chunked stage.  Every completed chunk is independently
hash-verifiable and reusable after a runtime loss.  Scientific choices are fixed in
``REPRESENTATION_POLICY`` and written to a hash-bound representation manifest before extraction;
the analysis stage has no model- or outcome-dependent switches.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import typing as t

import numpy as np
from scipy import stats

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from hmm_geometry.evaluate import (  # noqa: E402
    PairStates,
    fit_whitener,
    h1_predictive_equivalence,
    h2_relative_geometry,
    h3_posterior_decodability,
    partial_spearman,
    whitened_euclidean_distance,
)
from hmm_geometry.forward import js_divergence  # noqa: E402
from hmm_geometry.extraction_cache import ExtractionCache, ExtractionCacheError  # noqa: E402
from lurestar.durable_checkpoint import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402

SCHEMA = "nextlat_forgetting/hmm_geometry/1"
REPRESENTATION_SCHEMA = "nextlat_forgetting/hmm_representation_plan/1"
PINNED_UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
MODELS = ("gpt", "nextlat")
FIT_ROWS = (0, 5_000)
SCORE_ROWS = (5_000, 10_000)
FIT_PREFIXES = tuple(range(16, 33))
LENGEN_PREFIXES = tuple(range(33, 65))
NEIGHBOR_K = 10
RIDGE = 1e-2

REPRESENTATION_POLICY: dict[str, object] = {
    "state": "final post-normalization hidden state at prefix final token (zero-based t-1)",
    "h3_fit": {"split": "val", "rows": list(FIT_ROWS), "prefixes": [16, 32]},
    "h3_score": {"split": "val", "rows": list(SCORE_ROWS), "prefixes": [16, 32]},
    "h3_lengen_score_without_refit": {
        "split": "lengen", "rows": list(SCORE_ROWS), "prefixes": [33, 64]
    },
    "neighborhood": {
        "population": "all unique frozen pair-bank endpoints within each pool", "k": NEIGHBOR_K
    },
    "primary_distance": "centered cosine",
    "co_primary_distances": ["centered cosine", "held-out-whitened Mahalanobis"],
    "whitener_fit": {"split": "val", "rows": list(FIT_ROWS), "prefixes": list(FIT_PREFIXES)},
    "probe": {"kind": "closed-form linear ridge", "ridge": RIDGE},
    "outcome_dependent_selection": False,
}


class HMMEvaluationError(RuntimeError):
    """A frozen identity, extraction, cache, or scoring invariant failed closed."""


def _git_head(upstream: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HMMEvaluationError("cannot determine upstream git commit") from exc


def _manifest_records(paths: t.Sequence[pathlib.Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise HMMEvaluationError(f"frozen manifest is missing: {resolved}")
        records.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    return records


def verify_inventory_binding(
    manifests: t.Sequence[pathlib.Path], inputs: t.Sequence[pathlib.Path]
) -> None:
    """Require the frozen sha256sum inventory to bind every large/scientific input by path."""
    inventories = [path.resolve() for path in manifests if path.name.endswith("inventory.sha256")]
    if len(inventories) != 1:
        raise HMMEvaluationError("exactly one manifest_inventory.sha256 is required")
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(inventories[0].read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise HMMEvaluationError(f"malformed inventory line {line_number}")
        rows[fields[1].lstrip("*")] = fields[0].lower()
    for input_path in inputs:
        resolved = input_path.resolve()
        exact_suffix = [digest for name, digest in rows.items() if resolved.as_posix().endswith("/" + name)]
        matches = exact_suffix or [
            digest for name, digest in rows.items() if pathlib.Path(name).name == resolved.name
        ]
        if len(matches) != 1 or matches[0] != sha256_file(resolved):
            raise HMMEvaluationError(f"frozen inventory does not bind {resolved.name}")


def load_pair_bank(path: pathlib.Path) -> list[dict[str, object]]:
    """Load the frozen JSONL bank and validate the fields used by every estimator."""
    rows: list[dict[str, object]] = []
    allowed_types = {"equivalent", "near_lure", "matched_control"}
    allowed_pools = {"test32", "test64"}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HMMEvaluationError(f"invalid pair JSON at line {line_number}") from exc
        if row.get("pair_type") not in allowed_types or row.get("pool") not in allowed_pools:
            raise HMMEvaluationError(f"invalid pair type/pool at line {line_number}")
        prefix_len = row.get("prefix_len")
        if not isinstance(prefix_len, int) or isinstance(prefix_len, bool):
            raise HMMEvaluationError(f"invalid prefix length at line {line_number}")
        for endpoint_name in ("a", "b"):
            endpoint = row.get(endpoint_name)
            if not isinstance(endpoint, dict):
                raise HMMEvaluationError(f"missing endpoint {endpoint_name} at line {line_number}")
            prefix = endpoint.get("prefix")
            belief = endpoint.get("belief")
            if (
                not isinstance(prefix, list) or len(prefix) != prefix_len
                or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 4 for x in prefix)
                or not isinstance(belief, list) or len(belief) != 4
                or not np.isfinite(np.asarray(belief, dtype=np.float64)).all()
            ):
                raise HMMEvaluationError(f"invalid endpoint {endpoint_name} at line {line_number}")
        rows.append(row)
    if not rows:
        raise HMMEvaluationError("pair bank is empty")
    return rows


def _endpoint_key(prefix: t.Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in prefix)


def unique_endpoints(
    pair_rows: t.Sequence[dict[str, object]], pool: str
) -> dict[tuple[int, ...], np.ndarray]:
    """All unique frozen endpoints, rejecting inconsistent duplicate posterior labels."""
    endpoints: dict[tuple[int, ...], np.ndarray] = {}
    for row in pair_rows:
        if row["pool"] != pool:
            continue
        for side in ("a", "b"):
            endpoint = t.cast(dict, row[side])
            key = _endpoint_key(endpoint["prefix"])
            belief = np.asarray(endpoint["belief"], dtype=np.float64)
            if key in endpoints and not np.allclose(endpoints[key], belief, atol=1e-12, rtol=0.0):
                raise HMMEvaluationError("one frozen prefix has conflicting exact posteriors")
            endpoints[key] = belief
    if len(endpoints) <= NEIGHBOR_K + 1:
        raise HMMEvaluationError(f"{pool} has too few unique endpoints for k={NEIGHBOR_K}")
    return endpoints


def representation_manifest(pair_rows: t.Sequence[dict[str, object]]) -> dict[str, object]:
    endpoint_counts = {pool: len(unique_endpoints(pair_rows, pool)) for pool in ("test32", "test64")}
    return {
        "schema": REPRESENTATION_SCHEMA,
        "policy": REPRESENTATION_POLICY,
        "pair_counts": {
            pool: {
                kind: sum(row["pool"] == pool and row["pair_type"] == kind for row in pair_rows)
                for kind in ("equivalent", "near_lure", "matched_control")
            }
            for pool in ("test32", "test64")
        },
        "unique_endpoint_counts": endpoint_counts,
    }


def _planned_dataset_keys(label: str, start: int, stop: int, chunk_rows: int) -> list[str]:
    return [f"{label}_{lo:05d}_{min(lo + chunk_rows, stop):05d}" for lo in range(start, stop, chunk_rows)]


def _planned_endpoint_keys(
    pair_rows: t.Sequence[dict[str, object]], chunk_rows: int
) -> list[str]:
    keys: list[str] = []
    for pool in ("test32", "test64"):
        grouped: dict[int, int] = {}
        for prefix in unique_endpoints(pair_rows, pool):
            grouped[len(prefix)] = grouped.get(len(prefix), 0) + 1
        for length, count in sorted(grouped.items()):
            for lo in range(0, count, chunk_rows):
                keys.append(f"endpoints_{pool}_t{length:02d}_{lo:05d}_{min(lo+chunk_rows,count):05d}")
    return keys


def planned_cache_keys(pair_rows: t.Sequence[dict[str, object]], chunk_rows: int) -> list[str]:
    return [
        *_planned_dataset_keys("h3_fit", *FIT_ROWS, chunk_rows),
        *_planned_dataset_keys("h3_score", *SCORE_ROWS, chunk_rows),
        *_planned_dataset_keys("h3_lengen", *SCORE_ROWS, chunk_rows),
        *_planned_endpoint_keys(pair_rows, chunk_rows),
    ]


class TorchExtractor:
    """Thin tensor-touching adapter over the pinned GPT/NextLat hidden-state forwards."""

    def __init__(
        self, *, model_name: str, seed: int, checkpoint: pathlib.Path,
        config_path: pathlib.Path, upstream: pathlib.Path,
    ) -> None:
        try:
            import lightning as L
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - Colab-only dependency surface
            raise HMMEvaluationError("pinned torch/lightning runtime is unavailable") from exc
        if _git_head(upstream) != PINNED_UPSTREAM_COMMIT:
            raise HMMEvaluationError("upstream checkout is not at the pinned commit")
        document = OmegaConf.load(str(config_path))
        if int(document.seed) != seed or bool(document.use_nextlat) != (model_name == "nextlat"):
            raise HMMEvaluationError("materialized config does not match evaluator model/seed")
        config = OmegaConf.merge(OmegaConf.load(str(upstream / "defaults.yaml")), document)
        config.model.vocab_size = 5
        config.model.context_length = 0
        config.model.block_size = 65
        config.data.device_batch_size = int(config.data.effective_batch_size)
        config.data.micro_batch_size = int(config.data.effective_batch_size)

        sys.path.insert(0, str(upstream))
        prior = pathlib.Path.cwd()
        try:
            os.chdir(upstream)
            from core_train import initialize_model
            from hmm_geometry.datamodule import HMMTokenizer

            fabric = L.Fabric()
            fabric.seed_everything(seed)
            wrapper = initialize_model(
                fabric, config, HMMTokenizer(4), initialize_optimizer=False,
                checkpoint_path=str(checkpoint),
            )
        finally:
            os.chdir(prior)
        wrapper.eval()
        self.torch = torch
        self.fabric = fabric
        self.inner = wrapper.model
        self.model_name = model_name

    def __call__(self, tokens: np.ndarray, positions: t.Sequence[int]) -> np.ndarray:
        from lurestar.representations import forward_states_and_logits

        batch = self.torch.as_tensor(tokens, dtype=self.torch.long, device=self.fabric.device)
        with self.torch.inference_mode():
            hidden, _ = forward_states_and_logits(
                self.inner, batch, architecture=self.model_name
            )
            selected = hidden.index_select(
                1, self.torch.as_tensor(positions, dtype=self.torch.long, device=hidden.device)
            )
        return selected.float().cpu().numpy()


def populate_cache(
    cache: ExtractionCache, *, extractor: t.Callable[[np.ndarray, t.Sequence[int]], np.ndarray],
    pair_rows: t.Sequence[dict[str, object]], val_npz: pathlib.Path,
    lengen_npz: pathlib.Path, chunk_rows: int,
) -> list[str]:
    """Run only missing chunks; every loop iteration commits useful, resumable output."""
    with np.load(val_npz, mmap_mode="r", allow_pickle=False) as val, np.load(
        lengen_npz, mmap_mode="r", allow_pickle=False
    ) as lengen:
        # NPZ members are archive entries rather than true memmaps. Materialize each one once;
        # indexing ``payload[name]`` inside every chunk would decompress the entire member again.
        val_arrays = {
            name: val[name] for name in ("observations", "beliefs", "next_obs", "hidden_states")
        }
        lengen_arrays = {
            name: lengen[name]
            for name in ("observations", "beliefs", "next_obs", "hidden_states")
        }
        datasets = (
            ("h3_fit", val_arrays, FIT_ROWS, FIT_PREFIXES),
            ("h3_score", val_arrays, SCORE_ROWS, FIT_PREFIXES),
            ("h3_lengen", lengen_arrays, SCORE_ROWS, LENGEN_PREFIXES),
        )
        for label, payload, (start, stop), prefixes in datasets:
            if payload["observations"].shape[0] != 10_000:
                raise HMMEvaluationError(f"{label} split does not have exactly 10,000 rows")
            positions = tuple(prefix - 1 for prefix in prefixes)
            for lo in range(start, stop, chunk_rows):
                hi = min(lo + chunk_rows, stop)
                key = f"{label}_{lo:05d}_{hi:05d}"
                if cache.has(key):
                    continue
                observations = np.asarray(payload["observations"][lo:hi])
                states = extractor(observations, positions)
                cache.write(key, {
                    "row_index": np.arange(lo, hi, dtype=np.int32),
                    "prefix_lengths": np.asarray(prefixes, dtype=np.int16),
                    "states": states.astype(np.float32),
                    "beliefs": np.asarray(payload["beliefs"][lo:hi][:, positions], dtype=np.float64),
                    "next_obs": np.asarray(
                        payload["next_obs"][lo:hi][:, prefixes], dtype=np.float64
                    ),
                    "realized_states": np.asarray(
                        payload["hidden_states"][lo:hi][:, positions], dtype=np.int16
                    ),
                })

    for pool in ("test32", "test64"):
        endpoints = unique_endpoints(pair_rows, pool)
        grouped: dict[int, list[tuple[tuple[int, ...], np.ndarray]]] = {}
        for prefix, belief in sorted(endpoints.items()):
            grouped.setdefault(len(prefix), []).append((prefix, belief))
        for length, items in sorted(grouped.items()):
            for lo in range(0, len(items), chunk_rows):
                hi = min(lo + chunk_rows, len(items))
                key = f"endpoints_{pool}_t{length:02d}_{lo:05d}_{hi:05d}"
                if cache.has(key):
                    continue
                selected = items[lo:hi]
                prefixes = np.asarray([item[0] for item in selected], dtype=np.int8)
                cache.write(key, {
                    "prefixes": prefixes,
                    "states": extractor(prefixes, (length - 1,))[:, 0].astype(np.float32),
                    "beliefs": np.asarray([item[1] for item in selected], dtype=np.float64),
                })
    return planned_cache_keys(pair_rows, chunk_rows)


def _load_flat(cache: ExtractionCache, keys: t.Sequence[str]) -> dict[str, np.ndarray]:
    chunks = [cache.load(key) for key in keys]
    return {
        name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
        for name in chunks[0]
    }


def _load_endpoint_maps(
    cache: ExtractionCache, pair_rows: t.Sequence[dict[str, object]], chunk_rows: int
) -> tuple[dict[str, dict[tuple[int, ...], np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    maps: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
    neighborhoods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pool in ("test32", "test64"):
        grouped: dict[int, int] = {}
        for prefix in unique_endpoints(pair_rows, pool):
            grouped[len(prefix)] = grouped.get(len(prefix), 0) + 1
        keys = []
        for length, count in sorted(grouped.items()):
            keys.extend(
                f"endpoints_{pool}_t{length:02d}_{lo:05d}_{min(lo+chunk_rows,count):05d}"
                for lo in range(0, count, chunk_rows)
            )
        mapping: dict[tuple[int, ...], np.ndarray] = {}
        state_parts: list[np.ndarray] = []
        belief_parts: list[np.ndarray] = []
        for key in keys:
            data = cache.load(key)
            mapping.update({
                _endpoint_key(prefix): state
                for prefix, state in zip(data["prefixes"], data["states"])
            })
            state_parts.append(data["states"])
            belief_parts.append(data["beliefs"])
        maps[pool] = mapping
        neighborhoods[pool] = (
            np.concatenate(state_parts, axis=0), np.concatenate(belief_parts, axis=0)
        )
    return maps, neighborhoods


def _pair_states(
    rows: t.Sequence[dict[str, object]], states: dict[tuple[int, ...], np.ndarray],
    *, js_field: str = "js_divergence_bits",
) -> PairStates:
    return PairStates(
        a=np.asarray([states[_endpoint_key(t.cast(dict, row["a"])["prefix"])] for row in rows]),
        b=np.asarray([states[_endpoint_key(t.cast(dict, row["b"])["prefix"])] for row in rows]),
        edit_distance=np.asarray([row["edit_distance"] for row in rows], dtype=np.int32),
        prefix_len=np.asarray([row["prefix_len"] for row in rows], dtype=np.int16),
        js_bits=np.asarray(
            [row.get(js_field, row["js_divergence_bits"]) for row in rows], dtype=np.float64
        ),
    )


def _h2_whitened_euclidean(
    pairs: PairStates, mean: np.ndarray, transform: np.ndarray
) -> dict[str, object]:
    distance = whitened_euclidean_distance(pairs.a, pairs.b, mean, transform)
    rho, pvalue = stats.spearmanr(distance, pairs.js_bits)
    partial, partial_p = partial_spearman(
        distance, pairs.js_bits, [pairs.edit_distance, pairs.prefix_len]
    )
    return {
        "n_pairs": int(len(distance)), "spearman_distance_vs_js": float(rho),
        "spearman_p": float(pvalue),
        "partial_spearman_given_edit_and_length": partial,
        "partial_spearman_p": partial_p,
        "distance": "held_out_whitened_mahalanobis",
    }


def neighborhood_retrieval_chunked(
    states: np.ndarray, beliefs: np.ndarray, *, k: int = NEIGHBOR_K,
    center: np.ndarray | None = None, query_block: int = 64,
) -> dict[str, object]:
    """Exact kNN overlap without allocating either full ``n x n`` distance matrix.

    The frozen endpoint populations contain roughly 7k/9k items, so the otherwise-identical
    estimator in ``hmm_geometry.evaluate`` would retain two ~0.7GB matrices plus sorting work
    space.  Here each query block is compared to all endpoints and immediately reduced to its
    exact neighbours.  This changes memory complexity, not the statistic or population.
    """
    states = np.asarray(states, dtype=np.float64)
    beliefs = np.asarray(beliefs, dtype=np.float64)
    n = len(states)
    if n <= k + 1 or query_block <= 0:
        raise HMMEvaluationError("invalid neighborhood population/block size")
    center = states.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    normalized = states - center
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-30)
    overlaps: list[int] = []
    for lo in range(0, n, query_block):
        hi = min(lo + query_block, n)
        cosine = 1.0 - normalized[lo:hi] @ normalized.T
        js = js_divergence(beliefs[lo:hi, None, :], beliefs[None, :, :])
        rows = np.arange(hi - lo)
        cols = np.arange(lo, hi)
        cosine[rows, cols] = np.inf
        js[rows, cols] = np.inf
        # Stable sort fixes the tie rule by frozen endpoint order.
        state_knn = np.argsort(cosine, axis=1, kind="stable")[:, :k]
        belief_knn = np.argsort(js, axis=1, kind="stable")[:, :k]
        overlaps.extend(
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(state_knn, belief_knn)
        )
    overlap = np.asarray(overlaps, dtype=np.float64)
    chance = k * (k / (n - 1))
    return {
        "k": int(k), "n_items": int(n), "mean_overlap": float(overlap.mean()),
        "precision_at_k": float(overlap.mean() / k),
        "chance_precision_at_k": float(chance / k),
        "lift_over_chance": float(overlap.mean() / chance),
        "algorithm": "exact_query_blocked_stable_sort",
        "query_block": int(query_block),
    }


def compute_metrics(
    cache: ExtractionCache, pair_rows: t.Sequence[dict[str, object]], chunk_rows: int,
    *, seed: int,
) -> dict[str, dict[str, object]]:
    """Apply the frozen estimators without any outcome-dependent branching."""
    family_future_bound = all(
        "future_js_divergence_bits" in row and "belief_js_divergence_bits" in row
        for row in pair_rows
    )
    flatten = lambda x: x.reshape((-1,) + x.shape[2:])
    fit_keys = _planned_dataset_keys("h3_fit", *FIT_ROWS, chunk_rows)
    fit = _load_flat(cache, fit_keys)
    heldout_white_mean, heldout_white_transform = fit_whitener(flatten(fit["states"]))
    maps, neighborhoods = _load_endpoint_maps(cache, pair_rows, chunk_rows)
    h1_primary: dict[str, object] = {}
    h1_robust: dict[str, object] = {}
    h2_raw: dict[str, object] = {}
    h2_partial: dict[str, object] = {}
    h2_whitened: dict[str, object] = {}
    h2_belief: dict[str, object] = {}
    h2_belief_whitened: dict[str, object] = {}
    retrieval: dict[str, object] = {}
    for pool_index, pool in enumerate(("test32", "test64")):
        rows = [row for row in pair_rows if row["pool"] == pool]
        equivalent = _pair_states([row for row in rows if row["pair_type"] == "equivalent"], maps[pool])
        control = _pair_states([row for row in rows if row["pair_type"] == "matched_control"], maps[pool])
        selected_rows = [row for row in rows if row["pair_type"] != "matched_control"]
        h2_pairs = _pair_states(selected_rows, maps[pool])
        belief_pairs = _pair_states(
            selected_rows, maps[pool], js_field="belief_js_divergence_bits"
        )
        population_states, population_beliefs = neighborhoods[pool]
        center = population_states.mean(axis=0)
        white_mean, white_transform = heldout_white_mean, heldout_white_transform
        h1 = h1_predictive_equivalence(
            equivalent, control, center=center, whitener=(white_mean, white_transform),
            rng=np.random.default_rng(seed + pool_index),
        )
        robustness = t.cast(dict, h1.pop("robustness_whitened_euclidean"))
        robustness.update({"n_pairs": h1["n_pairs"], "distance": "whitened_euclidean"})
        h1_primary[pool] = h1
        h1_robust[pool] = robustness
        h2 = h2_relative_geometry(h2_pairs, center=center)
        h2_raw[pool] = {
            key: value for key, value in h2.items()
            if key in {"n_pairs", "spearman_distance_vs_js", "spearman_p", "mean_distance"}
        }
        h2_partial[pool] = {
            key: value for key, value in h2.items()
            if key in {"n_pairs", "partial_spearman_given_edit_and_length", "partial_spearman_p",
                       "spearman_distance_vs_edit_distance"}
        }
        h2w = _h2_whitened_euclidean(h2_pairs, white_mean, white_transform)
        h2_whitened[pool] = {
            key: value for key, value in h2w.items()
            if key in {"n_pairs", "partial_spearman_given_edit_and_length", "partial_spearman_p"}
        }
        h2b = h2_relative_geometry(belief_pairs, center=center)
        h2_belief[pool] = {
            key: value for key, value in h2b.items()
            if key in {"n_pairs", "partial_spearman_given_edit_and_length", "partial_spearman_p"}
        }
        h2bw = _h2_whitened_euclidean(belief_pairs, white_mean, white_transform)
        h2_belief_whitened[pool] = {
            key: value for key, value in h2bw.items()
            if key in {"n_pairs", "partial_spearman_given_edit_and_length", "partial_spearman_p"}
        }
        retrieval[pool] = neighborhood_retrieval_chunked(
            population_states, population_beliefs, k=NEIGHBOR_K, center=center
        )

    score_keys = _planned_dataset_keys("h3_score", *SCORE_ROWS, chunk_rows)
    lengen_keys = _planned_dataset_keys("h3_lengen", *SCORE_ROWS, chunk_rows)
    score = _load_flat(cache, score_keys)
    lengen = _load_flat(cache, lengen_keys)
    h3 = h3_posterior_decodability(
        flatten(fit["states"]), flatten(fit["beliefs"]),
        flatten(score["states"]), flatten(score["beliefs"]),
        next_obs_train=flatten(fit["next_obs"]), next_obs_test=flatten(score["next_obs"]),
        hidden_states_test=flatten(score["realized_states"]),
        h_lengen=flatten(lengen["states"]), b_lengen=flatten(lengen["beliefs"]),
        next_obs_lengen=flatten(lengen["next_obs"]), ridge=RIDGE,
        rng=np.random.default_rng(seed + 10_000),
    )
    result = {
        "h1_predictive_equivalence_centered_cosine": h1_primary,
        "h1_predictive_equivalence_whitened": h1_robust,
        "h2_spearman": h2_raw,
        "h2_partial_spearman": h2_partial,
        "h2_neighborhood_retrieval": retrieval,
        "h3_posterior_decoding_len32": t.cast(dict, h3["posterior"])["test"],
        "h3_future_distribution_decoding_len32": t.cast(dict, h3["next_obs"])["test"],
        "h3_posterior_decoding_len64": t.cast(dict, h3["posterior"])["lengen64"],
        "h3_future_distribution_decoding_len64": t.cast(dict, h3["next_obs"])["lengen64"],
    }
    if family_future_bound:
        result.update({
            "h2_partial_spearman_whitened": h2_whitened,
            "h2_belief_partial_spearman": h2_belief,
            "h2_belief_partial_spearman_whitened": h2_belief_whitened,
        })
    return result


def evaluate_checkpoint(
    *, job_id: str, model_name: str, seed: int, checkpoint: pathlib.Path,
    config: pathlib.Path, source_config: pathlib.Path, pair_bank: pathlib.Path,
    val_posteriors: pathlib.Path, lengen_posteriors: pathlib.Path,
    manifests: t.Sequence[pathlib.Path], upstream: pathlib.Path, output: pathlib.Path,
    cache_root: pathlib.Path, batch_size: int,
    regime: str | None = None,
    extractor_factory: t.Callable[..., object] = TorchExtractor,
) -> dict[str, object]:
    expected_job = f"{model_name}-seed{seed}-hmm" + (f"-{regime}" if regime else "")
    if model_name not in MODELS or job_id != expected_job:
        raise HMMEvaluationError(f"job/model/seed identity mismatch: {job_id!r}")
    if batch_size <= 0:
        raise HMMEvaluationError("batch size must be positive")
    inputs = [checkpoint, config, source_config, pair_bank, val_posteriors, lengen_posteriors]
    if any(not path.resolve().is_file() for path in inputs):
        raise HMMEvaluationError("one or more checkpoint/evaluation inputs are missing")
    manifest_records = _manifest_records(manifests)
    verify_inventory_binding(manifests, (pair_bank, val_posteriors, lengen_posteriors))
    pair_rows = load_pair_bank(pair_bank)
    if regime is not None and not all(
        "future_js_divergence_bits" in row and "belief_js_divergence_bits" in row
        for row in pair_rows
    ):
        raise HMMEvaluationError(
            "family pair bank is not bound to future-JS selection plus belief-JS secondary labels"
        )
    representation = representation_manifest(pair_rows)
    representation_path = output.parent / "representation_manifest.json"
    atomic_write_json(representation_path, representation)
    representation_record = {
        "path": str(representation_path.resolve()), "sha256": sha256_file(representation_path)
    }
    evaluator_sources = [
        pathlib.Path(__file__).resolve(),
        (_REPO / "src/hmm_geometry/extraction_cache.py").resolve(),
        (_REPO / "src/hmm_geometry/evaluate.py").resolve(),
        (_REPO / "src/hmm_geometry/forward.py").resolve(),
        (_REPO / "src/hmm_geometry/pair_bank.py").resolve(),
    ]
    identity = {
        "job_id": job_id,
        "model": model_name,
        "seed": seed,
        "regime": regime or "primary",
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "source_config_sha256": sha256_file(source_config),
        "pair_bank_sha256": sha256_file(pair_bank),
        "val_posteriors_sha256": sha256_file(val_posteriors),
        "lengen_posteriors_sha256": sha256_file(lengen_posteriors),
        "representation_manifest_sha256": representation_record["sha256"],
        "evaluator_source_sha256": {str(path): sha256_file(path) for path in evaluator_sources},
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "chunk_rows": batch_size,
    }
    cache = ExtractionCache(cache_root, identity)
    expected_keys = planned_cache_keys(pair_rows, batch_size)
    if not all(cache.has(key) for key in expected_keys):
        # Model construction/checkpoint load is intentionally deferred. If extraction completed
        # before a disconnect and only CPU scoring/receipt emission remains, reconnecting does not
        # reload weights or repeat a single GPU forward.
        extractor = extractor_factory(
            model_name=model_name, seed=seed, checkpoint=checkpoint.resolve(),
            config_path=config.resolve(), upstream=upstream.resolve(),
        )
        populate_cache(
            cache, extractor=t.cast(t.Callable, extractor), pair_rows=pair_rows,
            val_npz=val_posteriors, lengen_npz=lengen_posteriors, chunk_rows=batch_size,
        )
    cache_record = cache.receipt(expected_keys=expected_keys)
    metrics = compute_metrics(cache, pair_rows, batch_size, seed=seed)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "job_id": job_id,
        "model": model_name,
        "seed": seed,
        "regime": regime or "primary",
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "all_preregistered_metrics_reported": True,
        "metric_selection_performed": False,
        "evaluator": {"path": str(evaluator_sources[0]), "sha256": sha256_file(evaluator_sources[0])},
        "evaluator_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in evaluator_sources
        ],
        "inputs": {
            "source_config": {"path": str(source_config.resolve()), "sha256": sha256_file(source_config)},
            "materialized_config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
            "pair_bank": {"path": str(pair_bank.resolve()), "sha256": sha256_file(pair_bank)},
            "val_posteriors": {
                "path": str(val_posteriors.resolve()), "sha256": sha256_file(val_posteriors)
            },
            "lengen_posteriors": {
                "path": str(lengen_posteriors.resolve()), "sha256": sha256_file(lengen_posteriors)
            },
        },
        "manifests": manifest_records,
        "representation_manifest": representation_record,
        "representation_cache": cache_record,
        "metrics": metrics,
    }
    # allow_nan=False is a final recursive finite-value gate before any DONE receipt exists.
    json.dumps(result, allow_nan=False)
    atomic_write_json(output, result)
    atomic_write_text(output.with_name(output.name + ".sha256"), f"{sha256_file(output)}  {output.name}\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--regime", choices=(
            "persistent_moderate", "fast_mixing_moderate", "persistent_high_aliasing"
        )
    )
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--source-config", required=True, type=pathlib.Path)
    ap.add_argument("--pair-bank", required=True, type=pathlib.Path)
    ap.add_argument("--val-posteriors", required=True, type=pathlib.Path)
    ap.add_argument("--lengen-posteriors", required=True, type=pathlib.Path)
    ap.add_argument("--manifest", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--upstream", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--cache-root", required=True, type=pathlib.Path)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)
    try:
        result = evaluate_checkpoint(
            job_id=args.job_id, model_name=args.model, seed=args.seed,
            checkpoint=args.checkpoint, config=args.config, source_config=args.source_config,
            pair_bank=args.pair_bank, val_posteriors=args.val_posteriors,
            lengen_posteriors=args.lengen_posteriors, manifests=args.manifest,
            upstream=args.upstream, output=args.output, cache_root=args.cache_root,
            batch_size=args.batch_size, regime=args.regime,
        )
    except (HMMEvaluationError, ExtractionCacheError, OSError, ValueError) as exc:
        print(f"[evaluate_hmm_checkpoints] REFUSED/FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
