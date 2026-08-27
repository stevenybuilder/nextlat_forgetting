"""Production HMM evaluator: frozen splits, endpoint population, and durable extraction."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from evaluate_hmm_checkpoints import (
    FIT_PREFIXES,
    FIT_ROWS,
    LENGEN_PREFIXES,
    NEIGHBOR_K,
    REPRESENTATION_POLICY,
    SCORE_ROWS,
    HMMEvaluationError,
    load_pair_bank,
    neighborhood_retrieval_chunked,
    planned_cache_keys,
    populate_cache,
    representation_manifest,
    unique_endpoints,
    verify_inventory_binding,
)
from hmm_geometry.extraction_cache import ExtractionCache, ExtractionCacheError
from hmm_geometry.evaluate import neighborhood_retrieval
from lurestar.durable_checkpoint import sha256_file


def _pair(pool: str, kind: str, index: int, length: int) -> dict:
    a = [index % 4] * (length - 1) + [(index // 4) % 4]
    b = [(index + 1) % 4] * (length - 1) + [((index + 1) // 4) % 4]
    belief_a = np.full(4, 0.1)
    belief_a[index % 4] = 0.7
    belief_b = np.full(4, 0.1)
    belief_b[(index + 1) % 4] = 0.7
    return {
        "pair_id": f"{pool}-{kind}-{index}", "pool": pool, "pair_type": kind,
        "prefix_len": length, "edit_distance": length,
        "js_divergence_bits": 0.2,
        "a": {"prefix": a, "belief": belief_a.tolist()},
        "b": {"prefix": b, "belief": belief_b.tolist()},
    }


def _rows() -> list[dict]:
    rows = []
    # Unique endpoints must exceed k+1 in both pools. Vary two suffix symbols so the prefixes are
    # genuinely distinct even after the four-symbol leading pattern repeats.
    for pool, length in (("test32", 16), ("test64", 33)):
        for index in range(12):
            row = _pair(pool, "near_lure", index, length)
            row["a"]["prefix"][-2] = (index // 4) % 4
            row["b"]["prefix"][-2] = ((index + 2) // 4) % 4
            rows.append(row)
    return rows


def test_policy_freezes_disjoint_fit_score_and_no_refit() -> None:
    assert FIT_ROWS == (0, 5_000)
    assert SCORE_ROWS == (5_000, 10_000)
    assert FIT_PREFIXES == tuple(range(16, 33))
    assert LENGEN_PREFIXES == tuple(range(33, 65))
    assert REPRESENTATION_POLICY["h3_lengen_score_without_refit"]["prefixes"] == [33, 64]
    assert REPRESENTATION_POLICY["neighborhood"]["k"] == 10 == NEIGHBOR_K
    assert REPRESENTATION_POLICY["outcome_dependent_selection"] is False


def test_neighborhood_population_is_every_unique_frozen_endpoint() -> None:
    rows = _rows()
    manifest = representation_manifest(rows)
    for pool in ("test32", "test64"):
        endpoints = unique_endpoints(rows, pool)
        direct = {
            tuple(endpoint["prefix"])
            for row in rows if row["pool"] == pool for endpoint in (row["a"], row["b"])
        }
        assert set(endpoints) == direct
        assert manifest["unique_endpoint_counts"][pool] == len(direct)


def test_chunked_neighborhood_is_exactly_the_declared_estimator() -> None:
    rng = np.random.default_rng(12)
    states = rng.normal(size=(37, 8))
    beliefs = rng.random((37, 4))
    beliefs /= beliefs.sum(axis=1, keepdims=True)
    expected = neighborhood_retrieval(states, beliefs, k=10)
    actual = neighborhood_retrieval_chunked(states, beliefs, k=10, query_block=7)
    for key in ("k", "n_items", "mean_overlap", "precision_at_k",
                "chance_precision_at_k", "lift_over_chance"):
        assert actual[key] == pytest.approx(expected[key])


def test_pair_loader_rejects_malformed_prefix(tmp_path: pathlib.Path) -> None:
    row = _rows()[0]
    row["a"]["prefix"] = row["a"]["prefix"][:-1]
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(HMMEvaluationError, match="invalid endpoint"):
        load_pair_bank(path)


def test_cache_commits_npz_then_sidecar_then_progress_and_resumes(tmp_path: pathlib.Path) -> None:
    identity = {"checkpoint_sha256": "a" * 64, "policy": {"rows": [0, 5_000]}}
    cache = ExtractionCache(tmp_path / "cache", identity)
    record = cache.write("fit_000", {"states": np.arange(12).reshape(3, 4).astype(np.float32)})
    assert pathlib.Path(record["path"]).is_file()
    assert pathlib.Path(record["sidecar"]).is_file()
    assert sha256_file(record["path"]) == record["sha256"]
    assert json.loads(cache.progress_path.read_text())["chunks"]["fit_000"]["sha256"] == record["sha256"]
    np.testing.assert_array_equal(cache.load("fit_000")["states"], np.arange(12).reshape(3, 4))

    # A reconnect with the same identity reuses the committed data without rewriting it.
    resumed = ExtractionCache(tmp_path / "cache", identity)
    assert resumed.has("fit_000")
    assert resumed.write("fit_000", {"states": np.ones((3, 4))}) == record


def test_cache_fails_closed_on_identity_corruption_and_nonfinite(tmp_path: pathlib.Path) -> None:
    cache = ExtractionCache(tmp_path / "cache", {"checkpoint": "one"})
    cache.write("chunk", {"states": np.ones((2, 2))})
    with pytest.raises(ExtractionCacheError, match="identity mismatch"):
        ExtractionCache(tmp_path / "cache", {"checkpoint": "two"})
    with pytest.raises(ExtractionCacheError, match="non-finite"):
        ExtractionCache(tmp_path / "other", {"bad": float("nan")})
    with pytest.raises(ExtractionCacheError, match="non-finite"):
        cache.write("bad", {"states": np.asarray([[np.inf]])})


def test_corrupt_cache_chunk_is_never_considered_resumable(tmp_path: pathlib.Path) -> None:
    cache = ExtractionCache(tmp_path / "cache", {"checkpoint": "one"})
    record = cache.write("chunk", {"states": np.ones((2, 2))})
    pathlib.Path(record["path"]).write_bytes(b"truncated")
    assert cache.has("chunk") is False
    with pytest.raises(ExtractionCacheError, match="SHA-256"):
        cache.load("chunk")
    repaired = cache.write("chunk", {"states": np.full((2, 2), 3.0)})
    assert sha256_file(repaired["path"]) == repaired["sha256"]
    np.testing.assert_array_equal(cache.load("chunk")["states"], np.full((2, 2), 3.0))


def test_inventory_must_bind_every_scientific_input(tmp_path: pathlib.Path) -> None:
    inputs = []
    rows = []
    for name in ("pairs.jsonl", "val.npz", "lengen.npz"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        inputs.append(path)
        rows.append(f"{sha256_file(path)}  data/{name}\n")
    inventory = tmp_path / "manifest_inventory.sha256"
    inventory.write_text("".join(rows))
    verify_inventory_binding([inventory], inputs)
    inputs[1].write_bytes(b"changed")
    with pytest.raises(HMMEvaluationError, match="does not bind val.npz"):
        verify_inventory_binding([inventory], inputs)


def test_populate_cache_skips_all_verified_chunks_after_reconnect(tmp_path: pathlib.Path) -> None:
    rows = _rows()
    rng = np.random.default_rng(4)

    def make_npz(path: pathlib.Path, length: int) -> None:
        observations = rng.integers(0, 4, (10_000, length), dtype=np.int8)
        beliefs = rng.random((10_000, length, 4))
        beliefs /= beliefs.sum(axis=-1, keepdims=True)
        next_obs = rng.random((10_000, length + 1, 4))
        next_obs /= next_obs.sum(axis=-1, keepdims=True)
        hidden = rng.integers(0, 4, (10_000, length), dtype=np.int8)
        np.savez(path, observations=observations, beliefs=beliefs, next_obs=next_obs,
                 hidden_states=hidden)

    val = tmp_path / "val.npz"
    lengen = tmp_path / "lengen.npz"
    make_npz(val, 32)
    make_npz(lengen, 64)
    calls = []

    def extractor(tokens, positions):
        calls.append((tokens.shape, tuple(positions)))
        base = tokens.sum(axis=1, dtype=np.float32)
        return np.stack([
            np.stack((base + position, base * 0 + position, base * 0 + 1), axis=1)
            for position in positions
        ], axis=1)

    cache = ExtractionCache(tmp_path / "cache", {"checkpoint": "one"})
    expected = populate_cache(
        cache, extractor=extractor, pair_rows=rows, val_npz=val,
        lengen_npz=lengen, chunk_rows=5_000,
    )
    assert sorted(expected) == sorted(planned_cache_keys(rows, 5_000))
    assert cache.receipt(expected_keys=expected)["n_chunks"] == len(expected)
    first_calls = len(calls)
    assert first_calls == len(expected)
    populate_cache(
        ExtractionCache(tmp_path / "cache", {"checkpoint": "one"}), extractor=extractor,
        pair_rows=rows, val_npz=val, lengen_npz=lengen, chunk_rows=5_000,
    )
    assert len(calls) == first_calls
