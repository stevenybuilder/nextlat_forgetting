from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfs1 import generate as G  # noqa: E402
from cfs1 import validate as V  # noqa: E402


def _bundle():
    return G.build_bundle(n_probes=8, n_updates=20)


def test_model_blind_generator_builds_complete_balanced_solver_valid_2x2_bundle() -> None:
    bundle = _bundle()
    retention, updates = bundle.retention, bundle.updates
    global_controls, codebook = bundle.global_controls, bundle.codebook
    assert len(retention) == 8
    assert set(updates) == set(V.CONDITIONS)
    assert all(len(rows) == 20 for rows in updates.values())
    assert codebook["schema"] == "nextlat_forgetting/cfs1_hash_codebook/1"
    assert len(codebook["episodes"]) == 2
    V.validate_bundle(retention, updates, codebook, expected_probes=8, expected_updates=20, global_controls=global_controls, expected_global_controls=8)
    assert len(global_controls) == 8
    by_unit = {}
    for condition, rows in updates.items():
        for row in rows:
            by_unit.setdefault(row["unit_id"], {})[condition] = row
    assert len(by_unit) == 20
    for cells in by_unit.values():
        assert cells[("high", "same")]["answer_sha256"] == cells[("low", "same")]["answer_sha256"]
        assert cells[("high", "different")]["answer_sha256"] == cells[("low", "different")]["answer_sha256"]
        assert cells[("high", "same")]["answer_sha256"] != cells[("high", "different")]["answer_sha256"]


def test_exact_required_overlap_is_verified_not_trusted_from_bookkeeping() -> None:
    bundle = _bundle()
    retention, updates = bundle.retention, bundle.updates
    global_controls, codebook = bundle.global_controls, bundle.codebook
    tampered = {key: [dict(row) for row in rows] for key, rows in updates.items()}
    tampered[("low", "different")][0]["edge_overlap_with_probe"] = 999
    with pytest.raises(V.CFS1ValidationError, match="wrong edge overlap"):
        V.validate_bundle(retention, tampered, codebook, expected_probes=8, expected_updates=20, global_controls=global_controls, expected_global_controls=8)


def test_duplicate_graph_identity_is_refused_across_condition_banks() -> None:
    bundle = _bundle()
    retention, updates = bundle.retention, bundle.updates
    global_controls, codebook = bundle.global_controls, bundle.codebook
    tampered = copy.deepcopy(updates)
    victim = tampered[("low", "same")][0]
    source = tampered[("high", "same")][0]
    for field in ("line", "prompt_sha256", "graph_key", "answer_sha256"):
        victim[field] = source[field]
    victim["edge_overlap_with_probe"] = source["edge_overlap_with_probe"]
    victim["future_same_as_probe"] = source["future_same_as_probe"]
    with pytest.raises(V.CFS1ValidationError, match="duplicates a CFS-1 graph/prompt identity|unexpected overlap"):
        V.validate_bundle(retention, tampered, codebook, expected_probes=8, expected_updates=20, global_controls=global_controls, expected_global_controls=8)


def test_legacy_or_corpus_collision_is_rejected_by_deterministic_candidate_retry() -> None:
    first = G._base_graph(0, 0).serialize()
    legacy = V.LegacyIndex(
        prompt_hashes=frozenset({V.prompt_sha256(first)}),
        graph_keys=frozenset({V.line_witness(first)["graph_key"]}),
        identifiers=frozenset(), sources=("positive-control",),
    )
    bundle = G.build_bundle(n_probes=8, n_updates=20, legacy=legacy)
    retention, updates = bundle.retention, bundle.updates
    global_controls, codebook = bundle.global_controls, bundle.codebook
    assert retention[0]["candidate_attempt"] > 0
    V.validate_bundle(retention, updates, codebook, expected_probes=8, expected_updates=20, global_controls=global_controls, expected_global_controls=8, legacy=legacy)


def test_codebook_has_exact_two_or_three_probe_reuse_and_two_fixed_episodes() -> None:
    codebook = G.make_codebook(8, 20)
    counts = {}
    for unit in codebook["units"]:
        counts[unit["probe_index"]] = counts.get(unit["probe_index"], 0) + 1
    assert sorted(counts.values()) == [2, 2, 2, 2, 3, 3, 3, 3]
    assert [entry["episode"] for entry in codebook["episodes"]] == [0, 1]
    assert all(set(entry["unit_order"]) == set(codebook["unit_order"]) for entry in codebook["episodes"])
