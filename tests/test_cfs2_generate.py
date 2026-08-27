from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_cfs2_banks as M  # noqa: E402
from cfs2 import generate as G  # noqa: E402
from cfs2 import validate as V  # noqa: E402


def _bundle():
    return G.build_bundle(n_probes=8, n_updates=20)


def test_g55_exact_equal_overlap_is_feasible_for_all_construction_variants() -> None:
    base = G._base_graph(0, 0)
    base_line = base.serialize()
    for occurrence in range(3):
        cells = G._unit_graphs(base, occurrence)
        measured = {
            condition: V.edge_overlap(base_line, graph.serialize())
            for condition, graph in cells.items()
        }
        assert measured == V.EXPECTED_OVERLAPS
        assert measured[("high", "same")] == measured[("high", "different")]
        assert measured[("low", "same")] == measured[("low", "different")]


def test_balanced_bundle_is_solver_valid_answer_balanced_and_two_episode() -> None:
    bundle = _bundle()
    V.validate_bundle(
        bundle.retention,
        bundle.updates,
        bundle.codebook,
        expected_probes=8,
        expected_updates=20,
        global_controls=bundle.global_controls,
        expected_global_controls=8,
    )
    assert len(bundle.retention) == 8
    assert len(bundle.global_controls) == 8
    assert [row["episode"] for row in bundle.codebook["episodes"]] == [0, 1]
    by_unit = {}
    for condition, rows in bundle.updates.items():
        for row in rows:
            by_unit.setdefault(row["unit_id"], {})[condition] = row
    assert len(by_unit) == 20
    for cells in by_unit.values():
        assert {
            condition: row["edge_overlap_with_probe"]
            for condition, row in cells.items()
        } == V.EXPECTED_OVERLAPS
        assert {
            condition: row["answer_edge_overlap_with_probe"]
            for condition, row in cells.items()
        } == {
            ("high", "same"): 4,
            ("low", "same"): 4,
            ("high", "different"): 3,
            ("low", "different"): 3,
        }
        assert {
            condition: row["nonanswer_edge_overlap_with_probe"]
            for condition, row in cells.items()
        } == {
            ("high", "same"): 14,
            ("low", "same"): 4,
            ("high", "different"): 15,
            ("low", "different"): 5,
        }
        assert (
            cells[("high", "same")]["answer_sha256"]
            == cells[("low", "same")]["answer_sha256"]
        )
        assert (
            cells[("high", "different")]["answer_sha256"]
            == cells[("low", "different")]["answer_sha256"]
        )
        assert (
            cells[("high", "same")]["answer_sha256"]
            != cells[("high", "different")]["answer_sha256"]
        )


def test_validator_rejects_the_cfs1_seven_edge_low_different_confound() -> None:
    bundle = _bundle()
    tampered = copy.deepcopy(bundle.updates)
    base = bundle.retention[0]
    victim = tampered[("low", "different")][0]
    unit_occurrence = int(victim["probe_occurrence"])
    partner = G._variants(unit_occurrence)[1]

    # Reconstruct the predecessor CFS-1 low/different mapping: its answer is
    # balanced, but it shares only seven edges with the retention probe.
    graph = G._base_graph(0, int(base["candidate_attempt"]), domain="cfs2-retention")
    groups = list(range(5))
    p0 = [partner] + [group for group in groups if group != partner]
    rest1 = G._derangement(
        [group for group in groups if group != partner],
        p0[1:],
        tag=("predecessor-low-different", unit_occurrence, 1),
    )
    p1 = [partner] + rest1
    rest2 = G._derangement(
        groups[1:], p1[1:], tag=("predecessor-low-different", unit_occurrence, 2)
    )
    p2 = [0] + rest2
    rest3 = G._derangement(
        groups[1:], p2[1:], tag=("predecessor-low-different", unit_occurrence, 3)
    )
    p3 = [0] + rest3
    positions = [p0, p1, p2, p3]
    old_graph = G._graph_with_arms(
        graph,
        [
            [graph.arms[positions[depth][arm]][depth] for depth in range(4)]
            for arm in groups
        ],
    )
    old_line = old_graph.serialize()
    old_witness = V.line_witness(old_line)
    assert V.edge_overlap(base["line"], old_line) == 7
    for key in ("line", "prompt_sha256", "graph_key", "answer_sha256"):
        victim[key] = old_line if key == "line" else old_witness[key]
    victim["edge_overlap_with_probe"] = 7
    answer_overlap, nonanswer_overlap = V.overlap_decomposition(base["line"], old_line)
    victim["answer_edge_overlap_with_probe"] = answer_overlap
    victim["nonanswer_edge_overlap_with_probe"] = nonanswer_overlap
    victim["future_same_as_probe"] = False
    with pytest.raises(V.CFS2ValidationError, match="unexpected overlap"):
        V.validate_bundle(
            bundle.retention,
            tampered,
            bundle.codebook,
            expected_probes=8,
            expected_updates=20,
            global_controls=bundle.global_controls,
            expected_global_controls=8,
        )


def test_legacy_collision_causes_deterministic_retry_without_model_inputs() -> None:
    first = G._base_graph(0, 0, domain="cfs2-retention").serialize()
    legacy = V.LegacyIndex(
        prompt_hashes=frozenset({V.prompt_sha256(first)}),
        graph_keys=frozenset({V.line_witness(first)["graph_key"]}),
        identifiers=frozenset(),
        sources=("positive-control",),
    )
    bundle = G.build_bundle(n_probes=8, n_updates=20, legacy=legacy)
    assert bundle.retention[0]["candidate_attempt"] > 0
    V.validate_bundle(
        bundle.retention,
        bundle.updates,
        bundle.codebook,
        expected_probes=8,
        expected_updates=20,
        global_controls=bundle.global_controls,
        expected_global_controls=8,
        legacy=legacy,
    )


def test_codebook_preserves_two_or_three_reuse_and_two_fixed_episodes() -> None:
    codebook = G.make_codebook(8, 20)
    counts = {}
    for unit in codebook["units"]:
        counts[unit["probe_index"]] = counts.get(unit["probe_index"], 0) + 1
    assert sorted(counts.values()) == [2, 2, 2, 2, 3, 3, 3, 3]
    assert [entry["episode"] for entry in codebook["episodes"]] == [0, 1]
    assert all(
        set(entry["unit_order"]) == set(codebook["unit_order"])
        for entry in codebook["episodes"]
    )


def test_materializer_writes_four_banks_two_stream_episodes_and_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        M.V,
        "build_legacy_index",
        lambda _root: V.LegacyIndex(frozenset(), frozenset(), frozenset(), ()),
    )
    output = tmp_path / "cfs2"
    result = M.materialize(
        root=ROOT,
        output_dir=output,
        n_probes=8,
        n_updates=20,
        dry_run=False,
    )
    assert result["status"] == "PASS"
    receipt = json.loads((output / "construction_receipt.json").read_text())
    assert receipt["factorial_contract"]["within_overlap_future_relation_exactly_balanced"]
    assert receipt["counts"] == {
        "retention": 8,
        "global_controls": 8,
        "each_update_bank": 20,
        "episodes": 2,
    }
    assert len(list(output.glob("updates_*.jsonl"))) == 4
    assert len(list((output / "streams").glob("*.txt"))) == 8
    assert all(
        len(path.read_text().splitlines()) == 20
        for path in (output / "streams").glob("*.txt")
    )
    # Identical replay is accepted; divergent bytes are never overwritten.
    assert M.materialize(
        root=ROOT,
        output_dir=output,
        n_probes=8,
        n_updates=20,
        dry_run=False,
    )["status"] == "PASS"
