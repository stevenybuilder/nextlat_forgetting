from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import materialize_cfs1_banks as M  # noqa: E402
from cfs1 import validate as V  # noqa: E402
from cfs1.adaptation import validate_update_manifest  # noqa: E402


def _empty_legacy(_root: Path) -> V.LegacyIndex:
    return V.LegacyIndex(frozenset(), frozenset(), frozenset(), ())


def test_no_model_dry_run_materializes_nothing_and_reports_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(M.V, "build_legacy_index", _empty_legacy)
    result = M.materialize(root=ROOT, output_dir=tmp_path / "out", n_probes=8, n_updates=20, dry_run=True)
    assert result["status"] == "PASS"
    assert result["model_imported"] is False
    assert result["loss_or_pilot_selection_used"] is False
    assert not (tmp_path / "out").exists()


def test_materialization_binds_generator_retention_adaptation_and_global_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(M.V, "build_legacy_index", _empty_legacy)
    out = tmp_path / "cfs1"
    result = M.materialize(root=ROOT, output_dir=out, n_probes=8, n_updates=20, dry_run=False)
    assert result["status"] == "PASS"
    outer_manifest = json.loads((out / "cfs1_update_manifest.json").read_text())
    receipt = json.loads((out / "materialization_receipt.json").read_text())
    assert outer_manifest["schema"] == V.UPDATE_SCHEMA
    assert outer_manifest["status"] == "FROZEN"
    assert tuple(outer_manifest["evaluation_inputs"]) == (
        "margin", "retention_ce", "retention_exact_path", "global_controls",
        "state_drift", "pregeometry",
    )
    normalized = validate_update_manifest(out / "cfs1_update_manifest.json")
    assert set(normalized["episodes"]) == {0, 1}
    for episode in (0, 1):
        for arm in ("high_different", "low_different", "high_same", "low_same"):
            stream = out / normalized["episodes"][episode]["arms"][arm]["path"]
            assert stream.suffix == ".txt"
            assert len(stream.read_text().splitlines()) == 20
            assert not stream.read_text().lstrip().startswith("{")
    assert len(outer_manifest["branch_plan"]) == 64
    assert {branch["parent_id"] for branch in outer_manifest["branch_plan"]} == {
        "nextlat-seed1234-base", "nextlat-seed1235-base", "nextlat-seed1236-base", "nextlat-seed1237-base", "nextlat-seed1238-base",
        "nextlat-seed2234-cfs1-base", "nextlat-seed2235-cfs1-base", "nextlat-seed2236-cfs1-base",
    }
    assert {branch["episode"] for branch in outer_manifest["branch_plan"]} == {0, 1}
    for branch in outer_manifest["branch_plan"]:
        assert {"parent_id", "episode", "overlap", "future_relation"} <= set(branch)
    assert receipt["cfs1_update_manifest_sha256"] == V.sha256_file(out / "cfs1_update_manifest.json")
    assert receipt["retention_sha256"] == V.sha256_file(out / "retention.jsonl")
    assert set(receipt["adaptation_sha256"]) == {
        "updates_high_same.jsonl", "updates_high_different.jsonl",
        "updates_low_same.jsonl", "updates_low_different.jsonl",
    }
    assert M.materialize(root=ROOT, output_dir=out, n_probes=8, n_updates=20, dry_run=False)["status"] == "PASS"


def test_immutable_materialization_refuses_divergent_rewrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(M.V, "build_legacy_index", _empty_legacy)
    out = tmp_path / "cfs1"
    M.materialize(root=ROOT, output_dir=out, n_probes=8, n_updates=20, dry_run=False)
    with pytest.raises(V.CFS1ValidationError, match="immutable artifact"):
        M.materialize(root=ROOT, output_dir=out, n_probes=9, n_updates=22, dry_run=False)
