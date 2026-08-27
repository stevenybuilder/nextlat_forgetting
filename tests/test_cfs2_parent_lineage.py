from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_cfs2_parent_lineage as L  # noqa: E402
import run_cfs2_parents as P  # noqa: E402
from cfs2.adaptation import sha256_file  # noqa: E402


def _matrix_module():
    spec = importlib.util.spec_from_file_location("run_cfs2_matrix_lineage_tested", ROOT / "scripts/run_cfs2_matrix.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({
            "training_steps": 20_000,
            "model": {"weight": 1},
            "optimizer": {"state": {}},
            "lurestar_rng_state_v1": {"schema": 1},
        }, handle)
    return str(path.resolve()), sha256_file(path)


def _entry(tmp_path: Path, seed: int) -> dict:
    path, digest = _checkpoint(tmp_path / f"seed{seed}" / "ckpt_iter_20000.pt")
    return {
        "job_id": L.expected_source_id(seed), "status": "TRAINED",
        "model": "nextlat", "seed": seed, "phase": "base",
        "step": 20_000, "updates": 20_000,
        "final_checkpoint": path, "final_checkpoint_sha256": digest,
    }


def _ledger(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "entries": entries}), encoding="utf-8")
    return path


def test_cfs2_parent_plan_is_exact_nextlat_20k_and_only_three_frozen_seeds(tmp_path: Path) -> None:
    jobs = P.build_cfs2_parent_jobs(tmp_path, competence_dataset=ROOT / "data/stargraph/graph_5_5_test_20000.txt")
    assert [job.seed for job in jobs] == [2234, 2235, 2236]
    assert [job.job_id for job in jobs] == [
        "nextlat-s2234-cfs2-base", "nextlat-s2235-cfs2-base", "nextlat-s2236-cfs2-base",
    ]
    assert all(job.model == "nextlat" and job.phase == "base" and job.train_batches == 20_000 for job in jobs)
    assert len({Path(job.out_root).resolve() for job in jobs}) == 3
    with pytest.raises(P.CFS2ParentError, match="frozen CFS-2-only roster"):
        P.source_parent_id(1234)
    with pytest.raises(P.CFS2ParentError, match="unique subset"):
        P.build_cfs2_parent_jobs(tmp_path, seeds=(2234, 2234), competence_dataset=ROOT / "data/stargraph/graph_5_5_test_20000.txt")


def test_cfs2_parent_launcher_emits_exact_scratch_target(tmp_path: Path) -> None:
    job = P.build_cfs2_parent_jobs(
        tmp_path, seeds=(2234,),
        competence_dataset=ROOT / "data/stargraph/graph_5_5_test_20000.txt",
    )[0]
    from run_matrix import FabricLauncher, ResumePlan

    command = FabricLauncher(tmp_path / "upstream", dry_run=True).command(ResumePlan(job, True, 0))
    assert "seed=2234" in command
    assert "trainer.train_batches=20000" in command
    assert "trainer.init_from=scratch" in command
    assert "--checkpoint_path" not in command


def test_materialized_v2_receipt_loads_all_eight_real_source_ids(tmp_path: Path) -> None:
    main = [_entry(tmp_path, seed) for seed in range(1234, 1239)]
    cfs2 = [_entry(tmp_path, seed) for seed in range(2234, 2237)]
    ledger_a = _ledger(tmp_path / "worker-a.json", main[:3])
    ledger_b = _ledger(tmp_path / "worker-b.json", main[3:] + cfs2)
    parent_ledger, receipt = L.materialize_lineage([ledger_a, ledger_b], output_dir=tmp_path / "frozen")

    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["schema"] == "nextlat_forgetting/cfs2_parent_lineage_receipt/2"
    assert len(document["parents"]) == 8
    assert {row["source_parent_id"] for row in document["parents"]} == {
        *(f"nextlat-s{seed}-base" for seed in range(1234, 1239)),
        *(f"nextlat-s{seed}-cfs2-base" for seed in range(2234, 2237)),
    }
    module = _matrix_module()
    parents = module.load_parents(parent_ledger, lineage_receipt=receipt)
    assert len(parents) == 8
    assert parents[module.parent_id_for_seed(1234)].source_parent_id == "nextlat-s1234-base"
    assert parents[module.parent_id_for_seed(2234)].source_parent_id == "nextlat-s2234-cfs2-base"
    assert len({parent.lineage_receipt_sha256 for parent in parents.values()}) == 1


def test_lineage_refuses_duplicate_sources_and_stale_checkpoint(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, seed) for seed in L.ALL_PARENT_SEEDS]
    one = _ledger(tmp_path / "one.json", entries)
    duplicate = _ledger(tmp_path / "duplicate.json", [entries[0]])
    with pytest.raises(L.CFS2LineageError, match="exactly one latest source"):
        L.materialize_lineage([one, duplicate], output_dir=tmp_path / "duplicate-out")

    Path(entries[-1]["final_checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(L.CFS2LineageError, match="missing or stale"):
        L.materialize_lineage([one], output_dir=tmp_path / "stale-out")
