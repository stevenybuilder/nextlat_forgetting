from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_cfs2_banks as M  # noqa: E402
from cfs2 import validate as V  # noqa: E402
from cfs2.adaptation import canonical_json_sha256, sha256_file, validate_update_manifest  # noqa: E402
from lurestar.durable_checkpoint import DurableCheckpointer, pickle_serializer  # noqa: E402

SER = pickle_serializer()


def _module():
    spec = importlib.util.spec_from_file_location("run_cfs2_matrix_tested", ROOT / "scripts/run_cfs2_matrix.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _empty_legacy(_root: Path) -> V.LegacyIndex:
    return V.LegacyIndex(frozenset(), frozenset(), frozenset(), ())


def _manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(M.V, "build_legacy_index", _empty_legacy)
    output = tmp_path / "inputs"
    M.materialize(root=ROOT, output_dir=output, n_probes=8, n_updates=20, dry_run=False)
    return output / "cfs2_update_manifest.json"


def _parents(tmp_path: Path, module, *, aliases: bool) -> tuple[Path, Path | None]:
    entries = []
    for seed in (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236):
        source_id = f"nextlat-seed{seed}" + ("-base" if seed < 2000 else "-cfs1-base")
        checkpoint = DurableCheckpointer(tmp_path / f"parent{seed}", f"parent{seed}", serializer=SER).save(
            {"training_steps": 20_000, "model": {}, "optimizer": {}, "lurestar_rng_state_v1": {"schema": 1}}, 20_000
        )
        entries.append({"job_id": source_id, "model": "nextlat", "status": "TRAINED", "step": 20_000,
                        "final_checkpoint": checkpoint.path, "final_checkpoint_sha256": checkpoint.sha256})
    ledger = tmp_path / "parents.json"
    ledger.write_text(json.dumps({"schema": "test", "entries": entries}))
    if not aliases:
        return ledger, None
    by_id = {entry["job_id"]: entry for entry in entries}
    aliases_rows = []
    for seed in (2234, 2235, 2236):
        source_id = f"nextlat-seed{seed}-cfs1-base"
        source = by_id[source_id]
        aliases_rows.append({
            "seed": seed, "canonical_parent_id": module.parent_id_for_seed(seed), "source_parent_id": source_id,
            "source_ledger_entry_sha256": canonical_json_sha256(source),
            "parent_checkpoint": {"path": source["final_checkpoint"], "sha256": source["final_checkpoint_sha256"]},
        })
    receipt = tmp_path / "cfs2-parent-lineage.json"
    receipt.write_text(json.dumps({
        "schema": module.CFS2_PARENT_LINEAGE_SCHEMA, "status": "FROZEN",
        "source_parent_ledger": {"path": str(ledger), "sha256": sha256_file(ledger)}, "aliases": aliases_rows,
    }))
    return ledger, receipt


class _Launcher:
    def __init__(self, module, *, interrupt_once: bool = False):
        self.module, self.interrupt_once, self.calls = module, interrupt_once, []

    def __call__(self, plan):
        self.calls.append(plan.spec.job_id)
        checkpoint = DurableCheckpointer(plan.spec.out_root, plan.spec.job_id, experiment_name=plan.spec.experiment_name, serializer=SER)
        if self.interrupt_once:
            self.interrupt_once = False
            checkpoint.save({"training_steps": plan.parent.step + 50}, plan.parent.step + 50)
            return self.module.LaunchResult(137, "simulated disconnect")
        checkpoint.save({"training_steps": plan.target_step}, plan.target_step)
        experiment = Path(plan.spec.experiment_dir)
        (experiment / "version_0").mkdir(parents=True, exist_ok=True)
        (experiment / "materialized_config.yaml").write_text("frozen: true\n")
        (experiment / "version_0" / "metrics.csv").write_text("step,loss\n20500,1\n")
        return self.module.LaunchResult(0, "ok")


def test_materialized_cfs2_envelope_binds_repaired_stimuli_and_state_commitment(tmp_path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path, monkeypatch)
    normalized = validate_update_manifest(manifest_path)
    assert set(normalized["episodes"]) == {0, 1}
    assert {arm["sha256"] for episode in normalized["episodes"].values() for arm in episode["arms"].values()}
    raw = json.loads(manifest_path.read_text())
    assert raw["construction"]["exact_total_edge_overlap"] == {
        "high_same": 18, "high_different": 18, "low_same": 8, "low_different": 8,
    }
    assert raw["state_interchange_activation_patching"]["path"].startswith("state_interchange")
    assert len(raw["branch_plan"]) == 64


def test_cfs_only_parent_alias_is_refused_without_hash_bound_lineage_receipt(tmp_path, monkeypatch) -> None:
    module = _module(); _manifest(tmp_path, monkeypatch)
    ledger, _ = _parents(tmp_path, module, aliases=False)
    with pytest.raises(module.CFS2MatrixError, match="lineage receipt"):
        module.load_parents(ledger)
    ledger, receipt = _parents(tmp_path / "with-receipt", module, aliases=True)
    parents = module.load_parents(ledger, lineage_receipt=receipt)
    assert set(parents) == {module.parent_id_for_seed(seed) for seed in range(1234, 1239)} | {
        module.parent_id_for_seed(seed) for seed in range(2234, 2237)
    }
    assert parents[module.parent_id_for_seed(2234)].source_parent_id.endswith("-cfs1-base")
    assert parents[module.parent_id_for_seed(2234)].lineage_receipt_sha256 == sha256_file(receipt)


def test_cfs2_crash_recovery_exact_clone_parity_and_atomic_64_branch_preflight(tmp_path, monkeypatch) -> None:
    module = _module(); manifest_path = _manifest(tmp_path, monkeypatch)
    jobs, manifest = module.build_cfs2_matrix(tmp_path / "root", manifest_path)
    parent_ledger, receipt = _parents(tmp_path / "parents", module, aliases=True)
    parents = module.load_parents(parent_ledger, lineage_receipt=receipt)
    ledger = module.CFS2Ledger(tmp_path / "root" / "cfs2_run_ledger.json")
    states = module.CFS2Runner(ledger, _Launcher(module, interrupt_once=True), serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert sum(value["status"] == module.CFS2_INTERRUPTED for value in states.values()) == 1
    complete = _Launcher(module)
    states = module.CFS2Runner(ledger, complete, serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert all(value["status"] == module.CFS2_TRAINED and value["updates"] == 500 for value in states.values())
    calls_before = len(complete.calls)
    module.CFS2Runner(ledger, complete, serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert len(complete.calls) == calls_before
    readiness = json.loads(module.preflight_cfs2_evaluation(tmp_path / "root", jobs, ledger, manifest).read_text())
    assert readiness["status"] == "ALL_64_BRANCHES_TRAINED" and len(readiness["branches"]) == 64
    assert readiness["state_interchange_activation_patching"] == manifest["state_interchange_activation_patching"]


def test_tampered_cfs2_stream_refuses_before_any_paid_branch_launch(tmp_path, monkeypatch) -> None:
    module = _module(); manifest_path = _manifest(tmp_path, monkeypatch)
    jobs, manifest = module.build_cfs2_matrix(tmp_path / "root", manifest_path)
    Path(jobs[0].update_bank).write_text("tampered")
    parent_ledger, receipt = _parents(tmp_path / "parents", module, aliases=True)
    launcher = _Launcher(module)
    with pytest.raises(module.CFS2MatrixError, match="update-stream hash changed"):
        module.CFS2Runner(module.CFS2Ledger(tmp_path / "ledger.json"), launcher, serializer=SER, echo=lambda _: None).run(
            jobs, module.load_parents(parent_ledger, lineage_receipt=receipt), manifest
        )
    assert launcher.calls == []
