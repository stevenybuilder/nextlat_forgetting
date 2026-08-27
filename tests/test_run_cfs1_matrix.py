from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cfs1.adaptation import CFS1_ARMS, CFS1_EPISODES
from lurestar.durable_checkpoint import DurableCheckpointer, pickle_serializer, sha256_file


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cfs1_matrix.py"
SER = pickle_serializer()


def _module():
    spec = importlib.util.spec_from_file_location("run_cfs1_matrix_tested", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(root: Path, name: str, payload: str = "frozen") -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return {"path": name, "sha256": sha256_file(path)}


def _manifest(root: Path) -> Path:
    construction = _artifact(root, "receipts/construction.json")
    generator = _artifact(root, "receipts/generator.json")
    generator_manifest = _artifact(root, "manifests/global.json")
    retention = _artifact(root, "manifests/retention.jsonl")
    global_controls = _artifact(root, "manifests/global_controls.jsonl")
    probes = {name: _artifact(root, f"probes/{name}.jsonl") for name in (
        "margin", "retention_ce", "retention_exact_path", "global_controls", "state_drift",
        "pregeometry")}
    episodes = []
    for episode in CFS1_EPISODES:
        arms = {}
        for arm in CFS1_ARMS:
            overlap, relation = arm.split("_", 1)
            arms[arm] = {**_artifact(root, f"updates/graph_5_5_cfs1_episode{episode}_{arm}.txt", arm),
                         "overlap": overlap, "future_relation": relation}
        episodes.append({"episode": episode,
                         "episode_sha256": ("a" if episode == 0 else "b") * 64, "arms": arms})
    path = root / "cfs1_update_manifest.json"
    path.write_text(json.dumps({
        "schema": "nextlat_forgetting/cfs1_update_manifest/1", "status": "FROZEN",
        "construction": {"model_outcomes_inspected": False, "training_outcomes_inspected": False,
                         "retention_outcomes_inspected": False, "matching": "construction_matched",
                         "randomized_assignment": True, "receipt": construction},
        "generator_receipt": generator,
        "generator_manifest": generator_manifest,
        "retention_probes": retention,
        "global_control_manifest": global_controls,
        "design": {"model": "nextlat", "adaptation_steps": 500, "full_parameter": True,
                   "loss": "teacher_forced_next_token_cross_entropy", "arms": list(CFS1_ARMS),
                   "episodes": list(CFS1_EPISODES)},
        "execution_order_algorithm": "sha256-sort-v1", "execution_order_salt_sha256": "c" * 64,
        "evaluation_inputs": probes, "episodes": episodes,
    }))
    # Synthetic fixture clearance exists only so the retired runner's durability
    # mechanics remain testable. Production CFS-1 has a hash-bound BLOCKED marker.
    disposition = root / "STIMULUS_VALIDITY_BLOCK.json"
    disposition.write_text(json.dumps({
        "schema": "nextlat_forgetting/cfs1_stimulus_validity_block/1",
        "status": "CLEARED_FOR_CONFIRMATORY_LAUNCH",
        "fixture_only": True,
    }))
    disposition.with_name(disposition.name + ".sha256").write_text(
        f"{sha256_file(disposition)}  {disposition.name}\n")
    return path


def _config(root: Path) -> Path:
    path = root / "cfs1.yaml"
    path.write_text("""
use_nextlat: true
use_bst: false
trainer: {train_batches: 500}
data:
  stargraph_train_data_path: /content/cfs1/UNBOUND_CFS1_UPDATE_BANK_FORBIDDEN.txt
  stargraph_test_data_path: /content/cfs1/UNBOUND_CFS1_UPDATE_BANK_FORBIDDEN.txt
model: {lambda_mse: 0.0, lambda_kl: 0.0, lambda_ce: 0.0}
cfs1:
  adaptation_steps: 500
  unbound_bank_sentinel: /content/cfs1/UNBOUND_CFS1_UPDATE_BANK_FORBIDDEN.txt
""")
    return path


def _parents(root: Path, module) -> Path:
    entries = []
    for seed in (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236):
        ck = DurableCheckpointer(root / f"parent{seed}", f"parent{seed}", serializer=SER)
        record = ck.save({"training_steps": 20_000, "model": {}, "optimizer": {},
                          "lurestar_rng_state_v1": {"schema": 1}}, 20_000)
        entries.append({"job_id": module.parent_id_for_seed(seed), "status": "TRAINED",
                        "model": "nextlat", "step": 20_000, "final_checkpoint": record.path,
                        "final_checkpoint_sha256": record.sha256})
    path = root / "parents.json"
    path.write_text(json.dumps({"schema": "test", "entries": entries}))
    return path


class _Launcher:
    def __init__(self, module, *, interrupt_once: bool = False):
        self.module, self.calls, self.interrupt_once = module, [], interrupt_once

    def __call__(self, plan):
        self.calls.append(plan.spec.job_id)
        target = plan.target_step
        checkpoint = DurableCheckpointer(plan.spec.out_root, plan.spec.job_id,
                                         experiment_name=plan.spec.experiment_name, serializer=SER)
        if self.interrupt_once:
            self.interrupt_once = False
            checkpoint.save({"training_steps": plan.parent.step + 50}, plan.parent.step + 50)
            return self.module.LaunchResult(137, "simulated disconnect")
        checkpoint.save({"training_steps": target}, target)
        experiment = Path(plan.spec.experiment_dir)
        (experiment / "version_0").mkdir(parents=True, exist_ok=True)
        (experiment / "materialized_config.yaml").write_text("frozen: true\n")
        (experiment / "version_0" / "metrics.csv").write_text("step,loss\n20500,1\n")
        return self.module.LaunchResult(0, "ok")


def test_exact_matrix_is_64_isolated_hash_randomized_branches(tmp_path: Path) -> None:
    module = _module()
    jobs, manifest = module.build_cfs1_matrix(tmp_path / "runs", _manifest(tmp_path / "inputs"),
                                               config=_config(tmp_path))
    assert len(jobs) == 64
    assert len({job.out_root for job in jobs}) == 64
    assert jobs[0].parent_id == "nextlat-seed1234-base"
    assert module.parent_id_for_seed(2234) == "nextlat-seed2234-cfs1-base"
    assert {job.update_bank_sha256 for job in jobs if job.episode == 0 and
            job.arm == "high_different"} == {jobs[0].update_bank_sha256}
    assert len(module.cfs1_branch_order(manifest, [job.job_id for job in jobs])) == 64


def test_production_adjacent_stimulus_block_refuses_matrix_before_launch(tmp_path: Path) -> None:
    module = _module()
    inputs = tmp_path / "inputs"
    manifest_path = _manifest(inputs)
    block_path = inputs / module.CFS1_STIMULUS_BLOCK_FILENAME
    block_path.write_text(json.dumps({
        "schema": module.CFS1_STIMULUS_BLOCK_SCHEMA,
        "status": "BLOCKED_FOR_CONFIRMATORY_CAUSAL_LAUNCH",
        "reason": {"summary": "outcome-blind overlap imbalance"},
    }))
    block_path.with_name(block_path.name + ".sha256").write_text(
        f"{sha256_file(block_path)}  {block_path.name}\n")
    with pytest.raises(module.CFS1MatrixError, match="confirmatory launch is blocked"):
        module.build_cfs1_matrix(tmp_path / "runs", manifest_path, config=_config(tmp_path))


def test_missing_stimulus_disposition_also_fails_closed(tmp_path: Path) -> None:
    module = _module()
    inputs = tmp_path / "inputs"
    manifest_path = _manifest(inputs)
    (inputs / module.CFS1_STIMULUS_BLOCK_FILENAME).unlink()
    (inputs / f"{module.CFS1_STIMULUS_BLOCK_FILENAME}.sha256").unlink()
    with pytest.raises(module.CFS1MatrixError, match="disposition is missing"):
        module.build_cfs1_matrix(tmp_path / "runs", manifest_path, config=_config(tmp_path))


def test_crash_recovery_exact_clone_parity_duplicate_noop_and_atomic_preflight(tmp_path: Path) -> None:
    module = _module()
    inputs = tmp_path / "inputs"
    jobs, manifest = module.build_cfs1_matrix(tmp_path / "root", _manifest(inputs),
                                               config=_config(tmp_path))
    parents = module.load_parents(_parents(tmp_path, module))
    ledger = module.CFS1Ledger(tmp_path / "root" / "cfs1_run_ledger.json")
    interrupted = _Launcher(module, interrupt_once=True)
    states = module.CFS1Runner(ledger, interrupted, serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert sum(state["status"] == module.CFS1_INTERRUPTED for state in states.values()) == 1
    completed = _Launcher(module)
    states = module.CFS1Runner(ledger, completed, serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert all(state["status"] == module.CFS1_TRAINED for state in states.values())
    assert all(state["updates"] == 500 and state["step"] == 20_500 for state in states.values())
    assert all(state["parent_checkpoint_sha256"] == parents[job.parent_id].sha256
               for job_id, state in states.items() for job in jobs if job.job_id == job_id)
    calls_before = len(completed.calls)
    module.CFS1Runner(ledger, completed, serializer=SER, echo=lambda _: None).run(jobs, parents, manifest)
    assert len(completed.calls) == calls_before
    readiness = module.preflight_cfs1_evaluation(tmp_path / "root", jobs, ledger, manifest)
    receipt = json.loads(readiness.read_text())
    assert receipt["status"] == "ALL_BRANCHES_TRAINED" and len(receipt["branches"]) == 64


def test_refuses_tampered_update_input_before_any_branch_launch(tmp_path: Path) -> None:
    module = _module()
    inputs = tmp_path / "inputs"
    jobs, manifest = module.build_cfs1_matrix(tmp_path / "root", _manifest(inputs),
                                               config=_config(tmp_path))
    Path(jobs[0].update_bank).write_text("tampered")
    parents = module.load_parents(_parents(tmp_path, module))
    launcher = _Launcher(module)
    with pytest.raises(module.CFS1MatrixError, match="update bank hash changed"):
        module.CFS1Runner(module.CFS1Ledger(tmp_path / "ledger.json"), launcher, serializer=SER,
                           echo=lambda _: None).run(jobs, parents, manifest)
    assert launcher.calls == []
