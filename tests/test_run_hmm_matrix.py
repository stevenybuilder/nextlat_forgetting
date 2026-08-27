"""HMM confirmatory matrix: frozen plan, exact updates, recovery, and DONE gate."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest
import numpy as np

from lurestar.durable_checkpoint import DurableCheckpointer, pickle_serializer, sha256_file
from run_hmm_matrix import (
    HMM_CHECKPOINT_INTERVAL,
    HMM_EVALUATION_RECEIPT,
    HMM_EVALUATION_SCHEMA,
    HMM_EVALUATION_SIDECAR,
    HMM_FAMILY_REGIMES,
    HMM_REQUIRED_INVENTORY,
    HMM_REQUIRED_METRICS,
    HMM_RECEIPT_MANIFEST_NAMES,
    HMM_TRAIN_UPDATES,
    HMMFabricLauncher,
    HMMMatrixError,
    build_hmm_matrix,
    hmm_job_id,
    hmm_source_inputs,
    hmm_evaluator_command,
    load_runtime_recovery_barrier,
    preflight_hmm_evaluation_matrix,
    promote_hmm_evaluations,
    run_hmm_evaluators,
    _selected_jobs,
    verify_hmm_family_snapshot,
    verify_hmm_snapshot,
    main,
)
from run_matrix import DONE, FAILED, INTERRUPTED, TRAINED, JobSpec, LaunchResult, Ledger, MatrixRunner
from hmm_geometry.extraction_cache import ExtractionCache

SER = pickle_serializer()


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    for model in ("gpt", "nextlat"):
        (project / "configs" / f"{model}_hmm.yaml").write_text(
            "trainer:\n  train_batches: 3000\n", encoding="utf-8"
        )
    evaluator = project / "src" / "hmm_geometry" / "evaluate.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# frozen estimator\n", encoding="utf-8")
    extraction = project / "src" / "hmm_geometry" / "extraction_cache.py"
    extraction.write_text("# frozen extraction cache\n", encoding="utf-8")
    runner = project / "scripts" / "evaluate_hmm_checkpoints.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# frozen evaluator\n", encoding="utf-8")
    for name in HMM_RECEIPT_MANIFEST_NAMES:
        path = project / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {name}\n", encoding="utf-8")
    return project


def _jobs(tmp_path: pathlib.Path, **kwargs) -> list[JobSpec]:
    project = _project(tmp_path)
    identity = project / "identity.txt"
    identity.write_text("frozen\n", encoding="utf-8")
    inputs = [
        identity,
        project / "scripts" / "evaluate_hmm_checkpoints.py",
        project / "src" / "hmm_geometry" / "extraction_cache.py",
    ]
    inputs.extend(project / "manifests" / name for name in HMM_RECEIPT_MANIFEST_NAMES)
    for name in ("hmm4x4_val_posteriors.npz", "hmm4x4_lengen_posteriors.npz"):
        path = project / "data/hmm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen {name}\n".encode())
        inputs.append(path)
    return build_hmm_matrix(
        tmp_path / "durable", project_root=project,
        identity_inputs=tuple(str(path) for path in inputs), **kwargs,
    )


class FakeHMMLauncher:
    def __init__(self, *, stop_at: int | None = None, final_step: int | None = None):
        self.stop_at = stop_at
        self.final_step = final_step
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        spec = plan.spec
        checkpointer = DurableCheckpointer(
            spec.out_root, spec.job_id, experiment_name=spec.experiment_dir_name,
            serializer=SER,
        )
        start = plan.resume_step
        target = self.final_step if self.final_step is not None else spec.train_batches
        stop = self.stop_at if self.stop_at is not None and start < self.stop_at else target
        step = start
        while step < min(stop, target):
            step += 1
            if step % HMM_CHECKPOINT_INTERVAL == 0:
                checkpointer.save({"step": step, "job": spec.job_id}, step)
        if self.stop_at is not None and step == self.stop_at and step < target:
            self.stop_at = None
            return LaunchResult(137, step, "simulated disconnect with useful checkpoint")
        checkpointer.save({"step": target, "job": spec.job_id}, target, kind="final")
        experiment = pathlib.Path(spec.checkpoint_dir)
        (experiment / "version_0").mkdir(parents=True, exist_ok=True)
        (experiment / "materialized_config.yaml").write_text("frozen: true\n")
        (experiment / "version_0" / "metrics.csv").write_text(
            f"step,train_loss\n{target},0.1\n"
        )
        contract = pathlib.Path(spec.out_root) / "metrics" / "step_0_contract.json"
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text("{}\n")
        return LaunchResult(0, target, "ok")


def _runner(tmp_path: pathlib.Path, launcher) -> tuple[Ledger, MatrixRunner]:
    ledger = Ledger(tmp_path / "hmm_ledger.json")
    return ledger, MatrixRunner(ledger, launcher, serializer=SER, echo=lambda _: None)


def _canonical_trained_family(
        tmp_path: pathlib.Path) -> tuple[list[JobSpec], Ledger, dict[str, dict]]:
    project = _project(tmp_path)
    manifest = project / "family-identity.json"
    manifest.write_text('{"frozen": true}\n')
    ledger = Ledger(tmp_path / "family-ledger.json")
    jobs: list[JobSpec] = []
    states: dict[str, dict] = {}
    for regime in HMM_FAMILY_REGIMES:
        for model in ("gpt", "nextlat"):
            for seed in range(1234, 1239):
                job_id = hmm_job_id(model, seed, regime)
                out = tmp_path / "family-runs" / regime / model / f"seed{seed}"
                out.mkdir(parents=True)
                checkpoint = out / "ckpt_iter_3000_1.0.pt"
                checkpoint.write_bytes(f"{job_id} checkpoint".encode())
                training_artifact = out / "materialized_config.yaml"
                training_artifact.write_text("frozen: true\n")
                training_artifacts = {
                    training_artifact.name: sha256_file(training_artifact),
                }
                summary = out / "final_summary.json"
                summary.write_text(json.dumps({
                    "schema": "nextlat_forgetting/training_completion/1",
                    "kind": "training_completion",
                    "job_id": job_id,
                    "model": model,
                    "seed": seed,
                    "phase": "hmm",
                    "condition": regime,
                    "step": HMM_TRAIN_UPDATES,
                    "updates": HMM_TRAIN_UPDATES,
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": sha256_file(checkpoint),
                    },
                    "training_artifacts": training_artifacts,
                }, sort_keys=True) + "\n")
                config = project / "configs" / f"{model}_hmm.yaml"
                job = JobSpec(
                    job_id=job_id, model=model, seed=seed, phase="hmm",
                    condition=regime, config=str(config), out_root=str(out),
                    manifests=(str(manifest),), train_batches=HMM_TRAIN_UPDATES,
                )
                jobs.append(job)
                state = ledger.append({
                    "job_id": job_id,
                    "status": TRAINED,
                    "step": HMM_TRAIN_UPDATES,
                    "updates": HMM_TRAIN_UPDATES,
                    "model": model,
                    "seed": seed,
                    "phase": "hmm",
                    "condition": regime,
                    "out_root": str(out.resolve()),
                    "config_sha256": sha256_file(config),
                    "manifest_sha256": {str(manifest): sha256_file(manifest)},
                    "final_checkpoint": str(checkpoint.resolve()),
                    "final_checkpoint_sha256": sha256_file(checkpoint),
                    "artifacts": {
                        **training_artifacts,
                        summary.name: sha256_file(summary),
                    },
                })
                states[job_id] = state
    return jobs, ledger, states


def test_confirmatory_plan_is_exactly_ten_frozen_jobs(tmp_path: pathlib.Path) -> None:
    jobs = _jobs(tmp_path)
    assert len(jobs) == 10
    assert [job.job_id for job in jobs] == [
        *(f"gpt-seed{seed}-hmm" for seed in (1234, 1235, 1236, 1237, 1238)),
        *(f"nextlat-seed{seed}-hmm" for seed in (1234, 1235, 1236, 1237, 1238)),
    ]
    assert {job.train_batches for job in jobs} == {3_000}
    assert {job.phase for job in jobs} == {"hmm"}
    assert len({pathlib.Path(job.out_root).resolve() for job in jobs}) == 10
    assert all("seed" in job.experiment_name for job in jobs)
    assert all(job.experiment_dir_name == job.job_id for job in jobs)
    assert hmm_job_id("nextlat", 1238) == "nextlat-seed1238-hmm"
    with pytest.raises(ValueError, match="not preregistered"):
        hmm_job_id("gpt", 9)


def test_snapshot_preflight_rehashes_every_required_input(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "project"
    rows = []
    for index, rel in enumerate(sorted(HMM_REQUIRED_INVENTORY)):
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-{index}".encode())
        rows.append(f"{sha256_file(path)}  {rel}\n")
    inventory = project / "manifests" / "manifest_inventory.sha256"
    inventory.write_text("".join(rows), encoding="utf-8")
    verified = verify_hmm_snapshot(project)
    assert len(verified) == len(HMM_REQUIRED_INVENTORY) + 1

    victim = project / sorted(HMM_REQUIRED_INVENTORY)[0]
    victim.write_bytes(b"tampered")
    with pytest.raises(HMMMatrixError, match="hash mismatch"):
        verify_hmm_snapshot(project)


def test_checked_in_family_snapshot_passes_full_cross_artifact_preflight() -> None:
    """Exercise the real frozen schema, including nested threshold-to-HMM binding."""
    project = pathlib.Path(__file__).resolve().parent.parent
    verified = verify_hmm_family_snapshot(project)
    assert tuple(verified) == HMM_FAMILY_REGIMES
    assert all(verified[regime] for regime in verified)


def test_production_source_identity_covers_actual_pinned_layout() -> None:
    sources = {pathlib.Path(path).name for path in hmm_source_inputs()}
    assert {
        "core_train.py", "model_base.py", "model_gpt.py", "model_nextlat.py",
        "evaluate_hmm_checkpoints.py", "extraction_cache.py",
    } <= sources


def test_launcher_uses_hmm_shim_exact_target_and_verified_resume(tmp_path: pathlib.Path) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    durable = tmp_path / "durable-data"
    launcher = HMMFabricLauncher(
        tmp_path / "project", tmp_path / "upstream", data_root=durable
    )
    from run_matrix import ResumePlan

    fresh = launcher.command(ResumePlan(job, fresh=True))
    assert str(tmp_path / "project" / "scripts" / "train_hmm.py") in fresh
    assert "trainer.train_batches=3000" in fresh
    assert "trainer.save_recovery_checkpoint=250" in fresh
    assert "trainer.init_from=scratch" in fresh
    assert "--checkpoint_path" not in fresh
    assert f"data.hmm_train_data_path={durable / 'data/hmm/hmm4x4_train_len32_100000.npy'}" in fresh
    assert not any("project/data/hmm" in arg for arg in fresh)

    resumed = launcher.command(ResumePlan(
        job, fresh=False, resume_step=1_750, checkpoint_path="/durable/ckpt_iter_1750.pt",
        checkpoint_sha256="a" * 64,
    ))
    assert resumed[resumed.index("--checkpoint_path") + 1] == "/durable/ckpt_iter_1750.pt"
    assert "trainer.init_from=resume" in resumed
    assert "trainer.train_batches=3000" in resumed


def test_build_uses_durable_snapshot_root_not_source_root(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    project = _project(tmp_path)
    snapshot = tmp_path / "durable"
    rows = []
    for index, rel in enumerate(sorted(HMM_REQUIRED_INVENTORY)):
        path = snapshot / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"durable-{index}".encode())
        rows.append(f"{sha256_file(path)}  {rel}\n")
    (snapshot / "manifests" / "manifest_inventory.sha256").write_text("".join(rows))
    source = project / "source.txt"
    source.write_text("code\n")
    import run_hmm_matrix as module

    monkeypatch.setattr(module, "hmm_source_inputs", lambda *args: (str(source),))
    jobs = build_hmm_matrix(
        snapshot, project_root=project, snapshot_root=snapshot,
        models=("gpt",), seeds=(1234,),
    )
    assert jobs[0].config == str(project / "configs" / "gpt_hmm.yaml")
    assert str(snapshot / "manifests" / "manifest_inventory.sha256") in jobs[0].manifests


def test_dry_run_is_pure_and_does_not_construct_ledger(tmp_path: pathlib.Path, monkeypatch, capsys) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    import run_hmm_matrix as module

    monkeypatch.setattr(module, "build_hmm_matrix", lambda *args, **kwargs: [job])
    monkeypatch.setattr(
        module, "Ledger", lambda *args, **kwargs: pytest.fail("dry-run constructed Ledger")
    )
    assert main([
        "--root", str(tmp_path / "durable"),
        "--project-root", str(tmp_path / "project"),
        "--upstream", str(tmp_path / "upstream"),
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mutated_ledger"] is False
    assert output["commands"][0][-3].startswith("data.hmm_train_data_path=")


def test_operational_selection_is_nonempty_and_cannot_name_legacy_primary(
        tmp_path: pathlib.Path) -> None:
    family_job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    family_job = dataclasses.replace(
        family_job, job_id="gpt-seed1234-hmm-persistent_moderate",
        condition="persistent_moderate",
    )

    with pytest.raises(HMMMatrixError, match="may not be empty"):
        _selected_jobs([family_job], [])
    with pytest.raises(HMMMatrixError, match="unknown --only HMM job ids"):
        _selected_jobs([family_job], ["gpt-seed1234-hmm"])


def test_family_recovery_plan_is_explicitly_aggregate_ineligible(
        tmp_path: pathlib.Path, monkeypatch, capsys) -> None:
    family_job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    family_job = dataclasses.replace(
        family_job, job_id="gpt-seed1234-hmm-persistent_moderate",
        condition="persistent_moderate",
    )
    import run_hmm_matrix as module
    monkeypatch.setattr(module, "build_hmm_matrix", lambda *args, **kwargs: [family_job])

    assert main([
        "--root", str(tmp_path / "durable"), "--family", "--print-plan",
        "--only", family_job.job_id,
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["confirmatory"] is False
    assert plan["confirmatory_aggregate_eligible"] is False
    assert plan["operational_recovery_selection_only"] is True


def test_mutating_phase_requires_driver_owned_durability(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    import run_hmm_matrix as module

    monkeypatch.setattr(module, "build_hmm_matrix", lambda *args, **kwargs: [job])
    monkeypatch.setattr(
        module, "Ledger", lambda *args, **kwargs: pytest.fail("refusal constructed Ledger")
    )
    assert main(["--root", str(tmp_path / "durable")]) == 2
    assert "direct execution could lose paid progress" in capsys.readouterr().err


def test_cli_refuses_unverified_data_root_before_build(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import run_hmm_matrix as module

    monkeypatch.setattr(
        module, "build_hmm_matrix", lambda *args, **kwargs: pytest.fail("built mismatched plan")
    )
    assert main([
        "--root", str(tmp_path / "durable"),
        "--snapshot-root", str(tmp_path / "verified"),
        "--data-root", str(tmp_path / "different"),
        "--print-plan",
    ]) == 2
    assert "arrays that were hash-verified" in capsys.readouterr().err


def test_deprecated_bucket_flag_fails_closed_before_ledger(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    import run_hmm_matrix as module

    monkeypatch.setattr(module, "build_hmm_matrix", lambda *args, **kwargs: [job])
    monkeypatch.setattr(
        module, "Ledger", lambda *args, **kwargs: pytest.fail("refusal constructed Ledger")
    )
    assert main([
        "--root", str(tmp_path / "durable"), "--bucket", "misleading-bucket",
        "--driver-managed-durability",
    ]) == 2
    assert "not a live checkpoint daemon" in capsys.readouterr().err


def test_disconnect_records_partial_progress_then_resumes_exactly(tmp_path: pathlib.Path) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    launcher = FakeHMMLauncher(stop_at=1_500)
    ledger, runner = _runner(tmp_path, launcher)
    first = runner.run([job])[job.job_id]
    assert first["status"] == INTERRUPTED
    assert first["step"] == 1_500
    assert "useful checkpoint" in first["reason"]

    second = runner.run([job])[job.job_id]
    assert second["status"] == TRAINED
    assert second["step"] == HMM_TRAIN_UPDATES
    assert second["updates"] == HMM_TRAIN_UPDATES
    assert launcher.calls[-1].resume_step == 1_500
    assert sha256_file(second["final_checkpoint"]) == second["final_checkpoint_sha256"]

    calls = len(launcher.calls)
    MatrixRunner(ledger, launcher, serializer=SER, echo=lambda _: None).run([job])
    assert len(launcher.calls) == calls, "hash-verified TRAINED job was relaunched"


@pytest.mark.parametrize("final_step", [2_999, 3_001])
def test_success_shaped_under_or_overrun_is_failed(tmp_path: pathlib.Path, final_step: int) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    _, runner = _runner(tmp_path, FakeHMMLauncher(final_step=final_step))
    state = runner.run([job])[job.job_id]
    assert state["status"] == FAILED
    assert state["updates"] == final_step
    assert "exactly 3000" in state["reason"]


def _write_evaluation(job: JobSpec, state: dict, *, metrics=None) -> None:
    root = pathlib.Path(job.out_root)
    evaluator = next(
        pathlib.Path(path) for path in job.manifests
        if pathlib.Path(path).name == "evaluate_hmm_checkpoints.py"
    )
    extraction = next(
        pathlib.Path(path) for path in job.manifests
        if pathlib.Path(path).name == "extraction_cache.py"
    )
    manifests = [
        pathlib.Path(path) for path in job.manifests
        if pathlib.Path(path).name in HMM_RECEIPT_MANIFEST_NAMES
    ]
    receipt = root / HMM_EVALUATION_RECEIPT
    receipt.parent.mkdir(parents=True, exist_ok=True)
    representation = receipt.parent / "representation_manifest.json"
    representation.write_text(json.dumps({
        "schema": "nextlat_forgetting/hmm_representation_plan/1",
        "policy": {"outcome_dependent_selection": False},
    }) + "\n")
    source_config = pathlib.Path(job.config).resolve()
    materialized_config = pathlib.Path(state["final_checkpoint"]).parent / "materialized_config.yaml"
    pair_bank = next(
        pathlib.Path(path) for path in job.manifests if pathlib.Path(path).name == "hmm_eval_pairs.jsonl"
    )
    val_posteriors = next(
        pathlib.Path(path) for path in job.manifests
        if pathlib.Path(path).name == "hmm4x4_val_posteriors.npz"
    )
    lengen_posteriors = next(
        pathlib.Path(path) for path in job.manifests
        if pathlib.Path(path).name == "hmm4x4_lengen_posteriors.npz"
    )
    cache = ExtractionCache(receipt.parent / "representation_cache", {
        "job_id": job.job_id,
        "model": job.model,
        "seed": job.seed,
        "checkpoint_sha256": state["final_checkpoint_sha256"],
        "config_sha256": sha256_file(materialized_config),
        "source_config_sha256": sha256_file(source_config),
        "pair_bank_sha256": sha256_file(pair_bank),
        "val_posteriors_sha256": sha256_file(val_posteriors),
        "lengen_posteriors_sha256": sha256_file(lengen_posteriors),
        "representation_manifest_sha256": sha256_file(representation),
        "evaluator_source_sha256": {
            str(path): sha256_file(path) for path in (evaluator, extraction)
        },
        "upstream_commit": "3770be6009cea2b3c455a9ce7f2ca88b504bb955",
        "chunk_rows": 256,
    })
    cache.write("chunk_000", {"states": np.zeros((2, 3))})
    cache_record = cache.receipt(expected_keys=["chunk_000"])
    receipt.write_text(json.dumps({
        "schema": HMM_EVALUATION_SCHEMA,
        "job_id": job.job_id,
        "model": job.model,
        "seed": job.seed,
        "checkpoint_sha256": state["final_checkpoint_sha256"],
        "all_preregistered_metrics_reported": True,
        "metric_selection_performed": False,
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "evaluator_sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (evaluator, extraction)
        ],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "source_config": source_config,
                "materialized_config": materialized_config,
                "pair_bank": pair_bank,
                "val_posteriors": val_posteriors,
                "lengen_posteriors": lengen_posteriors,
            }.items()
        },
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "representation_manifest": {
            "path": str(representation), "sha256": sha256_file(representation),
        },
        "representation_cache": cache_record,
        "metrics": metrics if metrics is not None else {
            key: {"estimate": 0.0} for key in HMM_REQUIRED_METRICS
        },
    }, sort_keys=True) + "\n")
    sidecar = root / HMM_EVALUATION_SIDECAR
    sidecar.write_text(f"{sha256_file(receipt)}  {receipt.name}\n")


def test_done_requires_checkpoint_bound_all_metrics_receipt(tmp_path: pathlib.Path) -> None:
    job = _jobs(tmp_path, models=("nextlat",), seeds=(1238,))[0]
    ledger, runner = _runner(tmp_path, FakeHMMLauncher())
    trained = runner.run([job])[job.job_id]
    assert trained["status"] == TRAINED
    _write_evaluation(job, trained)

    done = promote_hmm_evaluations([job], ledger, echo=lambda _: None)[job.job_id]
    assert done["status"] == DONE
    assert set(done["evaluation_artifacts"]) == {
        HMM_EVALUATION_RECEIPT, HMM_EVALUATION_SIDECAR,
    }
    before = len(ledger.entries())
    promote_hmm_evaluations([job], ledger, echo=lambda _: None)
    assert len(ledger.entries()) == before, "verified DONE promotion was not idempotent"


def test_done_gate_refuses_metric_selection_or_missing_metric(tmp_path: pathlib.Path) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    ledger, runner = _runner(tmp_path, FakeHMMLauncher())
    trained = runner.run([job])[job.job_id]
    incomplete = {key: {"estimate": 0.0} for key in HMM_REQUIRED_METRICS}
    incomplete.pop("h2_partial_spearman")
    _write_evaluation(job, trained, metrics=incomplete)
    with pytest.raises(HMMMatrixError, match="frozen metric set"):
        promote_hmm_evaluations([job], ledger, echo=lambda _: None)
    assert ledger.state_of(job.job_id)["status"] == TRAINED


def test_done_gate_recursively_rejects_nested_nonfinite_metric(tmp_path: pathlib.Path) -> None:
    job = _jobs(tmp_path, models=("gpt",), seeds=(1234,))[0]
    ledger, runner = _runner(tmp_path, FakeHMMLauncher())
    trained = runner.run([job])[job.job_id]
    metrics = {key: {"nested": {"estimate": 0.0}} for key in HMM_REQUIRED_METRICS}
    metrics["h2_spearman"]["nested"]["estimate"] = float("nan")
    _write_evaluation(job, trained, metrics=metrics)
    with pytest.raises(HMMMatrixError, match="non-finite.*h2_spearman"):
        promote_hmm_evaluations([job], ledger, echo=lambda _: None)
    assert ledger.state_of(job.job_id)["status"] == TRAINED


def test_evaluator_command_binds_checkpoint_frozen_data_and_resumable_cache(
    tmp_path: pathlib.Path,
) -> None:
    job = _jobs(tmp_path, models=("nextlat",), seeds=(1235,))[0]
    ledger, runner = _runner(tmp_path, FakeHMMLauncher())
    trained = runner.run([job])[job.job_id]
    command = hmm_evaluator_command(
        job, trained, project_root=tmp_path / "project", snapshot_root=tmp_path / "snapshot",
        upstream_root=tmp_path / "upstream", batch_size=192,
    )
    assert command[:7] == [
        "fabric", "run", "--devices", "1", "--precision", "bf16-mixed",
        str(tmp_path / "project/scripts/evaluate_hmm_checkpoints.py"),
    ]
    assert command[command.index("--checkpoint") + 1] == trained["final_checkpoint"]
    assert command[command.index("--pair-bank") + 1].endswith("manifests/hmm_eval_pairs.jsonl")
    assert command[command.index("--cache-root") + 1].endswith(
        "evaluation/representation_cache"
    )
    assert command[command.index("--batch-size") + 1] == "192"
    assert command.count("--manifest") == len(HMM_RECEIPT_MANIFEST_NAMES)


def test_evaluator_preflight_accepts_exact_canonical_30_before_any_invocation(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    jobs, ledger, _ = _canonical_trained_family(tmp_path)
    states = preflight_hmm_evaluation_matrix(jobs, ledger)
    assert tuple(states[job.job_id]["status"] for job in jobs) == (TRAINED,) * 30

    invoked: list[str] = []
    import run_hmm_matrix as module
    monkeypatch.setattr(
        module, "hmm_evaluator_command",
        lambda spec, *args, **kwargs: ["evaluate", spec.job_id],
    )
    monkeypatch.setattr(module, "verify_hmm_evaluation_receipt", lambda *args: {})

    def command_runner(command, **kwargs):
        invoked.append(command[-1])
        return 0

    run_hmm_evaluators(
        jobs, ledger, project_root=tmp_path, snapshot_root=tmp_path,
        upstream_root=tmp_path, command_runner=command_runner,
    )
    assert invoked == [job.job_id for job in jobs]


@pytest.mark.parametrize("mutation", ("not_trained", "checkpoint_hash", "provenance", "subset"))
def test_atomic_30_job_evaluation_preflight_refuses_before_first_cell(
        tmp_path: pathlib.Path, monkeypatch, mutation: str) -> None:
    jobs, ledger, states = _canonical_trained_family(tmp_path)
    last = jobs[-1]
    if mutation == "not_trained":
        ledger.append({**states[last.job_id], "status": INTERRUPTED})
    elif mutation == "checkpoint_hash":
        pathlib.Path(states[last.job_id]["final_checkpoint"]).write_bytes(b"mutated")
    elif mutation == "provenance":
        artifacts = dict(states[last.job_id]["artifacts"])
        artifacts.pop("final_summary.json")
        ledger.append({**states[last.job_id], "artifacts": artifacts})
    else:
        jobs = jobs[:-1]

    invoked: list[list[str]] = []
    import run_hmm_matrix as module
    monkeypatch.setattr(
        module, "hmm_evaluator_command",
        lambda spec, *args, **kwargs: ["evaluate", spec.job_id],
    )
    with pytest.raises(HMMMatrixError, match="canonical 30|TRAINED|checkpoint|provenance"):
        run_hmm_evaluators(
            jobs, ledger, project_root=tmp_path, snapshot_root=tmp_path,
            upstream_root=tmp_path,
            command_runner=lambda command, **kwargs: invoked.append(command) or 0,
        )
    assert invoked == [], "a cell evaluator ran before the complete matrix passed preflight"


@pytest.mark.parametrize("failure", ["missing", "overstep", "empty_artifacts"])
def test_atomic_exact_ten_barrier_refuses_incomplete_or_overstep_before_launch(
        tmp_path: pathlib.Path, failure: str) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 3000\n")
    jobs = []
    records = []
    for model in ("gpt", "nextlat"):
        for seed in range(1234, 1239):
            job_id = hmm_job_id(model, seed, "persistent_moderate")
            out = tmp_path / "runs" / job_id
            out.mkdir(parents=True)
            checkpoint = out / "ckpt_iter_3000_1.0.pt"
            checkpoint.write_bytes((job_id + " checkpoint").encode())
            artifact = out / "materialized_config.yaml"
            artifact.write_text("frozen\n")
            jobs.append(JobSpec(
                job_id=job_id, model=model, seed=seed, phase="hmm",
                condition="persistent_moderate", config=str(cfg), out_root=str(out),
                train_batches=HMM_TRAIN_UPDATES,
            ))
            records.append({
                "job_id": job_id,
                "target_step": 3001 if failure == "overstep" and not records else 3000,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "recovery_provenance": {
                    "checkpoint_sha256": sha256_file(checkpoint),
                },
                "authoritative_artifacts": {str(artifact): sha256_file(artifact)},
            })
    if failure == "missing":
        records.pop()
    elif failure == "empty_artifacts":
        records[0]["authoritative_artifacts"] = {}
    barrier = tmp_path / "barrier.json"
    barrier.write_text(json.dumps({
        "schema": "nextlat_forgetting/runtime_recovery_barrier/1",
        "status": "PASS",
        "job_ids": [record["job_id"] for record in records],
        "jobs": records,
    }))

    with pytest.raises(HMMMatrixError, match="canonical exact ten|deep verification"):
        load_runtime_recovery_barrier(barrier, jobs)
