from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cfs2_patching_matrix as M  # noqa: E402


def _materialize_training_matrix(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    retention = tmp_path / "retention.jsonl"
    retention.write_text('{"probe_id":"p0"}\n', encoding="utf-8")
    parents: dict[int, tuple[Path, str]] = {}
    for seed in M.CFS2_PARENT_SEEDS:
        path = tmp_path / "parents" / f"seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"parent-{seed}".encode())
        parents[seed] = (path, M.sha256_file(path))

    state_commitment = {"path": "state_interchange_activation_patching_commitment.json", "sha256": "c" * 64}
    entries, ready = [], []
    for job_id, (parent_id, seed, episode, overlap, relation) in M.expected_jobs().items():
        out_root = tmp_path / "training" / job_id
        out_root.mkdir(parents=True)
        branch = out_root / "branch.pt"
        branch.write_bytes(f"branch-{job_id}".encode())
        branch_sha = M.sha256_file(branch)
        parent, parent_sha = parents[seed]
        completion = out_root / "cfs2_training_completion.json"
        completion.write_text(json.dumps({
            "schema": M.COMPLETION_SCHEMA,
            "job_id": job_id,
            "parent_id": parent_id,
            "parent_checkpoint": {"path": str(parent), "sha256": parent_sha, "training_steps": 20_000},
            "branch_checkpoint": {"path": str(branch), "sha256": branch_sha, "training_steps": 20_500},
            "adaptation": {"updates": 500},
            "inputs": {"state_interchange_activation_patching": state_commitment},
            "scientific_evaluation_started": False,
        }), encoding="utf-8")
        completion_sha = M.sha256_file(completion)
        entries.append({
            "job_id": job_id, "status": "TRAINED", "out_root": str(out_root),
            "completion_sha256": completion_sha,
            "parent_checkpoint_sha256": parent_sha,
            "final_checkpoint": str(branch), "final_checkpoint_sha256": branch_sha,
        })
        ready.append({
            "job_id": job_id, "parent_id": parent_id, "seed": seed,
            "episode": episode, "overlap": overlap, "future_relation": relation,
            "completion_sha256": completion_sha,
            "parent_checkpoint_sha256": parent_sha,
            "branch_checkpoint_sha256": branch_sha,
        })

    ledger = tmp_path / "cfs2_run_ledger.json"
    ledger.write_text(json.dumps({"schema": M.LEDGER_SCHEMA, "entries": entries}), encoding="utf-8")
    readiness = tmp_path / "cfs2_pre_evaluation_readiness.json"
    readiness.write_text(json.dumps({
        "schema": M.READINESS_SCHEMA, "status": "ALL_64_BRANCHES_TRAINED",
        "n_branches": 64, "scientific_evaluation_started": False,
        "state_interchange_activation_patching": state_commitment,
        "branches": ready,
    }), encoding="utf-8")
    return readiness, ledger, retention, tmp_path / "patches"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        upstream_root=str(ROOT / "upstream/NextLat"),
        config=str(ROOT / "configs/cfs2_nextlat_adapt.yaml"),
        device="cuda", batch_size=256, patch_layers=M.DEFAULT_PATCH_LAYERS,
        analysis_seed=20260824,
    )


def _write_patch(path: Path, job: M.PatchJob, retention_sha: str, *, seed: int = 20260824) -> None:
    values: dict[str, object] = {
        "schema": np.asarray(M.PATCH_ARTIFACT_SCHEMA),
        "branch_id": np.asarray(job.job_id),
        "probe_ids": np.asarray(["p0", "p1"]),
        "patch_position": np.asarray(M.PATCH_POSITION),
        "patch_layers": np.asarray(M.DEFAULT_PATCH_LAYERS),
        "analysis_seed": np.asarray(seed),
        "baseline_margin": np.asarray([1.0, -1.0]),
        "parent_checkpoint_sha256": np.asarray(job.parent_checkpoint_sha256),
        "adapted_checkpoint_sha256": np.asarray(job.branch_checkpoint_sha256),
        "retention_manifest_sha256": np.asarray(retention_sha),
    }
    for layer in M.DEFAULT_PATCH_LAYERS:
        for stem in M.CONTROL_STEMS:
            values[f"layer_{layer}_{stem}_margin"] = np.asarray([0.0, 0.0])
            # Deliberately arbitrary effects: scheduling must never select on them.
            values[f"layer_{layer}_patch_{stem}_effect"] = np.asarray([999.0, -999.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


def test_full_sweep_runs_all_64_then_resumes_only_hash_verified_artifacts(tmp_path: Path) -> None:
    readiness, ledger, retention, output = _materialize_training_matrix(tmp_path)
    inputs = M.load_matrix_inputs(readiness, ledger, retention, output)
    assert len(inputs.jobs) == 64
    by_id = {job.job_id: job for job in inputs.jobs}
    calls: list[str] = []

    def launch(command) -> int:
        branch_id = command[command.index("--branch-id") + 1]
        path = Path(command[command.index("--output") + 1])
        calls.append(branch_id)
        _write_patch(path, by_id[branch_id], inputs.retention_sha256)
        return 0

    index = output / "cfs2_patching_matrix.json"
    final = M.run_matrix(inputs, index, _args(), launcher=launch)
    assert len(calls) == 64
    assert final["status"] == M.FINAL_STATUS and final["n_branches"] == 64
    assert [row["job_id"] for row in final["branches"]] == sorted(M.expected_jobs())
    assert final["outcome_filtering"] is False

    M.run_matrix(inputs, index, _args(), launcher=lambda _command: pytest.fail("resume relaunched work"))
    artifact = Path(inputs.jobs[0].output)
    artifact.write_bytes(b"tampered")
    with pytest.raises(M.CFS2PatchingMatrixError, match="unreadable"):
        M.run_matrix(inputs, index, _args(), launcher=lambda _command: 0)


def test_incomplete_readiness_is_refused_before_any_patching(tmp_path: Path) -> None:
    readiness, ledger, retention, output = _materialize_training_matrix(tmp_path)
    value = json.loads(readiness.read_text())
    value["branches"].pop()
    readiness.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(M.CFS2PatchingMatrixError, match="64 branches"):
        M.load_matrix_inputs(readiness, ledger, retention, output)


def test_artifact_must_bind_exact_checkpoint_and_fixed_layers(tmp_path: Path) -> None:
    readiness, ledger, retention, output = _materialize_training_matrix(tmp_path)
    inputs = M.load_matrix_inputs(readiness, ledger, retention, output)
    job = inputs.jobs[0]
    artifact = Path(job.output)
    _write_patch(artifact, job, inputs.retention_sha256)
    assert len(M.validate_patch_artifact(
        artifact, job, retention_sha256=inputs.retention_sha256,
        layers=M.DEFAULT_PATCH_LAYERS, analysis_seed=20260824,
    )) == 64
    _write_patch(artifact, dataclasses.replace(job, branch_checkpoint_sha256="0" * 64), inputs.retention_sha256)
    with pytest.raises(M.CFS2PatchingMatrixError, match="branch hash is stale"):
        M.validate_patch_artifact(
            artifact, job, retention_sha256=inputs.retention_sha256,
            layers=M.DEFAULT_PATCH_LAYERS, analysis_seed=20260824,
        )
