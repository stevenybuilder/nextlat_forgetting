from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_cfs2 as V  # noqa: E402
import run_cfs2_analysis as R  # noqa: E402


def _file(root: Path, name: str, payload: bytes = b"opaque") -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": R.sha256_file(path)}


def _inputs(tmp_path: Path, *, n_items: int) -> tuple[Path, Path]:
    state = _file(tmp_path, "state.json")
    generator = _file(tmp_path, "generator.json")
    retention = _file(tmp_path, "retention.jsonl")
    stream = _file(tmp_path, "stream.txt")
    global_controls = _file(tmp_path, "global.jsonl")
    completion = _file(tmp_path, "completion.json")
    parent_checkpoint = _file(tmp_path, "parent.ckpt")
    branch_checkpoint = _file(tmp_path, "branch.ckpt")
    ledger = _file(tmp_path, "ledger.json")
    parent_ids = [
        f"nextlat-seed{seed}" + ("-base" if seed < 2000 else "-cfs2-base")
        for seed in (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236)
    ]
    ids = np.asarray([f"probe-{index}" for index in range(n_items)])
    ids_sha = R._item_ids_sha256(ids)
    ready_rows = []
    evidence_rows = []
    patch_rows = []
    for parent_number, parent_id in enumerate(parent_ids):
        for episode in (0, 1):
            for arm in ("high_different", "low_different", "high_same", "low_same"):
                overlap, relation = arm.split("_", 1)
                job_id = f"cfs2-{parent_id}-{episode}-{arm}"
                ready_rows.append({
                    "job_id": job_id,
                    "parent_id": parent_id,
                    "seed": 0,
                    "episode": episode,
                    "overlap": overlap,
                    "future_relation": relation,
                    "completion_sha256": completion["sha256"],
                    "parent_checkpoint_sha256": parent_checkpoint["sha256"],
                    "branch_checkpoint_sha256": branch_checkpoint["sha256"],
                })
                erosion = {
                    "high_different": 0.5,
                    "high_same": 0.2,
                    "low_different": 0.15,
                    "low_same": 0.05,
                }[arm] + parent_number / 1000
                evidence_path = tmp_path / "evidence" / f"{job_id}.npz"
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                pre = np.full(n_items, 2.0)
                np.savez_compressed(
                    evidence_path,
                    schema=np.asarray(R.BRANCH_EVIDENCE_SCHEMA),
                    job_id=np.asarray(job_id),
                    parent_id=np.asarray(parent_id),
                    episode=np.asarray(episode),
                    overlap=np.asarray(overlap),
                    future_relation=np.asarray(relation),
                    adaptation_steps=np.asarray(500),
                    parent_checkpoint_sha256=np.asarray(parent_checkpoint["sha256"]),
                    branch_checkpoint_sha256=np.asarray(branch_checkpoint["sha256"]),
                    generator_manifest_sha256=np.asarray(generator["sha256"]),
                    retention_probe_manifest_sha256=np.asarray(retention["sha256"]),
                    adaptation_stream_manifest_sha256=np.asarray(stream["sha256"]),
                    global_control_manifest_sha256=np.asarray(global_controls["sha256"]),
                    retention_probe_item_ids=ids,
                    retention_probe_item_ids_sha256=np.asarray(ids_sha),
                    pre_correct_first_branch_margin=pre,
                    post_correct_first_branch_margin=pre - erosion,
                )
                evidence_rows.append({
                    "job_id": job_id,
                    "parent_id": parent_id,
                    "episode": episode,
                    "overlap": overlap,
                    "future_relation": relation,
                    "parent_checkpoint": parent_checkpoint,
                    "branch_checkpoint": branch_checkpoint,
                    "completion_receipt": completion,
                    "generator_manifest": generator,
                    "retention_probe_manifest": retention,
                    "adaptation_stream_manifest": stream,
                    "global_control_manifest": global_controls,
                    "state_interchange_activation_patching": state,
                    "pre_post_paired_endpoints": [
                        "margin", "retention_ce", "retention_exact_path", "global_controls",
                        "state_drift", "pregeometry",
                    ],
                    "evidence_npz": _record(evidence_path),
                })
                patch_path = tmp_path / "patches" / f"{job_id}.npz"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                patch_arrays: dict[str, object] = {
                    "schema": np.asarray(R.PATCH_ARTIFACT_SCHEMA),
                    "branch_id": np.asarray(job_id),
                    "probe_ids": ids,
                    "patch_position": np.asarray(63),
                    "patch_layers": np.asarray([3, 7, 10]),
                    "analysis_seed": np.asarray(17),
                    "parent_checkpoint_sha256": np.asarray(parent_checkpoint["sha256"]),
                    "adapted_checkpoint_sha256": np.asarray(branch_checkpoint["sha256"]),
                    "retention_manifest_sha256": np.asarray(retention["sha256"]),
                }
                for layer in (3, 7, 10):
                    patch_arrays[f"layer_{layer}_patch_parent_state_effect"] = np.full(n_items, 0.3)
                    patch_arrays[f"layer_{layer}_patch_unrelated_anchor_effect"] = np.full(n_items, 0.05)
                    patch_arrays[f"layer_{layer}_patch_norm_matched_random_subspace_effect"] = np.full(n_items, 0.1)
                np.savez_compressed(patch_path, **patch_arrays)
                patch_rows.append({
                    "job_id": job_id,
                    "parent_id": parent_id,
                    "seed": 0,
                    "episode": episode,
                    "overlap": overlap,
                    "future_relation": relation,
                    "path": str(patch_path),
                    "sha256": R.sha256_file(patch_path),
                    "parent_checkpoint_sha256": parent_checkpoint["sha256"],
                    "branch_checkpoint_sha256": branch_checkpoint["sha256"],
                })

    readiness_payload = {
        "schema": V.READINESS_SCHEMA,
        "status": "ALL_64_BRANCHES_TRAINED",
        "n_branches": 64,
        "update_manifest_sha256": "a" * 64,
        "evaluation_input_sha256s": {
            name: "b" * 64 for name in (
                "margin", "retention_ce", "retention_exact_path", "global_controls",
                "state_drift", "pregeometry",
            )
        },
        "state_interchange_activation_patching": state,
        "branches": ready_rows,
        "scientific_evaluation_started": False,
    }
    readiness = _file(tmp_path, "readiness.json", json.dumps(readiness_payload, sort_keys=True).encode())
    evidence_job = {
        "schema": V.EVIDENCE_JOB_SCHEMA,
        "analysis_seed": 17,
        "readiness": readiness,
        "state_interchange_activation_patching": state,
        "parents": [
            {"parent_id": parent_id, "base_checkpoint": parent_checkpoint, "base_training_steps": 20_000}
            for parent_id in parent_ids
        ],
        "branches": evidence_rows,
    }
    job = _file(tmp_path, "evidence-job.json", json.dumps(evidence_job, sort_keys=True).encode())
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "schema": V.EVALUATION_MANIFEST_SCHEMA,
        "analysis_seed": 17,
        "n_boot": 100,
        "readiness": readiness,
        "evidence_job": job,
    }))
    patch_manifest = tmp_path / "patches" / "cfs2_patching_matrix.json"
    patch_manifest.write_text(json.dumps({
        "schema": R.PATCH_MATRIX_SCHEMA,
        "status": "ALL_64_BRANCHES_PATCHED",
        "n_branches": 64,
        "expected_branches": 64,
        "readiness": readiness,
        "ledger": ledger,
        "retention_manifest": retention,
        "state_interchange_activation_patching": state,
        "analysis_seed": 17,
        "patch_layers": [3, 7, 10],
        "branches": patch_rows,
        "outcome_filtering": False,
    }))
    return evaluation, patch_manifest


def test_cli_analysis_consumes_complete_envelope_and_patch_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "N_RETENTION_PROBES", 4)
    evaluation, patch_manifest = _inputs(tmp_path, n_items=4)
    report = R.evaluate(evaluation, patch_manifest)
    assert report["status"] == "COMPLETE"
    assert report["branch_count"] == 64
    assert len(report["branch_cells"]) == 64
    assert report["activation_patching"]["status"] == "COMPLETE_ALL_64_BRANCHES_ALL_FIXED_LAYERS"

