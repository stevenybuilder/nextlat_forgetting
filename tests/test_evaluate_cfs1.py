from __future__ import annotations

import importlib
import json
import pathlib

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
X = importlib.import_module("extract_cfs1_evidence")
M = importlib.import_module("evaluate_cfs1")


def _record(path: pathlib.Path) -> dict:
    return {"path": str(path), "sha256": X.sha256_file(path)}


def _ready_job(tmp_path: pathlib.Path) -> pathlib.Path:
    shared = {}
    for name in ("generator", "retention", "adaptation", "global"):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        shared[name] = _record(path)
    ids = np.asarray([f"probe-{index:04d}" for index in range(2000)])
    parents, branches = [], []
    for parent_number in range(8):
        parent_id = f"parent-{parent_number}"
        base = tmp_path / f"{parent_id}.pt"
        base.write_bytes(parent_id.encode())
        parent = _record(base)
        parents.append({"parent_id": parent_id, "base_checkpoint": parent, "base_training_steps": 20_000})
        for episode in (0, 1):
            for overlap, relation, erosion in (("high", "different", .60), ("low", "different", .15), ("high", "same", .20), ("low", "same", .05)):
                stem = f"{parent_id}-{episode}-{overlap}-{relation}"
                checkpoint = tmp_path / f"{stem}.pt"
                checkpoint.write_bytes(stem.encode())
                evidence_path = tmp_path / f"{stem}.npz"
                branch = {
                    "parent_id": parent_id, "episode": episode, "overlap": overlap, "future_relation": relation,
                    "parent_checkpoint": parent, "branch_checkpoint": _record(checkpoint), "adaptation_steps": 500,
                    "generator_manifest": shared["generator"], "retention_probe_manifest": shared["retention"],
                    "adaptation_stream_manifest": shared["adaptation"], "global_control_manifest": shared["global"],
                    "evidence_npz": {"path": str(evidence_path), "sha256": "0" * 64},
                }
                pre = np.full(2000, 2.0 + parent_number / 100)
                np.savez(
                    evidence_path,
                    schema=np.asarray(X.BRANCH_EVIDENCE_SCHEMA), parent_id=np.asarray(parent_id), episode=np.asarray(episode),
                    overlap=np.asarray(overlap), future_relation=np.asarray(relation),
                    parent_checkpoint_sha256=np.asarray(parent["sha256"]), parent_training_steps=np.asarray(20_000),
                    branch_checkpoint_sha256=np.asarray(branch["branch_checkpoint"]["sha256"]), adaptation_steps=np.asarray(500),
                    generator_manifest_sha256=np.asarray(shared["generator"]["sha256"]), retention_probe_manifest_sha256=np.asarray(shared["retention"]["sha256"]),
                    adaptation_stream_manifest_sha256=np.asarray(shared["adaptation"]["sha256"]), global_control_manifest_sha256=np.asarray(shared["global"]["sha256"]),
                    retention_probe_item_ids=ids, retention_probe_item_ids_sha256=np.asarray(X.item_ids_sha256(ids)),
                    pre_correct_first_branch_margin=pre, post_correct_first_branch_margin=pre - erosion,
                    pre_retention_cross_entropy=np.full(2000, .2), post_retention_cross_entropy=np.full(2000, .2 + erosion),
                    pre_retention_exact_path_accuracy=np.full(2000, .9), post_retention_exact_path_accuracy=np.full(2000, .9 - erosion / 2),
                    adaptation_acquisition=np.full(2000, .8), pre_global_control_margin=pre, post_global_control_margin=pre - .1,
                    penultimate_state_drift=np.full(2000, .1), pre_adaptation_predictive_geometry=np.asarray(float(parent_number)),
                    penultimate_state_patching_status=np.asarray("NOT_RUN"),
                )
                branch["evidence_npz"] = _record(evidence_path)
                branches.append(branch)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"schema": X.JOB_SCHEMA, "analysis_seed": 19, "parents": parents, "branches": branches}))
    return job


def _manifest(job: pathlib.Path) -> pathlib.Path:
    path = job.with_name("evaluation.json")
    path.write_text(json.dumps({"schema": M.SCHEMA, "analysis_seed": 19, "n_boot": 100, "extraction_job": _record(job)}))
    return path


def test_evaluator_emits_parent_did_holm_noncausal_moderation_and_no_mediation_claim(tmp_path):
    manifest = _manifest(_ready_job(tmp_path))
    report, receipt = M.evaluate_manifest(manifest)
    assert report["status"] == "COMPLETE"
    assert report["primary"]["estimate"] == pytest.approx(.3)
    assert report["primary"]["exact_two_sided_sign_flip_p"] == 1 / 128
    assert len(report["primary"]["conditional_item_bootstrap_by_parent"]) == 8
    assert set(report["secondary_endpoints"]) == {
        "retention_cross_entropy_increase", "retention_exact_path_accuracy_loss", "adaptation_acquisition",
        "global_control_margin_erosion", "penultimate_state_drift",
    }
    assert report["geometry_moderation"]["causal_mediation_claim_permitted"] is False
    assert report["penultimate_state_patching"]["causal_mediation_claim_permitted"] is False
    assert receipt["branch_count"] == 64


def test_evaluator_fail_closes_to_terminal_invalid_report_before_partial_metrics(tmp_path):
    job = _ready_job(tmp_path)
    payload = json.loads(job.read_text())
    bad = pathlib.Path(payload["branches"][0]["evidence_npz"]["path"])
    with np.load(bad, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["parent_training_steps"] = np.asarray(19_999)
    np.savez(bad, **arrays)
    payload["branches"][0]["evidence_npz"] = _record(bad)
    job.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(_manifest(job))
    assert report["status"] == "INVALID_INCOMPLETE"
    assert report["primary"] is None
    assert report["secondary_endpoints"] is None
    assert receipt["status"] == "INVALID_INCOMPLETE"


def test_cli_writes_terminal_invalid_output_for_missing_branch(tmp_path):
    job = _ready_job(tmp_path)
    payload = json.loads(job.read_text())
    payload["branches"].pop()
    job.write_text(json.dumps(payload))
    manifest = _manifest(job)
    output = tmp_path / "report.json"
    assert M.main(["--manifest", str(manifest), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "INVALID_INCOMPLETE"
    assert report["primary"] is None
