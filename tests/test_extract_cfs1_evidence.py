from __future__ import annotations

import hashlib
import importlib
import json
import pathlib

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
X = importlib.import_module("extract_cfs1_evidence")


def _record(path: pathlib.Path) -> dict:
    return {"path": str(path), "sha256": X.sha256_file(path)}


def _parents_and_job(tmp_path: pathlib.Path, *, evidence: bool) -> pathlib.Path:
    shared = {}
    for name in ("generator", "retention", "adaptation", "global"):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        shared[name] = _record(path)
    parents = []
    branches = []
    ids = np.asarray([f"probe-{index:04d}" for index in range(X.N_RETENTION_PROBES)])
    ids_sha = X.item_ids_sha256(ids)
    for parent_number in range(8):
        parent_id = f"parent-{parent_number}"
        parent_checkpoint = tmp_path / f"{parent_id}.pt"
        parent_checkpoint.write_bytes(parent_id.encode())
        parent_record = _record(parent_checkpoint)
        parents.append({"parent_id": parent_id, "base_checkpoint": parent_record, "base_training_steps": 20_000})
        for episode in (0, 1):
            for overlap, relation in (("high", "different"), ("low", "different"), ("high", "same"), ("low", "same")):
                stem = f"{parent_id}-{episode}-{overlap}-{relation}"
                branch_checkpoint = tmp_path / f"{stem}.pt"
                branch_checkpoint.write_bytes(stem.encode())
                evidence_path = tmp_path / f"{stem}.npz"
                branch = {
                    "parent_id": parent_id, "episode": episode, "overlap": overlap, "future_relation": relation,
                    "parent_checkpoint": parent_record, "branch_checkpoint": _record(branch_checkpoint),
                    "adaptation_steps": 500, "generator_manifest": shared["generator"],
                    "retention_probe_manifest": shared["retention"], "adaptation_stream_manifest": shared["adaptation"],
                    "global_control_manifest": shared["global"], "evidence_npz": {"path": str(evidence_path), "sha256": "0" * 64},
                }
                if evidence:
                    erosion = {("high", "different"): .5, ("high", "same"): .2, ("low", "different"): .15, ("low", "same"): .05}[(overlap, relation)]
                    pre = np.full(X.N_RETENTION_PROBES, 2.0)
                    np.savez(
                        evidence_path,
                        schema=np.asarray(X.BRANCH_EVIDENCE_SCHEMA), parent_id=np.asarray(parent_id), episode=np.asarray(episode),
                        overlap=np.asarray(overlap), future_relation=np.asarray(relation),
                        parent_checkpoint_sha256=np.asarray(parent_record["sha256"]), parent_training_steps=np.asarray(20_000),
                        branch_checkpoint_sha256=np.asarray(branch["branch_checkpoint"]["sha256"]), adaptation_steps=np.asarray(500),
                        generator_manifest_sha256=np.asarray(shared["generator"]["sha256"]),
                        retention_probe_manifest_sha256=np.asarray(shared["retention"]["sha256"]),
                        adaptation_stream_manifest_sha256=np.asarray(shared["adaptation"]["sha256"]),
                        global_control_manifest_sha256=np.asarray(shared["global"]["sha256"]),
                        retention_probe_item_ids=ids, retention_probe_item_ids_sha256=np.asarray(ids_sha),
                        pre_correct_first_branch_margin=pre, post_correct_first_branch_margin=pre - erosion,
                        pre_retention_cross_entropy=np.full(X.N_RETENTION_PROBES, .2), post_retention_cross_entropy=np.full(X.N_RETENTION_PROBES, .2 + erosion),
                        pre_retention_exact_path_accuracy=np.full(X.N_RETENTION_PROBES, .9), post_retention_exact_path_accuracy=np.full(X.N_RETENTION_PROBES, .9 - erosion / 2),
                        adaptation_acquisition=np.full(X.N_RETENTION_PROBES, .8),
                        pre_global_control_margin=pre, post_global_control_margin=pre - .1,
                        penultimate_state_drift=np.full(X.N_RETENTION_PROBES, .1),
                        pre_adaptation_predictive_geometry=np.asarray(float(parent_number)),
                        penultimate_state_patching_status=np.asarray("NOT_RUN"),
                    )
                    branch["evidence_npz"] = _record(evidence_path)
                branches.append(branch)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"schema": X.JOB_SCHEMA, "analysis_seed": 91, "parents": parents, "branches": branches}))
    return job


def test_load_job_requires_exact_complete_64_branch_lattice(tmp_path):
    job = _parents_and_job(tmp_path, evidence=False)
    loaded = X.load_job(job)
    assert len(loaded.parent_by_id) == 8
    assert len(loaded.branch_by_key) == 64
    payload = json.loads(job.read_text())
    payload["branches"].pop()
    job.write_text(json.dumps(payload))
    with pytest.raises(X.ExtractionRefused, match="atomic 64-branch lattice"):
        X.load_job(job)


def test_evidence_contract_refuses_step_or_parent_tampering(tmp_path):
    job = _parents_and_job(tmp_path, evidence=True)
    loaded = X.load_job(job)
    key, branch = next(iter(loaded.branch_by_key.items()))
    evidence_path = pathlib.Path(branch["evidence_npz"]["path"])
    with np.load(evidence_path, allow_pickle=False) as old:
        arrays = {name: np.asarray(old[name]) for name in old.files}
    arrays["adaptation_steps"] = np.asarray(499)
    np.savez(evidence_path, **arrays)
    with pytest.raises(X.ExtractionRefused, match="adaptation steps"):
        X.validate_branch_evidence(evidence_path, branch=branch, parent=loaded.parent_by_id[key[0]])


def test_complete_preflight_is_outcome_blind_and_refuses_a_partial_cell(tmp_path):
    job = _parents_and_job(tmp_path, evidence=True)
    receipt = X.preflight_job(job)
    assert receipt["status"] == "COMPLETE"
    assert receipt["valid_branch_count"] == 64
    assert "arrays" not in json.dumps(receipt)
    loaded = X.load_job(job)
    _, branch = next(iter(loaded.branch_by_key.items()))
    pathlib.Path(branch["evidence_npz"]["path"]).unlink()
    receipt = X.preflight_job(job)
    assert receipt["status"] == "INVALID_INCOMPLETE"
    assert receipt["valid_branch_count"] == 63
