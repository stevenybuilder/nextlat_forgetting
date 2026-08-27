from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("evaluate_cfs2_tested", ROOT / "scripts/evaluate_cfs2.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file(root: Path, name: str, payload: bytes = b"opaque") -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _envelope(tmp_path: Path, module) -> Path:
    state = _file(tmp_path, "state-commitment.json")
    generator = _file(tmp_path, "generator.json")
    retention = _file(tmp_path, "retention.jsonl")
    stream = _file(tmp_path, "cfs2-stream.txt")
    global_controls = _file(tmp_path, "global.jsonl")
    evidence = _file(tmp_path, "evidence.npz")
    completion = _file(tmp_path, "completion.json")
    parent_checkpoint = _file(tmp_path, "parent.ckpt")
    branch_checkpoint = _file(tmp_path, "branch.ckpt")
    parent_ids = [f"nextlat-seed{seed}" + ("-base" if seed < 2000 else "-cfs2-base")
                  for seed in (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236)]
    branches = []
    for parent_id in parent_ids:
        for episode in (0, 1):
            for arm in ("high_different", "low_different", "high_same", "low_same"):
                overlap, relation = arm.split("_", 1)
                branches.append({
                    "job_id": f"cfs2-{parent_id}-{episode}-{arm}", "parent_id": parent_id,
                    "seed": 0, "episode": episode, "overlap": overlap, "future_relation": relation,
                    "completion_sha256": completion["sha256"], "parent_checkpoint_sha256": parent_checkpoint["sha256"],
                    "branch_checkpoint_sha256": branch_checkpoint["sha256"],
                })
    readiness_payload = {
        "schema": module.READINESS_SCHEMA, "status": "ALL_64_BRANCHES_TRAINED", "n_branches": 64,
        "update_manifest_sha256": "a" * 64,
        "evaluation_input_sha256s": {name: "b" * 64 for name in (
            "margin", "retention_ce", "retention_exact_path", "global_controls", "state_drift", "pregeometry")},
        "state_interchange_activation_patching": state, "branches": branches,
        "scientific_evaluation_started": False,
    }
    readiness = _file(tmp_path, "readiness.json", json.dumps(readiness_payload, sort_keys=True).encode())
    evidence_job = {
        "schema": module.EVIDENCE_JOB_SCHEMA, "analysis_seed": 17, "readiness": readiness,
        "state_interchange_activation_patching": state,
        "parents": [{"parent_id": parent_id, "base_checkpoint": parent_checkpoint, "base_training_steps": 20_000}
                    for parent_id in parent_ids],
        "branches": [{
            "job_id": branch["job_id"], "parent_id": branch["parent_id"], "episode": branch["episode"],
            "overlap": branch["overlap"], "future_relation": branch["future_relation"],
            "parent_checkpoint": parent_checkpoint, "branch_checkpoint": branch_checkpoint,
            "completion_receipt": completion, "generator_manifest": generator,
            "retention_probe_manifest": retention, "adaptation_stream_manifest": stream,
            "global_control_manifest": global_controls, "state_interchange_activation_patching": state,
            "pre_post_paired_endpoints": ["margin", "retention_ce", "retention_exact_path", "global_controls", "state_drift", "pregeometry"],
            "evidence_npz": evidence,
        } for branch in branches],
    }
    job = _file(tmp_path, "evidence-job.json", json.dumps(evidence_job, sort_keys=True).encode())
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps({"schema": module.EVALUATION_MANIFEST_SCHEMA, "analysis_seed": 17, "n_boot": 10_000,
                                    "readiness": readiness, "evidence_job": job}))
    return manifest


def test_cfs2_evaluator_contract_requires_complete_new_study_64_branch_envelope(tmp_path) -> None:
    module = _module()
    envelope = module.validate_evaluation_envelope(_envelope(tmp_path, module))
    assert envelope.branch_count == 64
    assert envelope.analysis_seed == 17


def test_cfs2_evaluator_refuses_partial_or_cross_study_evidence_job(tmp_path) -> None:
    module = _module(); manifest = _envelope(tmp_path, module)
    outer = json.loads(manifest.read_text())
    job_path = Path(outer["evidence_job"]["path"])
    job = json.loads(job_path.read_text())
    job["schema"] = "nextlat_forgetting/cfs1_evidence_extraction_job/1"
    job["branches"] = job["branches"][:-1]
    job_path.write_text(json.dumps(job, sort_keys=True))
    outer["evidence_job"]["sha256"] = hashlib.sha256(job_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(outer))
    with pytest.raises(module.CFS2EvaluationRefused, match="schema"):
        module.validate_evaluation_envelope(manifest)
