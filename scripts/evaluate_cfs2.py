#!/usr/bin/env python3
"""Fail-closed CFS-2 evaluator envelope validator.

This is intentionally an outcome-free gate: it authorizes a later CFS-2
extractor/evaluator only when it receives all 64 hash-bound, paired pre/post
branch hand-offs.  It neither accepts CFS-1 schemas nor opens NPZ/model values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Mapping

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cfs2.adaptation import (  # noqa: E402
    CFS2_ARMS, CFS2_EPISODES, CFS2_EVALUATION_INPUTS, CFS2_PARENT_SEEDS, sha256_file,
)


EVALUATION_MANIFEST_SCHEMA = "nextlat_forgetting/cfs2_evaluation_manifest/1"
EVIDENCE_JOB_SCHEMA = "nextlat_forgetting/cfs2_evidence_extraction_job/1"
READINESS_SCHEMA = "nextlat_forgetting/cfs2_pre_evaluation_readiness/1"
CONTRACT_RECEIPT_SCHEMA = "nextlat_forgetting/cfs2_evaluation_contract_receipt/1"


class CFS2EvaluationRefused(RuntimeError):
    """The CFS-2 evaluation envelope is incomplete, stale, or cross-study."""


@dataclasses.dataclass(frozen=True)
class CFS2EvaluationEnvelope:
    manifest_path: str
    manifest_sha256: str
    analysis_seed: int
    n_boot: int
    readiness: dict[str, str]
    evidence_job: dict[str, str]
    branch_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS2EvaluationRefused(message)


def _record(value: Any, *, base: pathlib.Path, label: str) -> dict[str, str]:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
             f"{label} must bind exactly path and SHA-256")
    raw_path, digest = value["path"], value["sha256"]
    _require(isinstance(raw_path, str) and isinstance(digest, str) and len(digest) == 64,
             f"{label} is malformed")
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    _require(path.is_file() and sha256_file(path) == digest, f"{label} is missing or stale")
    return {"path": str(path), "sha256": digest}


def _load_json(record: Mapping[str, str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(pathlib.Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2EvaluationRefused(f"{label} is unreadable") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    try:
        value = (str(row["parent_id"]), int(row["episode"]), str(row["overlap"]), str(row["future_relation"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CFS2EvaluationRefused("branch identity is malformed") from exc
    _require(value[1] in CFS2_EPISODES and f"{value[2]}_{value[3]}" in CFS2_ARMS and bool(value[0]),
             "branch identity is outside the CFS-2 lattice")
    return value


def _expected(parent_ids: set[str]) -> set[tuple[str, int, str, str]]:
    _require(len(parent_ids) == 8, "CFS-2 needs exactly eight parents")
    return {(parent, episode, *arm.split("_", 1)) for parent in parent_ids for episode in CFS2_EPISODES for arm in CFS2_ARMS}


def _validate_readiness(record: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _load_json(record, "CFS-2 readiness")
    required = {"schema", "status", "n_branches", "update_manifest_sha256", "evaluation_input_sha256s",
                "state_interchange_activation_patching", "branches", "scientific_evaluation_started"}
    _require(set(value) == required and value["schema"] == READINESS_SCHEMA and value["status"] == "ALL_64_BRANCHES_TRAINED",
             "CFS-2 readiness schema/status is invalid")
    _require(value["n_branches"] == 64 and value["scientific_evaluation_started"] is False,
             "CFS-2 readiness cannot authorize a partial or already-evaluated matrix")
    _require(isinstance(value["evaluation_input_sha256s"], Mapping) and set(value["evaluation_input_sha256s"]) == set(CFS2_EVALUATION_INPUTS),
             "CFS-2 readiness lacks exact paired endpoints")
    state = value["state_interchange_activation_patching"]
    _require(isinstance(state, Mapping) and set(state) == {"path", "sha256"}, "CFS-2 readiness lacks state-interchange commitment")
    branches = value["branches"]
    _require(isinstance(branches, list) and len(branches) == 64, "CFS-2 readiness must bind exactly 64 branches")
    by_id: dict[str, dict[str, Any]] = {}
    parent_ids: set[str] = set()
    for branch in branches:
        _require(isinstance(branch, Mapping) and isinstance(branch.get("job_id"), str), "CFS-2 readiness branch is malformed")
        key = _key(branch); parent_ids.add(key[0])
        _require(branch["job_id"] not in by_id, "duplicate CFS-2 readiness job ID")
        for name in ("completion_sha256", "parent_checkpoint_sha256", "branch_checkpoint_sha256"):
            _require(isinstance(branch.get(name), str) and len(branch[name]) == 64, f"CFS-2 readiness {name} is invalid")
        by_id[branch["job_id"]] = dict(branch)
    _require({_key(branch) for branch in by_id.values()} == _expected(parent_ids),
             "CFS-2 readiness does not cover the exact 64-branch lattice")
    return value, by_id


def _validate_evidence_job(record: Mapping[str, str], readiness_record: Mapping[str, str], readiness: Mapping[str, Any],
                           ready_by_id: Mapping[str, Mapping[str, Any]], analysis_seed: int) -> None:
    value = _load_json(record, "CFS-2 evidence job")
    required = {"schema", "analysis_seed", "readiness", "state_interchange_activation_patching", "parents", "branches"}
    _require(set(value) == required and value["schema"] == EVIDENCE_JOB_SCHEMA,
             "CFS-2 evidence job schema is invalid")
    _require(value["readiness"] == readiness_record and value["state_interchange_activation_patching"] == readiness["state_interchange_activation_patching"],
             "CFS-2 evidence job is not bound to the authorized readiness/state commitment")
    _require(isinstance(value["analysis_seed"], int) and not isinstance(value["analysis_seed"], bool) and value["analysis_seed"] == analysis_seed,
             "CFS-2 evidence job analysis seed is invalid")
    parents = value["parents"]
    _require(isinstance(parents, list) and len(parents) == 8, "CFS-2 evidence job needs eight parents")
    parent_by_id: dict[str, dict[str, str]] = {}
    for row in parents:
        _require(isinstance(row, Mapping) and set(row) == {"parent_id", "base_checkpoint", "base_training_steps"},
                 "CFS-2 evidence parent record is invalid")
        parent_id = row["parent_id"]
        _require(isinstance(parent_id, str) and parent_id and row["base_training_steps"] == 20_000 and parent_id not in parent_by_id,
                 "CFS-2 evidence parent identity/step is invalid")
        parent_by_id[parent_id] = _record(row["base_checkpoint"], base=pathlib.Path(record["path"]).parent, label="CFS-2 base checkpoint")
    _require(set(parent_by_id) == {row["parent_id"] for row in ready_by_id.values()}, "CFS-2 evidence parents differ from readiness")
    branch_required = {"job_id", "parent_id", "episode", "overlap", "future_relation", "parent_checkpoint", "branch_checkpoint",
                       "completion_receipt", "generator_manifest", "retention_probe_manifest", "adaptation_stream_manifest",
                       "global_control_manifest", "state_interchange_activation_patching", "pre_post_paired_endpoints", "evidence_npz"}
    branches = value["branches"]
    _require(isinstance(branches, list) and len(branches) == 64, "CFS-2 evidence job must contain 64 branches")
    keys = set()
    for row in branches:
        _require(isinstance(row, Mapping) and set(row) == branch_required, "CFS-2 evidence branch fields are invalid")
        job_id = row["job_id"]; _require(isinstance(job_id, str) and job_id in ready_by_id, "CFS-2 evidence job ID is not ready")
        key = _key(row); _require(key == _key(ready_by_id[job_id]) and key not in keys, "duplicate/mismatched CFS-2 evidence branch")
        keys.add(key); ready = ready_by_id[job_id]
        parent = _record(row["parent_checkpoint"], base=pathlib.Path(record["path"]).parent, label="CFS-2 branch parent checkpoint")
        branch = _record(row["branch_checkpoint"], base=pathlib.Path(record["path"]).parent, label="CFS-2 branch checkpoint")
        completion = _record(row["completion_receipt"], base=pathlib.Path(record["path"]).parent, label="CFS-2 completion receipt")
        _require(parent["sha256"] == ready["parent_checkpoint_sha256"] and branch["sha256"] == ready["branch_checkpoint_sha256"] and completion["sha256"] == ready["completion_sha256"],
                 "CFS-2 evidence branch checkpoint/completion differs from readiness")
        _require(parent == parent_by_id[key[0]], "CFS-2 evidence branch parent differs from parent table")
        for name in ("generator_manifest", "retention_probe_manifest", "adaptation_stream_manifest", "global_control_manifest", "state_interchange_activation_patching", "evidence_npz"):
            _record(row[name], base=pathlib.Path(record["path"]).parent, label=f"CFS-2 {name}")
        _require(tuple(row["pre_post_paired_endpoints"]) == CFS2_EVALUATION_INPUTS,
                 "CFS-2 evidence branch lacks exact paired endpoint commitment")
    _require(keys == _expected(set(parent_by_id)), "CFS-2 evidence job does not cover all 64 branches")


def validate_evaluation_envelope(path: os.PathLike[str] | str) -> CFS2EvaluationEnvelope:
    """Validate every non-outcome binding necessary before CFS-2 evaluation."""
    manifest_path = pathlib.Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2EvaluationRefused("CFS-2 evaluation manifest is unreadable") from exc
    required = {"schema", "analysis_seed", "n_boot", "readiness", "evidence_job"}
    _require(set(value) == required and value.get("schema") == EVALUATION_MANIFEST_SCHEMA,
             "CFS-2 evaluation manifest schema is invalid")
    _require(isinstance(value["analysis_seed"], int) and not isinstance(value["analysis_seed"], bool) and value["analysis_seed"] >= 0 and
             isinstance(value["n_boot"], int) and not isinstance(value["n_boot"], bool) and value["n_boot"] >= 100,
             "CFS-2 evaluation seed/bootstrap contract is invalid")
    readiness_record = _record(value["readiness"], base=manifest_path.parent, label="CFS-2 readiness")
    evidence_record = _record(value["evidence_job"], base=manifest_path.parent, label="CFS-2 evidence job")
    readiness, ready_by_id = _validate_readiness(readiness_record)
    _validate_evidence_job(evidence_record, readiness_record, readiness, ready_by_id, value["analysis_seed"])
    return CFS2EvaluationEnvelope(str(manifest_path), sha256_file(manifest_path), value["analysis_seed"], value["n_boot"], readiness_record, evidence_record, 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        envelope = validate_evaluation_envelope(args.manifest)
    except CFS2EvaluationRefused as exc:
        print(f"[evaluate_cfs2] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": CONTRACT_RECEIPT_SCHEMA, "status": "AUTHORIZED_FOR_COMPLETE_CFS2_EVALUATION",
                      "manifest_sha256": envelope.manifest_sha256, "branch_count": envelope.branch_count,
                      "scientific_outcomes_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
