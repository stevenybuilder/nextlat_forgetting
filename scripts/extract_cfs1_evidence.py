#!/usr/bin/env python3
"""Hash-bound evidence contract for CFS-1 GPU extractors.

This module deliberately defines an *opaque* hand-off between the future CFS-1 generator /
durable runner and its evaluator.  It does not construct stimuli, train a model, or inspect
outcomes.  A GPU extractor may write the documented NPZ fields incrementally elsewhere, but
the evaluator accepts only a final atomic evidence file that passes this contract.

The strict boundary is important: a branch with the right loss curve but the wrong parent
checkpoint, update count, or stimulus manifest is not evidence for the randomized causal
intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from cfs1 import evaluate as E  # noqa: E402


JOB_SCHEMA = "nextlat_forgetting/cfs1_evidence_extraction_job/1"
BRANCH_EVIDENCE_SCHEMA = "nextlat_forgetting/cfs1_branch_evidence/1"
EXTRACTION_RECEIPT_SCHEMA = "nextlat_forgetting/cfs1_evidence_preflight/1"
N_RETENTION_PROBES = 2_000


class ExtractionRefused(RuntimeError):
    """Raised when an opaque CFS-1 evidence hand-off is incomplete or forged."""


@dataclass(frozen=True)
class BoundJob:
    path: pathlib.Path
    sha256: str
    payload: dict
    parent_by_id: dict[str, dict]
    branch_by_key: dict[tuple[str, int, str, str], dict]


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


def item_ids_sha256(values: np.ndarray) -> str:
    normalized = [str(value) for value in np.asarray(values).ravel().tolist()]
    if any("\n" in value or "\r" in value for value in normalized):
        raise ExtractionRefused("item IDs may not contain line breaks")
    return hashlib.sha256("".join(value + "\n" for value in normalized).encode("utf-8")).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _record(value: Any, *, base: pathlib.Path, label: str, require_file: bool = True) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ExtractionRefused(f"{label} must bind exactly path and SHA-256")
    raw_path, digest = value["path"], value["sha256"]
    if not isinstance(raw_path, str) or not _is_sha(digest):
        raise ExtractionRefused(f"{label} path/SHA-256 record is malformed")
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if require_file:
        if not path.is_file():
            raise ExtractionRefused(f"{label} is missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise ExtractionRefused(f"{label} SHA-256 mismatch: expected {digest}, got {actual}")
    return {"path": str(path), "sha256": digest}


def _require_int(value: Any, *, label: str, exact: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != exact:
        raise ExtractionRefused(f"{label} must be exactly {exact}")
    return int(value)


def _branch_key(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    try:
        parent_id = str(record["parent_id"])
        episode = int(record["episode"])
        overlap = str(record["overlap"])
        relation = str(record["future_relation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExtractionRefused("branch identity needs parent_id, episode, overlap, future_relation") from exc
    if episode not in E.EPISODES:
        raise ExtractionRefused(f"branch episode must be one of {list(E.EPISODES)}")
    E.condition_key(overlap, relation)
    if not parent_id:
        raise ExtractionRefused("branch parent_id may not be empty")
    return parent_id, episode, overlap, relation


def _validated_parent(record: Any, *, base: pathlib.Path) -> tuple[str, dict]:
    required = {"parent_id", "base_checkpoint", "base_training_steps"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ExtractionRefused(f"parent record must contain exactly {sorted(required)}")
    parent_id = record["parent_id"]
    if not isinstance(parent_id, str) or not parent_id:
        raise ExtractionRefused("parent_id must be a nonempty string")
    return parent_id, {
        "parent_id": parent_id,
        "base_checkpoint": _record(record["base_checkpoint"], base=base, label=f"parent {parent_id} checkpoint"),
        "base_training_steps": _require_int(
            record["base_training_steps"], label=f"parent {parent_id} base_training_steps",
            exact=E.BASE_TRAINING_STEPS,
        ),
    }


def _validated_branch(record: Any, *, base: pathlib.Path, parents: Mapping[str, dict]) -> tuple[tuple[str, int, str, str], dict]:
    required = {
        "parent_id", "episode", "overlap", "future_relation", "parent_checkpoint",
        "branch_checkpoint", "adaptation_steps", "generator_manifest",
        "retention_probe_manifest", "adaptation_stream_manifest", "global_control_manifest",
        "evidence_npz",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ExtractionRefused(f"branch record must contain exactly {sorted(required)}")
    key = _branch_key(record)
    parent_id, _, _, _ = key
    if parent_id not in parents:
        raise ExtractionRefused(f"branch refers to undeclared parent {parent_id!r}")
    parent_checkpoint = _record(record["parent_checkpoint"], base=base, label=f"branch {key} parent checkpoint")
    if parent_checkpoint != parents[parent_id]["base_checkpoint"]:
        raise ExtractionRefused(f"branch {key} parent checkpoint differs from its declared parent")
    return key, {
        "identity": {
            "parent_id": parent_id, "episode": key[1], "overlap": key[2],
            "future_relation": key[3],
        },
        "parent_checkpoint": parent_checkpoint,
        "branch_checkpoint": _record(record["branch_checkpoint"], base=base, label=f"branch {key} checkpoint"),
        "adaptation_steps": _require_int(
            record["adaptation_steps"], label=f"branch {key} adaptation_steps", exact=E.ADAPTATION_STEPS
        ),
        "generator_manifest": _record(record["generator_manifest"], base=base, label=f"branch {key} generator manifest"),
        "retention_probe_manifest": _record(record["retention_probe_manifest"], base=base, label=f"branch {key} retention probe manifest"),
        "adaptation_stream_manifest": _record(record["adaptation_stream_manifest"], base=base, label=f"branch {key} adaptation stream manifest"),
        "global_control_manifest": _record(record["global_control_manifest"], base=base, label=f"branch {key} global control manifest"),
        "evidence_npz": _record(record["evidence_npz"], base=base, label=f"branch {key} evidence NPZ", require_file=False),
    }


def load_job(path: os.PathLike[str] | str) -> BoundJob:
    """Validate the job's complete 64-branch lattice without loading outcomes."""
    path = pathlib.Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionRefused("CFS-1 extraction job is unreadable") from exc
    required = {"schema", "analysis_seed", "parents", "branches"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ExtractionRefused(f"job must contain exactly {sorted(required)}")
    if payload["schema"] != JOB_SCHEMA:
        raise ExtractionRefused(f"job schema must be {JOB_SCHEMA}")
    if isinstance(payload["analysis_seed"], bool) or not isinstance(payload["analysis_seed"], int) or payload["analysis_seed"] < 0:
        raise ExtractionRefused("analysis_seed must be a nonnegative integer")
    if not isinstance(payload["parents"], list) or not isinstance(payload["branches"], list):
        raise ExtractionRefused("parents and branches must be lists")
    parents: dict[str, dict] = {}
    for row in payload["parents"]:
        parent_id, parent = _validated_parent(row, base=path.parent)
        if parent_id in parents:
            raise ExtractionRefused(f"duplicate parent_id {parent_id!r}")
        parents[parent_id] = parent
    if len(parents) != E.EXPECTED_PARENT_COUNT:
        raise ExtractionRefused(f"job requires exactly {E.EXPECTED_PARENT_COUNT} parents")
    branches: dict[tuple[str, int, str, str], dict] = {}
    for row in payload["branches"]:
        key, branch = _validated_branch(row, base=path.parent, parents=parents)
        if key in branches:
            raise ExtractionRefused(f"duplicate branch identity {key}")
        branches[key] = branch
    expected = E.expected_branch_keys(tuple(parents))
    if set(branches) != expected:
        raise ExtractionRefused(
            f"job does not contain the atomic 64-branch lattice; missing={sorted(expected-set(branches))}, "
            f"extra={sorted(set(branches)-expected)}"
        )
    if len(branches) != 64:
        raise ExtractionRefused("CFS-1 job must contain exactly 64 branches")
    return BoundJob(
        path=path, sha256=sha256_file(path), payload=payload, parent_by_id=parents,
        branch_by_key=branches,
    )


def _scalar(z: Mapping[str, Any], key: str) -> str:
    if key not in z:
        raise ExtractionRefused(f"evidence missing {key}")
    value = np.asarray(z[key])
    if value.size != 1:
        raise ExtractionRefused(f"evidence {key} must be scalar")
    scalar = value.reshape(()).item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar)


def _scalar_int(z: Mapping[str, Any], key: str) -> int:
    raw = _scalar(z, key)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ExtractionRefused(f"evidence {key} must be an integer") from exc
    if str(value) != raw:
        raise ExtractionRefused(f"evidence {key} must canonically encode an integer")
    return value


def _finite_array(z: Mapping[str, Any], key: str, *, size: int) -> np.ndarray:
    if key not in z:
        raise ExtractionRefused(f"evidence missing {key}")
    value = np.asarray(z[key], dtype=np.float64)
    if value.ndim != 1 or value.size != size or not np.all(np.isfinite(value)):
        raise ExtractionRefused(f"evidence {key} must be a finite 1-D {size}-item array")
    return value


def validate_branch_evidence(
    evidence_path: os.PathLike[str] | str, *, branch: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and load one final evidence NPZ against its branch and parent contract."""
    path = pathlib.Path(evidence_path).resolve()
    if not path.is_file():
        raise ExtractionRefused(f"evidence NPZ is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            z = {key: np.asarray(loaded[key]) for key in loaded.files}
    except (OSError, ValueError) as exc:
        raise ExtractionRefused(f"evidence NPZ is unreadable: {path}") from exc
    expected_identity = branch["identity"]
    identity = {
        "parent_id": _scalar(z, "parent_id"),
        "episode": _scalar_int(z, "episode"),
        "overlap": _scalar(z, "overlap"),
        "future_relation": _scalar(z, "future_relation"),
    }
    if _scalar(z, "schema") != BRANCH_EVIDENCE_SCHEMA:
        raise ExtractionRefused(f"evidence schema must be {BRANCH_EVIDENCE_SCHEMA}")
    if identity != expected_identity:
        raise ExtractionRefused(f"evidence identity {identity} differs from branch identity {expected_identity}")
    E.condition_key(identity["overlap"], identity["future_relation"])
    if _scalar(z, "parent_checkpoint_sha256") != parent["base_checkpoint"]["sha256"]:
        raise ExtractionRefused("evidence parent checkpoint SHA-256 differs from declared parent")
    if _scalar_int(z, "parent_training_steps") != E.BASE_TRAINING_STEPS:
        raise ExtractionRefused("evidence parent training steps differ from frozen base contract")
    scalar_bindings = {
        "branch_checkpoint_sha256": branch["branch_checkpoint"]["sha256"],
        "generator_manifest_sha256": branch["generator_manifest"]["sha256"],
        "retention_probe_manifest_sha256": branch["retention_probe_manifest"]["sha256"],
        "adaptation_stream_manifest_sha256": branch["adaptation_stream_manifest"]["sha256"],
        "global_control_manifest_sha256": branch["global_control_manifest"]["sha256"],
    }
    for key, expected in scalar_bindings.items():
        if _scalar(z, key) != expected:
            raise ExtractionRefused(f"evidence {key} differs from branch-bound artifact")
    if _scalar_int(z, "adaptation_steps") != E.ADAPTATION_STEPS:
        raise ExtractionRefused("evidence adaptation steps differ from frozen 500-update contract")
    ids = np.asarray(z.get("retention_probe_item_ids", []))
    if ids.ndim != 1 or ids.size != N_RETENTION_PROBES or len(set(map(str, ids.tolist()))) != ids.size:
        raise ExtractionRefused(f"retention_probe_item_ids must be {N_RETENTION_PROBES} unique ordered IDs")
    ids_hash = item_ids_sha256(ids)
    if _scalar(z, "retention_probe_item_ids_sha256") != ids_hash:
        raise ExtractionRefused("retention probe ID hash does not match evidence IDs")
    arrays = {
        key: _finite_array(z, key, size=N_RETENTION_PROBES)
        for key in (
            "pre_correct_first_branch_margin", "post_correct_first_branch_margin",
            "pre_retention_cross_entropy", "post_retention_cross_entropy",
            "pre_retention_exact_path_accuracy", "post_retention_exact_path_accuracy",
            "adaptation_acquisition", "pre_global_control_margin", "post_global_control_margin",
            "penultimate_state_drift",
        )
    }
    if not np.all((arrays["pre_retention_exact_path_accuracy"] >= 0.0) & (arrays["pre_retention_exact_path_accuracy"] <= 1.0)) or not np.all((arrays["post_retention_exact_path_accuracy"] >= 0.0) & (arrays["post_retention_exact_path_accuracy"] <= 1.0)):
        raise ExtractionRefused("exact-path accuracy arrays must lie in [0, 1]")
    geometry = float(_scalar(z, "pre_adaptation_predictive_geometry"))
    if not np.isfinite(geometry):
        raise ExtractionRefused("pre_adaptation_predictive_geometry must be finite")
    patch_status = _scalar(z, "penultimate_state_patching_status")
    patch_report: dict[str, Any] = {"status": patch_status}
    if patch_status == "NOT_RUN":
        pass
    elif patch_status == "COMPLETE_WITH_NAMED_CONTROLS":
        controls = {
            key: float(_scalar(z, key))
            for key in (
                "patch_parent_state_effect", "patch_unrelated_anchor_effect",
                "patch_norm_matched_random_subspace_effect",
            )
        }
        if not all(np.isfinite(value) for value in controls.values()):
            raise ExtractionRefused("state-patching effects must be finite")
        patch_report["named_controls"] = controls
    else:
        raise ExtractionRefused(
            "penultimate_state_patching_status must be NOT_RUN or COMPLETE_WITH_NAMED_CONTROLS"
        )
    return {
        "path": str(path), "sha256": sha256_file(path), "identity": identity,
        "retention_probe_item_ids_sha256": ids_hash, "arrays": arrays,
        "pre_adaptation_predictive_geometry": geometry, "state_patching": patch_report,
    }


def preflight_job(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Read-only all-64 branch preflight, returning no outcome-derived statistics."""
    job = load_job(path)
    valid, invalid = [], []
    for key, branch in sorted(job.branch_by_key.items()):
        parent = job.parent_by_id[key[0]]
        try:
            evidence = validate_branch_evidence(branch["evidence_npz"]["path"], branch=branch, parent=parent)
            if evidence["sha256"] != branch["evidence_npz"]["sha256"]:
                raise ExtractionRefused("evidence NPZ SHA-256 differs from job record")
            valid.append({"identity": evidence["identity"], "path": evidence["path"], "sha256": evidence["sha256"]})
        except ExtractionRefused as exc:
            invalid.append({"identity": branch["identity"], "reason": str(exc)})
    return {
        "schema": EXTRACTION_RECEIPT_SCHEMA, "job": {"path": str(job.path), "sha256": job.sha256},
        "required_branch_count": 64, "valid_branch_count": len(valid), "invalid_branches": invalid,
        "status": "COMPLETE" if not invalid and len(valid) == 64 else "INVALID_INCOMPLETE",
        "valid_branch_identities": valid,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="opaque CFS-1 extraction/evaluation job JSON")
    args = ap.parse_args(argv)
    try:
        # Do not print branch metrics: this is an integrity preflight, not an evaluator.
        receipt = preflight_job(args.job)
    except ExtractionRefused as exc:
        print(f"[extract_cfs1_evidence] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"], "required_branch_count": receipt["required_branch_count"],
        "valid_branch_count": receipt["valid_branch_count"], "invalid_branch_count": len(receipt["invalid_branches"]),
    }, sort_keys=True))
    return 0 if receipt["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
