#!/usr/bin/env python3
"""Run the fixed CFS-2 analysis after all 64 evidence and patch cells exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cfs2 import analysis as A  # noqa: E402
from cfs2.adaptation import CFS2_UPDATE_STEPS, sha256_file  # noqa: E402
from evaluate_cfs2 import validate_evaluation_envelope  # noqa: E402


BRANCH_EVIDENCE_SCHEMA = "nextlat_forgetting/cfs2_branch_evidence/1"
PATCH_MATRIX_SCHEMA = "nextlat_forgetting/cfs2_patching_matrix/1"
PATCH_ARTIFACT_SCHEMA = "nextlat_forgetting/cfs2_activation_patching/1"
N_RETENTION_PROBES = 2_000


class CFS2ScientificEvaluationRefused(RuntimeError):
    """The all-branch scientific input contract is incomplete or stale."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS2ScientificEvaluationRefused(message)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _record(value: Any, *, base: pathlib.Path, label: str) -> dict[str, str]:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "sha256"},
        f"{label} must bind exactly path and SHA-256",
    )
    path = pathlib.Path(str(value["path"]))
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    digest = value["sha256"]
    _require(path.is_file() and _is_sha(digest), f"{label} is missing or malformed")
    _require(sha256_file(path) == digest, f"{label} SHA-256 mismatch")
    return {"path": str(path), "sha256": str(digest)}


def _scalar(z: Mapping[str, np.ndarray], key: str) -> Any:
    _require(key in z, f"NPZ is missing scalar {key}")
    value = np.asarray(z[key])
    _require(value.size == 1, f"NPZ {key} must be scalar")
    scalar = value.reshape(()).item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else scalar


def _scalar_text(z: Mapping[str, np.ndarray], key: str) -> str:
    return str(_scalar(z, key))


def _scalar_int(z: Mapping[str, np.ndarray], key: str) -> int:
    raw = _scalar(z, key)
    _require(
        not isinstance(raw, (bool, np.bool_)) and isinstance(raw, (int, np.integer)),
        f"NPZ {key} must be an integer",
    )
    return int(raw)


def _finite(z: Mapping[str, np.ndarray], key: str, *, size: int) -> np.ndarray:
    _require(key in z, f"NPZ is missing {key}")
    value = np.asarray(z[key], dtype=np.float64)
    _require(
        value.ndim == 1 and value.size == size and np.all(np.isfinite(value)),
        f"NPZ {key} must be a finite {size}-item vector",
    )
    return value


def _item_ids_sha256(values: np.ndarray) -> str:
    normalized = [str(value) for value in np.asarray(values).ravel().tolist()]
    _require(not any("\n" in value or "\r" in value for value in normalized), "probe IDs contain line breaks")
    return hashlib.sha256("".join(value + "\n" for value in normalized).encode("utf-8")).hexdigest()


def _load_npz(path: pathlib.Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError) as exc:
        raise CFS2ScientificEvaluationRefused(f"{label} is unreadable") from exc


def _load_branch_evidence(row: Mapping[str, Any], *, base: pathlib.Path) -> A.BranchOutcome:
    evidence = _record(row["evidence_npz"], base=base, label=f"{row.get('job_id')} evidence")
    z = _load_npz(pathlib.Path(evidence["path"]), label=f"{row.get('job_id')} evidence")
    identity = {
        "job_id": str(row["job_id"]),
        "parent_id": str(row["parent_id"]),
        "episode": int(row["episode"]),
        "overlap": str(row["overlap"]),
        "future_relation": str(row["future_relation"]),
    }
    _require(_scalar_text(z, "schema") == BRANCH_EVIDENCE_SCHEMA, "unexpected CFS-2 branch-evidence schema")
    for key in ("job_id", "parent_id", "overlap", "future_relation"):
        _require(_scalar_text(z, key) == identity[key], f"evidence {key} differs from evidence job")
    _require(_scalar_int(z, "episode") == identity["episode"], "evidence episode differs from evidence job")
    _require(_scalar_int(z, "adaptation_steps") == CFS2_UPDATE_STEPS, "evidence must bind exactly 500 updates")
    scalar_bindings = {
        "parent_checkpoint_sha256": row["parent_checkpoint"]["sha256"],
        "branch_checkpoint_sha256": row["branch_checkpoint"]["sha256"],
        "generator_manifest_sha256": row["generator_manifest"]["sha256"],
        "retention_probe_manifest_sha256": row["retention_probe_manifest"]["sha256"],
        "adaptation_stream_manifest_sha256": row["adaptation_stream_manifest"]["sha256"],
        "global_control_manifest_sha256": row["global_control_manifest"]["sha256"],
    }
    for key, expected in scalar_bindings.items():
        _require(_scalar_text(z, key) == expected, f"evidence {key} differs from its hash-bound input")
    ids = np.asarray(z.get("retention_probe_item_ids", []))
    _require(
        ids.ndim == 1
        and ids.size == N_RETENTION_PROBES
        and len(set(map(str, ids.tolist()))) == ids.size,
        f"evidence must contain {N_RETENTION_PROBES} unique ordered probe IDs",
    )
    _require(
        _scalar_text(z, "retention_probe_item_ids_sha256") == _item_ids_sha256(ids),
        "evidence probe-ID hash mismatch",
    )
    known_arrays = (
        "pre_correct_first_branch_margin",
        "post_correct_first_branch_margin",
        "pre_retention_cross_entropy",
        "post_retention_cross_entropy",
        "pre_retention_exact_path_accuracy",
        "post_retention_exact_path_accuracy",
        "adaptation_acquisition",
        "pre_global_control_margin",
        "post_global_control_margin",
        "penultimate_state_drift",
    )
    arrays = {key: _finite(z, key, size=ids.size) for key in known_arrays if key in z}
    if "pre_retention_exact_path_accuracy" in arrays:
        for key in ("pre_retention_exact_path_accuracy", "post_retention_exact_path_accuracy"):
            _require(np.all((arrays[key] >= 0.0) & (arrays[key] <= 1.0)), f"{key} lies outside [0,1]")
    geometry = None
    if "pre_adaptation_predictive_geometry" in z:
        geometry = float(_scalar(z, "pre_adaptation_predictive_geometry"))
        _require(np.isfinite(geometry), "pregeometry is nonfinite")
    return A.BranchOutcome(
        job_id=identity["job_id"],
        parent_id=identity["parent_id"],
        episode=identity["episode"],
        overlap=identity["overlap"],
        future_relation=identity["future_relation"],
        item_ids=ids,
        arrays=arrays,
        pregeometry=geometry,
    )


def _patch_manifest(
    path: pathlib.Path,
    *,
    evidence_by_job: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, str],
    analysis_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2ScientificEvaluationRefused("patch matrix manifest is unreadable") from exc
    _require(isinstance(payload, Mapping), "patch matrix manifest must be an object")
    _require(payload.get("schema") == PATCH_MATRIX_SCHEMA, "unexpected patch matrix schema")
    _require(payload.get("status") == "ALL_64_BRANCHES_PATCHED", "patch matrix is not complete")
    _require(payload.get("n_branches") == 64, "patch matrix must bind 64 branches")
    _require(payload.get("expected_branches") == 64, "patch matrix expected-branch contract changed")
    _require(payload.get("outcome_filtering") is False, "patch sweep may not filter branches by outcomes")
    _require(payload.get("analysis_seed") == analysis_seed, "patch and evaluation analysis seeds differ")
    _require(tuple(payload.get("patch_layers", ())) == A.PATCH_LAYERS, "patch matrix must retain layers 3, 7, and 10")
    patch_readiness = _record(payload.get("readiness"), base=path.parent, label="patch readiness")
    _require(patch_readiness["sha256"] == readiness["sha256"], "patch matrix is bound to different readiness")
    readiness_payload = json.loads(pathlib.Path(patch_readiness["path"]).read_text(encoding="utf-8"))
    expected_state = readiness_payload.get("state_interchange_activation_patching")
    patch_state = _record(
        payload.get("state_interchange_activation_patching"),
        base=path.parent,
        label="patch state-interchange commitment",
    )
    _require(
        isinstance(expected_state, Mapping) and patch_state["sha256"] == expected_state.get("sha256"),
        "patch matrix is bound to a different state-interchange commitment",
    )
    _record(payload.get("ledger"), base=path.parent, label="patch ledger")
    retention = _record(payload.get("retention_manifest"), base=path.parent, label="patch retention manifest")
    evidence_retention_hashes = {
        row.get("retention_probe_manifest", {}).get("sha256")
        for row in evidence_by_job.values()
        if isinstance(row.get("retention_probe_manifest"), Mapping)
    }
    _require(
        evidence_retention_hashes == {retention["sha256"]},
        "patch and evidence matrices use different retention manifests",
    )
    rows = payload.get("branches")
    _require(isinstance(rows, list) and len(rows) == 64, "patch matrix needs exactly 64 branch records")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "patch branch record is malformed")
        required = {
            "job_id", "parent_id", "episode", "overlap", "future_relation", "path", "sha256",
            "parent_checkpoint_sha256", "branch_checkpoint_sha256",
        }
        _require(required <= set(row) and set(row) <= required | {"seed"}, "patch branch record fields are invalid")
        job_id = row["job_id"]
        _require(isinstance(job_id, str) and job_id in evidence_by_job and job_id not in seen, "patch job is duplicate or unknown")
        seen.add(job_id)
        evidence = evidence_by_job[job_id]
        for key in ("parent_id", "episode", "overlap", "future_relation"):
            _require(row[key] == evidence[key], f"patch {job_id} identity differs from evidence job")
        _require(
            row["parent_checkpoint_sha256"] == evidence["parent_checkpoint"]["sha256"]
            and row["branch_checkpoint_sha256"] == evidence["branch_checkpoint"]["sha256"],
            f"patch {job_id} checkpoint hashes differ from evidence job",
        )
        artifact = _record(
            {"path": row["path"], "sha256": row["sha256"]},
            base=path.parent,
            label=f"patch {job_id}",
        )
        normalized.append(dict(row) | {"path": artifact["path"], "retention_sha256": retention["sha256"]})
    _require(seen == set(evidence_by_job), "patch matrix does not cover the evidence matrix")
    return normalized, dict(payload)


def _load_patch(row: Mapping[str, Any], branch: A.BranchOutcome, *, analysis_seed: int) -> A.PatchOutcome:
    z = _load_npz(pathlib.Path(row["path"]), label=f"{row['job_id']} patch artifact")
    _require(_scalar_text(z, "schema") == PATCH_ARTIFACT_SCHEMA, "unexpected patch artifact schema")
    _require(_scalar_text(z, "branch_id") == row["job_id"], "patch artifact branch ID mismatch")
    _require(_scalar_int(z, "analysis_seed") == analysis_seed, "patch artifact analysis seed mismatch")
    _require(_scalar_text(z, "parent_checkpoint_sha256") == row["parent_checkpoint_sha256"], "patch parent hash mismatch")
    _require(_scalar_text(z, "adapted_checkpoint_sha256") == row["branch_checkpoint_sha256"], "patch branch hash mismatch")
    _require(_scalar_text(z, "retention_manifest_sha256") == row["retention_sha256"], "patch retention hash mismatch")
    layers = tuple(int(value) for value in np.asarray(z.get("patch_layers", [])).tolist())
    _require(layers == A.PATCH_LAYERS, "patch artifact does not retain exact layers 3, 7, and 10")
    _require(_scalar_int(z, "patch_position") == 63, "patch artifact did not intervene at index 63")
    probes = np.asarray(z.get("probe_ids", []))
    _require(
        np.array_equal(probes.astype(str), np.asarray(branch.item_ids).astype(str)),
        "patch artifact probe order differs from evidence",
    )
    source_names = {
        "matching_parent": "parent_state",
        "unrelated_anchor": "unrelated_anchor",
        "norm_matched_random_subspace": "norm_matched_random_subspace",
    }
    effects = {
        layer: {
            control: _finite(
                z,
                f"layer_{layer}_patch_{source}_effect",
                size=probes.size,
            )
            for control, source in source_names.items()
        }
        for layer in A.PATCH_LAYERS
    }
    return A.PatchOutcome(
        job_id=branch.job_id,
        parent_id=branch.parent_id,
        episode=branch.episode,
        overlap=branch.overlap,
        future_relation=branch.future_relation,
        probe_ids=probes,
        effects=effects,
    )


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(
    manifest_path: os.PathLike[str] | str,
    patch_manifest_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    envelope = validate_evaluation_envelope(manifest_path)
    job_path = pathlib.Path(envelope.evidence_job["path"])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    branch_rows = job["branches"]
    evidence_by_job = {row["job_id"]: row for row in branch_rows}
    _require(len(evidence_by_job) == 64, "evidence job IDs are not unique")

    # Complete hash preflight happens before opening the first scientific value.
    for row in branch_rows:
        _record(row["evidence_npz"], base=job_path.parent, label=f"{row['job_id']} evidence")
    patch_rows, patch_manifest = _patch_manifest(
        pathlib.Path(patch_manifest_path).resolve(),
        evidence_by_job=evidence_by_job,
        readiness=envelope.readiness,
        analysis_seed=envelope.analysis_seed,
    )

    branches = [_load_branch_evidence(row, base=job_path.parent) for row in branch_rows]
    branch_by_job = {branch.job_id: branch for branch in branches}
    patches = [
        _load_patch(row, branch_by_job[row["job_id"]], analysis_seed=envelope.analysis_seed)
        for row in patch_rows
    ]
    report = A.analyze_complete_matrix(
        branches,
        patches,
        analysis_seed=envelope.analysis_seed,
        n_boot=envelope.n_boot,
    )
    report["inputs"] = {
        "evaluation_manifest": {"path": envelope.manifest_path, "sha256": envelope.manifest_sha256},
        "evidence_job": envelope.evidence_job,
        "patch_manifest": {
            "path": str(pathlib.Path(patch_manifest_path).resolve()),
            "sha256": sha256_file(patch_manifest_path),
        },
        "patch_manifest_status": patch_manifest["status"],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="complete CFS-2 evaluation envelope")
    parser.add_argument("--patch-manifest", required=True, help="complete 64-branch patch sweep manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = pathlib.Path(args.output).resolve()
    try:
        report = evaluate(args.manifest, args.patch_manifest)
        _atomic_json(output, report)
    except (CFS2ScientificEvaluationRefused, A.CFS2AnalysisError, OSError, ValueError) as exc:
        print(f"[run_cfs2_analysis] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
