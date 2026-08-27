#!/usr/bin/env python3
"""Run mandatory activation patching for the complete 64-branch CFS-2 matrix.

This orchestrator is deliberately outcome-blind: it validates the frozen training
readiness and checkpoint lineage, then schedules every branch exactly once.  It
never reads an effect value to decide whether a branch should run or be retained.
Completed per-branch artifacts are resumed only after their input hashes and
required arrays have been verified.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cfs2.adaptation import CFS2_ARMS, CFS2_EPISODES, CFS2_PARENT_SEEDS  # noqa: E402
from cfs2.patching import DEFAULT_PATCH_LAYERS, PATCH_EFFECT_NAMES, PATCH_POSITION  # noqa: E402


READINESS_SCHEMA = "nextlat_forgetting/cfs2_pre_evaluation_readiness/1"
LEDGER_SCHEMA = "nextlat_forgetting/cfs2_run_ledger/1"
COMPLETION_SCHEMA = "nextlat_forgetting/cfs2_training_completion/1"
PATCH_ARTIFACT_SCHEMA = "nextlat_forgetting/cfs2_activation_patching/1"
PATCH_MATRIX_SCHEMA = "nextlat_forgetting/cfs2_patching_matrix/1"
FINAL_STATUS = "ALL_64_BRANCHES_PATCHED"
TERMINAL = frozenset(("TRAINED", "DONE"))
CONTROL_STEMS = (
    "parent_state",
    "unrelated_anchor",
    "norm_matched_random_subspace",
)


class CFS2PatchingMatrixError(RuntimeError):
    """The full CFS-2 patching matrix or one of its artifacts is invalid."""


@dataclasses.dataclass(frozen=True)
class PatchJob:
    job_id: str
    parent_id: str
    seed: int
    episode: int
    overlap: str
    future_relation: str
    parent_checkpoint: str
    parent_checkpoint_sha256: str
    branch_checkpoint: str
    branch_checkpoint_sha256: str
    completion_sha256: str
    output: str


@dataclasses.dataclass(frozen=True)
class MatrixInputs:
    readiness_path: str
    readiness_sha256: str
    ledger_path: str
    ledger_sha256: str
    retention_path: str
    retention_sha256: str
    state_commitment: Mapping[str, str]
    jobs: tuple[PatchJob, ...]


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS2PatchingMatrixError(message)


def _json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2PatchingMatrixError(f"{label} is unreadable") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _checkpoint_path(value: Any, *, base: pathlib.Path, label: str) -> tuple[pathlib.Path, str]:
    _require(isinstance(value, Mapping), f"{label} record is missing")
    text, digest = value.get("path"), value.get("sha256")
    _require(isinstance(text, str) and text, f"{label} path is missing")
    _require(isinstance(digest, str) and len(digest) == 64, f"{label} SHA-256 is invalid")
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve(), digest


def _parent_id(seed: int) -> str:
    return f"nextlat-seed{seed}" + ("-base" if seed < 2000 else "-cfs2-base")


def _job_id(seed: int, episode: int, arm: str) -> str:
    return f"cfs2-nextlat-seed{seed}-episode{episode}-{arm}"


def expected_jobs() -> dict[str, tuple[str, int, int, str, str]]:
    expected: dict[str, tuple[str, int, int, str, str]] = {}
    for seed in CFS2_PARENT_SEEDS:
        for episode in CFS2_EPISODES:
            for arm in CFS2_ARMS:
                overlap, relation = arm.split("_", 1)
                expected[_job_id(seed, episode, arm)] = (
                    _parent_id(seed), seed, episode, overlap, relation
                )
    _require(len(expected) == 64, "internal CFS-2 patch lattice is not 64 branches")
    return expected


def load_matrix_inputs(
    readiness_path: os.PathLike[str] | str,
    ledger_path: os.PathLike[str] | str,
    retention_path: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
) -> MatrixInputs:
    """Validate all 64 checkpoint lineages before any inference can launch."""

    readiness_file = pathlib.Path(readiness_path).resolve()
    ledger_file = pathlib.Path(ledger_path).resolve()
    retention_file = pathlib.Path(retention_path).resolve()
    output_dir = pathlib.Path(output_root).resolve()
    _require(retention_file.is_file(), "CFS-2 retention manifest is absent")
    readiness = _json(readiness_file, "CFS-2 readiness")
    ledger = _json(ledger_file, "CFS-2 training ledger")
    _require(
        readiness.get("schema") == READINESS_SCHEMA
        and readiness.get("status") == "ALL_64_BRANCHES_TRAINED"
        and readiness.get("n_branches") == 64
        and readiness.get("scientific_evaluation_started") is False,
        "CFS-2 readiness does not authorize the complete untouched branch matrix",
    )
    state_commitment = readiness.get("state_interchange_activation_patching")
    _require(
        isinstance(state_commitment, Mapping)
        and set(state_commitment) == {"path", "sha256"}
        and isinstance(state_commitment.get("path"), str)
        and isinstance(state_commitment.get("sha256"), str)
        and len(state_commitment["sha256"]) == 64,
        "CFS-2 readiness lacks its frozen activation-patching commitment",
    )
    branches = readiness.get("branches")
    _require(isinstance(branches, list) and len(branches) == 64, "CFS-2 readiness must contain 64 branches")
    expected = expected_jobs()
    ready_by_id = {
        row.get("job_id"): row for row in branches if isinstance(row, Mapping) and isinstance(row.get("job_id"), str)
    }
    _require(set(ready_by_id) == set(expected), "CFS-2 readiness is not the exact frozen 64-branch lattice")
    _require(ledger.get("schema") == LEDGER_SCHEMA, "CFS-2 training ledger schema is invalid")
    entries = ledger.get("entries")
    _require(isinstance(entries, list), "CFS-2 training ledger has no append-only entries")
    latest = {
        row["job_id"]: row
        for row in entries
        if isinstance(row, Mapping) and isinstance(row.get("job_id"), str)
    }

    digest_cache: dict[pathlib.Path, str] = {}

    def verified(path: pathlib.Path, expected_sha: str, label: str) -> None:
        _require(path.is_file(), f"{label} is absent: {path}")
        if path not in digest_cache:
            digest_cache[path] = sha256_file(path)
        actual = digest_cache[path]
        _require(actual == expected_sha, f"{label} SHA-256 mismatch: {path}")

    jobs: list[PatchJob] = []
    for job_id in sorted(expected):
        parent_id, seed, episode, overlap, relation = expected[job_id]
        ready = ready_by_id[job_id]
        _require(
            ready.get("parent_id") == parent_id
            and ready.get("seed") == seed
            and ready.get("episode") == episode
            and ready.get("overlap") == overlap
            and ready.get("future_relation") == relation,
            f"{job_id} readiness identity differs from the frozen lattice",
        )
        state = latest.get(job_id)
        _require(isinstance(state, Mapping) and state.get("status") in TERMINAL, f"{job_id} is not terminal in the CFS-2 ledger")
        for field in ("completion_sha256", "parent_checkpoint_sha256", "branch_checkpoint_sha256"):
            _require(isinstance(ready.get(field), str) and len(ready[field]) == 64, f"{job_id} readiness {field} is invalid")
        _require(
            state.get("completion_sha256") == ready["completion_sha256"]
            and state.get("parent_checkpoint_sha256") == ready["parent_checkpoint_sha256"]
            and state.get("final_checkpoint_sha256") == ready["branch_checkpoint_sha256"],
            f"{job_id} ledger hashes differ from readiness",
        )
        out_root = state.get("out_root")
        _require(isinstance(out_root, str) and out_root, f"{job_id} ledger lacks out_root")
        completion_path = pathlib.Path(out_root).resolve() / "cfs2_training_completion.json"
        verified(completion_path, ready["completion_sha256"], f"{job_id} completion receipt")
        completion = _json(completion_path, f"{job_id} completion receipt")
        _require(
            completion.get("schema") == COMPLETION_SCHEMA
            and completion.get("job_id") == job_id
            and completion.get("parent_id") == parent_id
            and completion.get("scientific_evaluation_started") is False
            and completion.get("adaptation", {}).get("updates") == 500
            and completion.get("inputs", {}).get("state_interchange_activation_patching") == state_commitment,
            f"{job_id} completion identity is invalid",
        )
        _require(
            completion.get("parent_checkpoint", {}).get("training_steps") == 20_000
            and completion.get("branch_checkpoint", {}).get("training_steps") == 20_500,
            f"{job_id} completion does not bind the exact parent/+500 checkpoints",
        )
        parent_path, parent_sha = _checkpoint_path(
            completion.get("parent_checkpoint"), base=completion_path.parent, label=f"{job_id} parent checkpoint"
        )
        branch_path, branch_sha = _checkpoint_path(
            completion.get("branch_checkpoint"), base=completion_path.parent, label=f"{job_id} branch checkpoint"
        )
        _require(
            parent_sha == ready["parent_checkpoint_sha256"]
            and branch_sha == ready["branch_checkpoint_sha256"],
            f"{job_id} completion checkpoint hashes differ from readiness",
        )
        state_checkpoint = pathlib.Path(str(state.get("final_checkpoint", "")))
        if not state_checkpoint.is_absolute():
            state_checkpoint = completion_path.parent / state_checkpoint
        _require(state_checkpoint.resolve() == branch_path,
                 f"{job_id} ledger and completion name different branch checkpoints")
        verified(parent_path, parent_sha, f"{job_id} parent checkpoint")
        verified(branch_path, branch_sha, f"{job_id} branch checkpoint")
        jobs.append(PatchJob(
            job_id=job_id, parent_id=parent_id, seed=seed, episode=episode,
            overlap=overlap, future_relation=relation,
            parent_checkpoint=str(parent_path), parent_checkpoint_sha256=parent_sha,
            branch_checkpoint=str(branch_path), branch_checkpoint_sha256=branch_sha,
            completion_sha256=ready["completion_sha256"],
            output=str(output_dir / job_id / "activation_patching.npz"),
        ))

    return MatrixInputs(
        readiness_path=str(readiness_file), readiness_sha256=sha256_file(readiness_file),
        ledger_path=str(ledger_file), ledger_sha256=sha256_file(ledger_file),
        retention_path=str(retention_file), retention_sha256=sha256_file(retention_file),
        state_commitment=dict(state_commitment),
        jobs=tuple(jobs),
    )


def _scalar(arrays: Mapping[str, Any], name: str) -> Any:
    _require(name in arrays, f"patch artifact lacks {name}")
    value = arrays[name]
    _require(getattr(value, "shape", None) == (), f"patch artifact {name} must be scalar")
    return value.item()


def validate_patch_artifact(
    path: os.PathLike[str] | str,
    job: PatchJob,
    *,
    retention_sha256: str,
    layers: Sequence[int],
    analysis_seed: int,
) -> str:
    """Validate structure and frozen-input bindings without selecting on effects."""

    artifact = pathlib.Path(path).resolve()
    _require(artifact.is_file(), f"{job.job_id} patch artifact is absent")
    try:
        with np.load(artifact, allow_pickle=False) as arrays:
            _require(_scalar(arrays, "schema") == PATCH_ARTIFACT_SCHEMA, f"{job.job_id} patch schema is invalid")
            _require(_scalar(arrays, "branch_id") == job.job_id, f"{job.job_id} patch branch ID is invalid")
            _require(_scalar(arrays, "parent_checkpoint_sha256") == job.parent_checkpoint_sha256,
                     f"{job.job_id} patch parent hash is stale")
            _require(_scalar(arrays, "adapted_checkpoint_sha256") == job.branch_checkpoint_sha256,
                     f"{job.job_id} patch branch hash is stale")
            _require(_scalar(arrays, "retention_manifest_sha256") == retention_sha256,
                     f"{job.job_id} patch retention hash is stale")
            _require(int(_scalar(arrays, "patch_position")) == PATCH_POSITION,
                     f"{job.job_id} patch position is invalid")
            _require(int(_scalar(arrays, "analysis_seed")) == int(analysis_seed),
                     f"{job.job_id} patch analysis seed is invalid")
            actual_layers = tuple(int(value) for value in arrays["patch_layers"].tolist())
            _require(actual_layers == tuple(layers), f"{job.job_id} patch layers are invalid")
            probe_ids = arrays["probe_ids"]
            baseline = arrays["baseline_margin"]
            _require(probe_ids.ndim == baseline.ndim == 1 and len(probe_ids) > 0 and len(probe_ids) == len(baseline),
                     f"{job.job_id} patch probe/baseline shapes are invalid")
            for layer in layers:
                for stem in CONTROL_STEMS:
                    for suffix in ("margin", "effect"):
                        key = f"layer_{layer}_{stem}_margin" if suffix == "margin" else f"layer_{layer}_patch_{stem}_effect"
                        _require(key in arrays and arrays[key].shape == baseline.shape,
                                 f"{job.job_id} patch artifact lacks complete {key}")
            expected_effect_names = tuple(f"patch_{stem}_effect" for stem in CONTROL_STEMS)
            _require(expected_effect_names == PATCH_EFFECT_NAMES, "patch control names drifted from the frozen contract")
    except (OSError, ValueError, KeyError) as exc:
        raise CFS2PatchingMatrixError(f"{job.job_id} patch artifact is unreadable") from exc
    return sha256_file(artifact)


def command_for(job: PatchJob, inputs: MatrixInputs, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(REPO / "scripts" / "run_cfs2_patching.py"),
        "--branch-id", job.job_id,
        "--upstream-root", str(pathlib.Path(args.upstream_root).resolve()),
        "--config", str(pathlib.Path(args.config).resolve()),
        "--parent-checkpoint", job.parent_checkpoint,
        "--adapted-checkpoint", job.branch_checkpoint,
        "--retention-manifest", inputs.retention_path,
        "--output", job.output,
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--layers", ",".join(str(layer) for layer in args.patch_layers),
        "--analysis-seed", str(args.analysis_seed),
    ]


def _branch_entry(job: PatchJob, digest: str) -> dict[str, Any]:
    return {
        "job_id": job.job_id, "parent_id": job.parent_id, "seed": job.seed,
        "episode": job.episode, "overlap": job.overlap, "future_relation": job.future_relation,
        "path": job.output, "sha256": digest,
        "parent_checkpoint_sha256": job.parent_checkpoint_sha256,
        "branch_checkpoint_sha256": job.branch_checkpoint_sha256,
    }


def _index_value(
    inputs: MatrixInputs,
    completed: Mapping[str, Mapping[str, Any]],
    *,
    layers: Sequence[int],
    analysis_seed: int,
) -> dict[str, Any]:
    done = [dict(completed[job_id]) for job_id in sorted(completed)]
    return {
        "schema": PATCH_MATRIX_SCHEMA,
        "status": FINAL_STATUS if len(done) == 64 else "IN_PROGRESS",
        "n_branches": len(done), "expected_branches": 64,
        "readiness": {"path": inputs.readiness_path, "sha256": inputs.readiness_sha256},
        "ledger": {"path": inputs.ledger_path, "sha256": inputs.ledger_sha256},
        "retention_manifest": {"path": inputs.retention_path, "sha256": inputs.retention_sha256},
        "state_interchange_activation_patching": dict(inputs.state_commitment),
        "analysis_seed": int(analysis_seed), "patch_layers": list(layers), "branches": done,
        "outcome_filtering": False,
    }


def _load_prior_index(
    path: pathlib.Path,
    inputs: MatrixInputs,
    *,
    layers: Sequence[int],
    analysis_seed: int,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = _json(path, "CFS-2 patching matrix index")
    _require(value.get("schema") == PATCH_MATRIX_SCHEMA, "patching matrix index schema is invalid")
    _require(value.get("readiness") == {"path": inputs.readiness_path, "sha256": inputs.readiness_sha256},
             "patching index is bound to different readiness")
    _require(value.get("ledger") == {"path": inputs.ledger_path, "sha256": inputs.ledger_sha256},
             "patching index is bound to a different ledger")
    _require(value.get("retention_manifest") == {"path": inputs.retention_path, "sha256": inputs.retention_sha256},
             "patching index is bound to a different retention manifest")
    _require(value.get("state_interchange_activation_patching") == dict(inputs.state_commitment),
             "patching index is bound to a different activation-patching commitment")
    _require(value.get("patch_layers") == list(layers) and value.get("analysis_seed") == analysis_seed,
             "patching index analysis contract changed")
    rows = value.get("branches")
    _require(isinstance(rows, list), "patching index branches are invalid")
    by_id = {row.get("job_id"): dict(row) for row in rows if isinstance(row, Mapping)}
    _require(len(by_id) == len(rows), "patching index has duplicate or invalid branch rows")
    _require(set(by_id) <= {job.job_id for job in inputs.jobs}, "patching index contains an unknown branch")
    return by_id


Launcher = Callable[[Sequence[str]], int]


def run_matrix(
    inputs: MatrixInputs,
    index_path: os.PathLike[str] | str,
    args: argparse.Namespace,
    *,
    launcher: Launcher | None = None,
) -> dict[str, Any]:
    """Run/resume all branches; every successful branch atomically advances the index."""

    index = pathlib.Path(index_path).resolve()
    completed = _load_prior_index(
        index, inputs, layers=args.patch_layers, analysis_seed=args.analysis_seed
    )
    launch = launcher or (lambda command: subprocess.run(command, check=False).returncode)
    for job in inputs.jobs:
        artifact = pathlib.Path(job.output)
        digest: str | None = None
        if artifact.is_file():
            digest = validate_patch_artifact(
                artifact, job, retention_sha256=inputs.retention_sha256,
                layers=args.patch_layers, analysis_seed=args.analysis_seed,
            )
            prior = completed.get(job.job_id)
            if prior is not None:
                _require(prior.get("sha256") == digest and prior.get("path") == job.output,
                         f"{job.job_id} artifact differs from its recorded hash")
        if digest is None:
            command = command_for(job, inputs, args)
            returncode = int(launch(command))
            _require(returncode == 0, f"{job.job_id} patching exited with status {returncode}")
            digest = validate_patch_artifact(
                artifact, job, retention_sha256=inputs.retention_sha256,
                layers=args.patch_layers, analysis_seed=args.analysis_seed,
            )
        completed[job.job_id] = _branch_entry(job, digest)
        atomic_write_json(index, _index_value(
            inputs, completed, layers=args.patch_layers, analysis_seed=args.analysis_seed
        ))

    _require(len(completed) == 64, "patching sweep ended without all 64 branches")
    final = _index_value(inputs, completed, layers=args.patch_layers, analysis_seed=args.analysis_seed)
    atomic_write_json(index, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--index")
    parser.add_argument("--retention-manifest", default=str(REPO / "manifests/cfs2/retention.jsonl"))
    parser.add_argument("--upstream-root", default=str(REPO / "upstream/NextLat"))
    parser.add_argument("--config", default=str(REPO / "configs/cfs2_nextlat_adapt.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_PATCH_LAYERS))
    parser.add_argument("--analysis-seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the complete plan without inference")
    parser.add_argument("--smoke", action="store_true", help="also verify that the one-branch runner can start")
    args = parser.parse_args(argv)
    try:
        args.patch_layers = tuple(int(value) for value in args.layers.split(","))
    except ValueError:
        parser.error("--layers must be comma-separated integers")
    if args.batch_size <= 0 or args.patch_layers != DEFAULT_PATCH_LAYERS:
        parser.error("CFS-2 requires positive batches and exact patch layers 3,7,10")
    index = pathlib.Path(args.index).resolve() if args.index else pathlib.Path(args.output_root).resolve() / "cfs2_patching_matrix.json"
    try:
        inputs = load_matrix_inputs(args.readiness, args.ledger, args.retention_manifest, args.output_root)
        if args.smoke:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "run_cfs2_patching.py"), "--help"],
                check=False, stdout=subprocess.DEVNULL,
            )
            _require(result.returncode == 0, "one-branch patching runner smoke test failed")
        if args.dry_run or args.smoke:
            print(json.dumps({
                "status": "DRY_RUN_VALIDATED", "n_branches": len(inputs.jobs),
                "commands": [command_for(job, inputs, args) for job in inputs.jobs],
            }, sort_keys=True))
            return 0
        result = run_matrix(inputs, index, args)
    except (OSError, ValueError, CFS2PatchingMatrixError) as exc:
        print(f"[run_cfs2_patching_matrix] FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "n_branches": result["n_branches"], "index": str(index)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
