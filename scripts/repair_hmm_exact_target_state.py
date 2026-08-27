#!/usr/bin/env python3
"""Repair a poisoned HMM durability pointer without modifying checkpoint payloads.

An interrupted controller can leave the per-job checkpoint object at the exact
preregistered target while a later stale-ledger resume publishes a target+1 state.
This tool archives the bad commit record, independently hashes the already-existing
exact-target checkpoint and metadata objects, and publishes a minimal replacement
state that makes only those immutable objects resumable.  Training/evaluation metrics
are neither read nor interpreted.

The command is dry-run by default.  ``--apply`` is required to replace state.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tempfile
import time
from typing import Any


STATE_SCHEMA = "nextlat_forgetting/colab_state/1"
REPAIR_SCHEMA = "nextlat_forgetting/exact_target_state_repair/1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RepairError(RuntimeError):
    pass


def run(*args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise RepairError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return proc.stdout


def gcs_bytes(uri: str) -> bytes:
    return run("gcloud", "storage", "cat", uri)


def gcs_json(uri: str) -> dict[str, Any]:
    try:
        value = json.loads(gcs_bytes(uri))
    except json.JSONDecodeError as exc:
        raise RepairError(f"invalid JSON at {uri}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"expected a JSON object at {uri}")
    return value


def describe(uri: str) -> dict[str, Any]:
    try:
        value = json.loads(run(
            "gcloud", "storage", "objects", "describe", uri, "--format=json"
        ))
    except json.JSONDecodeError as exc:
        raise RepairError(f"invalid object metadata for {uri}") from exc
    if not isinstance(value, dict) or not value.get("generation"):
        raise RepairError(f"missing object generation for {uri}")
    return value


def unique_object(pattern: str) -> str:
    objects = [line.strip() for line in run(
        "gcloud", "storage", "ls", pattern
    ).decode("utf-8").splitlines() if line.strip()]
    if len(objects) != 1:
        raise RepairError(f"expected exactly one object for {pattern}, found {objects}")
    return objects[0]


def pinned_artifact(uri: str, local_path: str) -> tuple[dict[str, Any], bytes]:
    """Read one object at the generation observed by describe, never at latest twice."""
    metadata = describe(uri)
    generation = str(metadata["generation"])
    payload = gcs_bytes(f"{uri}#{generation}")
    size = len(payload)
    if int(metadata.get("size", -1)) != size:
        raise RepairError(f"size changed while reading {uri}")
    record = {
        "local_path": local_path,
        "remote": str(metadata["name"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": size,
        "generation": generation,
    }
    return record, payload


def artifact(uri: str, local_path: str) -> dict[str, Any]:
    return pinned_artifact(uri, local_path)[0]


def checkpoint_payload_step(payload: bytes) -> int:
    try:
        import torch
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RepairError("checkpoint payload cannot be deserialized") from exc
    if not isinstance(state, dict) or type(state.get("training_steps")) is not int:
        raise RepairError("checkpoint payload has no integer training_steps")
    return state["training_steps"]


def validate_checkpoint_bundle(
    checkpoint_uri: str, metadata_uri: str, *, job_id: str, target_step: int,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_name = checkpoint_uri.rsplit("/", 1)[-1]
    match = re.match(r"(?:recovery_)?ckpt_iter_(\d+)(?:_[^/]*)?\.pt$", checkpoint_name)
    if not match or int(match.group(1)) != target_step:
        raise RepairError(f"checkpoint filename does not encode exact target {target_step}")
    # Resolve and read each object once at its observed immutable generation.
    provisional_metadata, metadata_payload = pinned_artifact(metadata_uri, "/pending")
    try:
        sidecar = json.loads(metadata_payload)
    except json.JSONDecodeError as exc:
        raise RepairError("checkpoint metadata sidecar is invalid JSON") from exc
    local = str(sidecar.get("path", ""))
    if pathlib.PurePosixPath(local).name != checkpoint_name:
        raise RepairError("checkpoint sidecar path disagrees with object name")
    checkpoint_artifact, checkpoint_payload = pinned_artifact(checkpoint_uri, local)
    metadata_artifact = dict(provisional_metadata, local_path=local + ".meta.json")
    digest = checkpoint_artifact["sha256"]
    if (sidecar.get("run_id") != job_id or type(sidecar.get("step")) is not int or
            sidecar["step"] != target_step or
            sidecar.get("sha256") != digest or
            type(sidecar.get("size_bytes")) is not int or
            sidecar["size_bytes"] != checkpoint_artifact["size_bytes"]):
        raise RepairError("checkpoint sidecar identity/step/hash/size mismatch")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RepairError("checkpoint disagrees with committed state hash")
    if checkpoint_payload_step(checkpoint_payload) != target_step:
        raise RepairError("checkpoint payload training_steps does not equal exact target")
    return checkpoint_artifact, metadata_artifact


def upload_bytes(
    uri: str, payload: bytes, *, commit_metadata: bool = False,
    if_generation_match: str | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hmm-state-repair-") as temporary:
        source = pathlib.Path(temporary) / "payload.json"
        source.write_bytes(payload)
        command = ["gcloud", "storage", "cp"]
        if if_generation_match is not None:
            command.append("--if-generation-match=" + if_generation_match)
        if commit_metadata:
            command.append(
                "--custom-metadata=sha256=" + hashlib.sha256(payload).hexdigest()
            )
        run(*command, str(source), uri)


def repair_one(
    *, bucket: str, prefix: str, job_id: str, target_step: int,
    source_sha256: str, apply: bool,
) -> dict[str, Any]:
    state_uri = f"gs://{bucket}/{prefix}/runs/{job_id}/state.json"
    before_description = describe(state_uri)
    before_generation = str(before_description["generation"])
    before_bytes = gcs_bytes(f"{state_uri}#{before_generation}")
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    before_metadata = (before_description.get("metadata") or
                       before_description.get("custom_fields") or {})
    if before_metadata.get("sha256") != before_sha:
        raise RepairError(f"{job_id}: state metadata hash mismatch")
    before = json.loads(before_bytes)
    if before.get("schema") != STATE_SCHEMA or before.get("run_id") != job_id:
        raise RepairError(f"{job_id}: state identity/schema mismatch")
    if before.get("source_snapshot_sha256") != source_sha256:
        raise RepairError(f"{job_id}: state source is not the declared predecessor")
    if (before.get("status") in {"TRAINED", "RUNNING"} and
            type(before.get("step")) is int and before["step"] == target_step):
        checkpoint = before.get("checkpoint") or {}
        checkpoint_path = checkpoint.get("path")
        artifacts = {item.get("local_path"): item for item in before.get("artifacts", [])}
        committed_artifact = artifacts.get(checkpoint_path)
        committed_metadata = artifacts.get(str(checkpoint_path) + ".meta.json")
        if (not committed_artifact or not committed_artifact.get("remote") or
                not committed_metadata or not committed_metadata.get("remote") or
                not checkpoint.get("sha256") or checkpoint.get("size_bytes") is None or
                checkpoint.get("generation") is None or
                committed_artifact.get("generation") is None or
                committed_metadata.get("generation") is None):
            raise RepairError(f"{job_id}: exact state lacks its checkpoint artifact")
        checkpoint_uri = f"gs://{bucket}/{committed_artifact['remote']}"
        metadata_uri = f"gs://{bucket}/{committed_metadata['remote']}"
        verified_checkpoint, verified_metadata = validate_checkpoint_bundle(
            checkpoint_uri, metadata_uri, job_id=job_id,
            target_step=target_step, expected_sha256=checkpoint.get("sha256"),
        )
        for committed, verified, label in (
            (committed_artifact, verified_checkpoint, "checkpoint"),
            (committed_metadata, verified_metadata, "metadata"),
        ):
            for key in ("sha256", "size_bytes", "remote"):
                if committed.get(key) != verified.get(key):
                    raise RepairError(
                        f"{job_id}: committed {label} {key} disagrees with live object"
                    )
            if str(committed["generation"]) != str(verified["generation"]):
                raise RepairError(
                    f"{job_id}: committed {label} generation disagrees with live object"
                )
        if (checkpoint.get("sha256") != verified_checkpoint["sha256"] or
                type(checkpoint["size_bytes"]) is not int or
                checkpoint["size_bytes"] != verified_checkpoint["size_bytes"] or
                str(checkpoint["generation"]) != str(verified_checkpoint["generation"])):
            raise RepairError(f"{job_id}: checkpoint state record disagrees with artifact")
        candidates = before.get("recovery_candidates")
        if not isinstance(candidates, list) or not any(
            isinstance(candidate, dict) and
            candidate.get("path") == checkpoint_path and
            candidate.get("metadata_path") == str(checkpoint_path) + ".meta.json" and
            candidate.get("sha256") == verified_checkpoint["sha256"] and
            type(candidate.get("size_bytes")) is int and
            candidate["size_bytes"] == verified_checkpoint["size_bytes"] and
            str(candidate.get("generation")) == str(verified_checkpoint["generation"]) and
            candidate.get("metadata_sha256") == verified_metadata["sha256"] and
            type(candidate.get("metadata_size_bytes")) is int and
            candidate["metadata_size_bytes"] == verified_metadata["size_bytes"] and
            str(candidate.get("metadata_generation")) == str(verified_metadata["generation"]) and
            type(candidate.get("step")) is int and candidate["step"] == target_step
            for candidate in candidates
        ):
            raise RepairError(f"{job_id}: exact state lacks a complete recovery candidate")
        operational_repair = before.get("operational_repair") or {}
        return {
            "job_id": job_id,
            "action": (
                "already_exact_terminal" if before.get("status") == "TRAINED"
                else "already_exact_repaired"
            ),
            "before_sha256": before_sha,
            "checkpoint_sha256": checkpoint.get("sha256"),
            "checkpoint_generation": operational_repair.get(
                "selected_checkpoint_generation"
            ) or verified_checkpoint.get("generation"),
            "repaired_from_state_sha256": operational_repair.get(
                "repaired_from_state_sha256"
            ),
        }
    if (before.get("status") != "FAILED" or type(before.get("step")) is not int or
            before["step"] != target_step + 1):
        raise RepairError(
            f"{job_id}: refusing unexpected state {before.get('status')} "
            f"step={before.get('step')}"
        )

    checkpoint_uri = unique_object(
        f"gs://{bucket}/{prefix}/runs/{job_id}/{job_id}/ckpt_iter_{target_step}_*.pt"
    )
    metadata_uri = checkpoint_uri + ".meta.json"
    checkpoint_artifact, metadata_artifact = validate_checkpoint_bundle(
        checkpoint_uri, metadata_uri, job_id=job_id, target_step=target_step,
    )
    checkpoint_local = checkpoint_artifact["local_path"]

    repaired = {
        "schema": STATE_SCHEMA,
        "run_id": job_id,
        # RUNNING is deliberate: the patched runner must independently adopt and attest the
        # exact-target checkpoint before publishing TRAINED under the successor source.
        "status": "RUNNING",
        "step": target_step,
        "out_root": before["out_root"],
        "pointer": str(pathlib.PurePosixPath(before["out_root"]) / "latest_ckpt"),
        "checkpoint": {
            key: checkpoint_artifact[key]
            for key in ("local_path", "sha256", "size_bytes", "generation")
        },
        "recovery_candidates": [{
            "path": checkpoint_local,
            "metadata_path": checkpoint_local + ".meta.json",
            "sha256": checkpoint_artifact["sha256"],
            "size_bytes": checkpoint_artifact["size_bytes"],
            "generation": checkpoint_artifact["generation"],
            "metadata_sha256": metadata_artifact["sha256"],
            "metadata_size_bytes": metadata_artifact["size_bytes"],
            "metadata_generation": metadata_artifact["generation"],
            "step": target_step,
        }],
        "artifacts": [checkpoint_artifact, metadata_artifact],
        "source_snapshot_sha256": source_sha256,
        "synced_at": time.time(),
        "operational_repair": {
            "schema": REPAIR_SCHEMA,
            "reason": "stale ledger resumed an already complete checkpoint to target+1",
            "repaired_from_state_sha256": before_sha,
            "discarded_step": target_step + 1,
            "selected_step": target_step,
            "selected_checkpoint_generation": checkpoint_artifact["generation"],
            "scientific_metrics_inspected": False,
        },
    }
    # RuntimeDurability expects `path`, not `local_path`, in checkpoint.
    repaired["checkpoint"]["path"] = repaired["checkpoint"].pop("local_path")
    repaired_bytes = (json.dumps(repaired, indent=2, sort_keys=True) + "\n").encode("utf-8")
    repaired_sha = hashlib.sha256(repaired_bytes).hexdigest()
    archive_uri = (
        f"gs://{bucket}/{prefix}/recovery_audit/exact-target-{target_step}/"
        f"{job_id}/state-before-{before_sha}.json"
    )

    if apply:
        try:
            archived = gcs_bytes(archive_uri)
        except RepairError:
            upload_bytes(archive_uri, before_bytes)
            archived = gcs_bytes(archive_uri)
        if archived != before_bytes:
            raise RepairError(f"{job_id}: recovery archive content mismatch")
        # Refuse a race between audit and replacement.
        if hashlib.sha256(gcs_bytes(f"{state_uri}#{before_generation}")).hexdigest() != before_sha:
            raise RepairError(f"{job_id}: state changed before repair commit")
        upload_bytes(
            state_uri, repaired_bytes, commit_metadata=True,
            if_generation_match=before_generation,
        )
        if gcs_bytes(state_uri) != repaired_bytes:
            raise RepairError(f"{job_id}: repaired state readback mismatch")
        committed = describe(state_uri)
        custom = committed.get("metadata") or committed.get("custom_fields") or {}
        if custom.get("sha256") != repaired_sha:
            raise RepairError(f"{job_id}: repaired state metadata hash mismatch")

    return {
        "job_id": job_id,
        "action": "repaired" if apply else "would_repair",
        "before_sha256": before_sha,
        "after_sha256": repaired_sha,
        "archive_uri": archive_uri,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_generation": checkpoint_artifact["generation"],
        "checkpoint_sha256": checkpoint_artifact["sha256"],
        "metadata_sha256": metadata_artifact["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="lurestar")
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if not SHA256_RE.fullmatch(args.source_sha256):
        raise SystemExit("--source-sha256 must be 64 lowercase hex characters")
    if args.target_step <= 0:
        raise SystemExit("--target-step must be positive")
    if len(set(args.job_id)) != len(args.job_id):
        raise SystemExit("duplicate --job-id")

    results = [repair_one(
        bucket=args.bucket,
        prefix=args.prefix.strip("/"),
        job_id=job_id,
        target_step=args.target_step,
        source_sha256=args.source_sha256,
        apply=args.apply,
    ) for job_id in args.job_id]
    receipt = {
        "schema": REPAIR_SCHEMA,
        "mode": "apply" if args.apply else "dry_run",
        "source_snapshot_sha256": args.source_sha256,
        "target_step": args.target_step,
        "scientific_metrics_inspected": False,
        "jobs": results,
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        pathlib.Path(args.receipt).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
