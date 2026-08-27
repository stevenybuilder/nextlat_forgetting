#!/usr/bin/env python3
"""Outcome-blind, generation-exact GCS soft-delete recovery.

Every current live object is archived by generation and content hash before the exact
declared soft-deleted generation is restored.  The restored bytes are verified against
the plan and copied to a content-addressed audit path.  Dry-run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any


PLAN_SCHEMA = "nextlat_forgetting/gcs_soft_delete_recovery_plan/1"
RECEIPT_SCHEMA = "nextlat_forgetting/gcs_soft_delete_recovery_receipt/1"
SHA_RE = re.compile(r"[0-9a-f]{64}")


class RecoveryError(RuntimeError):
    pass


def command(*args: str) -> bytes:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RecoveryError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )
    return proc.stdout


def describe(uri: str) -> dict[str, Any]:
    value = json.loads(command(
        "gcloud", "storage", "objects", "describe", uri, "--format=json"
    ))
    if not isinstance(value, dict) or not value.get("generation"):
        raise RecoveryError(f"invalid live object description: {uri}")
    return value


def object_bytes(uri: str) -> bytes:
    return command("gcloud", "storage", "cat", uri)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def soft_deleted_record(uri: str, generation: str) -> dict[str, Any]:
    records = json.loads(command(
        "gcloud", "storage", "ls", "--soft-deleted", "--json", uri
    ))
    matched = [record for record in records
               if str((record.get("metadata") or {}).get("generation")) == generation]
    if len(matched) != 1:
        raise RecoveryError(
            f"expected one soft-deleted generation {generation} for {uri}, found {len(matched)}"
        )
    return matched[0]


def archive_uri(uri: str, generation: str, digest: str, *, kind: str) -> str:
    if not uri.startswith("gs://"):
        raise RecoveryError(f"not a GCS URI: {uri}")
    bucket, name = uri[5:].split("/", 1)
    return (
        f"gs://{bucket}/lurestar/recovery_audit/soft-delete/{kind}/"
        f"{name.removeprefix('lurestar/')}--g{generation}--sha256-{digest}"
    )


def transaction_uri(uri: str, generation: str) -> str:
    bucket, _name = uri[5:].split("/", 1)
    identity = hashlib.sha256(f"{uri}#{generation}".encode("utf-8")).hexdigest()
    return (
        f"gs://{bucket}/lurestar/recovery_audit/soft-delete/transactions/"
        f"{identity}.json"
    )


def copy_and_verify(source: str, destination: str, expected: bytes) -> None:
    command("gcloud", "storage", "cp", source, destination)
    if object_bytes(destination) != expected:
        raise RecoveryError(f"archive readback mismatch: {destination}")


def recover_one(record: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    uri = record.get("uri")
    generation = str(record.get("generation", ""))
    expected_sha = record.get("sha256")
    if (not isinstance(uri, str) or not uri.startswith("gs://") or
            not generation.isdigit() or not SHA_RE.fullmatch(str(expected_sha))):
        raise RecoveryError(f"invalid recovery-plan record: {record!r}")

    transaction_object = transaction_uri(uri, generation)
    try:
        live = describe(uri)
    except RecoveryError:
        # A previous invocation may have archived and generation-conditionally deleted the
        # live object before its restore call was interrupted. Only our verified transaction
        # commit authorizes resuming across that intentionally narrow absent-live window.
        try:
            transaction = json.loads(object_bytes(transaction_object))
        except (RecoveryError, json.JSONDecodeError) as exc:
            raise RecoveryError(
                f"{uri} is absent and has no valid recovery transaction"
            ) from exc
        if (transaction.get("schema") != RECEIPT_SCHEMA or
                transaction.get("uri") != uri or
                transaction.get("requested_generation") != generation or
                transaction.get("expected_sha256") != expected_sha or
                not SHA_RE.fullmatch(str(transaction.get("previous_live_sha256", ""))) or
                not str(transaction.get("previous_live_generation", "")).isdigit()):
            raise RecoveryError(f"invalid interrupted recovery transaction: {transaction_object}")
        soft = soft_deleted_record(uri, generation)
        declared = ((soft.get("metadata") or {}).get("metadata") or {}).get("sha256")
        if declared != expected_sha:
            raise RecoveryError(f"soft-deleted generation changed during retry: {uri}")
        result = dict(transaction)
        result.pop("schema", None)
        result.update({
            "action": "would_resume_restore" if not apply else "resuming_restore",
            "transaction_uri": transaction_object,
        })
        if not apply:
            return result
        command(
            "gcloud", "storage", "restore", f"{uri}#{generation}",
            "--if-generation-match=0",
        )
        restored = describe(uri)
        restored_generation = str(restored["generation"])
        restored_payload = object_bytes(f"{uri}#{restored_generation}")
        if sha256(restored_payload) != expected_sha:
            raise RecoveryError(f"resumed restore hash mismatch: {uri}")
        restored_metadata = restored.get("metadata") or restored.get("custom_fields") or {}
        if restored_metadata.get("sha256") != expected_sha:
            raise RecoveryError(f"resumed restore metadata hash mismatch: {uri}")
        restored_archive = transaction["restored_archive"]
        try:
            archived = object_bytes(restored_archive)
        except RecoveryError:
            copy_and_verify(f"{uri}#{restored_generation}", restored_archive, restored_payload)
        else:
            if archived != restored_payload:
                raise RecoveryError(f"existing restored archive disagrees: {restored_archive}")
        result.update({
            "action": "restored_after_interruption",
            "restored_live_generation": restored_generation,
            "restored_live_sha256": expected_sha,
        })
        return result

    live_generation = str(live["generation"])
    live_payload = object_bytes(f"{uri}#{live_generation}")
    live_sha = sha256(live_payload)
    before_archive = archive_uri(uri, live_generation, live_sha, kind="before-restore")
    restored_archive = archive_uri(uri, generation, expected_sha, kind="restored-exact")
    if live_sha == expected_sha:
        live_metadata = live.get("metadata") or live.get("custom_fields") or {}
        if live_metadata.get("sha256") != expected_sha:
            raise RecoveryError(f"already-restored live metadata hash mismatch: {uri}")
        if apply:
            try:
                archived = object_bytes(restored_archive)
            except RecoveryError:
                copy_and_verify(
                    f"{uri}#{live_generation}", restored_archive, live_payload
                )
            else:
                if archived != live_payload:
                    raise RecoveryError(
                        f"existing restored archive disagrees: {restored_archive}"
                    )
        return {
            "uri": uri,
            "requested_generation": generation,
            "expected_sha256": expected_sha,
            "restored_live_generation": live_generation,
            "restored_live_sha256": live_sha,
            "restored_archive": restored_archive,
            "action": "already_restored" if apply else "already_restored_dry_run",
        }

    soft = soft_deleted_record(uri, generation)
    soft_metadata = soft.get("metadata") or {}
    declared_metadata_sha = (soft_metadata.get("metadata") or {}).get("sha256")
    if declared_metadata_sha != expected_sha:
        raise RecoveryError(
            f"soft-deleted generation metadata hash mismatch for {uri}#{generation}"
        )
    result = {
        "uri": uri,
        "requested_generation": generation,
        "expected_sha256": expected_sha,
        "previous_live_generation": live_generation,
        "previous_live_sha256": live_sha,
        "previous_live_archive": before_archive,
        "restored_archive": restored_archive,
        "transaction_uri": transaction_object,
        "action": "would_restore",
    }
    if not apply:
        return result

    # Archive the current live bytes first. Reuse is allowed only when byte-identical.
    try:
        archived = object_bytes(before_archive)
    except RecoveryError:
        copy_and_verify(f"{uri}#{live_generation}", before_archive, live_payload)
    else:
        if archived != live_payload:
            raise RecoveryError(f"existing pre-restore archive disagrees: {before_archive}")

    transaction = {
        "schema": RECEIPT_SCHEMA,
        "uri": uri,
        "requested_generation": generation,
        "expected_sha256": expected_sha,
        "previous_live_generation": live_generation,
        "previous_live_sha256": live_sha,
        "previous_live_archive": before_archive,
        "restored_archive": restored_archive,
        "scientific_metrics_inspected": False,
    }
    transaction_payload = (json.dumps(transaction, indent=2, sort_keys=True) + "\n").encode()
    try:
        existing_transaction = object_bytes(transaction_object)
    except RecoveryError:
        # The name is derived from the exact requested object generation. A readback makes the
        # commit authoritative before the live generation is removed.
        copy_and_verify_bytes = transaction_payload
        # `copy_and_verify` accepts a GCS source, so stage this small receipt through stdin-free
        # temporary storage using the same helper as the recovery tools.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="soft-delete-transaction-") as temporary:
            local = pathlib.Path(temporary) / "transaction.json"
            local.write_bytes(copy_and_verify_bytes)
            command("gcloud", "storage", "cp", "--if-generation-match=0",
                    str(local), transaction_object)
        existing_transaction = object_bytes(transaction_object)
    if existing_transaction != transaction_payload:
        raise RecoveryError(f"recovery transaction disagrees: {transaction_object}")

    # This gcloud release rejects --allow-overwrite for synchronous restores. Remove only the
    # generation we just archived, with an exact precondition; soft-delete keeps it recoverable.
    # Then restore only the requested predecessor generation. A failure between these calls is
    # recoverable from both the content-addressed archive and the new soft-deleted generation.
    command(
        "gcloud", "storage", "rm", f"{uri}#{live_generation}",
        f"--if-generation-match={live_generation}",
    )
    command(
        "gcloud", "storage", "restore", f"{uri}#{generation}",
        "--if-generation-match=0",
    )
    restored = describe(uri)
    restored_generation = str(restored["generation"])
    restored_payload = object_bytes(f"{uri}#{restored_generation}")
    restored_sha = sha256(restored_payload)
    if restored_sha != expected_sha:
        raise RecoveryError(
            f"restored object hash {restored_sha} != expected {expected_sha}: {uri}"
        )
    restored_metadata = restored.get("metadata") or restored.get("custom_fields") or {}
    if restored_metadata.get("sha256") != expected_sha:
        raise RecoveryError(f"restored object metadata hash mismatch: {uri}")
    try:
        archived = object_bytes(restored_archive)
    except RecoveryError:
        copy_and_verify(
            f"{uri}#{restored_generation}", restored_archive, restored_payload
        )
    else:
        if archived != restored_payload:
            raise RecoveryError(f"existing restored archive disagrees: {restored_archive}")
    result.update({
        "action": "restored",
        "restored_live_generation": restored_generation,
        "restored_live_sha256": restored_sha,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan_path = pathlib.Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("schema") != PLAN_SCHEMA or
            plan.get("scientific_metrics_inspected") is not False or
            not isinstance(plan.get("objects"), list) or not plan["objects"]):
        raise SystemExit("invalid or outcome-aware recovery plan")
    uris = [record.get("uri") for record in plan["objects"]]
    if len(set(uris)) != len(uris):
        raise SystemExit("recovery plan contains duplicate object URIs")

    results = [recover_one(record, apply=args.apply) for record in plan["objects"]]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "mode": "apply" if args.apply else "dry_run",
        "plan_sha256": sha256(plan_path.read_bytes()),
        "scientific_metrics_inspected": False,
        "objects": results,
    }
    output = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        pathlib.Path(args.receipt).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
