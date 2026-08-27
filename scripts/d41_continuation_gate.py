#!/usr/bin/env python
"""Outcome-blind D41 continuation evidence and validation.

This module deliberately has no evaluator imports and never opens evaluation receipts.  It
records only source identities and recovery metadata needed to continue the interrupted HMM
training matrix without silently treating the continuation as a fresh pre-compute launch.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


PREDECESSOR_SOURCE_SHA256 = (
    "a962cdb94c865e16c2c7c86d5c18b9cc2d3bd301feeea12e42075751f52c9285"
)
TARGET_STEP = 3_000
REGIME = "persistent_moderate"
BUCKET = "nextlat-lurestar-project-flash-490419"
PINNED_UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
PREDECESSOR_SESSION_ID = "gpu-a100-s-kkb-ass1c2-3elvob8c0x50j"
PREDECESSOR_SOURCE_GENERATION = "1787566447414304"
REFERENCE_PATCH_RECEIPT_URI = (
    "gs://nextlat-lurestar-project-flash-490419/lurestar/profiles/"
    "a100-b15e1bc9d596-8d316efc9c53/artifacts/sha256/"
    "8290e69c5004d0e78e2da2635bf7d0dce350c4614b00e7f17e4e4ba915ba0b0a/"
    "provenance/runtime_patch_receipt.json"
)
EXPECTED_RUNTIME = {
    "device_name": "NVIDIA A100-SXM4-40GB",
    "torch_version": "2.11.0+cu128",
    "cuda_version": "12.8",
    "bf16_supported": True,
    "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
}
PREDECESSOR_REFERENCE_SCHEMA = "nextlat_forgetting/d41_predecessor_freeze_reference/1"
EXACT_TEN_SCHEMA = "nextlat_forgetting/d41_exact_ten_recovery/2"
EQUIVALENCE_SCHEMA = "nextlat_forgetting/d41_source_equivalence/1"
CONTINUATION_SCHEMA = "nextlat_forgetting/d41_continuation_clearance/1"
PREDECESSOR_REFERENCE_PATH = pathlib.Path(
    ".agent_state/d41-predecessor-freeze-reference.json"
)
PREDECESSOR_ARCHIVE_PATH = pathlib.Path(".agent_state/project-predecessor-d41.tar.gz")
EXACT_TEN_PATH = pathlib.Path(".agent_state/d41-exact-ten-recovery-receipt.json")
EQUIVALENCE_PATH = pathlib.Path(".agent_state/d41-source-equivalence-receipt.json")
D41_DECISION_PATH = pathlib.Path("docs/DECISION_D41_RUNTIME_RECOVERY_AMENDMENT.md")
RECOVERY_EVIDENCE_PATHS = (
    pathlib.Path(".agent_state/hmm-soft-delete-recovery-plan.json"),
    pathlib.Path(".agent_state/hmm-soft-delete-recovery-receipt.json"),
    pathlib.Path(".agent_state/hmm-exact-target-state-repair.json"),
)
RECOVERY_EVIDENCE_SCHEMAS = {
    ".agent_state/hmm-soft-delete-recovery-plan.json": (
        "nextlat_forgetting/gcs_soft_delete_recovery_plan/1"
    ),
    ".agent_state/hmm-soft-delete-recovery-receipt.json": (
        "nextlat_forgetting/gcs_soft_delete_recovery_receipt/1"
    ),
    ".agent_state/hmm-exact-target-state-repair.json": (
        "nextlat_forgetting/exact_target_state_repair/1"
    ),
}
EXACT_TEN_JOB_IDS = tuple(
    f"{model}-seed{seed}-hmm-{REGIME}"
    for model in ("gpt", "nextlat")
    for seed in range(1234, 1239)
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")

# Exact-byte scientific surface.  The operational controller and matrix lifecycle are excluded;
# their scientific portions are checked separately through normalized AST projections below.
_SCIENTIFIC_PREFIXES = (
    "configs/",
    "data/",
    "manifests/",
    "src/hmm_geometry/",
    "src/lurestar/",
    "upstream/",
)
_OPERATIONAL_MIXED_PATHS = frozenset({
    "scripts/run_hmm_matrix.py",
    "scripts/run_matrix.py",
    "src/lurestar/durable_checkpoint.py",
})
_SCIENTIFIC_EXACT_FILES = frozenset({
    "scripts/train_hmm.py",
    "scripts/runtime_bootstrap.py",
    "scripts/evaluate_hmm_checkpoints.py",
    "scripts/aggregate_hmm_family.py",
    "scripts/materialize_hmm_family.py",
})
_RUNNER_SCIENTIFIC_SYMBOLS = frozenset({
    "HMM_MODELS",
    "HMM_TRAIN_UPDATES",
    "HMM_CHECKPOINT_INTERVAL",
    "PINNED_UPSTREAM_COMMIT",
    "HMM_REQUIRED_METRICS",
    "HMM_FAMILY_REQUIRED_METRICS",
    "hmm_job_id",
    "_family_required_paths",
    "verify_hmm_family_snapshot",
    "hmm_source_inputs",
    "build_hmm_matrix",
    "HMMFabricLauncher.command",
    "verify_hmm_evaluation_receipt",
    "hmm_evaluator_command",
    "run_hmm_evaluators",
    "promote_hmm_evaluations",
})
_ALLOWED_OPERATIONAL_AST_DELTA = {
    "scripts/run_hmm_matrix.py": frozenset({
        "RUNTIME_RECOVERY_BARRIER_SCHEMA", "load_runtime_recovery_barrier", "main",
    }),
    "scripts/run_matrix.py": frozenset({
        "MatrixRunner.__init__", "MatrixRunner._terminalize_verified_checkpoint",
        "MatrixRunner.run", "MatrixRunner.run_job", "collect_training_artifacts",
        "write_completion_summary",
    }),
    "src/lurestar/durable_checkpoint.py": frozenset({
        "_loaded_step", "exact_sidecar_step", "DurableCheckpointer._verify_record",
        "DurableCheckpointer.adopt", "DurableCheckpointer.save",
    }),
}
_HMM_CLI_SCIENTIFIC_OPTIONS = frozenset({
    "--devices", "--precision", "--strategy", "--evaluation-batch-size", "--phase",
    "--family", "--data-root", "--snapshot-root",
})


class D41GateError(RuntimeError):
    """Continuation evidence is absent, malformed, stale, or scientifically non-equivalent."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(document: Any) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D41GateError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise D41GateError(f"{label} must be a JSON object")
    return value


def _binding(root: pathlib.Path, relative: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    path = root / relative
    document = _load_json(path, str(relative))
    if schema is not None and document.get("schema") != schema:
        raise D41GateError(f"{relative} schema mismatch")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "schema": document.get("schema"),
    }


def _recovery_bindings(root: pathlib.Path) -> list[dict[str, Any]]:
    bindings = []
    for path in RECOVERY_EVIDENCE_PATHS:
        schema = RECOVERY_EVIDENCE_SCHEMAS[path.as_posix()]
        document = _load_json(root / path, str(path))
        if document.get("schema") != schema:
            raise D41GateError(f"{path} schema mismatch")
        if document.get("scientific_metrics_inspected") is not False:
            raise D41GateError(f"{path} is not outcome-blind")
        if path.name.endswith("receipt.json") and document.get("mode") != "apply":
            raise D41GateError(f"{path} is not an applied recovery receipt")
        if path.name == "hmm-exact-target-state-repair.json" and (
            document.get("mode") != "apply" or document.get("target_step") != TARGET_STEP
            or document.get("source_snapshot_sha256") != PREDECESSOR_SOURCE_SHA256
        ):
            raise D41GateError("exact-target state repair contract mismatch")
        bindings.append(_binding(root, path, schema))
    return bindings


def preserve_predecessor_reference(root: pathlib.Path) -> pathlib.Path:
    """Embed the already-issued predecessor chain before successor receipts replace it."""
    root = root.resolve()
    archive = root / ".agent_state" / "project.tar.gz"
    clearance_path = root / ".agent_state" / "confirmatory-clearance.json"
    freeze_path = root / ".agent_state" / "preregistration-freeze-receipt.json"
    evidence_path = root / ".agent_state" / "preregistration-evidence.json"
    test_path = root / ".agent_state" / "confirmatory-test-receipt.json"
    review_path = root / ".agent_state" / "confirmatory-review-receipt.json"
    if not archive.is_file() or sha256_file(archive) != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("predecessor archive bytes do not match the frozen D41 source")
    preserved_archive = root / PREDECESSOR_ARCHIVE_PATH
    if preserved_archive.exists():
        if not preserved_archive.is_file() or sha256_file(preserved_archive) != PREDECESSOR_SOURCE_SHA256:
            raise D41GateError("existing preserved predecessor archive has the wrong bytes")
    else:
        partial = preserved_archive.with_name(preserved_archive.name + ".partial")
        with archive.open("rb") as source, partial.open("wb") as target_stream:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target_stream.write(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        if sha256_file(partial) != PREDECESSOR_SOURCE_SHA256:
            partial.unlink(missing_ok=True)
            raise D41GateError("copied predecessor archive failed byte verification")
        os.replace(partial, preserved_archive)
    clearance = _load_json(clearance_path, "predecessor clearance")
    freeze = _load_json(freeze_path, "predecessor all-eleven receipt")
    evidence = _load_json(evidence_path, "predecessor preregistration evidence")
    test = _load_json(test_path, "predecessor test receipt")
    review = _load_json(review_path, "predecessor review receipt")
    if clearance.get("authorization") != "GO" or clearance.get("source_sha256") != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("predecessor clearance is not source-bound GO")
    prereg = clearance.get("preregistration")
    if not isinstance(prereg, dict) or prereg.get("source_archive_sha256") != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("predecessor clearance lacks its source-bound all-eleven freeze")
    if prereg.get("receipt_sha256") != sha256_file(freeze_path):
        raise D41GateError("predecessor clearance/freeze hash mismatch")
    if prereg.get("evidence_sha256") != sha256_file(evidence_path):
        raise D41GateError("predecessor clearance/evidence hash mismatch")
    if clearance.get("test_receipt_sha256") != sha256_file(test_path):
        raise D41GateError("predecessor clearance/test hash mismatch")
    if clearance.get("review_receipt_sha256") != sha256_file(review_path):
        raise D41GateError("predecessor clearance/review hash mismatch")
    if freeze.get("status") != "PASS" or freeze.get("all_eleven_gates_pass") is not True:
        raise D41GateError("predecessor all-eleven freeze is not PASS")
    document = {
        "schema": PREDECESSOR_REFERENCE_SCHEMA,
        "created_at": utc_now(),
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
        "predecessor_archive": {
            "path": PREDECESSOR_ARCHIVE_PATH.as_posix(),
            "sha256": PREDECESSOR_SOURCE_SHA256,
            "size_bytes": preserved_archive.stat().st_size,
        },
        "issued_clearance": clearance,
        "all_eleven_freeze": freeze,
        "preregistration_evidence_sha256": sha256_file(evidence_path),
        "full_test_receipt": test,
        "independent_review_receipt": review,
        "confirmatory_lifecycle_at_reference": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
            "truthful_basis": "D41 incident occurred after ten training cells reached step 3000",
        },
    }
    target = root / PREDECESSOR_REFERENCE_PATH
    atomic_json(target, document)
    return target


def _run_json(command: list[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise D41GateError(f"{label} failed: {completed.stderr.decode(errors='replace').strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise D41GateError(f"{label} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise D41GateError(f"{label} did not return an object")
    return value


def _gcs_describe(uri: str) -> dict[str, Any]:
    return _run_json(["gcloud", "storage", "objects", "describe", "--format=json", uri], uri)


def _gcs_bytes(uri: str, generation: str) -> bytes:
    completed = subprocess.run(
        ["gcloud", "storage", "cat", f"{uri}#{generation}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise D41GateError(f"failed to read immutable GCS generation {uri}#{generation}")
    return completed.stdout


def _object_identity(uri: str, *, verify_bytes: bool = True) -> tuple[dict[str, Any], bytes | None]:
    metadata = _gcs_describe(uri)
    generation = str(metadata.get("generation", ""))
    size = metadata.get("size")
    custom = metadata.get("custom_fields") or {}
    expected = custom.get("sha256") if isinstance(custom, dict) else None
    if not generation.isdigit() or not isinstance(size, int) or not _SHA_RE.fullmatch(str(expected)):
        raise D41GateError(f"GCS identity is incomplete for {uri}")
    payload = _gcs_bytes(uri, generation) if verify_bytes else None
    if payload is not None and (len(payload) != size or hashlib.sha256(payload).hexdigest() != expected):
        raise D41GateError(f"immutable GCS bytes do not match metadata for {uri}#{generation}")
    return ({
        "uri": uri,
        "generation": generation,
        "sha256": expected,
        "size_bytes": size,
    }, payload)


def _exact_sidecar_step_parser(root: pathlib.Path):
    """Load the same sidecar-step parser used by the runtime restore barrier."""
    source = root / "src" / "lurestar" / "durable_checkpoint.py"
    spec = importlib.util.spec_from_file_location("d41_durable_checkpoint_semantics", source)
    if spec is None or spec.loader is None:
        raise D41GateError("cannot load the shared checkpoint-sidecar verifier")
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = prior
    parser = getattr(module, "exact_sidecar_step", None)
    if not callable(parser):
        raise D41GateError("shared exact_sidecar_step verifier is missing")
    return parser


def _checkpoint_filename_step(path: str) -> int:
    name = pathlib.PurePosixPath(path).name
    match = re.fullmatch(r"(?:recovery_)?ckpt_iter_(\d+)(?:_[^/]*)?\.pt", name)
    if match is None:
        raise D41GateError(f"checkpoint filename lacks an exact step: {path}")
    return int(match.group(1))


def _checkpoint_payload_training_steps(payload: bytes, job_id: str) -> int:
    """Deserialize receipt-pinned bytes and require the production step field literally."""
    try:
        import torch
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise D41GateError(f"{job_id}: exact checkpoint payload cannot be deserialized") from exc
    if not isinstance(state, dict):
        raise D41GateError(f"{job_id}: exact checkpoint payload is not a mapping")
    value = state.get("training_steps")
    if isinstance(value, bool) or not isinstance(value, int):
        raise D41GateError(
            f"{job_id}: checkpoint payload lacks literal integer training_steps"
        )
    return int(value)


def collect_exact_ten_recovery(root: pathlib.Path) -> pathlib.Path:
    """Deserialize and attest every exact-target checkpoint without reading outcomes."""
    root = root.resolve()
    jobs: list[dict[str, Any]] = []
    prefix = f"gs://{BUCKET}/lurestar/runs"
    exact_sidecar_step = _exact_sidecar_step_parser(root)
    for job_id in EXACT_TEN_JOB_IDS:
        model_match = re.fullmatch(
            rf"(gpt|nextlat)-seed(123[4-8])-hmm-{REGIME}", job_id
        )
        assert model_match is not None
        model, seed_text = model_match.groups()
        state_uri = f"{prefix}/{job_id}/state.json"
        state_object, state_bytes = _object_identity(state_uri)
        assert state_bytes is not None
        try:
            state = json.loads(state_bytes)
        except json.JSONDecodeError as exc:
            raise D41GateError(f"{job_id}: state JSON is invalid") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema") != "nextlat_forgetting/colab_state/1"
            or state.get("run_id") != job_id
            or state.get("status") != "TRAINED"
            or state.get("step") != TARGET_STEP
            or state.get("source_snapshot_sha256") != PREDECESSOR_SOURCE_SHA256
        ):
            raise D41GateError(f"{job_id}: state is not predecessor-bound TRAINED step 3000")
        checkpoint = state.get("checkpoint")
        candidates = state.get("recovery_candidates")
        if not isinstance(checkpoint, dict) or not isinstance(candidates, list):
            raise D41GateError(f"{job_id}: state lacks checkpoint recovery identity")
        exact = [item for item in candidates if isinstance(item, dict) and item.get("step") == TARGET_STEP]
        if len(exact) != 1 or any(
            exact[0].get(key) != checkpoint.get(key) for key in ("path", "sha256", "size_bytes")
        ):
            raise D41GateError(f"{job_id}: exact-target recovery candidate is missing or ambiguous")
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, list):
            raise D41GateError(f"{job_id}: state artifact inventory is missing")

        def artifact_for(local_path: str) -> tuple[str, dict[str, Any]]:
            matches = [item for item in artifacts if isinstance(item, dict) and item.get("local_path") == local_path]
            if len(matches) != 1 or not isinstance(matches[0].get("remote"), str):
                raise D41GateError(f"{job_id}: artifact mapping is missing or duplicated: {local_path}")
            record = matches[0]
            if (
                not _SHA_RE.fullmatch(str(record.get("sha256", "")))
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
            ):
                raise D41GateError(f"{job_id}: artifact mapping lacks byte identity")
            return f"gs://{BUCKET}/{record['remote']}", record

        checkpoint_path = str(checkpoint.get("path"))
        checkpoint_uri, checkpoint_artifact = artifact_for(checkpoint_path)
        sidecar_path = exact[0].get("metadata_path")
        if not isinstance(sidecar_path, str) or sidecar_path != checkpoint_path + ".meta.json":
            raise D41GateError(f"{job_id}: exact checkpoint sidecar path is missing")
        sidecar_uri, sidecar_artifact = artifact_for(sidecar_path)
        checkpoint_object, checkpoint_bytes = _object_identity(checkpoint_uri, verify_bytes=True)
        sidecar_object, sidecar_bytes = _object_identity(sidecar_uri, verify_bytes=True)
        assert checkpoint_bytes is not None and sidecar_bytes is not None
        try:
            sidecar = json.loads(sidecar_bytes)
        except json.JSONDecodeError as exc:
            raise D41GateError(f"{job_id}: exact checkpoint sidecar is invalid") from exc
        try:
            sidecar_step = exact_sidecar_step(sidecar)
        except ValueError as exc:
            raise D41GateError(f"{job_id}: checkpoint sidecar step is invalid: {exc}") from exc
        filename_step = _checkpoint_filename_step(checkpoint_path)
        payload_training_steps = _checkpoint_payload_training_steps(checkpoint_bytes, job_id)
        if (
            checkpoint_object["sha256"] != checkpoint.get("sha256")
            or checkpoint_object["size_bytes"] != checkpoint.get("size_bytes")
            or checkpoint_object["sha256"] != checkpoint_artifact.get("sha256")
            or checkpoint_object["size_bytes"] != checkpoint_artifact.get("size_bytes")
            or sidecar_object["sha256"] != sidecar_artifact.get("sha256")
            or sidecar_object["size_bytes"] != sidecar_artifact.get("size_bytes")
            or not isinstance(sidecar, dict)
            or sidecar.get("run_id") != job_id
            or sidecar.get("path") != checkpoint_path
            or set(sidecar).intersection({"step", "training_steps"}) != {"step"}
            or sidecar_step != TARGET_STEP
            or filename_step != TARGET_STEP
            or payload_training_steps != TARGET_STEP
            or sidecar.get("sha256") != checkpoint_object["sha256"]
            or sidecar.get("size_bytes") != checkpoint_object["size_bytes"]
        ):
            raise D41GateError(f"{job_id}: checkpoint/state/sidecar identity mismatch")
        jobs.append({
            "job_id": job_id,
            "model": model,
            "seed": int(seed_text),
            "regime": REGIME,
            "target_step": TARGET_STEP,
            "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
            "state_object": state_object,
            "checkpoint_object": checkpoint_object,
            "sidecar_object": sidecar_object,
            "checkpoint_semantics": {
                "filename_step": filename_step,
                "sidecar_step": sidecar_step,
                "sidecar_step_field": "step",
                "sidecar_path": sidecar_path,
                "sidecar_run_id": sidecar["run_id"],
                "payload_training_steps": payload_training_steps,
            },
            "verification": {
                "state_trained_exact_target": True,
                "checkpoint_bytes_sha256_verified": True,
                "sidecar_bytes_sha256_verified": True,
                "sidecar_binds_checkpoint": True,
                "source_identity_verified": True,
                "payload_training_steps_verified": True,
            },
        })
    recovery_bindings = _recovery_bindings(root)
    runtime_equivalence = _runtime_equivalence(root)
    document = {
        "schema": EXACT_TEN_SCHEMA,
        "created_at": utc_now(),
        "status": "PASS",
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
        "target_step": TARGET_STEP,
        "required_job_ids": list(EXACT_TEN_JOB_IDS),
        "jobs": jobs,
        "checkpoint_semantic_contract": {
            "verifier_path": "src/lurestar/durable_checkpoint.py",
            "verifier_symbol": "exact_sidecar_step",
            "verifier_source_sha256": sha256_file(
                root / "src" / "lurestar" / "durable_checkpoint.py"
            ),
            "target_step": TARGET_STEP,
            "predecessor_sidecar_schema": "legacy_step_only",
            "checkpoint_payload_step_field": "training_steps",
            "all_checkpoint_payloads_deserialized_on_host": True,
            "conflicting_dual_sidecar_step_fields_rejected": True,
            "migration_scope": "receipt_bound_source_migrated_only",
            "successor_sidecar_contract": "canonical_training_steps_required",
            "successor_retains_legacy_step": True,
            "successor_canonicalization_requires_from_to_hash_provenance": True,
            "current_source_retry_sidecar_contract": "training_steps_required",
        },
        "recovery_plan_and_receipts": recovery_bindings,
        "runtime_equivalence": runtime_equivalence,
        "scientific_metrics_inspected": False,
        "confirmatory_lifecycle": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
        },
    }
    validate_exact_ten_document(document)
    target = root / EXACT_TEN_PATH
    atomic_json(target, document)
    return target


def refresh_runtime_equivalence(root: pathlib.Path) -> pathlib.Path:
    """Refresh only the successor source/patch contract; immutable exact-ten bytes stay fixed."""
    root = root.resolve()
    target = root / EXACT_TEN_PATH
    document = _load_json(target, "D41 exact-ten recovery receipt")
    if document.get("schema") != EXACT_TEN_SCHEMA or document.get("status") != "PASS":
        raise D41GateError("cannot refresh a nonpassing exact-ten receipt")
    document["runtime_equivalence"] = _runtime_equivalence(root)
    document["runtime_equivalence_refreshed_at"] = utc_now()
    validate_exact_ten_document(document)
    atomic_json(target, document)
    return target


def _runtime_equivalence(root: pathlib.Path) -> dict[str, Any]:
    """Bind observed predecessor preflight facts and the successor's exact runtime contract.

    The predecessor's per-run runtime-patch receipt was not among the recursively retained run
    artifacts, which is stated explicitly.  We therefore bind the exact patch source plus a
    byte-verified immutable A100 receipt made by the same patch implementation and require the
    successor to emit a receipt with the same deterministic semantic projection.  This is a
    stronger actionable gate than pretending the missing predecessor receipt was retained.
    """
    source_uri = (
        f"gs://{BUCKET}/lurestar/source/project-{PREDECESSOR_SOURCE_SHA256}.tar.gz"
    )
    source_object, _ = _object_identity(source_uri, verify_bytes=True)
    if source_object["generation"] != PREDECESSOR_SOURCE_GENERATION:
        raise D41GateError("predecessor source archive generation changed")
    patch_object, patch_bytes = _object_identity(REFERENCE_PATCH_RECEIPT_URI, verify_bytes=True)
    assert patch_bytes is not None
    try:
        patch_receipt = json.loads(patch_bytes)
    except json.JSONDecodeError as exc:
        raise D41GateError("reference runtime patch receipt is invalid") from exc
    if (
        not isinstance(patch_receipt, dict)
        or patch_receipt.get("schema") != "nextlat_forgetting/runtime_patch/1"
        or patch_receipt.get("upstream_commit") != PINNED_UPSTREAM_COMMIT
        or not isinstance(patch_receipt.get("unified_diff"), str)
    ):
        raise D41GateError("reference runtime patch receipt contract mismatch")
    reference_projection = _patch_projection(patch_receipt)
    expected_projection = _materialize_current_patch_projection(root)
    patch_source = root / "scripts" / "runtime_bootstrap.py"
    if not patch_source.is_file():
        raise D41GateError("runtime patch source is missing")
    decision = root / D41_DECISION_PATH
    return {
        "predecessor_runtime_evidence": {
            "session_id": PREDECESSOR_SESSION_ID,
            "source_archive_object": source_object,
            "observed_preflight": dict(EXPECTED_RUNTIME),
            "observation_basis": {
                "kind": "host_controller_verified_preflight_console",
                "decision_path": D41_DECISION_PATH.as_posix(),
                "decision_sha256": sha256_file(decision),
            },
            "per_run_runtime_patch_receipt_retained": False,
        },
        "expected_successor_contract": dict(EXPECTED_RUNTIME),
        "runtime_patch": {
            "source_path": "scripts/runtime_bootstrap.py",
            "source_sha256": sha256_file(patch_source),
            "historical_reference_receipt_object": patch_object,
            "historical_reference_receipt_sha256": patch_object["sha256"],
            "historical_reference_receipt_projection": reference_projection,
            "historical_reference_receipt_projection_sha256": canonical_sha256(
                reference_projection
            ),
            "expected_receipt_projection": expected_projection,
            "expected_receipt_projection_sha256": canonical_sha256(expected_projection),
            "successor_must_emit_and_verify_own_receipt": True,
        },
    }


def _patch_projection(receipt: dict[str, Any]) -> dict[str, Any]:
    semantic_keys = (
        "schema", "patch_version", "upstream_commit", "before_sha256", "after_sha256",
        "helper_sha256", "adaptation_trainer_sha256", "adaptation_contract",
        "bst_parameter_count", "optimizer_update_rule", "optimizer_fusion_rule",
        "amp_scaler_checkpoint_rule", "deterministic_runtime_rule",
    )
    projection = {key: receipt.get(key) for key in semantic_keys}
    unified_diff = receipt.get("unified_diff")
    if not isinstance(unified_diff, str):
        raise D41GateError("runtime patch receipt lacks its deterministic diff")
    projection["unified_diff_sha256"] = hashlib.sha256(
        unified_diff.encode()
    ).hexdigest()
    return projection


def _materialize_current_patch_projection(root: pathlib.Path) -> dict[str, Any]:
    """Apply the exact checked-in bootstrap to a disposable pinned tree and project receipt."""
    source = root / "scripts" / "runtime_bootstrap.py"
    upstream = root / "upstream" / "NextLat"
    if not source.is_file() or not upstream.is_dir():
        raise D41GateError("current runtime patch or pinned upstream tree is missing")
    spec = importlib.util.spec_from_file_location("d41_runtime_bootstrap_projection", source)
    if spec is None or spec.loader is None:
        raise D41GateError("cannot import current runtime bootstrap")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="d41-runtime-patch-") as temporary:
        temp = pathlib.Path(temporary)
        temp_upstream = temp / "upstream"
        shutil.copytree(upstream, temp_upstream, symlinks=True)
        receipt = module.apply_runtime_patch(temp / "project", temp_upstream)
    if not isinstance(receipt, dict):
        raise D41GateError("current runtime bootstrap did not emit a receipt")
    return _patch_projection(receipt)


def validate_exact_ten_document(document: dict[str, Any]) -> None:
    if document.get("schema") != EXACT_TEN_SCHEMA or document.get("status") != "PASS":
        raise D41GateError("D41 exact-ten recovery receipt is not PASS")
    if document.get("predecessor_source_sha256") != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("D41 exact-ten predecessor source mismatch")
    if document.get("target_step") != TARGET_STEP:
        raise D41GateError("D41 exact-ten target step mismatch")
    if document.get("required_job_ids") != list(EXACT_TEN_JOB_IDS):
        raise D41GateError("D41 exact-ten canonical job inventory mismatch")
    semantic_contract = {
        "verifier_path": "src/lurestar/durable_checkpoint.py",
        "verifier_symbol": "exact_sidecar_step",
        "verifier_source_sha256": document.get(
            "checkpoint_semantic_contract", {}
        ).get("verifier_source_sha256"),
        "target_step": TARGET_STEP,
        "predecessor_sidecar_schema": "legacy_step_only",
        "checkpoint_payload_step_field": "training_steps",
        "all_checkpoint_payloads_deserialized_on_host": True,
        "conflicting_dual_sidecar_step_fields_rejected": True,
        "migration_scope": "receipt_bound_source_migrated_only",
        "successor_sidecar_contract": "canonical_training_steps_required",
        "successor_retains_legacy_step": True,
        "successor_canonicalization_requires_from_to_hash_provenance": True,
        "current_source_retry_sidecar_contract": "training_steps_required",
    }
    if (
        not _SHA_RE.fullmatch(str(semantic_contract["verifier_source_sha256"]))
        or document.get("checkpoint_semantic_contract") != semantic_contract
    ):
        raise D41GateError("D41 checkpoint semantic contract is missing or stale")
    recovery = document.get("recovery_plan_and_receipts")
    if not isinstance(recovery, list) or len(recovery) != len(RECOVERY_EVIDENCE_PATHS):
        raise D41GateError("D41 recovery plan/receipt binding set is incomplete")
    for binding in recovery:
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256", "schema"}
            or not _SHA_RE.fullmatch(str(binding.get("sha256", "")))
            or not isinstance(binding.get("schema"), str)
        ):
            raise D41GateError("D41 recovery plan/receipt binding is malformed")
    jobs = document.get("jobs")
    if not isinstance(jobs, list) or [item.get("job_id") for item in jobs if isinstance(item, dict)] != list(EXACT_TEN_JOB_IDS):
        raise D41GateError("D41 exact-ten recovered jobs are incomplete or out of order")
    required_verification = {
        "state_trained_exact_target": True,
        "checkpoint_bytes_sha256_verified": True,
        "sidecar_bytes_sha256_verified": True,
        "sidecar_binds_checkpoint": True,
        "source_identity_verified": True,
        "payload_training_steps_verified": True,
    }
    for expected, item in zip(EXACT_TEN_JOB_IDS, jobs):
        if not isinstance(item, dict):
            raise D41GateError("D41 exact-ten job record is not an object")
        if item.get("job_id") != expected or item.get("target_step") != TARGET_STEP or item.get("regime") != REGIME:
            raise D41GateError(f"D41 exact-ten job identity mismatch: {expected}")
        if item.get("predecessor_source_sha256") != PREDECESSOR_SOURCE_SHA256:
            raise D41GateError(f"D41 exact-ten job source mismatch: {expected}")
        for role in ("state_object", "checkpoint_object", "sidecar_object"):
            record = item.get(role)
            if (
                not isinstance(record, dict)
                or set(record) != {"uri", "generation", "sha256", "size_bytes"}
                or not str(record.get("generation", "")).isdigit()
                or not _SHA_RE.fullmatch(str(record.get("sha256", "")))
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
            ):
                raise D41GateError(f"D41 exact-ten {role} identity mismatch: {expected}")
        semantics = item.get("checkpoint_semantics")
        if (
            not isinstance(semantics, dict)
            or set(semantics) != {
                "filename_step", "sidecar_step", "sidecar_step_field", "sidecar_path",
                "sidecar_run_id", "payload_training_steps",
            }
            or semantics.get("filename_step") != TARGET_STEP
            or semantics.get("sidecar_step") != TARGET_STEP
            or semantics.get("sidecar_step_field") != "step"
            or semantics.get("payload_training_steps") != TARGET_STEP
            or semantics.get("sidecar_run_id") != expected
            or not isinstance(semantics.get("sidecar_path"), str)
            or pathlib.PurePosixPath(semantics["sidecar_path"]).name !=
            pathlib.PurePosixPath(item["sidecar_object"]["uri"]).name
        ):
            raise D41GateError(f"D41 exact-ten checkpoint semantics mismatch: {expected}")
        if item.get("verification") != required_verification:
            raise D41GateError(f"D41 exact-ten verification is nonpassing: {expected}")
    if document.get("scientific_metrics_inspected") is not False:
        raise D41GateError("D41 recovery receipt must remain outcome-blind")
    runtime = document.get("runtime_equivalence")
    if not isinstance(runtime, dict) or set(runtime) != {
        "predecessor_runtime_evidence", "expected_successor_contract", "runtime_patch"
    }:
        raise D41GateError("D41 runtime-equivalence binding is missing")
    predecessor_runtime = runtime["predecessor_runtime_evidence"]
    if (
        predecessor_runtime.get("session_id") != PREDECESSOR_SESSION_ID
        or predecessor_runtime.get("observed_preflight") != EXPECTED_RUNTIME
        or predecessor_runtime.get("per_run_runtime_patch_receipt_retained") is not False
        or runtime.get("expected_successor_contract") != EXPECTED_RUNTIME
    ):
        raise D41GateError("D41 predecessor/successor runtime contract mismatch")
    source_object = predecessor_runtime.get("source_archive_object")
    if (
        not isinstance(source_object, dict)
        or source_object.get("generation") != PREDECESSOR_SOURCE_GENERATION
        or source_object.get("sha256") != PREDECESSOR_SOURCE_SHA256
    ):
        raise D41GateError("D41 predecessor runtime source generation mismatch")
    patch = runtime["runtime_patch"]
    if (
        not isinstance(patch, dict)
        or not _SHA_RE.fullmatch(str(patch.get("source_sha256", "")))
        or patch.get("historical_reference_receipt_sha256") !=
        patch.get("historical_reference_receipt_object", {}).get("sha256")
        or canonical_sha256(patch.get("historical_reference_receipt_projection")) !=
        patch.get("historical_reference_receipt_projection_sha256")
        or canonical_sha256(patch.get("expected_receipt_projection")) !=
        patch.get("expected_receipt_projection_sha256")
        or patch.get("successor_must_emit_and_verify_own_receipt") is not True
    ):
        raise D41GateError("D41 runtime-patch source/receipt contract mismatch")
    lifecycle = document.get("confirmatory_lifecycle")
    if lifecycle != {"compute_started": True, "scientific_evaluations_inspected": False}:
        raise D41GateError("D41 exact-ten lifecycle is false or incomplete")


def validate_exact_ten_recovery_receipt(
    root: pathlib.Path,
    predecessor_sha256: str,
    expected_sha256: str | None = None,
    expected_job_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if predecessor_sha256 != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("D41 continuation names an unknown predecessor")
    path = root.resolve() / EXACT_TEN_PATH
    document = _load_json(path, "D41 exact-ten recovery receipt")
    validate_exact_ten_document(document)
    semantic_contract = document["checkpoint_semantic_contract"]
    verifier_source = root.resolve() / semantic_contract["verifier_path"]
    if (
        not verifier_source.is_file()
        or sha256_file(verifier_source) != semantic_contract["verifier_source_sha256"]
        or not callable(_exact_sidecar_step_parser(root.resolve()))
    ):
        raise D41GateError("D41 checkpoint semantic verifier source is stale")
    expected_recovery = _recovery_bindings(root.resolve())
    if document.get("recovery_plan_and_receipts") != expected_recovery:
        raise D41GateError("D41 exact-ten recovery plan/receipt binding drift")
    patch = document["runtime_equivalence"]["runtime_patch"]
    patch_source = root.resolve() / patch.get("source_path", "")
    if (
        not patch_source.is_file()
        or sha256_file(patch_source) != patch.get("source_sha256")
        or _materialize_current_patch_projection(root.resolve()) !=
        patch.get("expected_receipt_projection")
    ):
        raise D41GateError("D41 successor runtime-patch projection is stale")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise D41GateError("D41 exact-ten recovery receipt hash mismatch")
    if expected_job_ids is not None and list(expected_job_ids) != list(EXACT_TEN_JOB_IDS):
        raise D41GateError("D41 job spec recovery inventory is not exact-ten canonical order")
    return {
        "path": EXACT_TEN_PATH.as_posix(),
        "schema": EXACT_TEN_SCHEMA,
        "sha256": actual,
        "job_ids": list(EXACT_TEN_JOB_IDS),
    }


def _archive_files(path: pathlib.Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = pathlib.PurePosixPath(member.name)
            if member.isdir():
                continue
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise D41GateError(f"unsafe/nonregular archive member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise D41GateError(f"cannot read archive member: {member.name}")
            files[name.as_posix()] = extracted.read()
    return files


def _ast_projection(source: bytes) -> dict[str, str]:
    tree = ast.parse(source.decode("utf-8"))
    records: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            for name in names:
                if name in _RUNNER_SCIENTIFIC_SYMBOLS:
                    records[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _RUNNER_SCIENTIFIC_SYMBOLS:
            records[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, ast.ClassDef) and node.name == "HMMFabricLauncher":
            for child in node.body:
                qualified = f"{node.name}.{getattr(child, 'name', '')}"
                if qualified in _RUNNER_SCIENTIFIC_SYMBOLS:
                    records[qualified] = hashlib.sha256(ast.dump(child, include_attributes=False).encode()).hexdigest()
    missing = sorted(_RUNNER_SCIENTIFIC_SYMBOLS - set(records))
    if missing:
        raise D41GateError(f"HMM runner scientific projection is incomplete: {missing}")
    return records


def _module_symbol_projection(source: bytes) -> dict[str, str]:
    """Hash module constants/functions and individual class methods, not whole classes."""
    tree = ast.parse(source.decode("utf-8"))
    records: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    records[target.id] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest()
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            records[node.name] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{node.name}.{child.name}"
                    records[name] = hashlib.sha256(
                        ast.dump(child, include_attributes=False).encode()
                    ).hexdigest()
    return records


def _sanitized_module_ast_sha256(source: bytes, allowed: frozenset[str]) -> str:
    """Hash the full module AST after deleting only the exact reviewed delta nodes."""
    tree = copy.deepcopy(ast.parse(source.decode("utf-8")))
    retained: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names and names <= allowed:
                continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed:
            continue
        if isinstance(node, ast.ClassDef):
            node.body = [
                child for child in node.body
                if not (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and f"{node.name}.{child.name}" in allowed
                )
            ]
        retained.append(node)
    tree.body = retained
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _hmm_cli_projection(source: bytes) -> dict[str, str]:
    tree = ast.parse(source.decode("utf-8"))
    records: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value in _HMM_CLI_SCIENTIFIC_OPTIONS:
            records[str(first.value)] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
    if set(records) != _HMM_CLI_SCIENTIFIC_OPTIONS:
        raise D41GateError("HMM CLI scientific option projection is incomplete")
    return records


def _mixed_module_projection(
    before: dict[str, bytes], after: dict[str, bytes]
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for path, allowed in _ALLOWED_OPERATIONAL_AST_DELTA.items():
        if path not in before or path not in after:
            raise D41GateError(f"D41 mixed operational/scientific module is missing: {path}")
        old = _module_symbol_projection(before[path])
        new = _module_symbol_projection(after[path])
        changed = frozenset(
            symbol for symbol in set(old) | set(new) if old.get(symbol) != new.get(symbol)
        )
        if changed != allowed:
            unexpected = sorted(changed - allowed)
            missing = sorted(allowed - changed)
            raise D41GateError(
                f"D41 {path} AST delta differs from reviewed operational allowlist: "
                f"unexpected={unexpected}, missing={missing}"
            )
        protected = {
            symbol: digest for symbol, digest in sorted(new.items()) if symbol not in allowed
        }
        if any(old.get(symbol) != digest for symbol, digest in protected.items()):
            raise D41GateError(f"D41 changed protected symbols in {path}")
        sanitized_before = _sanitized_module_ast_sha256(before[path], allowed)
        sanitized_after = _sanitized_module_ast_sha256(after[path], allowed)
        if sanitized_before != sanitized_after:
            raise D41GateError(
                f"D41 changed non-allowlisted full-module AST in {path}"
            )
        projection[path] = {
            "reviewed_operational_delta_symbols": sorted(changed),
            "unchanged_symbol_projection_sha256": canonical_sha256(protected),
            "unchanged_symbol_count": len(protected),
            "sanitized_full_module_ast_sha256": sanitized_after,
        }
    cli_before = _hmm_cli_projection(before["scripts/run_hmm_matrix.py"])
    cli_after = _hmm_cli_projection(after["scripts/run_hmm_matrix.py"])
    if cli_before != cli_after:
        raise D41GateError("D41 changed HMM device/precision/strategy/evaluation CLI contract")
    projection["scripts/run_hmm_matrix.py"]["scientific_cli_projection"] = cli_after
    return projection


def _is_scientific_path(path: str) -> bool:
    return (
        path not in _OPERATIONAL_MIXED_PATHS
        and (path in _SCIENTIFIC_EXACT_FILES
             or any(path.startswith(prefix) for prefix in _SCIENTIFIC_PREFIXES))
    )


def create_source_equivalence(
    root: pathlib.Path,
    predecessor_archive: pathlib.Path,
    successor_archive: pathlib.Path,
) -> pathlib.Path:
    root = root.resolve()
    predecessor_archive = predecessor_archive.resolve()
    successor_archive = successor_archive.resolve()
    predecessor_sha = sha256_file(predecessor_archive)
    successor_sha = sha256_file(successor_archive)
    if predecessor_sha != PREDECESSOR_SOURCE_SHA256 or successor_sha == predecessor_sha:
        raise D41GateError("D41 source pair is not the expected predecessor and distinct successor")
    before = _archive_files(predecessor_archive)
    after = _archive_files(successor_archive)
    changed = sorted(
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    protected_changed = [path for path in changed if _is_scientific_path(path)]
    if protected_changed:
        raise D41GateError("D41 changed exact-byte scientific files: " + ", ".join(protected_changed))
    runner_path = "scripts/run_hmm_matrix.py"
    if runner_path not in before or runner_path not in after:
        raise D41GateError("D41 source archive lacks the HMM runner")
    runner_before = _ast_projection(before[runner_path])
    runner_after = _ast_projection(after[runner_path])
    if runner_before != runner_after:
        changed_symbols = sorted(key for key in runner_before if runner_before[key] != runner_after[key])
        raise D41GateError("D41 changed HMM scientific runner symbols: " + ", ".join(changed_symbols))
    mixed_modules = _mixed_module_projection(before, after)
    scientific_records = {
        path: hashlib.sha256(after[path]).hexdigest()
        for path in sorted(after)
        if _is_scientific_path(path)
    }
    projection = {
        "exact_byte_files": scientific_records,
        "hmm_runner_ast": runner_after,
        "mixed_module_ast": mixed_modules,
    }
    document = {
        "schema": EQUIVALENCE_SCHEMA,
        "created_at": utc_now(),
        "status": "PASS",
        "predecessor_archive": {
            "path": str(predecessor_archive.relative_to(root)),
            "sha256": predecessor_sha,
            "size_bytes": predecessor_archive.stat().st_size,
        },
        "successor_archive": {
            "path": str(successor_archive.relative_to(root)),
            "sha256": successor_sha,
            "size_bytes": successor_archive.stat().st_size,
        },
        "changed_paths": changed,
        "scientific_projection_sha256": canonical_sha256(projection),
        "scientific_projection": projection,
        "equivalence_claims": {
            "scientific_configs_unchanged": True,
            "model_implementations_unchanged": True,
            "data_and_manifests_unchanged": True,
            "optimizer_and_training_command_unchanged": True,
            "bootstrap_unchanged": True,
            "evaluator_and_aggregation_unchanged": True,
        },
        "scientific_metrics_inspected": False,
        "confirmatory_lifecycle": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
        },
    }
    target = root / EQUIVALENCE_PATH
    atomic_json(target, document)
    return target


def validate_source_equivalence_receipt(
    root: pathlib.Path, predecessor_sha256: str, successor_sha256: str
) -> dict[str, Any]:
    """Recompute the archive diff/projection instead of trusting stored PASS booleans."""
    root = root.resolve()
    path = root / EQUIVALENCE_PATH
    document = _load_json(path, "D41 source equivalence receipt")
    if document.get("schema") != EQUIVALENCE_SCHEMA or document.get("status") != "PASS":
        raise D41GateError("D41 source equivalence receipt is not PASS")
    archive_records = []
    for role, expected in (("predecessor_archive", predecessor_sha256),
                           ("successor_archive", successor_sha256)):
        record = document.get(role)
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise D41GateError(f"D41 {role} identity mismatch")
        relative = pathlib.Path(str(record.get("path", "")))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise D41GateError(f"D41 {role} escapes the project") from exc
        if (
            relative.is_absolute() or not candidate.is_file()
            or record.get("sha256") != expected
            or sha256_file(candidate) != expected
            or record.get("size_bytes") != candidate.stat().st_size
        ):
            raise D41GateError(f"D41 {role} byte identity mismatch")
        archive_records.append(candidate)
    before, after = map(_archive_files, archive_records)
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    if document.get("changed_paths") != changed:
        raise D41GateError("D41 stored source diff differs from recomputed archives")
    protected_changed = [name for name in changed if _is_scientific_path(name)]
    if protected_changed:
        raise D41GateError("D41 recomputation found changed scientific files")
    runner = "scripts/run_hmm_matrix.py"
    runner_before = _ast_projection(before[runner])
    runner_after = _ast_projection(after[runner])
    if runner_before != runner_after:
        raise D41GateError("D41 recomputation found changed scientific runner symbols")
    mixed_modules = _mixed_module_projection(before, after)
    projection = {
        "exact_byte_files": {
            name: hashlib.sha256(after[name]).hexdigest()
            for name in sorted(after) if _is_scientific_path(name)
        },
        "hmm_runner_ast": runner_after,
        "mixed_module_ast": mixed_modules,
    }
    claims = {
        "scientific_configs_unchanged": True,
        "model_implementations_unchanged": True,
        "data_and_manifests_unchanged": True,
        "optimizer_and_training_command_unchanged": True,
        "bootstrap_unchanged": True,
        "evaluator_and_aggregation_unchanged": True,
    }
    if (
        document.get("scientific_projection") != projection
        or document.get("scientific_projection_sha256") != canonical_sha256(projection)
        or document.get("equivalence_claims") != claims
        or document.get("scientific_metrics_inspected") is not False
        or document.get("confirmatory_lifecycle") != {
            "compute_started": True, "scientific_evaluations_inspected": False,
        }
    ):
        raise D41GateError("D41 source equivalence semantic projection mismatch")
    return {
        "path": EQUIVALENCE_PATH.as_posix(),
        "schema": EQUIVALENCE_SCHEMA,
        "sha256": sha256_file(path),
    }


def validate_d41_continuation_bundle(
    root: pathlib.Path,
    spec: dict[str, Any],
    successor_source_sha256: str,
) -> dict[str, Any]:
    """Return the exact binding inserted into the ordinary confirmatory clearance."""
    root = root.resolve()
    predecessor = spec.get("predecessor_source_sha256")
    if predecessor != PREDECESSOR_SOURCE_SHA256:
        raise D41GateError("D41 continuation job spec lacks the exact predecessor source")
    exact = validate_exact_ten_recovery_receipt(
        root,
        predecessor,
        expected_sha256=spec.get("recovery_receipt_sha256"),
        expected_job_ids=spec.get("recovery_job_ids"),
    )
    reference_path = root / PREDECESSOR_REFERENCE_PATH
    reference = _load_json(reference_path, "D41 predecessor freeze reference")
    if (
        reference.get("schema") != PREDECESSOR_REFERENCE_SCHEMA
        or reference.get("predecessor_source_sha256") != predecessor
        or reference.get("confirmatory_lifecycle_at_reference", {}).get("compute_started") is not True
        or reference.get("confirmatory_lifecycle_at_reference", {}).get("scientific_evaluations_inspected") is not False
    ):
        raise D41GateError("D41 predecessor reference is stale or lifecycle-false")
    issued = reference.get("issued_clearance")
    freeze = reference.get("all_eleven_freeze")
    if not isinstance(issued, dict) or issued.get("source_sha256") != predecessor or issued.get("authorization") != "GO":
        raise D41GateError("D41 predecessor reference lacks issued GO")
    if not isinstance(freeze, dict) or freeze.get("all_eleven_gates_pass") is not True:
        raise D41GateError("D41 predecessor reference lacks all-eleven PASS")
    equivalence_binding = validate_source_equivalence_receipt(
        root, predecessor, successor_source_sha256
    )
    decision = root / D41_DECISION_PATH
    if not decision.is_file():
        raise D41GateError("D41 decision record is missing")
    recovery = _recovery_bindings(root)
    successor_assurance = {
        "full_test_receipt": _binding(
            root, pathlib.Path(".agent_state/confirmatory-test-receipt.json"),
            "nextlat_forgetting/full_test_suite_receipt/1",
        ),
        "independent_review_receipt": _binding(
            root, pathlib.Path(".agent_state/confirmatory-review-receipt.json"),
            "nextlat_forgetting/independent_scientific_review/1",
        ),
    }
    return {
        "schema": CONTINUATION_SCHEMA,
        "decision": {
            "path": D41_DECISION_PATH.as_posix(),
            "sha256": sha256_file(decision),
        },
        "confirmatory_lifecycle": {
            "compute_started": True,
            "scientific_evaluations_inspected": False,
        },
        "predecessor": {
            "source_sha256": predecessor,
            "freeze_reference": {
                "path": PREDECESSOR_REFERENCE_PATH.as_posix(),
                "schema": PREDECESSOR_REFERENCE_SCHEMA,
                "sha256": sha256_file(reference_path),
            },
            "all_eleven_freeze_receipt_sha256": issued["preregistration"]["receipt_sha256"],
            "issued_clearance_sha256": canonical_sha256(issued),
        },
        "successor_source_sha256": successor_source_sha256,
        "recovery_plan_and_receipts": recovery,
        "exact_ten_recovery": exact,
        "runtime_equivalence": _load_json(
            root / EXACT_TEN_PATH, "D41 exact-ten recovery receipt"
        )["runtime_equivalence"],
        "source_equivalence": equivalence_binding,
        "successor_assurance": successor_assurance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(epilog=(
        "Order: preserve-predecessor before any snapshot overwrites project.tar.gz; then finish "
        "successor source/tests, snapshot; then run collect-exact-ten and equivalence; finally "
        "record successor tests/review and issue normal confirmatory clearance."
    ))
    parser.add_argument("mode", choices=(
        "preserve-predecessor", "collect-exact-ten", "refresh-runtime-equivalence",
        "equivalence",
    ))
    parser.add_argument("--project-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--predecessor-archive", default=".agent_state/project-predecessor-d41.tar.gz")
    parser.add_argument("--successor-archive", default=".agent_state/project.tar.gz")
    args = parser.parse_args()
    root = pathlib.Path(args.project_root).resolve()
    if args.mode == "preserve-predecessor":
        output = preserve_predecessor_reference(root)
    elif args.mode == "collect-exact-ten":
        output = collect_exact_ten_recovery(root)
    elif args.mode == "refresh-runtime-equivalence":
        output = refresh_runtime_equivalence(root)
    else:
        predecessor = pathlib.Path(args.predecessor_archive)
        successor = pathlib.Path(args.successor_archive)
        if not predecessor.is_absolute():
            predecessor = root / predecessor
        if not successor.is_absolute():
            successor = root / successor
        output = create_source_equivalence(root, predecessor, successor)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
