#!/usr/bin/env python3
"""Minimal, isolated Colab transport contract for CFS-1.

It deliberately has no legacy H3/HMM fallback path.  CFS-1 writes below the
``cfs1/`` GCS prefix only, binds a session id to both status observations, and
publishes mutable state only after all branch artifacts verify remotely.  This
module is transport plumbing: it cannot evaluate a model or select branches.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
import typing as t

REPO = pathlib.Path(__file__).resolve().parents[1]
sys_path = str(REPO / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from lurestar.durable_checkpoint import atomic_write_json, sha256_file  # noqa: E402


CFS1_PREFIX = "cfs1"
CFS1_RUNTIME_ROOT = "/content/cfs1"
CFS1_SPEC_SCHEMA = "nextlat_forgetting/cfs1_colab_job_spec/1"
CFS1_TERMINAL_SCHEMA = "nextlat_forgetting/cfs1_runtime_terminal/1"
MAX_CONSECUTIVE_SYNC_FAILURES = 3
_SHA = re.compile(r"[0-9a-f]{64}")


class CFS1ColabError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS1ColabError(message)


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             f"{label} must be lowercase SHA-256")
    return t.cast(str, value)


def validate_cfs1_job_spec(spec: t.Mapping[str, t.Any], *, require_session: bool = False) -> dict[str, t.Any]:
    """Validate one sidecar; no scope expansion or evaluator phase is representable."""
    allowed = {
        "schema", "runner", "gpu", "source_sha256", "source_object", "update_manifest_sha256",
        "update_manifest_object", "parent_ledger_sha256", "parent_ledger_object", "session_id",
        "hard_stop_balance_cu", "max_attempts",
    }
    unknown = set(spec) - allowed
    _require(not unknown, f"unknown CFS-1 job spec fields: {sorted(unknown)}")
    _require(spec.get("schema") == CFS1_SPEC_SCHEMA, "unexpected CFS-1 job spec schema")
    _require(spec.get("runner") == "cfs1_adaptation", "CFS-1 job spec must select adaptation runner")
    _require(spec.get("gpu") == "a100", "CFS-1 job spec requires exactly one A100")
    source = _sha(spec.get("source_sha256"), "source_sha256")
    source_object = spec.get("source_object")
    _require(source_object == f"{CFS1_PREFIX}/source/project-{source}.tar.gz",
             "CFS-1 source object/hash binding is wrong")
    manifest = _sha(spec.get("update_manifest_sha256"), "update_manifest_sha256")
    ledger = _sha(spec.get("parent_ledger_sha256"), "parent_ledger_sha256")
    _require(spec.get("update_manifest_object") == f"{CFS1_PREFIX}/inputs/{manifest}/cfs1_update_manifest.json",
             "CFS-1 update manifest object/hash binding is wrong")
    _require(spec.get("parent_ledger_object") == f"{CFS1_PREFIX}/inputs/{ledger}/parents.json",
             "CFS-1 parent ledger object/hash binding is wrong")
    if require_session:
        _require(isinstance(spec.get("session_id"), str) and str(spec["session_id"]).startswith("gpu-a100-"),
                 "CFS-1 runtime has no owned A100 session")
    return {"source_sha256": source, "update_manifest_sha256": manifest,
            "parent_ledger_sha256": ledger, "session_id": spec.get("session_id")}


def require_owned_session(first: t.Mapping[str, t.Any], second: t.Mapping[str, t.Any], expected: str) -> None:
    """Refuse to act unless two independent status reads name the same owned runtime."""
    for observed in (first, second):
        _require(observed.get("state") in {"running", "idle"}, "CFS-1 runtime is not active")
        _require(observed.get("session_id") == expected,
                 "CFS-1 runtime ownership is uncertain; no start/stop/upload is permitted")


def runtime_runner_argv(spec: t.Mapping[str, t.Any], *, project_root: str = "/content/project",
                       runtime_root: str = CFS1_RUNTIME_ROOT, bucket: str) -> list[str]:
    """The remote driver reads only verified sidecars and never accepts notebook argv."""
    validate_cfs1_job_spec(spec, require_session=True)
    _require(isinstance(bucket, str) and bucket and "/" not in bucket,
             "CFS-1 runtime requires one explicit GCS bucket name")
    return [
        "python", f"{project_root}/scripts/run_cfs1_matrix.py", "--root", runtime_root,
        "--manifest", f"{runtime_root}/inputs/cfs1_update_manifest.json", "--parent-ledger",
        f"{runtime_root}/inputs/parents.json", "--ledger", f"{runtime_root}/cfs1_run_ledger.json",
        "--upstream", f"{project_root}/upstream/NextLat",
        "--bucket", bucket, "--gcs-prefix", CFS1_PREFIX,
    ]


class CFS1StateLastDurability:
    """Content-verifying GCS publisher with a state-last commit boundary.

    ``uploader`` receives ``(local_path, remote, sha256)`` and must raise on an
    unsuccessful remote verification.  The implementation is injected because
    browser/Colab credentials are intentionally not present on the host tests.
    """

    def __init__(self, prefix: str, uploader: t.Callable[[pathlib.Path, str, str], None]) -> None:
        _require(prefix.strip("/") == CFS1_PREFIX, "CFS-1 must use its isolated GCS prefix")
        self.prefix, self.uploader = CFS1_PREFIX, uploader

    def _remote(self, relative: str) -> str:
        pure = pathlib.PurePosixPath(relative)
        _require(not pure.is_absolute() and ".." not in pure.parts and str(pure) != ".",
                 "unsafe CFS-1 GCS relative path")
        return f"{self.prefix}/{pure}"

    def publish_state_last(self, *, job_id: str, artifacts: t.Sequence[os.PathLike[str] | str],
                           state: os.PathLike[str] | str) -> list[str]:
        remotes: list[str] = []
        for raw in artifacts:
            path = pathlib.Path(raw)
            _require(path.is_file(), f"CFS-1 durability artifact is absent: {path}")
            remote = self._remote(f"runs/{job_id}/{path.name}")
            self.uploader(path, remote, sha256_file(path))
            remotes.append(remote)
        state_path = pathlib.Path(state)
        _require(state_path.is_file(), "CFS-1 mutable ledger/state is absent")
        remote = self._remote("control/cfs1_run_ledger.json")
        self.uploader(state_path, remote, sha256_file(state_path))
        remotes.append(remote)
        return remotes


def sync_with_circuit_breaker(sync_once: t.Callable[[], None], *, diagnostic: os.PathLike[str] | str,
                              source_sha256: str) -> None:
    """Stop paid work after three consecutive failed durability transactions."""
    errors: list[str] = []
    for _ in range(MAX_CONSECUTIVE_SYNC_FAILURES):
        try:
            sync_once()
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    diagnostic_path = pathlib.Path(diagnostic)
    atomic_write_json(diagnostic_path, {
        "schema": "nextlat_forgetting/cfs1_sync_failure/1", "training_complete": False,
        "source_sha256": _sha(source_sha256, "source_sha256"), "consecutive_failures": len(errors),
        "errors": errors,
    })
    raise CFS1ColabError("CFS-1 durability circuit breaker opened; runtime must stop")


def terminal_marker(*, session_id: str, source_sha256: str, returncode: int,
                    training_complete: bool) -> dict[str, t.Any]:
    _require(session_id.startswith("gpu-a100-"), "CFS-1 terminal marker lacks owned session")
    return {
        "schema": CFS1_TERMINAL_SCHEMA, "session_id": session_id,
        "source_snapshot_sha256": _sha(source_sha256, "source_sha256"),
        "returncode": int(returncode), "training_complete": bool(training_complete),
        "published_at": time.time(),
    }
