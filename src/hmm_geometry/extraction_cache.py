"""Durable, provenance-bound chunk cache for HMM representation extraction.

GPU inference is intentionally split into small independently useful chunks.  A chunk is first
written as an ``npz.partial``, flushed and fsynced, atomically renamed, hashed, and given a hash
sidecar.  Only then is the progress manifest advanced.  A reconnect therefore either sees the
old progress state or a complete, hash-verifiable new chunk; it never trusts a half-written file.

This module contains no torch dependency.  The evaluator supplies numpy arrays and can resume on
any runtime capable of reproducing the same frozen identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import typing as t

import numpy as np

from lurestar.durable_checkpoint import atomic_write_json, atomic_write_text, sha256_file

CACHE_SCHEMA = "nextlat_forgetting/hmm_representation_cache/1"
SIDECAR_SUFFIX = ".sha256"
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


class ExtractionCacheError(RuntimeError):
    """A cached representation is corrupt or belongs to another scientific identity."""


def _canonical_sha(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_finite(value: object, path: str = "root") -> None:
    """Reject non-finite values recursively, including numpy scalar values."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ExtractionCacheError(f"non-finite value at {path}")


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class ExtractionCache:
    """Append-only logical cache whose progress pointer is committed last."""

    def __init__(self, root: os.PathLike | str, identity: t.Mapping[str, object]):
        self.root = pathlib.Path(root).resolve()
        self.chunks = self.root / "chunks"
        self.progress_path = self.root / "progress.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        _require_finite(self.identity, "identity")
        self.identity_sha256 = _canonical_sha(self.identity)
        self._progress = self._load_progress()

    def _fresh_progress(self) -> dict[str, object]:
        return {
            "schema": CACHE_SCHEMA,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "chunks": {},
        }

    def _load_progress(self) -> dict[str, object]:
        if not self.progress_path.is_file():
            return self._fresh_progress()
        try:
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExtractionCacheError("representation cache progress is invalid JSON") from exc
        if (
            progress.get("schema") != CACHE_SCHEMA
            or progress.get("identity_sha256") != self.identity_sha256
            or progress.get("identity") != self.identity
            or not isinstance(progress.get("chunks"), dict)
        ):
            raise ExtractionCacheError("representation cache identity mismatch")
        _require_finite(progress, "progress")
        return progress

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ExtractionCacheError(f"unsafe cache chunk key {key!r}")

    def _paths(self, key: str) -> tuple[pathlib.Path, pathlib.Path]:
        self._validate_key(key)
        data = self.chunks / f"{key}.npz"
        return data, data.with_name(data.name + SIDECAR_SUFFIX)

    @staticmethod
    def _validate_arrays(arrays: t.Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
        if not arrays:
            raise ExtractionCacheError("a representation chunk cannot be empty")
        metadata: dict[str, dict[str, object]] = {}
        for name, raw in arrays.items():
            if not _KEY.fullmatch(str(name)):
                raise ExtractionCacheError(f"unsafe array name {name!r}")
            array = np.asarray(raw)
            if array.dtype.hasobject:
                raise ExtractionCacheError(f"object array {name!r} is forbidden")
            if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
                raise ExtractionCacheError(f"array {name!r} contains non-finite values")
            metadata[str(name)] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        return metadata

    def has(self, key: str) -> bool:
        """Return true only after re-verifying the committed chunk and its sidecar."""
        self._validate_key(key)
        record = t.cast(dict, self._progress["chunks"]).get(key)
        if not isinstance(record, dict):
            return False
        try:
            self._verify_record(key, record)
        except ExtractionCacheError:
            return False
        return True

    def _verify_record(self, key: str, record: t.Mapping[str, object]) -> pathlib.Path:
        data, sidecar = self._paths(key)
        expected = record.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ExtractionCacheError(f"chunk {key!r} has no valid hash")
        if not data.is_file() or not sidecar.is_file():
            raise ExtractionCacheError(f"chunk {key!r} is incomplete")
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if not fields or fields[0].lower() != expected or sha256_file(data) != expected:
            raise ExtractionCacheError(f"chunk {key!r} failed SHA-256 verification")
        return data

    def write(self, key: str, arrays: t.Mapping[str, np.ndarray]) -> dict[str, object]:
        """Atomically commit one chunk, its sidecar, then the resumable progress pointer."""
        self._validate_key(key)
        metadata = self._validate_arrays(arrays)
        existing = t.cast(dict, self._progress["chunks"]).get(key)
        if isinstance(existing, dict):
            try:
                self._verify_record(key, existing)
            except ExtractionCacheError:
                # Recompute in place. The atomic rename keeps the prior bytes present until the
                # replacement is complete; progress is still committed last below.
                pass
            else:
                return dict(existing)

        data, sidecar = self._paths(key)
        partial = data.with_name(data.name + ".partial")
        with open(partial, "wb") as handle:
            np.savez(handle, **{name: np.asarray(value) for name, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, data)
        _fsync_dir(data.parent)
        digest = sha256_file(data)
        atomic_write_text(sidecar, f"{digest}  {data.name}\n")
        record: dict[str, object] = {
            "path": str(data),
            "sha256": digest,
            "sidecar": str(sidecar),
            "sidecar_sha256": sha256_file(sidecar),
            "arrays": metadata,
            "bytes": data.stat().st_size,
        }
        t.cast(dict, self._progress["chunks"])[key] = record
        atomic_write_json(self.progress_path, self._progress)
        return record

    def load(self, key: str) -> dict[str, np.ndarray]:
        self._validate_key(key)
        record = t.cast(dict, self._progress["chunks"]).get(key)
        if not isinstance(record, dict):
            raise ExtractionCacheError(f"chunk {key!r} is not committed")
        data = self._verify_record(key, record)
        try:
            with np.load(data, allow_pickle=False) as payload:
                arrays = {name: payload[name] for name in payload.files}
        except (OSError, ValueError) as exc:
            raise ExtractionCacheError(f"chunk {key!r} cannot be deserialized") from exc
        metadata = self._validate_arrays(arrays)
        if metadata != record.get("arrays"):
            raise ExtractionCacheError(f"chunk {key!r} shape/dtype metadata changed")
        return arrays

    def receipt(self, *, expected_keys: t.Iterable[str] | None = None) -> dict[str, object]:
        """Return a state-last, fully verified cache attestation for the final receipt."""
        keys = sorted(t.cast(dict, self._progress["chunks"]))
        if expected_keys is not None:
            wanted = sorted(set(expected_keys))
            if keys != wanted:
                raise ExtractionCacheError(
                    f"cache is incomplete: missing={sorted(set(wanted)-set(keys))}, "
                    f"extra={sorted(set(keys)-set(wanted))}"
                )
        for key in keys:
            self._verify_record(key, t.cast(dict, self._progress["chunks"])[key])
        # Re-write so the manifest on disk is the exact state whose digest is reported.
        atomic_write_json(self.progress_path, self._progress)
        return {
            "schema": CACHE_SCHEMA,
            "identity_sha256": self.identity_sha256,
            "progress": {
                "path": str(self.progress_path),
                "sha256": sha256_file(self.progress_path),
            },
            "n_chunks": len(keys),
            "chunk_sha256": {
                key: t.cast(dict, self._progress["chunks"])[key]["sha256"] for key in keys
            },
            "total_bytes": sum(
                int(t.cast(dict, self._progress["chunks"])[key]["bytes"]) for key in keys
            ),
        }
