#!/usr/bin/env python3
"""Publish the frozen manifest inventory as an immutable GCS input bundle.

The SHA-256 of ``manifests/manifest_inventory.sha256`` names the bundle.  Every
listed object is created with a generation-zero precondition and verified by
downloading that exact generation.  The inventory itself is published last and
therefore acts as the bundle's commit record.

This program deliberately does not discover files.  Only entries in the strict
inventory, under the three allowed local prefixes, can be uploaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import typing as t


DEFAULT_BUCKET = "nextlat-lurestar-project-flash-490419"
DEFAULT_GCS_PREFIX = "lurestar"
INVENTORY_RELATIVE = "manifests/manifest_inventory.sha256"
RECEIPT_RELATIVE = ".agent_state/input-bundle-upload.json"
RECEIPT_SCHEMA = "nextlat_forgetting/input_bundle_upload/1"
LINE_RE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
SHA_RE = re.compile(r"[0-9a-f]{64}")


class UploadError(RuntimeError):
    """A local or remote invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UploadError(message)


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(document: dict[str, t.Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


@dataclasses.dataclass(frozen=True)
class InventoryEntry:
    sha256: str
    relative_path: str
    local_path: pathlib.Path
    remote_suffix: str
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class RemoteObject:
    name: str
    generation: str
    size_bytes: int
    custom_sha256: str


class Backend(t.Protocol):
    def resolve(self, name: str) -> RemoteObject | None: ...

    def create_file(self, name: str, local_path: pathlib.Path, sha256: str) -> None: ...

    def download_exact(
        self, name: str, generation: str, destination: pathlib.Path
    ) -> None: ...


class GcloudBackend:
    """Minimal gcloud transport with injectable subprocess runner."""

    def __init__(
        self,
        bucket: str,
        runner: t.Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.bucket = bucket
        self.runner = runner

    def _uri(self, name: str, generation: str | None = None) -> str:
        uri = "gs://%s/%s" % (self.bucket, name)
        return uri if generation is None else "%s#%s" % (uri, generation)

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return self.runner(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    @staticmethod
    def _error(completed: subprocess.CompletedProcess[bytes]) -> str:
        return completed.stderr.decode("utf-8", "replace").strip()

    @staticmethod
    def _missing(completed: subprocess.CompletedProcess[bytes]) -> bool:
        error = GcloudBackend._error(completed).lower()
        return ("not found" in error or "status=404" in error or
                "no urls matched" in error or "does not exist" in error)

    @staticmethod
    def _record(document: dict[str, t.Any]) -> RemoteObject:
        custom = document.get("custom_fields", document.get("metadata", {})) or {}
        generation = str(document.get("generation", ""))
        size = str(document.get("size", ""))
        require(generation.isdigit(), "remote object has no numeric generation")
        require(size.isdigit(), "remote object has no numeric size")
        digest = str(custom.get("sha256", ""))
        require(bool(SHA_RE.fullmatch(digest)), "remote object lacks valid sha256 metadata")
        return RemoteObject(
            name=str(document.get("name", "")), generation=generation,
            size_bytes=int(size), custom_sha256=digest,
        )

    def resolve(self, name: str) -> RemoteObject | None:
        completed = self._run([
            "gcloud", "storage", "objects", "describe", self._uri(name), "--format=json",
        ])
        if completed.returncode:
            if self._missing(completed):
                return None
            raise UploadError("gcloud describe failed: " + self._error(completed))
        try:
            document = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UploadError("gcloud describe returned invalid JSON") from exc
        record = self._record(document)
        require(record.name == name, "gcloud described a different object")
        return record

    def create_file(self, name: str, local_path: pathlib.Path, sha256: str) -> None:
        completed = self._run([
            "gcloud", "storage", "cp", "--if-generation-match=0",
            "--custom-metadata=sha256=%s" % sha256, str(local_path), self._uri(name),
        ])
        if completed.returncode:
            raise UploadError("gcloud create-only upload failed: " + self._error(completed))

    def download_exact(
        self, name: str, generation: str, destination: pathlib.Path
    ) -> None:
        require(str(generation).isdigit(), "refusing nonnumeric remote generation")
        completed = self._run([
            "gcloud", "storage", "cp", self._uri(name, generation), str(destination),
        ])
        if completed.returncode:
            raise UploadError("gcloud exact-generation readback failed: " + self._error(completed))


def _remote_suffix(relative_path: str) -> str:
    mappings = (
        ("data/hmm_family/", "corpus/hmm_family/"),
        ("data/hmm/", "corpus/hmm/"),
        ("manifests/", "manifests/"),
    )
    for local_prefix, remote_prefix in mappings:
        if relative_path.startswith(local_prefix) and len(relative_path) > len(local_prefix):
            return remote_prefix + relative_path[len(local_prefix):]
    raise UploadError("inventory path is outside allowed prefixes: %s" % relative_path)


def _require_plain_contained_file(root: pathlib.Path, relative: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(relative)
    require(not pure.is_absolute(), "absolute inventory path is forbidden")
    require(str(pure) == relative, "inventory path is not canonical POSIX")
    require("\\" not in relative, "backslash in inventory path is forbidden")
    require(all(part not in ("", ".", "..") for part in pure.parts),
            "inventory path traversal is forbidden")
    path = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "symlinked inventory path is forbidden: %s" % relative)
    resolved_root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise UploadError("inventory entry is absent or unreadable: %s" % relative) from exc
    require(resolved_root == resolved or resolved_root in resolved.parents,
            "inventory path escapes project root")
    require(path.is_file(), "inventory entry is not a regular file: %s" % relative)
    return path


def parse_inventory(project_root: pathlib.Path) -> tuple[pathlib.Path, str, list[InventoryEntry]]:
    root = project_root.resolve()
    inventory = root / INVENTORY_RELATIVE
    require(inventory.is_file() and not inventory.is_symlink(), "manifest inventory is absent")
    try:
        text = inventory.read_bytes().decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise UploadError("manifest inventory is not UTF-8") from exc
    require("\x00" not in text, "NUL in manifest inventory")
    lines = text.splitlines()
    require(bool(lines), "manifest inventory is empty")
    entries: list[InventoryEntry] = []
    seen_paths: set[str] = set()
    seen_remote: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        match = LINE_RE.fullmatch(line)
        require(match is not None, "malformed inventory line %d" % line_number)
        expected, relative = match.groups()
        require(relative != INVENTORY_RELATIVE, "inventory cannot recursively list itself")
        require(relative not in seen_paths, "duplicate inventory path: %s" % relative)
        remote = _remote_suffix(relative)
        local = _require_plain_contained_file(root, relative)
        actual = sha256_file(local)
        require(actual == expected, "local SHA mismatch: %s" % relative)
        require(remote not in seen_remote, "duplicate mapped remote path: %s" % remote)
        seen_paths.add(relative)
        seen_remote.add(remote)
        entries.append(InventoryEntry(expected, relative, local, remote, local.stat().st_size))
    paths = [entry.relative_path for entry in entries]
    require(paths == sorted(paths), "manifest inventory paths are not sorted")
    return inventory, sha256_file(inventory), entries


def _verify_remote(
    backend: Backend, record: RemoteObject, expected_name: str,
    expected_sha256: str, expected_size: int,
) -> dict[str, t.Any]:
    require(record.name == expected_name, "remote object name mismatch")
    require(record.generation.isdigit(), "remote generation is not numeric: %s" % expected_name)
    require(record.size_bytes == expected_size, "remote size mismatch: %s" % expected_name)
    require(record.custom_sha256 == expected_sha256,
            "remote sha256 metadata mismatch: %s" % expected_name)
    descriptor, temporary = tempfile.mkstemp(prefix="input-bundle-readback-")
    os.close(descriptor)
    destination = pathlib.Path(temporary)
    try:
        backend.download_exact(expected_name, record.generation, destination)
        require(destination.is_file(), "remote readback did not create a file")
        require(destination.stat().st_size == expected_size,
                "remote exact-generation readback size mismatch: %s" % expected_name)
        require(sha256_file(destination) == expected_sha256,
                "remote exact-generation readback SHA mismatch: %s" % expected_name)
    finally:
        destination.unlink(missing_ok=True)
    return {
        "name": expected_name,
        "generation": record.generation,
        "size_bytes": expected_size,
        "sha256": expected_sha256,
    }


def _create_or_verify(
    backend: Backend, name: str, local_path: pathlib.Path, digest: str, size: int,
    *, allow_create: bool,
) -> dict[str, t.Any]:
    require(local_path.is_file() and local_path.stat().st_size == size,
            "local input size changed after inventory validation: %s" % local_path)
    require(sha256_file(local_path) == digest,
            "local input changed after inventory validation: %s" % local_path)
    record = backend.resolve(name)
    if record is None:
        require(allow_create, "committed bundle is missing object: %s" % name)
        try:
            backend.create_file(name, local_path, digest)
        except Exception as create_error:
            # A transport failure can happen after GCS commits the object.  Re-resolve and
            # accept only a fully verified exact object; otherwise preserve the first error.
            record = backend.resolve(name)
            if record is None:
                raise create_error
        else:
            record = backend.resolve(name)
            require(record is not None, "uploaded object is not remotely visible: %s" % name)
    return _verify_remote(backend, record, name, digest, size)


def _write_receipt_create_only(path: pathlib.Path, document: dict[str, t.Any]) -> None:
    payload = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload,
                "existing input-bundle receipt differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.is_file() and path.read_bytes() == payload,
                    "racing input-bundle receipt differs")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def upload_bundle(
    project_root: pathlib.Path, backend: Backend, *, bucket: str,
    gcs_prefix: str = DEFAULT_GCS_PREFIX, plan: bool = False,
) -> dict[str, t.Any]:
    inventory, bundle_sha256, entries = parse_inventory(project_root)
    prefix = "%s/input_bundles/%s" % (gcs_prefix.rstrip("/"), bundle_sha256)
    commit_name = "%s/%s" % (prefix, INVENTORY_RELATIVE)
    if plan:
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "PLAN",
            "bucket": bucket,
            "bundle_prefix": prefix,
            "input_bundle_sha256": bundle_sha256,
            "object_count": len(entries),
            "commit_object": commit_name,
            "objects": [
                {"local_path": entry.relative_path,
                 "name": "%s/%s" % (prefix, entry.remote_suffix),
                 "size_bytes": entry.size_bytes, "sha256": entry.sha256}
                for entry in entries
            ],
        }

    existing_commit = backend.resolve(commit_name)
    committed = existing_commit is not None
    objects = []
    for entry in entries:
        name = "%s/%s" % (prefix, entry.remote_suffix)
        remote = _create_or_verify(
            backend, name, entry.local_path, entry.sha256, entry.size_bytes,
            allow_create=not committed,
        )
        objects.append({"local_path": entry.relative_path, **remote})

    inventory_digest = sha256_file(inventory)
    inventory_size = inventory.stat().st_size
    require(inventory_digest == bundle_sha256,
            "manifest inventory changed during upload; refusing commit")
    if committed:
        commit = _verify_remote(
            backend, t.cast(RemoteObject, existing_commit), commit_name,
            inventory_digest, inventory_size,
        )
    else:
        commit = _create_or_verify(
            backend, commit_name, inventory, inventory_digest, inventory_size,
            allow_create=True,
        )
    document = {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "bucket": bucket,
        "bundle_prefix": prefix,
        "input_bundle_sha256": bundle_sha256,
        "object_count": len(objects),
        "objects": objects,
        "commit": {"local_path": INVENTORY_RELATIVE, **commit},
    }
    _write_receipt_create_only(project_root / RECEIPT_RELATIVE, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--plan", action="store_true", help="validate locally; make no remote calls")
    args = parser.parse_args(argv)
    try:
        backend = GcloudBackend(args.bucket)
        document = upload_bundle(
            args.root.resolve(), backend, bucket=args.bucket,
            gcs_prefix=args.prefix, plan=args.plan,
        )
    except UploadError as exc:
        parser.exit(2, "BLOCKED: %s\n" % exc)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
