#!/usr/bin/env python3
"""Bounded Colab driver for durable nonconfirmatory H3 pilot scoring.

Scientific scoring remains in the already-frozen ``score_h3_pilot.py``.  This transport layer
monkey-patches only completed-chunk verification: every verified local chunk is committed to GCS
as content + scorer receipt + a generation-bound commit record.  On a replacement runtime those
records are read back and verified before the scorer decides whether another forward pass is
needed.  The final table/receipt and complete state are published last.

ADC is read from an explicitly uploaded mode-0600 file, exported only to this process, never
printed, copied, or uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_precompute as H  # noqa: E402
import score_h3_pilot as S  # noqa: E402


TRANSPORT_SCHEMA = "nextlat_forgetting/h3_pilot_gcs_transport/1"
COMMIT_SCHEMA = "nextlat_forgetting/h3_pilot_gcs_chunk_commit/1"
STATE_SCHEMA = "nextlat_forgetting/h3_pilot_gcs_state/1"
_SAFE_PREFIX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*[a-zA-Z0-9]$")


class DurableRefused(H.PrecomputeRefused):
    """A durable-transport identity, authentication, or read-back check failed."""


def _canonical(value: Any) -> bytes:
    return H.canonical_json(value)


def _raw_job(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DurableRefused("durable score job is unreadable") from exc
    transport = payload.get("durable_transport")
    remote = payload.get("durable_gcs")
    if not isinstance(transport, Mapping) or not isinstance(remote, Mapping):
        raise DurableRefused("score job lacks durable transport/GCS bindings")
    driver_path = pathlib.Path(str(transport.get("path", "")))
    if not driver_path.is_absolute():
        driver_path = path.parent / driver_path
    driver_path = driver_path.resolve()
    if driver_path != pathlib.Path(__file__).resolve():
        raise DurableRefused("score job binds a different durable driver")
    if transport.get("sha256") != H.sha256_file(driver_path):
        raise DurableRefused("durable driver SHA-256 changed after job freeze")
    bucket = remote.get("bucket")
    base = remote.get("base_prefix")
    project = remote.get("project")
    if not all(isinstance(value, str) and value for value in (bucket, base, project)):
        raise DurableRefused("durable GCS bucket/base/project are not frozen strings")
    if base.startswith("/") or base.endswith("/") or not _SAFE_PREFIX.fullmatch(base):
        raise DurableRefused("durable GCS base_prefix is unsafe")
    return payload


def _bootstrap_runtime() -> dict[str, str]:
    """Install only missing runtime libraries while proving CUDA torch is unchanged."""
    import torch

    before = torch.__version__
    command = [
        sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
        "lightning>=2.4,<3", "omegaconf>=2.3,<3", "google-cloud-storage>=2.16,<4",
        "google-auth>=2,<3",
    ]
    subprocess.run(command, check=True)
    # Import in a child process: if pip had to alter a module already imported in this process,
    # a clean interpreter is the only meaningful compatibility check.
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import json,numpy,torch,lightning,omegaconf; "
            "from google.cloud import storage; "
            "assert torch.cuda.is_available(); "
            "x=numpy.asarray([1.0],dtype=numpy.float32); "
            "assert torch.from_numpy(x).item()==1.0; "
            "print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,"
            "'lightning':lightning.__version__,'gpu':torch.cuda.get_device_name(0)}))"
        )], check=True, capture_output=True, text=True,
    )
    versions = json.loads(probe.stdout.strip().splitlines()[-1])
    if versions["torch"] != before:
        raise DurableRefused(f"bootstrap changed Colab CUDA torch: {before} -> {versions['torch']}")
    return {str(k): str(v) for k, v in versions.items()}


def _verify_runtime_without_bootstrap() -> dict[str, str]:
    try:
        import lightning
        import numpy
        import torch
        from google.cloud import storage as _storage  # noqa: F401
    except ImportError as exc:
        raise DurableRefused("runtime dependencies missing; use --bootstrap") from exc
    if not torch.cuda.is_available():
        raise DurableRefused("bounded pilot scoring requires a CUDA runtime")
    try:
        assert torch.from_numpy(numpy.asarray([1.0], dtype=numpy.float32)).item() == 1.0
    except Exception as exc:
        raise DurableRefused("torch/NumPy ABI probe failed") from exc
    return {
        "torch": str(torch.__version__), "numpy": str(numpy.__version__),
        "lightning": str(lightning.__version__), "gpu": str(torch.cuda.get_device_name(0)),
    }


class GcsDurability:
    def __init__(self, bucket: Any, *, base_prefix: str, job_sha256: str, driver_sha256: str):
        self.bucket = bucket
        self.prefix = f"{base_prefix}/{job_sha256}"
        self.job_sha256 = job_sha256
        self.driver_sha256 = driver_sha256

    def _name(self, relative: str) -> str:
        if relative.startswith("/") or ".." in pathlib.PurePosixPath(relative).parts:
            raise DurableRefused("unsafe remote artifact name")
        return f"{self.prefix}/{relative}"

    def _verify_blob(self, blob: Any, *, sha256: str, generation: int | None = None) -> dict[str, Any]:
        blob.reload()
        metadata = blob.metadata or {}
        if metadata.get("sha256") != sha256 or metadata.get("job_sha256") != self.job_sha256:
            raise DurableRefused(f"remote SHA/job metadata mismatch: {blob.name}")
        observed_generation = int(blob.generation)
        if generation is not None and observed_generation != int(generation):
            raise DurableRefused(f"remote generation changed: {blob.name}")
        payload = blob.download_as_bytes(if_generation_match=observed_generation)
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise DurableRefused(f"remote read-back SHA-256 mismatch: {blob.name}")
        return {"name": blob.name, "sha256": sha256, "generation": observed_generation, "size": len(payload)}

    def put_file(self, relative: str, path: pathlib.Path, *, kind: str) -> dict[str, Any]:
        digest = H.sha256_file(path)
        blob = self.bucket.blob(self._name(relative))
        blob.metadata = {
            "sha256": digest, "job_sha256": self.job_sha256,
            "driver_sha256": self.driver_sha256, "kind": kind,
        }
        try:
            blob.upload_from_filename(str(path), if_generation_match=0)
        except Exception as exc:
            # Only an already-existing byte-identical object is an acceptable create race/retry.
            try:
                return self._verify_blob(blob, sha256=digest)
            except Exception:
                raise DurableRefused(f"create-only GCS upload failed: {blob.name}") from exc
        return self._verify_blob(blob, sha256=digest)

    def put_bytes(self, relative: str, payload: bytes, *, kind: str) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(prefix="h3-gcs-", delete=False) as handle:
            handle.write(payload)
            temporary = pathlib.Path(handle.name)
        try:
            return self.put_file(relative, temporary, kind=kind)
        finally:
            temporary.unlink(missing_ok=True)

    def get_committed(self, relative: str, expected: Mapping[str, Any]) -> bytes:
        blob = self.bucket.blob(self._name(relative))
        if expected.get("name") != blob.name:
            raise DurableRefused("remote commit record names a different object")
        record = self._verify_blob(
            blob, sha256=str(expected["sha256"]), generation=int(expected["generation"])
        )
        if record["size"] != int(expected["size"]):
            raise DurableRefused(f"remote size changed: {blob.name}")
        return blob.download_as_bytes(if_generation_match=int(expected["generation"]))

    def chunk_commit_name(self, start: int, stop: int) -> str:
        return f"chunks/loss-{start:06d}-{stop:06d}.commit.json"

    def commit_chunk(self, output: pathlib.Path, start: int, stop: int) -> dict[str, Any]:
        data, receipt = S._chunk_paths(output, start, stop)
        data_record = self.put_file(f"chunks/{data.name}", data, kind="loss_chunk")
        receipt_record = self.put_file(f"chunks/{receipt.name}", receipt, kind="scorer_chunk_receipt")
        commit = {
            "schema": COMMIT_SCHEMA, "job_sha256": self.job_sha256,
            "driver_sha256": self.driver_sha256, "start": start, "stop": stop,
            "data": data_record, "receipt": receipt_record,
        }
        commit_record = self.put_bytes(
            self.chunk_commit_name(start, stop), _canonical(commit), kind="chunk_commit"
        )
        # Re-read the commit and both generation-pinned children before treating the compute as durable.
        commit_bytes = self.get_committed(self.chunk_commit_name(start, stop), commit_record)
        if json.loads(commit_bytes) != commit:
            raise DurableRefused("remote chunk commit changed during read-back")
        self.get_committed(f"chunks/{data.name}", data_record)
        self.get_committed(f"chunks/{receipt.name}", receipt_record)
        return {**commit, "commit": commit_record}

    def restore_chunk(self, output: pathlib.Path, start: int, stop: int) -> bool:
        commit_blob = self.bucket.blob(self._name(self.chunk_commit_name(start, stop)))
        try:
            commit_blob.reload()
        except Exception as exc:
            # google.api_core NotFound is intentionally checked by status code without importing
            # provider exceptions into local test environments.
            if getattr(exc, "code", None) == 404:
                return False
            raise DurableRefused("could not inspect remote chunk commit") from exc
        commit_generation = int(commit_blob.generation)
        commit_bytes = commit_blob.download_as_bytes(if_generation_match=commit_generation)
        metadata = commit_blob.metadata or {}
        if (
            metadata.get("job_sha256") != self.job_sha256
            or metadata.get("driver_sha256") != self.driver_sha256
            or metadata.get("sha256") != hashlib.sha256(commit_bytes).hexdigest()
        ):
            raise DurableRefused("remote chunk commit metadata/hash is invalid")
        commit = json.loads(commit_bytes)
        expected = {
            "schema": COMMIT_SCHEMA, "job_sha256": self.job_sha256,
            "driver_sha256": self.driver_sha256, "start": start, "stop": stop,
        }
        if any(commit.get(key) != value for key, value in expected.items()):
            raise DurableRefused("remote chunk commit belongs to another identity")
        data_path, receipt_path = S._chunk_paths(output, start, stop)
        for relative, target, record in (
            (f"chunks/{data_path.name}", data_path, commit.get("data")),
            (f"chunks/{receipt_path.name}", receipt_path, commit.get("receipt")),
        ):
            if not isinstance(record, Mapping):
                raise DurableRefused("remote chunk commit lacks a child record")
            payload = self.get_committed(relative, record)
            H.create_or_verify(target, payload)
        return True


def _download_frozen_input(bucket: Any, record: Mapping[str, Any], target: pathlib.Path) -> None:
    expected = str(record.get("sha256", ""))
    remote = record.get("source_object")
    generation = record.get("source_generation")
    if target.is_file():
        if H.sha256_file(target) != expected:
            raise DurableRefused(f"existing pilot input is stale: {target}")
        return
    if not isinstance(remote, str) or not remote or not str(generation).isdigit():
        raise DurableRefused(f"missing content-addressed source for pilot input {target.name}")
    blob = bucket.blob(remote)
    payload = blob.download_as_bytes(if_generation_match=int(generation))
    if hashlib.sha256(payload).hexdigest() != expected:
        raise DurableRefused(f"downloaded pilot input fails SHA-256: {target.name}")
    H.create_or_verify(target, payload)


def run(job_path: pathlib.Path, *, adc: pathlib.Path, bootstrap: bool, chunk_size: int,
        batch_size: int) -> dict[str, Any]:
    raw = _raw_job(job_path)
    if not adc.is_file():
        raise DurableRefused("uploaded ADC is absent")
    if adc.stat().st_mode & 0o077:
        raise DurableRefused("ADC permissions must be mode 0600")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc.resolve())
    os.environ["GOOGLE_CLOUD_PROJECT"] = str(raw["durable_gcs"]["project"])
    versions = _bootstrap_runtime() if bootstrap else _verify_runtime_without_bootstrap()
    from google.cloud import storage

    bucket = storage.Client(project=raw["durable_gcs"]["project"]).bucket(raw["durable_gcs"]["bucket"])
    # The job can be uploaded without the 568 MB checkpoint. Restore exact immutable profile
    # objects directly by generation before the scientific scorer validates local paths.
    for role in ("checkpoint", "config"):
        record = raw[role]
        target = pathlib.Path(record["path"])
        if not target.is_absolute():
            target = job_path.parent / target
        _download_frozen_input(bucket, record, target.resolve())

    job_sha = H.sha256_file(job_path)
    driver_sha = H.sha256_file(__file__)
    durable = GcsDurability(
        bucket, base_prefix=raw["durable_gcs"]["base_prefix"],
        job_sha256=job_sha, driver_sha256=driver_sha,
    )
    original_verified = S._verified_chunk
    bound = S._load_job(job_path)
    output = bound["_bound"]["output"]

    def durable_verified(local_output: pathlib.Path, local_job_sha: str, start: int, stop: int):
        if local_job_sha != job_sha:
            raise DurableRefused("scorer and transport disagree on job SHA-256")
        try:
            existing = original_verified(local_output, local_job_sha, start, stop)
        except H.PrecomputeRefused:
            raise
        if existing is None:
            if durable.restore_chunk(local_output, start, stop):
                existing = original_verified(local_output, local_job_sha, start, stop)
        if existing is not None:
            durable.commit_chunk(local_output, start, stop)
        return existing

    S._verified_chunk = durable_verified
    try:
        receipt = S.run_score(
            job_path, device="cuda", chunk_size=chunk_size, batch_size=batch_size
        )
    finally:
        S._verified_chunk = original_verified

    table_path = output / "pilot_losses.jsonl"
    receipt_path = output / "pilot_scoring_receipt.json"
    table_record = durable.put_file("final/pilot_losses.jsonl", table_path, kind="final_loss_table")
    receipt_record = durable.put_file(
        "final/pilot_scoring_receipt.json", receipt_path, kind="final_scoring_receipt"
    )
    state = {
        "schema": STATE_SCHEMA, "complete": True, "job_sha256": job_sha,
        "driver_sha256": driver_sha, "runtime": versions,
        "row_count": receipt["row_count"], "loss_table": table_record,
        "scoring_receipt": receipt_record,
    }
    state_record = durable.put_bytes("state.json", _canonical(state), kind="complete_state")
    if json.loads(durable.get_committed("state.json", state_record)) != state:
        raise DurableRefused("final state read-back mismatch")
    return {**state, "state": state_record, "remote_prefix": f"gs://{bucket.name}/{durable.prefix}"}


def plan(job_path: pathlib.Path) -> dict[str, Any]:
    raw = _raw_job(job_path)
    job_sha = H.sha256_file(job_path)
    return {
        "schema": TRANSPORT_SCHEMA, "mode": "READ_ONLY_PLAN",
        "job_sha256": job_sha,
        "remote_prefix": (
            f"gs://{raw['durable_gcs']['bucket']}/{raw['durable_gcs']['base_prefix']}/{job_sha}"
        ),
        "chunk_size": 1_000, "maximum_uncommitted_items_on_disconnect": 1_000,
        "publication_order": ["chunk_data", "chunk_receipt", "chunk_commit", "final_table", "final_receipt", "state_last"],
        "bootstrap": [
            "preserve Colab CUDA torch", "install missing Lightning/OmegaConf/GCS client only",
            "verify torch version unchanged", "verify CUDA and torch.from_numpy ABI",
        ],
        "credentials_persisted": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "run"), required=True)
    ap.add_argument("--job", type=pathlib.Path, required=True)
    ap.add_argument("--adc", type=pathlib.Path, default=pathlib.Path("/content/adc.json"))
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1_000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    job = args.job.resolve()
    result = plan(job) if args.mode == "plan" else run(
        job, adc=args.adc.resolve(), bootstrap=args.bootstrap,
        chunk_size=args.chunk_size, batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except H.PrecomputeRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
