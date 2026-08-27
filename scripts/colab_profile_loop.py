#!/usr/bin/env python3
"""Hardened, durable A100 launcher for the preregistered profiling gate.

The host role provisions exactly one owned A100 session after two agreeing status reads. The
runtime role restores any already-durable profile artifacts, runs the 50-step HMM smoke checks,
then invokes :mod:`scripts/profile.sh` for the five measured profiles. Runtime output is mirrored
to GCS every minute with ``state.json`` published last; a session terminal marker is published
only after a final verified sync. A lost ``colab exec`` stream therefore does not imply a lost
profile and never causes this launcher to provision a second runtime automatically.

The runtime receives no argv. Its immutable parameters arrive in ``/content/profile_job.json``.
All runtime GCS traffic uses ``google-cloud-storage`` and the uploaded mode-0600 ADC.
"""

from __future__ import annotations

import base64
import collections
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback


BUCKET = "nextlat-lurestar-project-flash-490419"
GCP_PROJECT = "project-flash-490419"
GCS_ROOT = "lurestar"
PINNED_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"
REMOTE_SPEC = "/content/profile_job.json"
ADC_PATH = "/content/adc.json"
PROFILE_SCHEMA = "nextlat_forgetting/colab_profile/2"
STATE_SCHEMA = "nextlat_forgetting/colab_profile_state/2"
TERMINAL_SCHEMA = "nextlat_forgetting/colab_profile_terminal/2"
INPUT_SCHEMA = "nextlat_forgetting/colab_profile_inputs/1"
SALVAGE_SCHEMA = "nextlat_forgetting/profile_salvage_clearance/1"
SYNC_SECONDS = 60
HEARTBEAT_SECONDS = 30
STATUS_DELAY_SECONDS = 30
STALL_WINDOWS = 6
HARD_STOP_BALANCE_CU = 1188.61
EXPECTED_GATE_JOBS = {
    "lurestar-gpt": (500, 100),
    "lurestar-nextlat": (500, 100),
    "lurestar-bst": (500, 100),
    "hmm-gpt": (300, 60),
    "hmm-nextlat": (300, 60),
}
EXPECTED_SMOKE_JOBS = ("hmm-smoke-gpt", "hmm-smoke-nextlat")
REQUIRED_PROFILE_FIELDS = (
    "seconds_per_step_median",
    "seconds_per_step_p95",
    "examples_per_second",
    "tokens_per_second",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "host_input_wait_seconds",
    "checkpoint_write_seconds_median",
    "checkpoint_bytes_median",
)
INPUT_PREFIXES = (
    GCS_ROOT + "/corpus/stargraph",
    GCS_ROOT + "/corpus/hmm",
    GCS_ROOT + "/manifests",
)


class ProfileError(RuntimeError):
    """Fail-closed profiling or lifecycle error."""


def probe_succeeded(probe: dict) -> bool:
    """Recognize both current and already-durable zero-exit probe receipts."""
    return probe.get("exit") in {"ok", "SystemExit(0)", "SystemExit(None)"}


def sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_base64_file(path: os.PathLike[str] | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def canonical_sha256(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as stream:
        stream.write((json.dumps(document, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def run_argv(
    argv: list[str],
    *,
    check: bool = True,
    relay: bool = True,
    max_lines: int | None = 200,
    cwd: str | None = None,
    env: dict | None = None,
    log_path: pathlib.Path | None = None,
) -> tuple[int, str]:
    """Run a fixed argv vector, relay output, and preserve the child's real status."""
    print("+ " + " ".join(argv), flush=True)
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = [] if max_lines is None else collections.deque(maxlen=max_lines)
    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
    try:
        assert process.stdout is not None
        for line in process.stdout:
            tail.append(line.rstrip())
            if log is not None:
                log.write(line)
                log.flush()
            if relay:
                print("  | " + line, end="", flush=True)
        rc = process.wait()
    finally:
        if log is not None:
            log.flush()
            os.fsync(log.fileno())
            log.close()
    output = "\n".join(tail)
    if check and rc:
        raise ProfileError("command failed rc=%d: %s\n%s" % (rc, argv[0], output))
    return rc, output


def parse_cli_json(text: str) -> dict:
    raw = str(text).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ProfileError("CLI output contained no JSON object")
        value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ProfileError("CLI JSON was not an object")
    return value


def package_project(project_root: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """Create a byte-reproducible source snapshot without data, results, or secrets."""
    excluded = {
        ".agent_state", ".git", ".secrets", ".venv", "__pycache__", "data",
        "docs", "output", "report", "results", "source_snapshot", "upstream",
    }

    def archive_filter(info: tarfile.TarInfo):
        parts = pathlib.PurePosixPath(info.name).parts
        if any(part in excluded for part in parts):
            return None
        if any(part in {".env", "adc.json", "application_default_credentials.json"}
               for part in parts):
            return None
        if info.name.endswith((".pt", ".ckpt", ".tar.gz", ".tgz")):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.pax_headers = {}
        return info

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for child in sorted(project_root.iterdir(), key=lambda item: item.name):
                    if child.name.startswith(".") or child.name in excluded:
                        continue
                    archive.add(child, arcname=child.name, filter=archive_filter)
    return destination


def _remote_object_record(name: str, generation, size, md5_hash, crc32c) -> dict:
    if not any(name.startswith(prefix + "/") for prefix in INPUT_PREFIXES):
        raise ProfileError("remote input object escapes frozen prefixes: %s" % name)
    generation = str(generation or "")
    size = str(size or "")
    if not generation.isdigit() or not size.isdigit():
        raise ProfileError("remote input lacks generation/size: %s" % name)
    if not md5_hash or not crc32c:
        raise ProfileError("remote input lacks content hashes: %s" % name)
    return {
        "name": name,
        "generation": generation,
        "size_bytes": int(size),
        "md5_base64": str(md5_hash),
        "crc32c_base64": str(crc32c),
    }


def build_remote_input_identity(objects: list[dict]) -> dict:
    """Canonical, complete identity of every GCS object consumed by profiling."""
    normalized = []
    for item in objects:
        metadata = item.get("metadata", item)
        normalized.append(_remote_object_record(
            str(metadata.get("name", "")), metadata.get("generation"),
            metadata.get("size", metadata.get("size_bytes")),
            metadata.get("md5Hash", metadata.get("md5_base64")),
            metadata.get("crc32c", metadata.get("crc32c_base64")),
        ))
    normalized.sort(key=lambda item: item["name"])
    names = [item["name"] for item in normalized]
    if len(names) != len(set(names)):
        raise ProfileError("remote input inventory contains duplicate object names")
    for prefix in INPUT_PREFIXES:
        if not any(name.startswith(prefix + "/") for name in names):
            raise ProfileError("remote input inventory is empty for %s" % prefix)
    unsigned = {
        "schema": INPUT_SCHEMA,
        "bucket": BUCKET,
        "prefixes": list(INPUT_PREFIXES),
        "objects": normalized,
    }
    return {**unsigned, "identity_sha256": canonical_sha256(unsigned)}


def validate_remote_input_identity(identity: dict) -> None:
    rebuilt = build_remote_input_identity(list(identity.get("objects", [])))
    if identity != rebuilt:
        raise ProfileError("remote input inventory identity/hash mismatch")


def validate_salvage_receipt(receipt: dict, *, target_source_sha256: str,
                             remote_inputs: dict) -> None:
    """Validate an explicitly supplied external audit; never discover orphan objects."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SALVAGE_SCHEMA:
        raise ProfileError("profile salvage receipt has an unsupported schema")
    if receipt.get("authorization") != "GO":
        raise ProfileError("profile salvage receipt is not explicitly authorized")
    if (receipt.get("target_source_sha256") != target_source_sha256 or
            receipt.get("input_identity_sha256") != remote_inputs["identity_sha256"] or
            receipt.get("remote_inputs") != remote_inputs):
        raise ProfileError("profile salvage receipt target/input binding mismatch")
    source_sha = str(receipt.get("source_sha256", ""))
    source_profile = str(receipt.get("source_profile_id", ""))
    if (not re.fullmatch(r"[0-9a-f]{64}", source_sha) or
            not re.fullmatch(r"a100-[0-9a-f]{12}-[0-9a-f]{12}", source_profile)):
        raise ProfileError("profile salvage receipt has invalid source identity")
    expected_source_profile = "a100-%s-%s" % (
        source_sha[:12], remote_inputs["identity_sha256"][:12])
    if source_profile != expected_source_profile:
        raise ProfileError("profile salvage source profile does not match source/input hashes")
    compatibility = receipt.get("training_compatibility", {})
    if (compatibility.get("verdict") != "BYTE_IDENTICAL" or
            compatibility.get("source_sha256") != source_sha or
            compatibility.get("target_source_sha256") != target_source_sha256 or
            not re.fullmatch(r"[0-9a-f]{64}", str(
                compatibility.get("compared_surface_sha256", "")))):
        raise ProfileError("profile salvage receipt lacks byte-identical training proof")
    audit = receipt.get("audit", {})
    if (not audit.get("auditor") or not audit.get("audited_at") or
            not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("evidence_sha256", "")))):
        raise ProfileError("profile salvage receipt lacks external audit identity")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProfileError("profile salvage receipt has no promoted artifacts")
    seen = set()
    for artifact in artifacts:
        relative = pathlib.PurePosixPath(str(artifact.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ProfileError("profile salvage artifact path is unsafe or duplicated")
        seen.add(str(relative))
        artifact_sha = str(artifact.get("sha256", ""))
        if (not str(artifact.get("object_generation", "")).isdigit() or
                not re.fullmatch(r"[0-9a-f]{64}", artifact_sha) or
                int(artifact.get("size_bytes", -1)) < 0 or
                not str(artifact.get("remote", "")).startswith(
                    "lurestar/profiles/%s/artifacts/sha256/%s/" % (
                        source_profile, artifact_sha))):
            raise ProfileError("profile salvage artifact identity is incomplete")
    if receipt.get("artifact_fingerprint") != canonical_sha256({"artifacts": artifacts}):
        raise ProfileError("profile salvage artifact fingerprint mismatch")
    completed = set(receipt.get("completed_jobs", []))
    resume_steps = receipt.get("resume_steps", {})
    if (not completed.issubset(EXPECTED_GATE_JOBS) or not isinstance(resume_steps, dict) or
            not completed.union(resume_steps)):
        raise ProfileError("profile salvage job disposition is invalid")
    for job, step in resume_steps.items():
        if (job not in EXPECTED_GATE_JOBS or job in completed or
                not 0 < int(step) < EXPECTED_GATE_JOBS[job][0]):
            raise ProfileError("profile salvage resume step is invalid: %s" % job)


def build_spec(source_sha256: str, remote_inputs: dict,
               session_id: str | None = None, *, salvage_receipt: dict | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ProfileError("invalid source digest")
    validate_remote_input_identity(remote_inputs)
    if salvage_receipt is not None:
        validate_salvage_receipt(
            salvage_receipt, target_source_sha256=source_sha256,
            remote_inputs=remote_inputs)
    input_sha = remote_inputs["identity_sha256"]
    profile_id = "a100-%s-%s" % (source_sha256[:12], input_sha[:12])
    document = {
        "schema": PROFILE_SCHEMA,
        "profile_id": profile_id,
        "session_id": session_id,
        "gpu": "a100",
        "precision": "bf16-mixed",
        "source_sha256": source_sha256,
        "source_object": "%s/source/project-%s.tar.gz" % (GCS_ROOT, source_sha256),
        "profile_prefix": "%s/profiles/%s" % (GCS_ROOT, profile_id),
        "remote_inputs": remote_inputs,
        "input_identity_sha256": input_sha,
        "nonconfirmatory": True,
        "salvage_receipt": salvage_receipt,
        "gate_jobs": {name: {"steps": steps, "warmup_steps": warmup}
                      for name, (steps, warmup) in sorted(EXPECTED_GATE_JOBS.items())},
        "smoke_jobs": {name: {"steps": 50} for name in EXPECTED_SMOKE_JOBS},
    }
    unsigned = dict(document)
    document["contract_sha256"] = canonical_sha256(unsigned)
    return document


def validate_spec(spec: dict, *, require_session: bool = True) -> None:
    contract = spec.get("contract_sha256")
    unsigned = dict(spec)
    unsigned.pop("contract_sha256", None)
    if contract != canonical_sha256(unsigned):
        raise ProfileError("profile sidecar contract hash mismatch")
    digest = str(spec.get("source_sha256", ""))
    remote_inputs = spec.get("remote_inputs")
    if not isinstance(remote_inputs, dict):
        raise ProfileError("profile sidecar has no frozen remote input identity")
    expected = build_spec(
        digest, remote_inputs, spec.get("session_id"),
        salvage_receipt=spec.get("salvage_receipt"))
    if spec != expected:
        raise ProfileError("profile sidecar differs from the frozen profiling contract")
    if require_session and not re.fullmatch(r"gpu-[A-Za-z0-9._-]+", str(spec.get("session_id", ""))):
        raise ProfileError("profile sidecar has no owned Colab session identity")


def status_pair(delay: int = STATUS_DELAY_SECONDS, sleeper=time.sleep) -> tuple[dict, dict]:
    first = parse_cli_json(run_argv(
        ["colab", "status", "--json"], relay=False, max_lines=None)[1])
    sleeper(delay)
    second = parse_cli_json(run_argv(
        ["colab", "status", "--json"], relay=False, max_lines=None)[1])
    return first, second


def quota_pair(delay: int = STATUS_DELAY_SECONDS, sleeper=time.sleep) -> tuple[dict, dict]:
    first = parse_cli_json(run_argv(
        ["colab", "quota", "--json"], relay=False, max_lines=None)[1])
    sleeper(delay)
    second = parse_cli_json(run_argv(
        ["colab", "quota", "--json"], relay=False, max_lines=None)[1])
    return first, second


def agreed_runtime_state(first: dict, second: dict) -> str:
    first_gone = first.get("status") == "no_runtime"
    second_gone = second.get("status") == "no_runtime"
    if first_gone and second_gone:
        return "gone"
    if not first_gone and not second_gone:
        return "active"
    return "uncertain"


def teardown_owned_runtime(session_id: str, *, stopper=None, status_reader=status_pair,
                           quota_reader=quota_pair) -> dict:
    """Release only ``session_id`` and prove both assignment and billing have settled."""
    if not re.fullmatch(r"gpu-[A-Za-z0-9._-]+", str(session_id)):
        raise ProfileError("refusing teardown without a valid owned session identity")
    if stopper is None:
        stopper = lambda sid: run_argv(
            ["colab", "stop", "--session", sid], check=False, relay=True)
    stopped = False
    for _ in range(2):
        stopper(session_id)
        first, second = status_reader()
        if agreed_runtime_state(first, second) == "gone":
            stopped = True
            break
    if not stopped:
        raise ProfileError("owned Colab runtime did not reach two-read no-runtime state")
    quota_first, quota_second = quota_reader()
    for quota in (quota_first, quota_second):
        if (int(quota.get("active_runtimes", -1)) != 0 or
                float(quota.get("burn_rate_hourly", -1)) != 0.0):
            raise ProfileError("post-stop quota did not settle to zero runtime/burn twice")
    return quota_second


def validate_gate_group(output_root: pathlib.Path, jobs: tuple[str, ...]) -> None:
    jobs_dir = output_root / "gate" / "jobs"
    for name in jobs:
        steps, warmup = EXPECTED_GATE_JOBS[name]
        manifest_path = jobs_dir / (name + ".job.json")
        if not manifest_path.is_file():
            raise ProfileError("missing profile manifest: %s" % name)
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("job") != name or manifest.get("returncode") != 0 or
                manifest.get("steps") != steps or manifest.get("warmup_steps") != warmup):
            raise ProfileError("invalid or failed profile manifest: %s" % name)
        log = pathlib.Path(manifest.get("log", ""))
        if not log.is_file() or log.stat().st_size == 0:
            raise ProfileError("profile raw log is missing or empty: %s" % name)
        probes = list(pathlib.Path(jobs_dir).glob(name + ".probe.*.json"))
        parsed_probes = [json.loads(path.read_text()) for path in probes]
        successful = [probe for probe in parsed_probes if probe_succeeded(probe)]
        ledger_path = manifest.get("attempt_ledger")
        if ledger_path:
            ledger = json.loads(pathlib.Path(ledger_path).read_text())
            attempts = ledger.get("attempts") or []
            final_attempt = len(attempts) - 1
            matching = [probe for probe in successful
                        if int(probe.get("profile_attempt", -1)) == final_attempt]
            if (not attempts or int(manifest.get("attempt", -1)) != final_attempt or
                    len(matching) != 1):
                raise ProfileError("terminal probe is not bound to the final attempt: %s" % name)
            probe = matching[0]
        else:
            if len(successful) != 1:
                raise ProfileError(
                    "profile must have exactly one successful terminal probe: %s" % name)
            probe = successful[0]
        if (probe.get("peak_allocated_bytes") is None or
                probe.get("peak_reserved_bytes") is None):
            raise ProfileError("profile probe lacks peak VRAM: %s" % name)


def validate_profile_summary(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise ProfileError("profile_summary.json is missing")
    summary = json.loads(path.read_text())
    records = summary.get("records")
    if not isinstance(records, dict) or set(records) != set(EXPECTED_GATE_JOBS):
        raise ProfileError("profile summary does not contain the exact five required jobs")
    for name, record in records.items():
        if record.get("returncode") not in (0, None):
            raise ProfileError("profile summary contains a failed job: %s" % name)
        if record.get("missing_required"):
            raise ProfileError("profile summary has unmeasured fields for %s" % name)
        missing = [field for field in REQUIRED_PROFILE_FIELDS if record.get(field) is None]
        if missing:
            raise ProfileError("profile summary lacks %s for %s" % (", ".join(missing), name))
    if summary.get("projection", {}).get("incomplete_for"):
        raise ProfileError("profile budget projection is incomplete")
    return summary


def validate_smoke_job(output_root: pathlib.Path, name: str) -> None:
    if name not in EXPECTED_SMOKE_JOBS:
        raise ProfileError("unknown HMM smoke job: %s" % name)
    path = output_root / "smoke" / "jobs" / (name + ".json")
    if not path.is_file():
        raise ProfileError("missing 50-step HMM smoke manifest: %s" % name)
    record = json.loads(path.read_text())
    if record.get("job") != name or record.get("returncode") != 0 or record.get("steps") != 50:
        raise ProfileError("failed or drifted 50-step HMM smoke: %s" % name)
    log = pathlib.Path(record.get("log", ""))
    probe = pathlib.Path(record.get("probe", ""))
    if not log.is_file() or log.stat().st_size == 0 or not probe.is_file():
        raise ProfileError("HMM smoke evidence is incomplete: %s" % name)
    probe_data = json.loads(probe.read_text())
    if (probe_data.get("peak_allocated_bytes") is None or
            probe_data.get("peak_reserved_bytes") is None):
        raise ProfileError("HMM smoke probe lacks peak VRAM: %s" % name)
    if not record.get("materialized_configs") or not record.get("checkpoints"):
        raise ProfileError("HMM smoke lacks config/checkpoint evidence: %s" % name)


def validate_smoke(output_root: pathlib.Path) -> None:
    for name in EXPECTED_SMOKE_JOBS:
        validate_smoke_job(output_root, name)


def profile_complete(output_root: pathlib.Path) -> dict:
    validate_smoke(output_root)
    validate_gate_group(output_root, tuple(EXPECTED_GATE_JOBS))
    return validate_profile_summary(output_root / "gate" / "profile_summary.json")


def materialize_salvage_attempt_ledgers(output_root: pathlib.Path, receipt: dict) -> None:
    """Bind promoted pre-checkpoint metrics to attempt 0 before a replacement launch."""
    if not receipt:
        return
    jobs_dir = output_root / "gate" / "jobs"
    for job, resume_step in receipt["resume_steps"].items():
        steps, warmup = EXPECTED_GATE_JOBS[job]
        task, model = job.split("-", 1)
        if task == "lurestar":
            exp_dir = (output_root / "gate" / "root" / "runs" / model /
                       "seed1234" / "base" / (model + "-seed1234-base"))
        else:
            exp_dir = (output_root / "gate" / "root" / "runs" / "hmm" / model /
                       "seed1234" / "base" / (model + "-seed1234-hmm"))
        metrics = exp_dir / "version_0" / "metrics.csv"
        if not metrics.is_file() or metrics.stat().st_size == 0:
            raise ProfileError(
                "salvage receipt did not restore pre-checkpoint metrics for %s" % job)
        ledger_path = jobs_dir / (job + ".attempts.json")
        expected = {
            "schema": "nextlat_forgetting/profile_attempts/1",
            "job": job,
            "target_steps": steps,
            "warmup_steps": warmup,
            "attempts": [{"attempt": 0, "resume_step": 0, "version_start_index": 0}],
            "salvage_boundary": int(resume_step),
        }
        if ledger_path.is_file():
            if json.loads(ledger_path.read_text()) != expected:
                raise ProfileError("existing salvage attempt ledger is not audit-identical")
        else:
            atomic_json(ledger_path, expected)


class ProfileDurability:
    """Verified GCS mirror whose state object is the commit record."""

    def __init__(self, bucket, output_root: pathlib.Path, spec: dict, logger=print):
        self.bucket = bucket
        self.output_root = output_root.resolve()
        self.spec = spec
        self.prefix = spec["profile_prefix"].strip("/")
        self.log = logger
        self._lock = threading.Lock()
        self._uploaded: set[tuple[str, str, int]] = set()
        self._generation = 0
        self._fingerprint = None
        self._complete_committed = False
        self._committed_artifacts: dict[str, dict] = {}
        self._salvage_restored = False

    def _artifact_remote(self, relative: str, digest: str) -> str:
        # A committed state must remain restorable even when the same logical file changes in a
        # later sync. The digest namespace makes payload objects immutable/content-addressed.
        return "%s/artifacts/sha256/%s/%s" % (self.prefix, digest, relative)

    @property
    def complete_committed(self) -> bool:
        return self._complete_committed

    @property
    def salvage_restored(self) -> bool:
        return self._salvage_restored

    def _upload_file(self, local: pathlib.Path, *, relative: str | None = None) -> dict:
        digest = sha256_file(local)
        md5_digest = md5_base64_file(local)
        size = local.stat().st_size
        relative = relative or str(local.relative_to(self.output_root))
        remote = self._artifact_remote(relative, digest)
        key = (remote, digest, size)
        blob = self.bucket.blob(remote)
        if key not in self._uploaded:
            blob.metadata = dict(blob.metadata or {}, sha256=digest)
            try:
                blob.upload_from_filename(str(local), if_generation_match=0)
            except Exception as upload_error:
                # A previous session may already have created these exact immutable bytes. Only
                # that verified race is acceptable; all other upload failures remain fatal.
                try:
                    blob.reload()
                except Exception:
                    raise upload_error
            if local.stat().st_size != size or sha256_file(local) != digest:
                raise ProfileError("artifact changed while uploading: %s" % local)
            self._uploaded.add(key)
        blob.reload()
        if (int(blob.size or -1) != size or
                (blob.metadata or {}).get("sha256") != digest or
                str(blob.md5_hash or "") != md5_digest or
                not str(blob.generation or "").isdigit()):
            raise ProfileError("GCS verification failed: %s" % remote)
        return {"relative_path": relative,
                "remote": remote, "object_generation": str(blob.generation),
                "sha256": digest, "size_bytes": size}

    @staticmethod
    def _is_live_mutable(relative: str) -> bool:
        """Files append-written by the trainer/telemetry sampler while a job is active."""
        return relative.endswith((".log", ".csv"))

    @staticmethod
    def _stat_identity(path: pathlib.Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _snapshot_live_mutable(
        self, local: pathlib.Path, snapshot_root: pathlib.Path
    ) -> tuple[pathlib.Path | None, str | None]:
        """Copy one append-style file only if its source is unchanged across the copy.

        The returned private copy is immutable for the remainder of the transaction, so a
        subsequent trainer append cannot invalidate an otherwise useful checkpoint commit.
        """
        relative = str(local.relative_to(self.output_root))
        snapshot = snapshot_root / hashlib.sha256(relative.encode()).hexdigest()
        try:
            before = self._stat_identity(local)
            with open(local, "rb") as source, open(snapshot, "xb") as destination:
                shutil.copyfileobj(source, destination, length=1 << 20)
                destination.flush()
                os.fsync(destination.fileno())
            after = self._stat_identity(local)
        except FileNotFoundError:
            snapshot.unlink(missing_ok=True)
            return None, "removed_during_snapshot"
        if before != after or snapshot.stat().st_size != before[2]:
            snapshot.unlink(missing_ok=True)
            return None, "changed_during_snapshot"
        return snapshot, None

    def _scan(self) -> list[pathlib.Path]:
        return sorted(
            path for path in self.output_root.rglob("*")
            if path.is_file() and not path.name.endswith(".partial")
        )

    def sync_once(self, *, complete: bool = False) -> dict:
        with self._lock:
            if self._complete_committed and not complete:
                raise ProfileError("refusing to downgrade a complete profile state")
            current: dict[str, dict] = {}
            deferred: list[dict] = []
            with tempfile.TemporaryDirectory(prefix="profile-sync-") as temporary:
                snapshot_root = pathlib.Path(temporary)
                for path in self._scan():
                    relative = str(path.relative_to(self.output_root))
                    if self._is_live_mutable(relative):
                        snapshot, reason = self._snapshot_live_mutable(path, snapshot_root)
                        if snapshot is None:
                            prior = self._committed_artifacts.get(relative)
                            if prior is not None:
                                current[relative] = prior
                            deferred.append({
                                "relative_path": relative,
                                "reason": reason,
                                "retained_sha256": prior["sha256"] if prior else None,
                            })
                            continue
                        current[relative] = self._upload_file(
                            snapshot, relative=relative)
                    else:
                        current[relative] = self._upload_file(path)

            if complete and deferred:
                names = ", ".join(item["relative_path"] for item in deferred)
                raise ProfileError(
                    "complete profile sync requires stable live artifacts: %s" % names)

            artifacts = [current[name] for name in sorted(current)]
            # Non-live files are uploaded directly and therefore retain the stronger original
            # check. Live files were uploaded from private, fsynced snapshots instead.
            for artifact in artifacts:
                relative = artifact["relative_path"]
                if self._is_live_mutable(relative):
                    continue
                local = self.output_root / relative
                if (local.stat().st_size != artifact["size_bytes"] or
                        sha256_file(local) != artifact["sha256"]):
                    raise ProfileError("artifact changed before state commit: %s" % local)
            fingerprint = canonical_sha256({
                "artifacts": artifacts,
                "deferred_live_mutable": deferred,
            })
            if fingerprint != self._fingerprint:
                self._generation += 1
                self._fingerprint = fingerprint
            state = {
                "schema": STATE_SCHEMA,
                "profile_id": self.spec["profile_id"],
                "source_sha256": self.spec["source_sha256"],
                "input_identity_sha256": self.spec["input_identity_sha256"],
                "remote_inputs": self.spec["remote_inputs"],
                "generation": self._generation,
                "complete": bool(complete),
                "artifacts": artifacts,
                "deferred_live_mutable": deferred,
                "artifact_fingerprint": fingerprint,
                "synced_at": time.time(),
            }
            payload = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()
            digest = hashlib.sha256(payload).hexdigest()
            blob = self.bucket.blob(self.prefix + "/state.json")
            blob.metadata = dict(blob.metadata or {}, sha256=digest)
            blob.upload_from_string(payload, content_type="application/json")
            blob.reload()
            if blob.download_as_bytes() != payload or (blob.metadata or {}).get("sha256") != digest:
                raise ProfileError("profile state verification failed")
            self._committed_artifacts = {
                artifact["relative_path"]: artifact for artifact in artifacts
            }
            self._complete_committed = bool(complete)
            state["state_sha256"] = digest
            self.log("[profile-sync] generation=%d artifacts=%d complete=%s" %
                     (self._generation, len(artifacts), complete))
            return state

    def restore(self, *, salvage_receipt: dict | None = None) -> int:
        self._salvage_restored = False
        blob = self.bucket.blob(self.prefix + "/state.json")
        try:
            payload = blob.download_as_bytes()
            blob.reload()
        except Exception:
            payload = None
        if payload is None:
            if salvage_receipt is None:
                return 0
            validate_salvage_receipt(
                salvage_receipt,
                target_source_sha256=self.spec["source_sha256"],
                remote_inputs=self.spec["remote_inputs"],
            )
            state = {
                "generation": 0,
                "complete": False,
                "artifacts": salvage_receipt["artifacts"],
                "artifact_fingerprint": salvage_receipt["artifact_fingerprint"],
            }
            self._salvage_restored = True
            self.log("[profile-restore] using explicit audited salvage receipt")
        else:
            digest = hashlib.sha256(payload).hexdigest()
            if (blob.metadata or {}).get("sha256") != digest:
                raise ProfileError("durable profile state metadata/hash mismatch")
            state = json.loads(payload)
            if (state.get("schema") != STATE_SCHEMA or
                    state.get("profile_id") != self.spec["profile_id"] or
                    state.get("source_sha256") != self.spec["source_sha256"] or
                    state.get("input_identity_sha256") != self.spec["input_identity_sha256"] or
                    state.get("remote_inputs") != self.spec["remote_inputs"]):
                raise ProfileError("durable profile state does not match this source contract")
        restored = 0
        for artifact in state.get("artifacts", []):
            relative = pathlib.PurePosixPath(str(artifact.get("relative_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ProfileError("durable artifact path escapes profile root")
            local = self.output_root.joinpath(*relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            partial = local.with_name(local.name + ".partial")
            object_generation = str(artifact.get("object_generation", ""))
            if not object_generation.isdigit():
                raise ProfileError("durable artifact lacks immutable object generation")
            artifact_blob = self.bucket.blob(
                artifact["remote"], generation=int(object_generation))
            artifact_blob.download_to_filename(str(partial))
            artifact_blob.reload()
            if str(artifact_blob.generation) != object_generation:
                partial.unlink(missing_ok=True)
                raise ProfileError("restored wrong GCS artifact generation: %s" % relative)
            if (partial.stat().st_size != int(artifact["size_bytes"]) or
                    sha256_file(partial) != artifact["sha256"]):
                partial.unlink(missing_ok=True)
                raise ProfileError("restored profile artifact failed verification: %s" % relative)
            os.replace(partial, local)
            restored += 1
        self._generation = int(state.get("generation", 0))
        self._fingerprint = state.get("artifact_fingerprint")
        self._committed_artifacts = {
            artifact["relative_path"]: artifact for artifact in state.get("artifacts", [])
        }
        self._complete_committed = bool(state.get("complete"))
        self.log("[profile-restore] artifacts=%d generation=%d" %
                 (restored, self._generation))
        return restored

    def publish_terminal(self, state: dict, *, success: bool, error: str | None = None) -> dict:
        if success and not state.get("complete"):
            raise ProfileError("refusing success terminal before complete committed state")
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "profile_id": self.spec["profile_id"],
            "session_id": self.spec["session_id"],
            "source_sha256": self.spec["source_sha256"],
            "input_identity_sha256": self.spec["input_identity_sha256"],
            "remote_inputs": self.spec["remote_inputs"],
            "state_sha256": state["state_sha256"],
            "generation": state["generation"],
            "success": bool(success),
            "complete": bool(state.get("complete")),
            "error": error,
            "published_at": time.time(),
        }
        payload = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        remote = "%s/sessions/%s/terminal.json" % (self.prefix, self.spec["session_id"])
        blob = self.bucket.blob(remote)
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        blob.upload_from_string(payload, content_type="application/json")
        blob.reload()
        if blob.download_as_bytes() != payload or (blob.metadata or {}).get("sha256") != digest:
            raise ProfileError("profile terminal verification failed")
        return terminal


def bucket_remote_input_identity(bucket) -> dict:
    objects = []
    for prefix in INPUT_PREFIXES:
        for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
            if blob.name.endswith("/"):
                continue
            blob.reload()
            objects.append({
                "name": blob.name,
                "generation": blob.generation,
                "size_bytes": blob.size,
                "md5_base64": blob.md5_hash,
                "crc32c_base64": blob.crc32c,
            })
    return build_remote_input_identity(objects)


def host_remote_input_identity() -> dict:
    """Freeze the current generations and object hashes before any runtime is provisioned."""
    objects = []
    for prefix in INPUT_PREFIXES:
        uri = "gs://%s/%s/**" % (BUCKET, prefix)
        raw = run_argv(
            ["gcloud", "storage", "ls", "--recursive", "--json", uri],
            relay=False, max_lines=None,
        )[1]
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProfileError("cannot parse remote input inventory for %s" % prefix) from exc
        if not isinstance(records, list):
            raise ProfileError("remote input inventory was not a list for %s" % prefix)
        objects.extend(records)
    return build_remote_input_identity(objects)


def download_prefix(bucket, remote_prefix: str, local_root: pathlib.Path,
                    remote_inputs: dict | None = None) -> int:
    count = 0
    prefix = remote_prefix.rstrip("/") + "/"
    if remote_inputs is None:
        bound = [
            {"name": blob.name, "generation": blob.generation}
            for blob in bucket.list_blobs(prefix=prefix)
            if not blob.name.endswith("/")
        ]
    else:
        validate_remote_input_identity(remote_inputs)
        bound = [item for item in remote_inputs["objects"]
                 if item["name"].startswith(prefix)]
    for item in bound:
        name = item["name"]
        relative = name[len(prefix):]
        if not relative or relative.endswith("/"):
            continue
        local = local_root / pathlib.PurePosixPath(relative)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".partial")
        blob = bucket.blob(name, generation=int(item["generation"]))
        blob.download_to_filename(str(partial))
        if remote_inputs is not None:
            blob.reload()
            observed = _remote_object_record(
                blob.name, blob.generation, blob.size, blob.md5_hash, blob.crc32c)
            if observed != item:
                partial.unlink(missing_ok=True)
                raise ProfileError("downloaded GCS generation identity mismatch: %s" % name)
        os.replace(partial, local)
        count += 1
    return count


def verify_sha_inventory(root: pathlib.Path, inventory: pathlib.Path) -> None:
    for line in inventory.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ProfileError("malformed SHA inventory line")
        relative = pathlib.PurePosixPath(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ProfileError("SHA inventory path escapes runtime root")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or sha256_file(path) != fields[0]:
            raise ProfileError("SHA inventory mismatch: %s" % relative)


def verify_runtime_gpu(requested: str, expected_torch: str | None = None) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise ProfileError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if requested.upper() != "A100" or "A100" not in name.upper():
        raise ProfileError("profiling requires an A100, got %s" % name)
    if not torch.cuda.is_bf16_supported():
        raise ProfileError("A100 runtime does not report BF16 support")
    if expected_torch is not None and str(torch.__version__) != expected_torch:
        raise ProfileError("dependency install replaced Colab torch")
    record = {"name": name, "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
              "bf16": True, "peak_vram_supported": True}
    print("PROFILE_GPU=" + json.dumps(record, sort_keys=True), flush=True)
    return record


def run_smoke_job(project: pathlib.Path, upstream: pathlib.Path, output: pathlib.Path,
                  model: str) -> None:
    name = "hmm-smoke-" + model
    manifest = output / "smoke" / "jobs" / (name + ".json")
    if manifest.is_file():
        try:
            validate_smoke_job(output, name)
            print("[resume] %s already complete" % name, flush=True)
            return
        except ProfileError:
            pass
    jobs = output / "smoke" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    probe_pattern = jobs / (name + ".probe.{pid}.json")
    log = jobs / (name + ".log")
    for stale_probe in jobs.glob(name + ".probe.*.json"):
        stale_probe.unlink()
    smoke_root = output / "smoke" / model / "root"
    env = dict(os.environ)
    env.update({
        "NEXTLAT_REPO": str(upstream),
        "LURESTAR_ROOT": str(smoke_root),
        "LURESTAR_PRECISION": "bf16-mixed",
        "LURESTAR_ENTRY": str(project / "scripts" / "profile_entry.py"),
        "PROFILE_TRAIN_PY": str(project / "scripts" / "train_hmm.py"),
        "PROFILE_PROBE_JSON": str(probe_pattern),
    })
    config = "%s_hmm.yaml" % model
    t0 = time.time()
    rc, _ = run_argv([
        "bash", str(project / "scripts" / "launch_train.sh"), config, "1234",
        "trainer.train_batches=50", "trainer.val_interval=50", "trainer.test_interval=50",
        "trainer.save_recovery_checkpoint=25",
    ], check=False, cwd=str(project), env=env, log_path=log)
    probes = sorted(jobs.glob(name + ".probe.*.json"))
    configs = sorted(str(path) for path in smoke_root.rglob("materialized_config.yaml"))
    checkpoints = sorted(str(path) for path in smoke_root.rglob("*.pt")
                         if not path.name.endswith(".partial"))
    atomic_json(manifest, {
        "job": name,
        "task": "hmm_smoke",
        "model": model,
        "seed": 1234,
        "steps": 50,
        "precision": "bf16-mixed",
        "returncode": rc,
        "wall_seconds": time.time() - t0,
        "log": str(log),
        "probe": str(probes[-1]) if len(probes) == 1 else "",
        "materialized_configs": configs,
        "checkpoints": checkpoints,
    })
    if rc:
        raise ProfileError("50-step HMM smoke failed for %s" % model)


def copy_runtime_provenance(project: pathlib.Path, output: pathlib.Path) -> None:
    source = project / "source_snapshot" / "runtime_patch"
    destination = output / "provenance"
    for name in ("runtime_patch.diff", "runtime_patch_receipt.json"):
        path = source / name
        if path.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            partial = destination / (name + ".partial")
            shutil.copyfile(path, partial)
            os.replace(partial, destination / name)


def runtime_driver() -> int:
    """Colab role. Durable traffic goes only through the Python storage client."""
    spec = json.loads(pathlib.Path(REMOTE_SPEC).read_text())
    validate_spec(spec)
    if not pathlib.Path(ADC_PATH).is_file():
        raise ProfileError("uploaded ADC is missing")
    os.chmod(ADC_PATH, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ADC_PATH
    os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT

    stop = threading.Event()

    def heartbeat() -> None:
        started = time.time()
        while not stop.wait(HEARTBEAT_SECONDS):
            print("[profile-heartbeat] elapsed_s=%d" % (time.time() - started), flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    salvage = spec.get("salvage_receipt")
    local_profile_id = (salvage["source_profile_id"] if salvage else spec["profile_id"])
    output = pathlib.Path("/content/lurestar/profiles") / local_profile_id
    output.mkdir(parents=True, exist_ok=True)
    durability = None
    sync_thread = None
    try:
        original_torch = verify_runtime_gpu(spec["gpu"])["torch"]
        run_argv([sys.executable, "-m", "pip", "install", "-q",
                  "google-cloud-storage", "google-auth"])
        from google.cloud import storage

        bucket = storage.Client(project=GCP_PROJECT).bucket(BUCKET)
        observed_inputs = bucket_remote_input_identity(bucket)
        if observed_inputs != spec["remote_inputs"]:
            raise ProfileError(
                "remote corpus/manifest inventory changed after sidecar freeze; refusing profile"
            )
        durability = ProfileDurability(bucket, output, spec)
        durability.restore(salvage_receipt=salvage)
        if durability.salvage_restored:
            materialize_salvage_attempt_ledgers(output, salvage)
        if durability.complete_committed:
            profile_complete(output)
            state = durability.sync_once(complete=True)
            terminal = durability.publish_terminal(state, success=True)
            print("PROFILE_TERMINAL=" + json.dumps(terminal, sort_keys=True), flush=True)
            print("PROFILE_COMPLETE=True", flush=True)
            return 0

        project = pathlib.Path("/content/project")
        project.mkdir(parents=True, exist_ok=True)
        archive = pathlib.Path("/content/project.tar.gz")
        source_blob = bucket.blob(spec["source_object"])
        source_blob.download_to_filename(str(archive) + ".partial")
        os.replace(str(archive) + ".partial", archive)
        if sha256_file(archive) != spec["source_sha256"]:
            raise ProfileError("immutable source snapshot hash mismatch")
        with tarfile.open(archive) as bundle:
            bundle.extractall(project)

        upstream = project / "upstream" / "NextLat"
        if not (upstream / ".git").is_dir():
            run_argv(["git", "clone", "-q", UPSTREAM_URL, str(upstream)])
        run_argv(["git", "checkout", "-q", PINNED_COMMIT], cwd=str(upstream))
        requirements = pathlib.Path("/content/requirements-no-torch.txt")
        requirements.write_text("\n".join(
            line for line in (upstream / "requirements.txt").read_text().splitlines()
            if not line.strip().startswith("torch")
        ) + "\n")
        run_argv([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
        verify_runtime_gpu(spec["gpu"], expected_torch=original_torch)
        run_argv([
            sys.executable, str(project / "scripts" / "runtime_bootstrap.py"),
            "--project-root", str(project), "--upstream", str(upstream),
        ], cwd=str(project))
        copy_runtime_provenance(project, output)

        data_root = pathlib.Path("/content/lurestar/data")
        manifests = pathlib.Path("/content/lurestar/manifests")
        if download_prefix(bucket, GCS_ROOT + "/corpus/stargraph", data_root / "stargraph",
                           spec["remote_inputs"]) == 0:
            raise ProfileError("stargraph corpus is absent from GCS")
        if download_prefix(bucket, GCS_ROOT + "/corpus/hmm", data_root / "hmm",
                           spec["remote_inputs"]) == 0:
            raise ProfileError("HMM arrays are absent from GCS")
        if download_prefix(bucket, GCS_ROOT + "/manifests", manifests,
                           spec["remote_inputs"]) == 0:
            raise ProfileError("frozen manifests are absent from GCS")
        verify_sha_inventory(data_root / "stargraph", manifests / "corpus.sha256")
        verify_sha_inventory(pathlib.Path("/content/lurestar"),
                             manifests / "manifest_inventory.sha256")
        atomic_json(output / "provenance" / "runtime_receipt.json", {
            "schema": PROFILE_SCHEMA,
            "profile_id": spec["profile_id"],
            "session_id": spec["session_id"],
            "source_sha256": spec["source_sha256"],
            "input_identity_sha256": spec["input_identity_sha256"],
            "remote_inputs": spec["remote_inputs"],
            "upstream_commit": PINNED_COMMIT,
            "gpu": verify_runtime_gpu(spec["gpu"], expected_torch=original_torch),
            "corpus_inventory_sha256": sha256_file(manifests / "corpus.sha256"),
            "manifest_inventory_sha256": sha256_file(
                manifests / "manifest_inventory.sha256"),
        })

        def sync_loop() -> None:
            while not stop.wait(SYNC_SECONDS):
                try:
                    durability.sync_once(complete=False)
                except Exception as exc:  # preserve training; retry on the next window
                    print("[profile-sync] failed: %s" % exc, flush=True)

        sync_thread = threading.Thread(target=sync_loop, daemon=True)
        sync_thread.start()

        for model in ("gpt", "nextlat"):
            run_smoke_job(project, upstream, output, model)
            durability.sync_once(complete=False)
        validate_smoke(output)

        lurestar_jobs = ("lurestar-gpt", "lurestar-nextlat", "lurestar-bst")
        hmm_jobs = ("hmm-gpt", "hmm-nextlat")
        try:
            validate_gate_group(output, lurestar_jobs)
            print("[resume] Lure-Star profile group already complete", flush=True)
        except ProfileError:
            rc, _ = run_argv([
                "bash", str(project / "scripts" / "profile.sh"), "--lurestar-only",
                "--out", str(output / "gate"),
            ], check=False, cwd=str(project), env=dict(
                os.environ, NEXTLAT_REPO=str(upstream), LURESTAR_PRECISION="bf16-mixed"),
               log_path=output / "wrapper-lurestar.log")
            # The group-only script deliberately returns nonzero because the final projection
            # lacks HMM. Job receipts, not that expected aggregate status, decide group success.
            print("PROFILE_LURESTAR_GROUP_RC=%d" % rc, flush=True)
            validate_gate_group(output, lurestar_jobs)
            durability.sync_once(complete=False)

        try:
            validate_gate_group(output, hmm_jobs)
            print("[resume] HMM profile group already complete", flush=True)
        except ProfileError:
            rc, _ = run_argv([
                "bash", str(project / "scripts" / "profile.sh"), "--hmm-only",
                "--out", str(output / "gate"),
            ], check=False, cwd=str(project), env=dict(
                os.environ, NEXTLAT_REPO=str(upstream), LURESTAR_PRECISION="bf16-mixed"),
               log_path=output / "wrapper-hmm.log")
            print("PROFILE_HMM_GROUP_RC=%d" % rc, flush=True)
            validate_gate_group(output, hmm_jobs)

        # Rebuild once from the complete set so the final summary cannot be the intentionally
        # incomplete intermediate summary emitted by the Lure-Star-only invocation.
        run_argv([
            sys.executable, str(project / "scripts" / "profile_summarize.py"),
            "--jobs-dir", str(output / "gate" / "jobs"),
            "--out", str(output / "gate" / "profile_summary.json"),
        ], cwd=str(project))
        profile_complete(output)
        state = durability.sync_once(complete=True)
        terminal = durability.publish_terminal(state, success=True)
        print("PROFILE_TERMINAL=" + json.dumps(terminal, sort_keys=True), flush=True)
        print("PROFILE_COMPLETE=True", flush=True)
        return 0
    except BaseException as exc:  # persist every useful partial artifact before refusal
        traceback.print_exc()
        atomic_json(output / "failure.json", {
            "schema": PROFILE_SCHEMA,
            "profile_id": spec.get("profile_id"),
            "session_id": spec.get("session_id"),
            "source_sha256": spec.get("source_sha256"),
            "input_identity_sha256": spec.get("input_identity_sha256"),
            "remote_inputs": spec.get("remote_inputs"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": time.time(),
        })
        if durability is not None:
            try:
                state = durability.sync_once(complete=False)
                durability.publish_terminal(state, success=False, error=str(exc))
            except Exception as sync_exc:
                print("PROFILE_FAILURE_SYNC_FAILED=%s" % sync_exc, flush=True)
        print("PROFILE_COMPLETE=False", flush=True)
        return 2
    finally:
        stop.set()
        if sync_thread is not None:
            sync_thread.join(timeout=30)
        heartbeat_thread.join(timeout=5)


def host_read_json(uri: str) -> dict | None:
    rc, raw = run_argv(
        ["gcloud", "storage", "cat", uri], check=False, relay=False, max_lines=None)
    if rc:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def verified_terminal(spec: dict) -> dict | None:
    uri = "gs://%s/%s/sessions/%s/terminal.json" % (
        BUCKET, spec["profile_prefix"], spec["session_id"])
    marker = host_read_json(uri)
    if not marker:
        return None
    if (marker.get("schema") != TERMINAL_SCHEMA or
            marker.get("profile_id") != spec["profile_id"] or
            marker.get("session_id") != spec["session_id"] or
            marker.get("source_sha256") != spec["source_sha256"] or
            marker.get("input_identity_sha256") != spec["input_identity_sha256"] or
            marker.get("remote_inputs") != spec["remote_inputs"]):
        return None
    return marker


def host_profile_generation(spec: dict) -> int:
    uri = "gs://%s/%s/state.json" % (BUCKET, spec["profile_prefix"])
    state = host_read_json(uri) or {}
    if (state.get("schema") != STATE_SCHEMA or
            state.get("profile_id") != spec["profile_id"] or
            state.get("source_sha256") != spec["source_sha256"] or
            state.get("input_identity_sha256") != spec["input_identity_sha256"] or
            state.get("remote_inputs") != spec["remote_inputs"]):
        return -1
    return int(state.get("generation", -1))


def monitor_owned_runtime(spec: dict, *, status_reader=status_pair,
                          terminal_reader=verified_terminal,
                          generation_reader=host_profile_generation,
                          stall_windows: int = STALL_WINDOWS) -> dict:
    """Keep an advancing runtime alive after an exec-stream return."""
    generation = generation_reader(spec)
    stalled = 0
    while True:
        marker = terminal_reader(spec)
        if marker is not None:
            return {"reason": "terminal", "terminal": marker}
        first, second = status_reader()
        runtime = agreed_runtime_state(first, second)
        if runtime == "gone":
            return {"reason": "gone", "terminal": terminal_reader(spec)}
        if runtime == "uncertain":
            continue
        current = generation_reader(spec)
        if current > generation:
            print("durable profile artifacts advanced; preserving owned runtime", flush=True)
            generation = current
            stalled = 0
            continue
        stalled += 1
        print("owned profile runtime has no durable advance %d/%d" %
              (stalled, stall_windows), flush=True)
        if stalled >= stall_windows:
            return {"reason": "stalled", "terminal": None}


def host_project_root() -> pathlib.Path:
    candidate = pathlib.Path(sys.argv[0]).resolve().parent.parent
    if not (candidate / "scripts" / "colab_profile_loop.py").is_file():
        candidate = pathlib.Path.cwd().resolve()
    if not (candidate / "scripts" / "colab_profile_loop.py").is_file():
        raise ProfileError("run scripts/colab_profile_loop.py from the project checkout")
    return candidate


def host_loop() -> int:
    project = host_project_root()
    first, second = status_pair()
    if agreed_runtime_state(first, second) != "gone":
        raise ProfileError("profiling launcher requires two agreeing no-runtime reads")
    quota = parse_cli_json(run_argv(
        ["colab", "quota", "--json"], relay=False, max_lines=None)[1])
    if float(quota.get("paid_balance", 0.0)) <= HARD_STOP_BALANCE_CU:
        raise ProfileError("project compute hard-stop balance reached")

    archive = package_project(project, project / ".agent_state" / "profile-project.tar.gz")
    source_sha = sha256_file(archive)
    remote_inputs = host_remote_input_identity()
    salvage_receipt = None
    salvage_path = os.environ.get("PROFILE_SALVAGE_RECEIPT")
    if salvage_path:
        salvage_receipt = json.loads(pathlib.Path(salvage_path).read_text())
    spec = build_spec(source_sha, remote_inputs, salvage_receipt=salvage_receipt)
    sidecar = project / ".agent_state" / "profile_job.json"
    atomic_json(sidecar, spec)
    source_uri = "gs://%s/%s" % (BUCKET, spec["source_object"])
    run_argv(["gcloud", "storage", "cp", str(archive), source_uri])

    adc = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if not adc.is_file() or (adc.stat().st_mode & 0o077):
        raise ProfileError("local ADC is missing or not mode 0600")
    owned = False
    terminal = None
    try:
        started = parse_cli_json(run_argv(
            ["colab", "start", "--gpu", "a100", "--json"], max_lines=None)[1])
        session_id = str(started.get("session", ""))
        spec = build_spec(
            source_sha, remote_inputs, session_id, salvage_receipt=salvage_receipt)
        validate_spec(spec)
        atomic_json(sidecar, spec)
        owned = True
        run_argv(["colab", "upload", "--session", session_id,
                  str(adc), ADC_PATH])
        run_argv(["colab", "upload", "--session", session_id,
                  str(sidecar), REMOTE_SPEC])
        launcher = project / "scripts" / "colab_profile_loop.py"
        rc, output = run_argv([
            "colab", "exec", "--session", session_id, "--timeout", "180m", str(launcher),
        ], check=False, max_lines=200)
        print("PROFILE_EXEC_RC=%d" % rc, flush=True)
        terminal = verified_terminal(spec)
        if terminal is None:
            outcome = monitor_owned_runtime(spec)
            terminal = outcome.get("terminal")
            print("PROFILE_MONITOR_REASON=%s" % outcome["reason"], flush=True)
        if terminal is None:
            raise ProfileError("profile runtime ended/stalled without a verified terminal marker")
        if not terminal.get("success") or not terminal.get("complete"):
            raise ProfileError("durable profile terminal reports failure: %s" %
                               terminal.get("error"))
        print("DURABLE_PROFILE_COMPLETE=True", flush=True)
        return 0
    finally:
        if owned:
            settled = teardown_owned_runtime(spec["session_id"])
            print("SETTLED_BALANCE_CU=%s ACTIVE_RUNTIMES=%s BURN_RATE=%s" %
                  (settled.get("paid_balance"), settled.get("active_runtimes"),
                   settled.get("burn_rate_hourly")), flush=True)


if __name__ == "__main__" or "get_ipython" in dir():
    raise SystemExit(runtime_driver() if pathlib.Path("/content").is_dir() else host_loop())
