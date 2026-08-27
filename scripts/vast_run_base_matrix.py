#!/usr/bin/env python3
"""Run the frozen Lure-Star base matrix durably on the rented Vast GPU.

This is a provider adapter, not a scientific-code fork.  It installs the exact
content-addressed project snapshot and pinned upstream checkout, restores any
durable state from GCS, then calls the frozen matrix/evaluator functions.  GCS
state is synchronized while a cell is running and after every evaluated cell.
"""

from __future__ import annotations

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
from dataclasses import dataclass


BUCKET = "nextlat-lurestar-project-flash-490419"
GCP_PROJECT = "project-flash-490419"
PREFIX = "lurestar"
SOURCE_SHA256 = "8aa491e55dc86f2b8700fb5f3f5376d61518110947832d3034933af9b2279013"
SOURCE_OBJECT = f"{PREFIX}/source/project-{SOURCE_SHA256}.tar.gz"
INPUT_SHA256 = "33fbf4358b7c7def932fb96c1f4a5c04cb8713925dccbff5d385982e910a5c43"
INPUT_PREFIX = f"{PREFIX}/input_bundles/{INPUT_SHA256}"
PINNED_UPSTREAM = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"

CONTENT = pathlib.Path("/content")
ROOT = CONTENT / "lurestar"
PROJECT = CONTENT / "project"
SOURCE_ARCHIVE = CONTENT / f"project-{SOURCE_SHA256}.tar.gz"
UPSTREAM = PROJECT / "upstream" / "NextLat"
ADC = pathlib.Path("/root/.nextlat-secrets/gcp-adc.json")
LOG_ROOT = pathlib.Path("/workspace/nextlat/logs")
INSTANCE_ID = os.environ.get("VAST_INSTANCE_ID", "48593365").strip()
ONLY_JOBS = tuple(
    value.strip()
    for value in os.environ.get("VAST_ONLY_JOBS", "").split(",")
    if value.strip()
)
if not INSTANCE_ID or not re.fullmatch(r"[0-9]+", INSTANCE_ID):
    raise RuntimeError("VAST_INSTANCE_ID must be the numeric Vast contract id")
if any(not re.fullmatch(r"(?:gpt|nextlat|bst)-s(?:1234|1235|1236|1237|1238)-base", job)
       for job in ONLY_JOBS):
    raise RuntimeError("VAST_ONLY_JOBS contains a non-canonical base job id")
if len(set(ONLY_JOBS)) != len(ONLY_JOBS):
    raise RuntimeError("VAST_ONLY_JOBS contains duplicate jobs")
LEDGER_OBJECT = f"run_ledger-{INSTANCE_ID}.json" if ONLY_JOBS else "run_ledger.json"
VAST_FAILURE_SCHEMA = "nextlat_forgetting/vast_base_failure_disposition/1"
VAST_QUARANTINE_SCHEMA = "nextlat_forgetting/vast_base_scientific_quarantine/1"
BASE_TARGET_STEP = 20_000
# Vast's 3090 worker has enough bandwidth for checkpoint uploads, but the default 120s
# write deadline is too short for a 542MB BST recovery generation.  Eight MiB is a
# valid GCS resumable chunk multiple and keeps each request bounded/restartable.
VAST_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
VAST_UPLOAD_TIMEOUT = (30, 900)  # (connect seconds, bounded read/write seconds)
VAST_BACKGROUND_SYNC_SECONDS = 300
VAST_FAST_RECOVERY_STEPS = 5_000
VAST_BST_RECOVERY_STEPS = 1_000


@dataclass(frozen=True)
class VastFailureDisposition:
    """Provider-local decision; this is deliberately not a Colab retry policy."""

    retry: bool
    kind: str
    reason: str


def _latest_ledger_states(ledger: pathlib.Path) -> dict[str, dict]:
    """Read only lifecycle/provenance state, never model or scientific outputs."""
    if not ledger.is_file():
        return {}
    try:
        document = json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Vast local run ledger is unreadable") from exc
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("Vast local run ledger lacks append-only entries")
    return {
        entry["job_id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("job_id"), str)
    }


def _safe_local_artifact(root: pathlib.Path, base: pathlib.Path, relative: object) -> pathlib.Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    pure = pathlib.PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        return None
    candidate = (base / pure).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def terminal_base_is_locally_complete(state: dict, root: pathlib.Path) -> tuple[bool, str]:
    """Verify all recorded terminal artifacts before a Vast restore can be skipped."""
    if state.get("status") not in {"TRAINED", "DONE"} or state.get("phase") != "base":
        return False, "not a terminal base state"
    if state.get("step") != BASE_TARGET_STEP:
        return False, "terminal base state does not have exact 20,000 steps"
    raw_root, artifacts = state.get("out_root"), state.get("artifacts")
    if not isinstance(raw_root, str) or not isinstance(artifacts, dict) or not artifacts:
        return False, "terminal base state lacks hashed artifact inventory"
    out_root = pathlib.Path(raw_root).resolve()
    try:
        out_root.relative_to((root / "runs").resolve())
    except ValueError:
        return False, "terminal base out_root escapes local run namespace"
    for relative, expected in artifacts.items():
        path = _safe_local_artifact(root, out_root, relative)
        if (path is None or not path.is_file() or path.is_symlink() or
                not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected):
            return False, f"terminal artifact missing/stale: {relative}"
    checkpoint_raw, checkpoint_sha = state.get("final_checkpoint"), state.get("final_checkpoint_sha256")
    if not isinstance(checkpoint_raw, str) or not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        return False, "terminal base lacks final checkpoint SHA"
    checkpoint = pathlib.Path(checkpoint_raw).resolve()
    try:
        checkpoint.relative_to(out_root)
    except ValueError:
        return False, "terminal checkpoint escapes its output root"
    if not checkpoint.is_file() or checkpoint.is_symlink() or sha256_file(checkpoint) != checkpoint_sha:
        return False, "terminal final checkpoint missing/stale"
    return True, "all terminal artifacts and checkpoint hash verify locally"


def remote_base_state_jobs(state_names: list[str], *, selected_jobs: tuple[str, ...]) -> set[str]:
    """Map only worker-scoped Lure-Star base state objects to job IDs."""
    prefix = f"{PREFIX}/runs/"
    selected = set(selected_jobs)
    jobs = set()
    for name in state_names:
        if not name.startswith(prefix) or not name.endswith("/state.json"):
            continue
        suffix = name[len(prefix):-len("/state.json")]
        if "/" in suffix or "-hmm-" in suffix:
            continue
        if not selected or suffix in selected:
            jobs.add(suffix)
    return jobs


def vast_restore_required(*, ledger: pathlib.Path, root: pathlib.Path, state_names: list[str],
                          selected_jobs: tuple[str, ...]) -> tuple[bool, str]:
    """Choose exact scoped restore versus local verified terminal reuse.

    No local checkpoint is trusted merely because its path exists.  Conversely,
    an already hash-verified terminal branch must not trigger a broad restore of
    unrelated old recovery generations before the evaluator can run.
    """
    remote_jobs = remote_base_state_jobs(state_names, selected_jobs=selected_jobs)
    if not remote_jobs:
        return False, "no selected remote base state; fresh launch"
    try:
        states = _latest_ledger_states(ledger)
    except RuntimeError:
        return True, "remote selected state exists but local ledger is unavailable"
    for job_id in sorted(remote_jobs):
        state = states.get(job_id)
        if state is None:
            return True, f"remote selected state {job_id} has no local ledger entry"
        verified, reason = terminal_base_is_locally_complete(state, root)
        if not verified:
            return True, f"{job_id} requires exact scoped restore: {reason}"
    return False, "every remote selected terminal base state is locally hash-verified"


def verified_local_active_checkpoint(durability, ledger: pathlib.Path,
                                     selected_jobs: tuple[str, ...]) -> dict[str, object] | None:
    """Prefer one newer verified Vast-local recovery generation over an older remote one.

    Supervisor restarts occur inside the same preserved Vast container. Downloading an older GCS
    pointer over a newer deeply verified local checkpoint creates needless replay. A recycled
    container has no local candidate and therefore still takes the exact scoped restore path.
    """
    latest = _latest_ledger_states(ledger)
    candidates: list[dict[str, object]] = []
    for job_id in selected_jobs:
        entry = latest.get(job_id)
        if not isinstance(entry, dict) or entry.get("status") in {
            "PENDING", "TRAINED", "DONE",
        }:
            continue
        raw_out_root = entry.get("out_root")
        if not isinstance(raw_out_root, str):
            continue
        out_root = pathlib.Path(raw_out_root)
        pointer_targets = durability._pointer_targets(out_root)
        if not pointer_targets:
            continue
        checkpoint = pointer_targets[0][1]
        # Generic current-source sidecars bind path/hash/size/step but predate the optional
        # run_id field. The same deep verification used by normal generic sync is sufficient.
        durability._verify_checkpoint_for_sync(checkpoint)
        match = re.search(r"_iter_(\d+)\.pt$", checkpoint.name)
        if match is None:
            raise RuntimeError(f"verified local checkpoint lacks exact step: {checkpoint}")
        step = int(match.group(1))
        ledger_step = int(entry.get("step", -1))
        if not (0 < step < BASE_TARGET_STEP) or step < ledger_step:
            continue
        candidates.append({
            "job_id": job_id,
            "step": step,
            "ledger_step": ledger_step,
            "checkpoint": str(checkpoint),
        })
    if len(candidates) > 1:
        raise RuntimeError("Vast worker has multiple active local base checkpoints")
    return candidates[0] if candidates else None


def _identity_mismatch_for_trained_base(states: dict[str, dict], job_ids: tuple[str, ...],
                                        project: pathlib.Path) -> str | None:
    """Check frozen evaluator identity only; do not invoke or parse an evaluator result."""
    trained = [
        state for job_id, state in states.items()
        if (not job_ids or job_id in job_ids) and state.get("status") == "TRAINED" and
        state.get("phase") == "base" and state.get("step") == BASE_TARGET_STEP
    ]
    if not trained:
        return None
    project_text = str(project)
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    try:
        from scripts.evaluate_trained_bases import LifecycleError, verify_parent_inputs
        from scripts.run_matrix import competence_identity_from_paths
        evaluator = project / "scripts" / "evaluate_base_competence.py"
        dataset = ROOT / "data" / "stargraph" / "graph_5_5_test_20000.txt"
        # The absolute path itself is frozen in the pre-training competence identity.
        # Runtime copies under ROOT/manifests have identical bytes, but are deliberately
        # not interchangeable with the source-snapshot path used at training time.
        manifests = [project / "manifests" / "corpus.sha256"]
        requested = competence_identity_from_paths(evaluator, dataset, manifests)
        for parent in trained:
            job_id = str(parent["job_id"])
            try:
                verify_parent_inputs(parent, job_id)
            except LifecycleError as exc:
                return f"{job_id} trained-base provenance is invalid: {exc}"
            if parent.get("competence_identity") != requested:
                return (
                    f"{job_id} evaluator/dataset/manifest identity differs from the "
                    "identity frozen before training"
                )
    except (RuntimeError, OSError, ValueError) as exc:
        # These files are part of the frozen scientific identity, so absent/bad hashes are
        # quarantined rather than treated as a provider disconnect.
        return f"frozen evaluator identity preflight failed: {exc}"
    return None


def classify_vast_failure(*, ledger: pathlib.Path, project: pathlib.Path, job_ids: tuple[str, ...],
                          returncode: int, exception: BaseException | None = None,
                          identity_checker=_identity_mismatch_for_trained_base) -> VastFailureDisposition:
    """Permit supervisor restart only for incomplete training or transport failure.

    A base stuck at ``TRAINED``/20,000 is no longer a training problem.  In
    particular, an evaluator identity/schema/hash refusal must stop the Vast
    supervisor before it can repeat the already-paid base cell.
    """
    detail = "" if exception is None else f"{type(exception).__name__}: {exception}"
    transport_tokens = ("timeout", "timed out", "connection", "network", "transport", "gcs", "upload", "download", "503", "429")
    if detail and any(token in detail.lower() for token in transport_tokens):
        return VastFailureDisposition(True, "TRANSIENT_TRANSPORT", detail)
    try:
        states = _latest_ledger_states(ledger)
    except RuntimeError as exc:
        # A locally unreadable ledger is not proof of scientific failure.  It is retryable
        # because the next supervised process can restore its last durable GCS generation.
        return VastFailureDisposition(True, "TRANSIENT_OR_UNREADABLE_LEDGER", str(exc))
    mismatch = identity_checker(states, job_ids, project)
    if mismatch is not None:
        return VastFailureDisposition(False, "SCIENTIFIC_IDENTITY_OR_SCHEMA_MISMATCH", mismatch)
    selected = [state for job_id, state in states.items() if not job_ids or job_id in job_ids]
    incomplete = [
        state for state in selected
        if state.get("status") in {"PENDING", "RUNNING", "INTERRUPTED", "FAILED"} and
        (not isinstance(state.get("step"), int) or state.get("step", 0) < BASE_TARGET_STEP)
    ]
    if incomplete:
        return VastFailureDisposition(
            True, "INCOMPLETE_TRAINING",
            "incomplete base training cells: " + ",".join(sorted(str(state.get("job_id")) for state in incomplete)),
        )
    if returncode == 0:
        return VastFailureDisposition(False, "SUCCESS", "base stage returned success")
    return VastFailureDisposition(
        False, "POST_TRAINING_OR_UNKNOWN_FAILURE",
        detail or "nonzero stage return after no incomplete training cell; quarantined for manual review",
    )


def write_vast_quarantine(root: pathlib.Path, disposition: VastFailureDisposition, *, ledger: pathlib.Path,
                          returncode: int) -> pathlib.Path:
    """Atomically retain a non-outcome failure receipt under an isolated Vast namespace."""
    record = {
        "schema": VAST_QUARANTINE_SCHEMA,
        "status": "QUARANTINED_NO_AUTORESTART",
        "provider": "vast.ai",
        "instance_id": INSTANCE_ID,
        "ledger_path": str(ledger),
        "ledger_sha256": sha256_file(ledger) if ledger.is_file() else None,
        "returncode": int(returncode),
        "failure": {"kind": disposition.kind, "reason": disposition.reason},
        "scientific_outcomes_opened": False,
    }
    payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    token = hashlib.sha256(payload).hexdigest()[:20]
    target = root / "quarantine" / "vast_scientific_integrity" / f"{token}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError("Vast quarantine receipt collision")
    else:
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(payload)
        os.replace(partial, target)
    return target


def publish_vast_quarantine(durability, receipt: pathlib.Path) -> None:
    """Upload a failure receipt, not a retry command or a scientific output."""
    # RuntimeDurability deliberately limits normal uploads to ROOT/runs.  This is a
    # provider-control receipt, so publish it under a separate prefix with the same
    # hash/size verification instead of weakening that run-artifact boundary.
    remote = f"{PREFIX}/quarantine/vast/{INSTANCE_ID}/{receipt.name}"
    digest, size = sha256_file(receipt), receipt.stat().st_size
    blob = durability.bucket.blob(remote)
    blob.metadata = dict(blob.metadata or {}, sha256=digest)
    blob.upload_from_filename(str(receipt))
    if sha256_file(receipt) != digest or receipt.stat().st_size != size:
        raise RuntimeError("Vast quarantine receipt changed during upload")
    blob.reload()
    if int(blob.size or -1) != size or (blob.metadata or {}).get("sha256") != digest:
        raise RuntimeError("Vast quarantine receipt remote hash/size verification failed")


class JobScopedBucket:
    """Expose only this worker's run namespaces during restore enumeration.

    Blob reads/writes remain canonical.  Filtering the broad ``lurestar/runs/`` listing is
    what prevents a second worker from restoring and then republishing another worker's live
    checkpoint state.
    """

    def __init__(self, bucket, allowed_jobs: tuple[str, ...]):
        self._bucket = bucket
        self._allowed_jobs = frozenset(allowed_jobs)
        self.name = bucket.name

    def blob(self, *args, **kwargs):
        return self._bucket.blob(*args, **kwargs)

    def list_blobs(self, *args, **kwargs):
        prefix = kwargs.get("prefix")
        if prefix is None and args:
            prefix = args[0]
        blobs = self._bucket.list_blobs(*args, **kwargs)
        if prefix != f"{PREFIX}/runs/":
            return blobs
        allowed_prefixes = tuple(f"{PREFIX}/runs/{job}/" for job in self._allowed_jobs)
        return (blob for blob in blobs if blob.name.startswith(allowed_prefixes))


class VastUploadBlob:
    """Provider-local upload policy wrapper; delegates all GCS identity checks unchanged."""

    def __init__(self, blob) -> None:
        object.__setattr__(self, "_blob", blob)
        # Setting chunk_size causes large filename uploads to use the resumable protocol.
        # Do not add an adapter retry loop: a bounded GCS attempt failure remains visible to
        # RuntimeDurability, which aborts safely and lets Supervisor make one later retry.
        try:
            blob.chunk_size = VAST_UPLOAD_CHUNK_BYTES
        except (AttributeError, ValueError):
            # Lightweight test doubles or old clients may not expose chunk_size.  Production
            # GCS Blob does; timeout delegation and post-upload hash checks still apply.
            pass

    @property
    def metadata(self):
        return self._blob.metadata

    @metadata.setter
    def metadata(self, value) -> None:
        self._blob.metadata = value

    def upload_from_filename(self, filename, *args, **kwargs):
        # Supervisor restarts reconstruct RuntimeDurability's in-memory upload cache.  If
        # the prior process already committed these exact bytes, do not spend minutes
        # re-uploading them while the GPU sits idle.  The caller still reloads and checks
        # remote size/hash after this method returns, so this is only an idempotent fast
        # path; it does not weaken the durable-state verification contract.
        if kwargs.get("if_generation_match") is None:
            local = pathlib.Path(filename)
            intended_metadata = dict(self._blob.metadata or {})
            if local.is_file():
                local_size = local.stat().st_size
                local_sha256 = hashlib.sha256(local.read_bytes()).hexdigest()
                try:
                    self._blob.reload()
                except Exception as exc:
                    if exc.__class__.__name__ != "NotFound":
                        raise
                    self._blob.metadata = intended_metadata
                else:
                    remote_sha256 = (self._blob.metadata or {}).get("sha256")
                    if int(self._blob.size or -1) == local_size and remote_sha256 == local_sha256:
                        print(
                            "[vast-upload] exact remote object already present; reusing "
                            + str(getattr(self._blob, "name", "object")),
                            flush=True,
                        )
                        return None
                    self._blob.metadata = intended_metadata
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = VAST_UPLOAD_TIMEOUT
        # Do not inherit the storage client's opaque 120-second aggregate retry budget.
        # The resumable chunk protocol plus this bounded 15-minute request is one Vast
        # attempt; a transport failure is surfaced to RuntimeDurability/Supervisor rather
        # than retried indefinitely or turned into a second training launch here.
        if "retry" not in kwargs:
            kwargs["retry"] = None
        return self._blob.upload_from_filename(filename, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._blob, name)


class VastUploadBucket:
    """Keep JobScopedBucket filtering while giving only Vast uploads a safe transport policy."""

    def __init__(self, bucket) -> None:
        self._bucket = bucket
        self.name = bucket.name

    def blob(self, *args, **kwargs):
        return VastUploadBlob(self._bucket.blob(*args, **kwargs))

    def list_blobs(self, *args, **kwargs):
        return self._bucket.list_blobs(*args, **kwargs)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive) as payload:
        members = payload.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsafe source member: {member.name}")
            if target != root and root not in target.parents:
                raise RuntimeError(f"source path traversal: {member.name}")
        payload.extractall(destination, members=members)


def download_blob(bucket, name: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    bucket.blob(name).download_to_filename(str(partial))
    os.replace(partial, destination)


def download_prefix(bucket, remote_prefix: str, destination: pathlib.Path) -> int:
    count = 0
    prefix = remote_prefix.rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=prefix):
        relative = pathlib.PurePosixPath(blob.name[len(prefix):])
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            raise RuntimeError(f"unsafe GCS path: {blob.name}")
        target = destination.joinpath(*relative.parts)
        download_blob(bucket, blob.name, target)
        count += 1
    return count


def install_project(bucket) -> None:
    CONTENT.mkdir(parents=True, exist_ok=True)
    marker = PROJECT / ".vast_source_sha256"
    if PROJECT.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != SOURCE_SHA256:
            raise RuntimeError("/content/project exists without the exact frozen-source marker")
    else:
        download_blob(bucket, SOURCE_OBJECT, SOURCE_ARCHIVE)
        if sha256_file(SOURCE_ARCHIVE) != SOURCE_SHA256:
            raise RuntimeError("immutable project archive hash mismatch")
        temporary = pathlib.Path(tempfile.mkdtemp(prefix="project-extract-", dir=CONTENT))
        temporary.rmdir()
        safe_extract(SOURCE_ARCHIVE, temporary)
        marker_tmp = temporary / ".vast_source_sha256"
        marker_tmp.write_text(SOURCE_SHA256 + "\n", encoding="utf-8")
        os.replace(temporary, PROJECT)

    inventory = PROJECT / "manifests" / "manifest_inventory.sha256"
    if not inventory.is_file() or sha256_file(inventory) != INPUT_SHA256:
        raise RuntimeError("frozen source/input inventory binding mismatch")


def install_upstream_and_dependencies() -> None:
    if not (UPSTREAM / ".git").is_dir():
        UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", UPSTREAM_URL, str(UPSTREAM)])
    run(["git", "checkout", "-q", PINNED_UPSTREAM], cwd=UPSTREAM)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    if head != PINNED_UPSTREAM:
        raise RuntimeError("pinned upstream checkout mismatch")

    receipt = UPSTREAM / ".lurestar_runtime_patch_receipt.json"
    if not receipt.is_file():
        filtered = CONTENT / "nextlat-requirements-no-torch.txt"
        requirements = (UPSTREAM / "requirements.txt").read_text(encoding="utf-8").splitlines()
        filtered.write_text(
            "\n".join(line for line in requirements if not line.strip().startswith("torch")) + "\n",
            encoding="utf-8",
        )
        run(["uv", "pip", "install", "-r", str(filtered)])
        run([
            sys.executable,
            str(PROJECT / "scripts" / "runtime_bootstrap.py"),
            "--project-root", str(PROJECT),
            "--upstream", str(UPSTREAM),
        ], cwd=PROJECT)


def apply_vast_operational_overrides() -> str:
    """Use a provider-sized cadence so one immutable generation can reach GCS.

    GPT/NextLat produce a 1,000-step checkpoint every few minutes on a 3090, faster than this
    host can upload and verify it before upstream replaces the pointer and deletes the preceding
    generation.  BST takes roughly 90 minutes per 1,000 steps, so its existing cadence is useful.
    The optimizer budget, data, model, precision and scientific evaluator remain untouched.  This
    provider-only source delta is hashed into every runtime receipt.
    """
    runner = PROJECT / "scripts" / "run_matrix.py"
    originals = (
        '"trainer.save_recovery_checkpoint=250",',
        '"trainer.save_recovery_checkpoint=1000",',
    )
    replacement = (
        '"trainer.save_recovery_checkpoint=%d" % ('
        f'{VAST_FAST_RECOVERY_STEPS} if spec.model in {{"gpt", "nextlat"}} '
        f'else {VAST_BST_RECOVERY_STEPS}),'
    )
    source = runner.read_text(encoding="utf-8")
    matches = [original for original in originals if original in source]
    if matches:
        if len(matches) != 1 or source.count(matches[0]) != 1:
            raise RuntimeError("ambiguous recovery-cadence override target")
        runner.write_text(source.replace(matches[0], replacement), encoding="utf-8")
    elif replacement not in source:
        raise RuntimeError("recovery-cadence override target is absent")
    return sha256_file(runner)


def vast_background_sync_loop(stop_event, durability, ledger, ledger_object, *,
                              interval=VAST_BACKGROUND_SYNC_SECONDS):
    """Best-effort Vast sync that can never terminate paid training.

    A moving checkpoint pointer is expected while training is live.  Colab treated three sync
    races as evidence that an ephemeral runtime was unsafe and killed its subprocess.  Vast
    stop/start preserves the container filesystem, so that policy only creates replay and long
    bootstrap stalls.  Here a failed snapshot is logged and the next interval captures the newest
    stable generation.  The synchronous per-cell/final sync remains mandatory before advancing.
    """
    consecutive = 0
    while not stop_event.wait(interval):
        try:
            states = durability.sync_once(ledger, ledger_object=ledger_object)
        except Exception as exc:
            consecutive += 1
            print(
                "VAST_BACKGROUND_SYNC_DEFERRED=" + json.dumps({
                    "consecutive": consecutive,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "training_continues": True,
                }, sort_keys=True),
                flush=True,
            )
        else:
            if consecutive:
                print(
                    "VAST_BACKGROUND_SYNC_RECOVERED=" + json.dumps({
                        "prior_consecutive": consecutive,
                        "states": sorted(states),
                    }, sort_keys=True),
                    flush=True,
                )
            consecutive = 0
    return None


def stage_inputs(bucket) -> None:
    manifest_dir = ROOT / "manifests"
    data_dir = ROOT / "data" / "stargraph"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if download_prefix(bucket, f"{INPUT_PREFIX}/manifests", manifest_dir) == 0:
        raise RuntimeError("frozen manifest bundle is absent")
    inventory = manifest_dir / "manifest_inventory.sha256"
    if not inventory.is_file() or sha256_file(inventory) != INPUT_SHA256:
        raise RuntimeError("staged manifest inventory mismatch")
    required = (
        "graph_5_5_sample_200000.txt",
        "graph_5_5_test_20000.txt",
    )
    if any(not (data_dir / name).is_file() for name in required):
        if download_prefix(bucket, f"{PREFIX}/corpus/stargraph", data_dir) == 0:
            raise RuntimeError("frozen StarGraph corpus is absent")
    if any(not (data_dir / name).is_file() for name in required):
        raise RuntimeError("required StarGraph train/test corpus is incomplete")


def runtime_fingerprint(provider_runner_sha256: str) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Vast CUDA device is unavailable")
    return {
        "provider": "vast.ai",
        "instance_id": INSTANCE_ID,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "pinned_upstream_commit": PINNED_UPSTREAM,
        "scientific_source_sha256": SOURCE_SHA256,
        "provider_runner_sha256": provider_runner_sha256,
        "operational_recovery_interval_steps": {
            "gpt": VAST_FAST_RECOVERY_STEPS,
            "nextlat": VAST_FAST_RECOVERY_STEPS,
            "bst": VAST_BST_RECOVERY_STEPS,
        },
        "job_allowlist": list(ONLY_JOBS),
    }


def vast_base_stage_paths() -> tuple[str, str]:
    """Return the deliberately distinct corpus and frozen-manifest locations."""
    return str(ROOT / "data" / "stargraph"), str(PROJECT / "manifests")


def main() -> int:
    if not ADC.is_file():
        raise RuntimeError("authorized GCS credential is absent")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ADC)
    os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage

    bucket = storage.Client(project=GCP_PROJECT).bucket(BUCKET)
    print("=== installing frozen runtime ===", flush=True)
    install_project(bucket)
    install_upstream_and_dependencies()
    provider_runner_sha256 = apply_vast_operational_overrides()
    stage_inputs(bucket)

    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "src"))
    from scripts import colab_train_loop as driver

    fingerprint = runtime_fingerprint(provider_runner_sha256)
    print("VAST_RUNTIME_FINGERPRINT=" + json.dumps(fingerprint, sort_keys=True), flush=True)
    scoped_bucket = JobScopedBucket(bucket, ONLY_JOBS) if ONLY_JOBS else bucket
    # This wrapper changes only Vast upload request transport (resumable chunk + bounded
    # deadline); JobScopedBucket isolation and RuntimeDurability hash verification remain.
    durability_bucket = VastUploadBucket(scoped_bucket)
    durability = driver.RuntimeDurability(
        durability_bucket,
        str(ROOT),
        PREFIX,
        source_sha256=SOURCE_SHA256,
        runtime_fingerprint=fingerprint,
    )

    # Download this worker's ledger *before* considering restore.  A valid local terminal
    # checkpoint plus its hash-bound ledger entry is sufficient to proceed to evaluation; a
    # path-only recovery-pointer heuristic previously redownloaded old GPT generations.
    ledger = ROOT / LEDGER_OBJECT
    durability.download_file(f"{PREFIX}/{LEDGER_OBJECT}", str(ledger), required=False)

    # HMM state shares the bucket namespace but is not an input to this base worker.  The
    # JobScopedBucket additionally prevents this worker from enumerating another worker's base
    # state when ONLY_JOBS is set.
    state_names = [
        blob.name
        for blob in durability_bucket.list_blobs(prefix=f"{PREFIX}/runs/")
        if blob.name.endswith("/state.json")
    ]
    restore_needed, restore_reason = vast_restore_required(
        ledger=ledger, root=ROOT, state_names=state_names, selected_jobs=ONLY_JOBS,
    )
    local_active = verified_local_active_checkpoint(durability, ledger, ONLY_JOBS)
    if restore_needed and local_active is not None:
        print(
            "=== no broad base restore: newer verified Vast-local active checkpoint "
            + json.dumps(local_active, sort_keys=True) + " ===",
            flush=True,
        )
    elif restore_needed:
        print(f"=== exact scoped base restore: {restore_reason} ===", flush=True)
        durability.restore()
    else:
        print(f"=== no broad base restore: {restore_reason} ===", flush=True)

    stop = threading.Event()
    sync_thread = threading.Thread(
        target=vast_background_sync_loop,
        args=(stop, durability, str(ledger), LEDGER_OBJECT),
        kwargs={"interval": VAST_BACKGROUND_SYNC_SECONDS},
        daemon=True,
    )
    sync_thread.start()

    args = [
        "--phase", "base",
        "--precision", "bf16-mixed",
        "--devices", "1",
        "--strategy", "ddp",
    ]
    if ONLY_JOBS:
        args.extend(("--only", *ONLY_JOBS))
    started = time.time()
    base_data_dir, competence_manifest_dir = vast_base_stage_paths()
    stage_exception: Exception | None = None
    rc = 70
    try:
        rc = driver.run_lurestar_base_stages(
            project=str(PROJECT),
            root=str(ROOT),
            ledger=str(ledger),
            upstream=str(UPSTREAM),
            args=args,
            data_dir=base_data_dir,
            # Base training froze this source-snapshot path in competence_identity.
            # ROOT/manifests remains the staged runtime input surface, but passing it to
            # the evaluator creates a deterministic absolute-path identity mismatch.
            manifest_dir=competence_manifest_dir,
            durability=durability,
            abort_event=None,
        )
    except Exception as exc:
        # The provider adapter owns this boundary.  Do not borrow Colab's re-exec
        # behavior: the disposition below decides whether supervisor may restart.
        stage_exception = exc
        print(f"VAST_STAGE_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
    finally:
        stop.set()
        sync_thread.join(timeout=10)

    try:
        states = durability.sync_once(str(ledger), ledger_object=LEDGER_OBJECT)
    except Exception as exc:
        states = {}
        if stage_exception is None:
            stage_exception = exc
        rc = 75
        print(f"VAST_FINAL_SYNC_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
    stage_rc = int(rc)
    disposition = classify_vast_failure(
        ledger=ledger, project=PROJECT, job_ids=ONLY_JOBS,
        returncode=stage_rc, exception=stage_exception,
    )
    if stage_rc != 0:
        print("VAST_FAILURE_DISPOSITION=" + json.dumps({
            "schema": VAST_FAILURE_SCHEMA, "retry": disposition.retry,
            "kind": disposition.kind, "reason": disposition.reason,
        }, sort_keys=True), flush=True)
    clean_quarantine_exit = False
    quarantine_path = None
    if stage_rc != 0 and not disposition.retry:
        quarantine_path = write_vast_quarantine(
            ROOT, disposition, ledger=ledger, returncode=stage_rc
        )
        try:
            publish_vast_quarantine(durability, quarantine_path)
        except Exception as exc:
            # Failure to publish the only durable diagnosis is a transport failure;
            # leave a nonzero exit for supervisor to retry the upload path.
            disposition = VastFailureDisposition(True, "TRANSIENT_QUARANTINE_UPLOAD", str(exc))
            print("VAST_FAILURE_DISPOSITION=" + json.dumps({
                "schema": VAST_FAILURE_SCHEMA, "retry": True,
                "kind": disposition.kind, "reason": disposition.reason,
            }, sort_keys=True), flush=True)
        else:
            clean_quarantine_exit = True
            rc = 0
    complete = stage_rc == 0
    durability.publish_terminal(
        f"vast-{INSTANCE_ID}-base",
        returncode=rc,
        training_complete=complete,
    )
    print(
        "VAST_BASE_MATRIX_DONE=" + json.dumps({
            "returncode": rc,
            "stage_returncode": stage_rc,
            "complete": complete,
            "quarantined": clean_quarantine_exit,
            "quarantine_path": str(quarantine_path) if quarantine_path else None,
            "elapsed_seconds": time.time() - started,
            "synced_states": sorted(states),
        }, sort_keys=True),
        flush=True,
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
