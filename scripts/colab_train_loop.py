#!/usr/bin/env python
"""Dual-role resumable driver for the confirmatory sweep on Colab.

Role is auto-detected by whether `/content` exists.

LOOP role (runs on the Mac)
    Package the project, push it to GCS, start a runtime, upload the credential, exec this
    same file on the runtime, and -- when the runtime drops, which it will -- start a fresh
    one and exec again. State lives in GCS, so a re-exec resumes rather than restarts. The
    loop ends when the ledger reports every job DONE.

DRIVER role (runs on the Colab GPU runtime)
    Pull the project, the pinned upstream repo and the immutable corpus from GCS, then hand
    control to scripts/run_matrix.py, which owns idempotency and checkpoint lineage. A
    background thread heartbeats and syncs checkpoints and metrics to GCS on a cadence, so a
    drop costs at most one sync interval rather than the whole run.

Three constraints from earlier failures in this project shape the design, and all three are
easy to reintroduce by accident:
  * `colab exec file.py -- args` does NOT forward argv. Every parameter arrives through the
    uploaded sidecar /content/job_spec.json, never the command line.
  * `__file__` is undefined under `colab exec`, and in DRIVER role the module executes top to
    bottom before any main() dispatch, so a top-level `__file__` reference crashes before the
    role is even chosen.
  * A child process's stdout does not reach the exec stream, and a silent long job starves the
    websocket. Child output is relayed line by line.
"""
import collections
import contextlib
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import queue
import re
import secrets
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # colab exec: no __file__ in the kernel
    HERE = os.getcwd()

BUCKET = "nextlat-lurestar-project-flash-490419"
GCP_PROJECT = "project-flash-490419"
PREFIX = "lurestar"
GCS = "gs://%s/%s" % (BUCKET, PREFIX)
PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"
SPEC_PATH = "/content/job_spec.json"
SYNC_SECONDS = 60
HEARTBEAT_SECONDS = 30
FAST_EXIT_SECONDS = 120
MAX_FAST_EXITS = 2
MAX_STALLED_SYNC_WINDOWS = 25
STATE_SCHEMA = "nextlat_forgetting/colab_state/1"
CONFIRMATORY_CLEARANCE_SCHEMA = "nextlat_forgetting/confirmatory_clearance/1"
# Gate 11 and the launch clearance deliberately consume the same receipts.  Keeping
# separate schema names here previously made it possible to attest one pair of files
# during preregistration and launch against another pair later.
FULL_TEST_SUITE_SCHEMA = "nextlat_forgetting/full_test_suite_receipt/1"
INDEPENDENT_SCIENTIFIC_REVIEW_SCHEMA = (
    "nextlat_forgetting/independent_scientific_review/1"
)
CONFIRMATORY_TEST_SCHEMA = FULL_TEST_SUITE_SCHEMA
CONFIRMATORY_REVIEW_SCHEMA = INDEPENDENT_SCIENTIFIC_REVIEW_SCHEMA
PREREGISTRATION_FREEZE_SCHEMA = "nextlat_forgetting/preregistration_freeze_receipt/1"
PREREGISTRATION_EVIDENCE_SCHEMA = "nextlat_forgetting/preregistration_evidence/1"
INPUT_BUNDLE_UPLOAD_SCHEMA = "nextlat_forgetting/input_bundle_upload/1"
INPUT_BUNDLE_RECEIPT_PATH = ".agent_state/input-bundle-upload.json"
CONFIRMATORY_PROTOCOL_PATHS = (
    "nextlat_v4_predictive_geometry_spec.md",
    "docs/PREREGISTRATION_AMENDMENT_2026-08-24.md",
    "docs/EVAL-REVIEW.md",
    "docs/FOUNDATIONS.md",
    "docs/EXTRACTION.md",
    "docs/DECISION_D41_RUNTIME_RECOVERY_AMENDMENT.md",
)
MIN_HARD_STOP_BALANCE_CU = 1188.61
MAX_CONFIRMATORY_ATTEMPTS = 20
FROZEN_BASE_MODELS = ("gpt", "nextlat", "bst")
FROZEN_BASE_SEEDS = (1234, 1235, 1236, 1237, 1238)
FROZEN_BASE_JOB_IDS = tuple(
    "%s-s%d-base" % (model, seed)
    for model in FROZEN_BASE_MODELS
    for seed in FROZEN_BASE_SEEDS
)
D41_RECOVERY_RECEIPT_PATH = ".agent_state/d41-exact-ten-recovery-receipt.json"
D41_RECOVERY_RECEIPT_SCHEMA = "nextlat_forgetting/d41_exact_ten_recovery/2"
D41_RECOVERY_JOB_IDS = tuple(
    "%s-seed%d-hmm-persistent_moderate" % (model, seed)
    for model in ("gpt", "nextlat") for seed in FROZEN_BASE_SEEDS
)
D43_CONTINUATION_GATE = "d43_measurement_amendment"
D43_CONTINUATION_SCHEMA = "nextlat_forgetting/d43_measurement_amendment/1"
D43_RECEIPT_PATH = ".agent_state/d43-measurement-amendment-receipt.json"
D43_PREDECESSOR_ARCHIVE_PATH = ".agent_state/project-predecessor-d41.tar.gz"
D43_D41_BASELINE_ARCHIVE_PATH = ".agent_state/project-d41-operational-baseline.tar.gz"
D43_SUCCESSOR_ARCHIVE_PATH = ".agent_state/project.tar.gz"
D43_LAUNCH_BINDING_SCHEMA = "nextlat_forgetting/d43_launch_continuation/1"
MAX_CONSECUTIVE_SYNC_FAILURES = 3
SYNC_FAILURE_DIAGNOSTIC = "sync_failure_circuit_breaker.json"
OWNERSHIP_UNCERTAIN_DIAGNOSTIC = ".agent_state/colab-ownership-uncertain.json"
_COMMON_JOB_SPEC_KEYS = {
    "gpu", "max_attempts", "hard_stop_balance_cu", "confirmatory_clearance_path",
    "predecessor_source_sha256", "recovery_job_ids", "recovery_receipt_sha256",
    "continuation_gate", "continuation_gate_schema", "continuation_receipt_sha256",
}
_RUNTIME_JOB_SPEC_KEYS = {
    "source_sha256", "source_object", "input_bundle_sha256", "input_bundle_prefix",
    "recovery_receipt_object", "recovery_receipt_generation",
}


def sh(cmd, check=True, timeout=None, cwd=None, quiet=False, silent=False, max_lines=200,
       abort_event=None):
    """Run one host command and return its output.

    Human-facing command output is bounded to the most recent 200 lines by default.  Callers
    consuming structured JSON must pass ``max_lines=None`` so a large document can never be
    silently truncated before parsing.
    """
    print("+ " + cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            executable="/bin/bash", start_new_session=abort_event is not None)
    tail = collections.deque(maxlen=max_lines)
    if abort_event is None:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            if not quiet:
                print("  | " + line, flush=True)
    else:
        lines = queue.Queue()

        def read_lines():
            for value in proc.stdout:
                lines.put(value)
            lines.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        stream_done = False
        interrupted = False
        while not stream_done or proc.poll() is None:
            if abort_event.is_set() and proc.poll() is None and not interrupted:
                interrupted = True
                print("  | [controller] durable-sync circuit breaker interrupt", flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                if interrupted and proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                continue
            if line is None:
                stream_done = True
                continue
            line = line.rstrip()
            tail.append(line)
            if not quiet:
                print("  | " + line, flush=True)
        reader.join(timeout=1)
    rc = proc.wait()
    if quiet and not silent:
        for ln in tail:
            print("  | " + ln, flush=True)
    if check and rc != 0:
        raise SystemExit("FAILED (%d): %s" % (rc, cmd))
    return rc, "\n".join(tail)


def ledger_progress(document, terminal_statuses=None):
    """Return ``(training_terminal, total)`` from the append-only Ledger document.

    ``scripts.run_matrix.Ledger`` stores ``{"schema": 1, "entries": [...]}``, with a
    new entry superseding the previous entry for the same job.  Completion must therefore
    be computed from the latest entry per ``job_id`` and from ``status`` (not ``state``).
    """
    entries = document.get("entries", []) if isinstance(document, dict) else []
    if not isinstance(entries, list):
        return 0, 0
    latest = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("job_id"):
            latest[entry["job_id"]] = entry
    terminal_statuses = ({"TRAINED", "DONE"} if terminal_statuses is None
                         else {str(value) for value in terminal_statuses})
    terminal = sum(1 for entry in latest.values()
                   if str(entry.get("status")) in terminal_statuses)
    return terminal, len(latest)


def requested_phase(args):
    """Return the explicit matrix phase from the frozen sidecar argument list."""
    args = list(args or [])
    try:
        return str(args[args.index("--phase") + 1])
    except (ValueError, IndexError):
        return None


def validate_confirmatory_job_spec(spec, *, runtime_overlay=False,
                                   require_session=False):
    """Validate semantic launch scope, not merely a sidecar's content hash.

    The same guard runs at clearance issuance/recomputation, host dispatch, and
    runtime dispatch. Runtime-only content-addressing fields are accepted solely
    after the host has generated them.
    """
    if not isinstance(spec, dict):
        raise SystemExit("confirmatory job spec must be a JSON object")

    runner = spec.get("runner", "lurestar")
    if runner not in {"lurestar", "hmm"}:
        raise SystemExit("confirmatory job spec runner must be lurestar or hmm")

    allowed = set(_COMMON_JOB_SPEC_KEYS)
    if runner == "lurestar":
        allowed.update({"runner", "run_matrix_args"})
    else:
        allowed.update({"runner", "runner_phase", "family"})
    if runtime_overlay:
        allowed.update(_RUNTIME_JOB_SPEC_KEYS)
        if require_session:
            allowed.add("session_id")
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise SystemExit("confirmatory job spec has unknown fields: %s" %
                         ", ".join(str(value) for value in unknown))

    if spec.get("gpu") != "a100":
        raise SystemExit("confirmatory job spec GPU must be exactly a100")
    attempts = spec.get("max_attempts", MAX_CONFIRMATORY_ATTEMPTS)
    if (isinstance(attempts, bool) or not isinstance(attempts, int) or
            not 1 <= attempts <= MAX_CONFIRMATORY_ATTEMPTS):
        raise SystemExit("confirmatory max_attempts must be an integer from 1 through %d" %
                         MAX_CONFIRMATORY_ATTEMPTS)
    hard_stop = spec.get("hard_stop_balance_cu", MIN_HARD_STOP_BALANCE_CU)
    if (isinstance(hard_stop, bool) or not isinstance(hard_stop, (int, float)) or
            not math.isfinite(float(hard_stop)) or
            float(hard_stop) < MIN_HARD_STOP_BALANCE_CU):
        raise SystemExit("confirmatory hard-stop balance may not be below %.2f CU" %
                         MIN_HARD_STOP_BALANCE_CU)
    if ("confirmatory_clearance_path" in spec and
            spec["confirmatory_clearance_path"] !=
            ".agent_state/confirmatory-clearance.json"):
        raise SystemExit("confirmatory clearance path is not the frozen project receipt")
    predecessor_source = spec.get("predecessor_source_sha256")
    if (predecessor_source is not None and
            not re.fullmatch(r"[0-9a-f]{64}", str(predecessor_source))):
        raise SystemExit("confirmatory predecessor source hash must be exact lowercase SHA-256")
    recovery_ids = spec.get("recovery_job_ids")
    recovery_sha = spec.get("recovery_receipt_sha256")
    continuation_gate = spec.get("continuation_gate")
    continuation_schema = spec.get("continuation_gate_schema")
    continuation_sha = spec.get("continuation_receipt_sha256")
    d43_requested = any(value is not None for value in (
        continuation_gate, continuation_schema, continuation_sha))
    if d43_requested:
        if (continuation_gate != D43_CONTINUATION_GATE or
                continuation_schema != D43_CONTINUATION_SCHEMA or
                not re.fullmatch(r"[0-9a-f]{64}", str(continuation_sha))):
            raise SystemExit("D43 continuation must select the exact gate/schema/receipt hash")
    recovery_requested = any(value is not None for value in (
        predecessor_source, recovery_ids, recovery_sha))
    if d43_requested and not recovery_requested:
        raise SystemExit("D43 continuation cannot fall back to a fresh launch")
    if recovery_requested:
        if runner != "hmm" or spec.get("runner_phase") != "train":
            raise SystemExit("D41 predecessor recovery is permitted only for HMM training")
        if predecessor_source is None or tuple(recovery_ids or ()) != D41_RECOVERY_JOB_IDS:
            raise SystemExit("D41 recovery must bind the exact canonical ten recovered job ids")
        if not re.fullmatch(r"[0-9a-f]{64}", str(recovery_sha)):
            raise SystemExit("D41 recovery receipt hash must be exact lowercase SHA-256")

    if runner == "lurestar":
        args = spec.get("run_matrix_args")
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            raise SystemExit("Lure-Star run_matrix_args must be a JSON string list")
        lowered = [value.lower() for value in args]
        if any("adapt" in value or "h3" in value for value in lowered):
            raise SystemExit("Lure-Star confirmatory scope excludes H3 and adaptation")
        if args[:2] != ["--phase", "base"]:
            raise SystemExit("Lure-Star confirmatory scope must be explicit base-only")
        only = []
        if len(args) > 2:
            if len(args) < 4 or args[2] != "--only":
                raise SystemExit("Lure-Star base scope permits only the optional --only IDs")
            only = args[3:]
            if (not only or len(set(only)) != len(only) or
                    any(job_id not in FROZEN_BASE_JOB_IDS for job_id in only)):
                raise SystemExit("Lure-Star --only must contain unique frozen base job IDs")
        result = {"runner": runner, "phase": "base", "only": tuple(only)}
    else:
        if spec.get("runner_phase") not in {"train", "evaluate"}:
            raise SystemExit("HMM runner_phase must be exactly train or evaluate")
        if spec.get("family") is not True:
            raise SystemExit("HMM confirmatory dispatch requires the complete frozen family")
        result = {"runner": runner, "phase": spec["runner_phase"], "family": True}

    if runtime_overlay:
        source_sha = spec.get("source_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source_sha)):
            raise SystemExit("runtime job spec has no valid source hash")
        if predecessor_source == source_sha:
            raise SystemExit("runtime predecessor source hash must differ from current source")
        transport_present = any(spec.get(key) is not None for key in (
            "recovery_receipt_object", "recovery_receipt_generation"))
        if recovery_requested and (require_session or transport_present):
            receipt_object = spec.get("recovery_receipt_object")
            expected_object = "%s/recovery_receipts/%s.json" % (PREFIX, recovery_sha)
            if receipt_object != expected_object:
                raise SystemExit("runtime recovery receipt object/hash mismatch")
            if not str(spec.get("recovery_receipt_generation", "")).isdigit():
                raise SystemExit("runtime recovery receipt lacks exact generation")
        expected_source = "%s/source/project-%s.tar.gz" % (PREFIX, source_sha)
        if spec.get("source_object") != expected_source:
            raise SystemExit("runtime job spec source object/hash mismatch")
        input_sha = spec.get("input_bundle_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(input_sha)):
            raise SystemExit("runtime job spec has no valid input-bundle hash")
        expected_prefix = "%s/input_bundles/%s" % (PREFIX, input_sha)
        if spec.get("input_bundle_prefix") != expected_prefix:
            raise SystemExit("runtime job spec input bundle prefix/hash mismatch")
        if require_session:
            session_id = spec.get("session_id")
            if (not isinstance(session_id, str) or
                    re.fullmatch(r"gpu-[a-z0-9-]+", session_id) is None):
                raise SystemExit("runtime job spec has no owned Colab session identity")
    elif require_session:
        raise SystemExit("session identity may only be required for a runtime overlay")

    return result


def attach_recovery_receipt_transport(project_root, runtime_spec_file, runtime_spec):
    """Publish the cleared D41 receipt immutably and bind its exact GCS generation."""
    if runtime_spec.get("predecessor_source_sha256") is None:
        return runtime_spec
    root = pathlib.Path(project_root).resolve()
    receipt = root / D41_RECOVERY_RECEIPT_PATH
    expected_sha = runtime_spec.get("recovery_receipt_sha256")
    if not receipt.is_file() or sha256_file(receipt) != expected_sha:
        raise SystemExit("cleared D41 recovery receipt is absent or hash-invalid")
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("cleared D41 recovery receipt is invalid: %s" % exc)
    if (document.get("schema") != D41_RECOVERY_RECEIPT_SCHEMA or
            document.get("status") != "PASS" or
            document.get("scientific_metrics_inspected") is not False):
        raise SystemExit("cleared D41 recovery receipt semantics changed")
    object_name = "%s/recovery_receipts/%s.json" % (PREFIX, expected_sha)
    record = publish_immutable_host_file(project_root, receipt, object_name)
    result = dict(runtime_spec)
    result["recovery_receipt_object"] = object_name
    result["recovery_receipt_generation"] = str(record["generation"])
    validate_confirmatory_job_spec(result, runtime_overlay=True)
    path = pathlib.Path(runtime_spec_file)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return result


def parse_cli_json(output):
    """Parse the JSON object emitted by a Colab/gcloud CLI command."""
    text = str(output).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("CLI output contained no JSON object")
        return json.loads(text[start:end + 1])


def colab_status_pair(delay_seconds=30, sleeper=time.sleep):
    """Two status reads; eventual-consistency disagreements are never acted on."""
    rc1, out1 = sh("colab status --json", check=False, quiet=True, silent=True,
                    max_lines=None)
    sleeper(delay_seconds)
    rc2, out2 = sh("colab status --json", check=False, quiet=True, silent=True,
                    max_lines=None)
    if rc1 or rc2:
        raise RuntimeError("colab status failed during lifecycle check")
    return parse_cli_json(out1), parse_cli_json(out2)


def colab_quota_pair(delay_seconds=30, sleeper=time.sleep):
    """Two full quota reads; a paid launch may use neither observation alone."""
    rc1, out1 = sh("colab quota --json", check=False, quiet=True, silent=True,
                    max_lines=None)
    sleeper(delay_seconds)
    rc2, out2 = sh("colab quota --json", check=False, quiet=True, silent=True,
                    max_lines=None)
    if rc1 or rc2:
        raise RuntimeError("colab quota failed during pre-provision check")
    return parse_cli_json(out1), parse_cli_json(out2)


def agreed_paid_balance(first, second):
    """Return the balance only when two quota documents agree on launch-critical state."""
    try:
        first_balance = float(first["paid_balance"])
        second_balance = float(second["paid_balance"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not math.isfinite(first_balance) or not math.isfinite(second_balance) or
            first_balance != second_balance):
        return None
    # These fields are not emitted by every CLI version. When present, however, a
    # disagreement means quota settlement or runtime lifecycle is still in flight.
    for key in ("active_runtimes", "burn_rate_hourly"):
        if key in first or key in second:
            if key not in first or key not in second or first[key] != second[key]:
                return None
    return first_balance


def agreed_runtime_state(first, second):
    """Return `gone`, `active`, or `uncertain` from two status documents."""
    first_gone = first.get("status") == "no_runtime"
    second_gone = second.get("status") == "no_runtime"
    if first_gone and second_gone:
        return "gone"
    if not first_gone and not second_gone:
        return "active"
    return "uncertain"


def authorize_provisioning(hard_floor, *, quota_reader=colab_quota_pair,
                           status_reader=colab_status_pair):
    """Authorize exactly one start after fresh paired quota and status observations.

    This helper belongs immediately beside ``colab start``. Calling it once before a
    retry loop is insufficient because a prior attempt can consume balance or leave a
    runtime whose lifecycle has not settled.
    """
    quota_first, quota_second = quota_reader()
    balance = agreed_paid_balance(quota_first, quota_second)
    if balance is None:
        raise SystemExit("Colab quota reads disagree or are invalid; refusing to provision")
    floor = float(hard_floor)
    print("COLAB_BALANCE_CU=%.6f HARD_STOP_FLOOR_CU=%.2f" %
          (balance, floor), flush=True)
    if balance <= floor:
        raise SystemExit("project compute hard stop reached; refusing to provision")

    status_first, status_second = status_reader()
    if agreed_runtime_state(status_first, status_second) != "gone":
        raise SystemExit(
            "fresh Colab status reads do not agree that no runtime exists; "
            "refusing to provision"
        )
    return balance


class OwnershipUncertain(RuntimeError):
    """Paired status reads cannot prove the active runtime is the session we own."""

    def __init__(self, expected, reported):
        super().__init__("active Colab runtime ownership is uncertain")
        self.expected = str(expected)
        self.reported = list(reported)


def require_owned_session(first, second, expected_session_id):
    """Require both active reads to explicitly name the same expected session."""
    reported = [
        document.get("session") or document.get("session_id")
        for document in (first, second)
    ]
    normalized = [str(value) if value is not None and str(value) else None
                  for value in reported]
    expected = str(expected_session_id)
    if normalized != [expected, expected]:
        raise OwnershipUncertain(expected, normalized)
    return expected


def write_ownership_uncertain_diagnostic(project_root, *, expected_session_id,
                                         status_first, status_second, stage):
    """Atomically persist a read-only handoff; this function performs no Colab mutation."""
    document = {
        "schema": "nextlat_forgetting/colab_ownership_uncertain/1",
        "status": "OWNERSHIP_UNCERTAIN_READ_ONLY",
        "stage": str(stage),
        "expected_session_id": str(expected_session_id),
        "status_reads": [status_first, status_second],
        "allowed_actions": ["manual_reconcile_session_identity"],
        "forbidden_actions": ["stop", "start", "exec", "upload"],
        "recorded_at_unix": time.time(),
    }
    path = pathlib.Path(project_root) / OWNERSHIP_UNCERTAIN_DIAGNOSTIC
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path


def latest_ledger_entries(document):
    """Fold an append-only Ledger document into its latest entry per job."""
    entries = document.get("entries", []) if isinstance(document, dict) else []
    latest = {}
    if not isinstance(entries, list):
        return latest
    for entry in entries:
        if isinstance(entry, dict) and entry.get("job_id"):
            latest[entry["job_id"]] = entry
    return latest


def state_required_jobs(document):
    """Jobs that have started and therefore need a committed durable state."""
    return {
        run_id for run_id, entry in latest_ledger_entries(document).items()
        if entry.get("status") != "PENDING"
    }


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _local_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


@contextlib.contextmanager
def controller_lock(project_root):
    """One local lifecycle writer; reclaim only a provably dead same-host owner."""
    state = pathlib.Path(project_root) / ".agent_state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "colab-controller.lock"
    hostname = socket.gethostname()
    controller_id = "%s-%s-%s" % (hostname, os.getpid(), secrets.token_hex(8))
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            raise SystemExit("Colab controller lock is unreadable; refusing duplicate ownership")
        same_host = existing.get("hostname") == hostname
        if not same_host or _local_pid_alive(existing.get("pid")):
            raise SystemExit("another Colab controller owns the lifecycle lock")
        stale = path.with_name("%s.stale.%d.%s" %
                               (path.name, int(time.time()), existing.get("controller_id", "unknown")))
        os.replace(path, stale)
        print("ARCHIVED_STALE_CONTROLLER_LOCK=%s" % stale, flush=True)
    document = {
        "schema": "nextlat_forgetting/colab_controller_lock/1",
        "controller_id": controller_id,
        "hostname": hostname,
        "pid": os.getpid(),
        "created_at_unix": time.time(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise SystemExit("another Colab controller acquired the lifecycle lock")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(document, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield document
    finally:
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            current = {}
        if current.get("controller_id") == controller_id:
            path.unlink(missing_ok=True)


def _load_bound_receipt(path, *, schema, label):
    try:
        document = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("%s is missing or invalid: %s" % (label, exc))
    if document.get("schema") != schema:
        raise SystemExit("%s schema mismatch" % label)
    return document


def _inside_project(root, raw_path, label):
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = pathlib.Path(root) / path
    path = path.resolve()
    try:
        path.relative_to(pathlib.Path(root).resolve())
    except ValueError:
        raise SystemExit("%s escapes the project root" % label)
    return path


def _input_remote_suffix(relative):
    mappings = (
        ("data/hmm_family/", "corpus/hmm_family/"),
        ("data/hmm/", "corpus/hmm/"),
        ("manifests/", "manifests/"),
    )
    for local_prefix, remote_prefix in mappings:
        if relative.startswith(local_prefix) and len(relative) > len(local_prefix):
            return remote_prefix + relative[len(local_prefix):]
    raise SystemExit("manifest inventory path is outside the frozen input domains: %s" % relative)


def validate_input_bundle_receipt(project_root):
    """Bind the exact local inventory to a generation-pinned completed upload receipt."""
    root = pathlib.Path(project_root).resolve()
    inventory = root / "manifests" / "manifest_inventory.sha256"
    receipt_path = root / INPUT_BUNDLE_RECEIPT_PATH
    if not inventory.is_file():
        raise SystemExit("manifest inventory is absent")
    receipt = _load_bound_receipt(
        receipt_path, schema=INPUT_BUNDLE_UPLOAD_SCHEMA,
        label="input-bundle upload receipt")
    if set(receipt) != {
            "schema", "status", "bucket", "bundle_prefix", "input_bundle_sha256",
            "object_count", "objects", "commit"}:
        raise SystemExit("input-bundle upload receipt field set mismatch")
    inventory_sha = sha256_file(inventory)
    prefix = "%s/input_bundles/%s" % (PREFIX, inventory_sha)
    if (receipt.get("status") != "COMPLETE" or receipt.get("bucket") != BUCKET or
            receipt.get("bundle_prefix") != prefix or
            receipt.get("input_bundle_sha256") != inventory_sha):
        raise SystemExit("input-bundle upload receipt identity mismatch")

    expected = {}
    line_re = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
    for number, line in enumerate(inventory.read_text(encoding="utf-8").splitlines(), 1):
        match = line_re.fullmatch(line)
        if match is None:
            raise SystemExit("malformed manifest inventory line %d" % number)
        digest, relative = match.groups()
        pure = pathlib.PurePosixPath(relative)
        if (pure.is_absolute() or str(pure) != relative or "\\" in relative or
                any(part in ("", ".", "..") for part in pure.parts) or
                relative == "manifests/manifest_inventory.sha256" or relative in expected):
            raise SystemExit("unsafe or duplicate manifest inventory path: %s" % relative)
        local = root.joinpath(*pure.parts)
        try:
            resolved = local.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise SystemExit("manifest inventory path escapes or is absent: %s" % relative)
        if not local.is_file() or local.is_symlink() or sha256_file(local) != digest:
            raise SystemExit("manifest inventory local binding mismatch: %s" % relative)
        expected[relative] = {
            "local_path": relative,
            "name": "%s/%s" % (prefix, _input_remote_suffix(relative)),
            "size_bytes": local.stat().st_size,
            "sha256": digest,
        }
    objects = receipt.get("objects")
    if (not isinstance(objects, list) or receipt.get("object_count") != len(expected) or
            len(objects) != len(expected)):
        raise SystemExit("input-bundle upload receipt object count mismatch")
    seen = set()
    for record in objects:
        if not isinstance(record, dict) or set(record) != {
                "local_path", "name", "generation", "size_bytes", "sha256"}:
            raise SystemExit("input-bundle upload object record mismatch")
        relative = record.get("local_path")
        if relative in seen or relative not in expected:
            raise SystemExit("input-bundle upload receipt has an extra or duplicate object")
        if (not str(record.get("generation", "")).isdigit() or
                any(record.get(key) != value for key, value in expected[relative].items())):
            raise SystemExit("input-bundle upload object binding mismatch: %s" % relative)
        seen.add(relative)
    if seen != set(expected):
        raise SystemExit("input-bundle upload receipt is incomplete")
    commit = receipt.get("commit")
    expected_commit = {
        "local_path": "manifests/manifest_inventory.sha256",
        "name": "%s/manifests/manifest_inventory.sha256" % prefix,
        "size_bytes": inventory.stat().st_size,
        "sha256": inventory_sha,
    }
    if (not isinstance(commit, dict) or set(commit) != {
            "local_path", "name", "generation", "size_bytes", "sha256"} or
            not str(commit.get("generation", "")).isdigit() or
            any(commit.get(key) != value for key, value in expected_commit.items())):
        raise SystemExit("input-bundle commit record mismatch")
    return {
        "receipt_path": INPUT_BUNDLE_RECEIPT_PATH,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_schema": INPUT_BUNDLE_UPLOAD_SCHEMA,
        "input_bundle_sha256": inventory_sha,
        "bundle_prefix": prefix,
        "object_count": len(expected),
        "commit_generation": str(commit["generation"]),
    }


def _load_input_uploader(project_root):
    path = pathlib.Path(project_root).resolve() / "scripts" / "upload_frozen_inputs.py"
    module_spec = importlib.util.spec_from_file_location("_nextlat_input_uploader", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit("frozen-input uploader could not be loaded")
    module = importlib.util.module_from_spec(module_spec)
    try:
        # Dataclasses inspect sys.modules while decorating classes.
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise SystemExit("frozen-input uploader load failed: %s" % exc)
    return module


def verify_remote_input_bundle(project_root, binding):
    """Verify every receipt generation exists before provisioning paid compute.

    Full payload read-back happened when the create-only receipt was minted.  Prelaunch checks
    compare each exact generation's authoritative name, size, and SHA metadata, then read back the
    small inventory commit again.  A deleted/replaced object therefore blocks before `colab start`.
    """
    root = pathlib.Path(project_root).resolve()
    if binding != validate_input_bundle_receipt(root):
        raise SystemExit("remote input verification received a stale local binding")
    receipt = json.loads((root / INPUT_BUNDLE_RECEIPT_PATH).read_text(encoding="utf-8"))
    uploader = _load_input_uploader(root)
    backend = uploader.GcloudBackend(BUCKET)
    records = list(receipt["objects"]) + [receipt["commit"]]
    for record in records:
        try:
            remote = backend.resolve(record["name"])
        except Exception as exc:
            raise SystemExit("could not verify frozen input object: %s" % exc)
        if (remote is None or str(remote.generation) != str(record["generation"]) or
                int(remote.size_bytes) != int(record["size_bytes"]) or
                remote.custom_sha256 != record["sha256"]):
            raise SystemExit("frozen input object is absent or changed: %s" % record["name"])
    commit = receipt["commit"]
    descriptor, temporary_name = tempfile.mkstemp(prefix="input-commit-preflight-")
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        backend.download_exact(commit["name"], commit["generation"], temporary)
        if sha256_file(temporary) != commit["sha256"]:
            raise SystemExit("frozen input commit exact-generation readback mismatch")
    finally:
        temporary.unlink(missing_ok=True)
    print("INPUT_BUNDLE_REMOTE_VERIFIED=%s OBJECTS=%d" %
          (binding["input_bundle_sha256"], binding["object_count"]), flush=True)
    return True


def publish_immutable_host_file(project_root, local_path, object_name):
    """Create-or-verify a host file at a content-addressed GCS name with exact read-back."""
    root = pathlib.Path(project_root).resolve()
    local = pathlib.Path(local_path).resolve()
    if not local.is_file():
        raise SystemExit("immutable upload source is absent: %s" % local)
    digest, size = sha256_file(local), local.stat().st_size
    uploader = _load_input_uploader(root)
    backend = uploader.GcloudBackend(BUCKET)
    try:
        record = uploader._create_or_verify(
            backend, object_name, local, digest, size, allow_create=True)
    except Exception as exc:
        raise SystemExit("immutable GCS publication failed: %s" % exc)
    print("IMMUTABLE_GCS_OBJECT=gs://%s/%s#%s SHA256=%s" %
          (BUCKET, object_name, record["generation"], digest), flush=True)
    return record


def recompute_preregistration_receipt(root, *, amendment, spec, evidence, validator):
    """Run the exact project validator and return its canonical in-memory receipt.

    Launch authorization never trusts a stored PASS merely because its outer shape looks
    plausible.  The validator source is loaded from the exact project path bound by the receipt,
    and every evidence artifact is re-read and rehashed on each clearance check.
    """
    root = pathlib.Path(root).resolve()
    expected_validator = (root / "scripts" / "validate_preregistration.py").resolve()
    validator = pathlib.Path(validator).resolve()
    if validator != expected_validator or not validator.is_file():
        raise SystemExit("preregistration validator authority path is stale")
    module_spec = importlib.util.spec_from_file_location(
        "_nextlat_preregistration_validator", validator)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit("preregistration validator could not be loaded")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
        computed = module.validate(
            pathlib.Path(evidence), amendment=pathlib.Path(amendment), spec=pathlib.Path(spec))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit("preregistration validator execution failed: %s" % exc)
    if not isinstance(computed, dict):
        raise SystemExit("preregistration validator returned a non-object receipt")
    return computed


def validate_preregistration_pass_receipt(project_root, source_sha256,
                                          receipt_path=None):
    """Bind the all-eleven PASS to the exact archive before any external lifecycle read."""
    root = pathlib.Path(project_root).resolve()
    receipt_path = _inside_project(
        root,
        receipt_path or root / ".agent_state" / "preregistration-freeze-receipt.json",
        "preregistration receipt path",
    )
    receipt = _load_bound_receipt(
        receipt_path, schema=PREREGISTRATION_FREEZE_SCHEMA,
        label="preregistration freeze receipt")
    expected_keys = {
        "schema", "status", "all_eleven_gates_pass", "authority",
        "missing_gate_blocks", "extra_gate_blocks", "global_issues", "gates", "meaning",
    }
    if set(receipt) != expected_keys:
        raise SystemExit("preregistration freeze receipt field set mismatch")
    if (receipt.get("status") != "PASS" or
            receipt.get("all_eleven_gates_pass") is not True or
            receipt.get("missing_gate_blocks") != [] or
            receipt.get("extra_gate_blocks") != [] or receipt.get("global_issues") != []):
        raise SystemExit("preregistration freeze receipt is not an all-eleven PASS")
    gates = receipt.get("gates")
    if (not isinstance(gates, list) or len(gates) != 11 or
            [gate.get("gate") for gate in gates if isinstance(gate, dict)] != list(range(1, 12)) or
            any(set(gate) != {"gate", "status", "issues"} or gate.get("status") != "PASS" or
                gate.get("issues") != [] for gate in gates if isinstance(gate, dict)) or
            any(not isinstance(gate, dict) for gate in gates)):
        raise SystemExit("preregistration freeze receipt gate set is incomplete or nonpassing")
    authority = receipt.get("authority")
    if not isinstance(authority, dict) or set(authority) != {
            "amendment", "spec", "evidence", "validator"}:
        raise SystemExit("preregistration freeze authority set mismatch")
    resolved = {}
    for role in ("amendment", "spec", "evidence", "validator"):
        record = authority.get(role)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SystemExit("preregistration %s authority record mismatch" % role)
        path = _inside_project(root, record.get("path", ""),
                               "preregistration %s authority" % role)
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise SystemExit("preregistration %s authority hash mismatch" % role)
        resolved[role] = path
    protocol_authorities = {
        "amendment": root / "docs" / "PREREGISTRATION_AMENDMENT_2026-08-24.md",
        "spec": root / "nextlat_v4_predictive_geometry_spec.md",
        "validator": root / "scripts" / "validate_preregistration.py",
    }
    for role, expected in protocol_authorities.items():
        if resolved[role] != expected.resolve():
            raise SystemExit("preregistration %s authority path is stale" % role)
    expected_evidence = (root / ".agent_state" / "preregistration-evidence.json").resolve()
    if resolved["evidence"] != expected_evidence:
        raise SystemExit(
            "preregistration evidence must be the archive-excluded .agent_state index")
    recomputed = recompute_preregistration_receipt(
        root,
        amendment=resolved["amendment"],
        spec=resolved["spec"],
        evidence=resolved["evidence"],
        validator=resolved["validator"],
    )
    if recomputed != receipt:
        raise SystemExit(
            "stored preregistration PASS does not equal hardened validator output")
    evidence = _load_bound_receipt(
        resolved["evidence"], schema=PREREGISTRATION_EVIDENCE_SCHEMA,
        label="preregistration evidence index")
    if set(evidence) != {"schema", "gates"} or not isinstance(evidence.get("gates"), dict) or \
            set(evidence["gates"]) != {str(index) for index in range(1, 12)}:
        raise SystemExit("preregistration evidence block set mismatch")
    gate_one = evidence["gates"].get("1")
    artifacts = gate_one.get("artifacts") if isinstance(gate_one, dict) else None
    if not isinstance(artifacts, list):
        raise SystemExit("preregistration gate 1 artifacts are missing")
    snapshots = [item for item in artifacts if isinstance(item, dict) and
                 item.get("role") == "source_snapshot"]
    if len(snapshots) != 1:
        raise SystemExit("preregistration gate 1 source snapshot binding is missing or duplicated")
    snapshot = snapshots[0]
    if set(snapshot) != {"role", "path", "sha256", "schema"} or \
            snapshot.get("schema") != "binary/source-snapshot":
        raise SystemExit("preregistration source snapshot record mismatch")
    archive = _inside_project(root, snapshot.get("path", ""),
                              "preregistration source snapshot")
    expected_archive = (root / ".agent_state" / "project.tar.gz").resolve()
    if archive != expected_archive:
        raise SystemExit("preregistration source snapshot path is not the packaged archive")
    if (not archive.is_file() or snapshot.get("sha256") != source_sha256 or
            sha256_file(archive) != source_sha256):
        raise SystemExit("preregistration PASS is stale for the exact source archive")

    return {
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_schema": PREREGISTRATION_FREEZE_SCHEMA,
        "evidence_sha256": sha256_file(resolved["evidence"]),
        "validator_sha256": sha256_file(resolved["validator"]),
        "source_archive_sha256": source_sha256,
    }


def validate_d43_continuation_bundle(project_root, spec, source_sha256):
    """Recompute the explicitly selected D43 receipt; never infer or downgrade its gate."""
    root = pathlib.Path(project_root).resolve()
    if (spec.get("continuation_gate") != D43_CONTINUATION_GATE or
            spec.get("continuation_gate_schema") != D43_CONTINUATION_SCHEMA):
        raise SystemExit("D43 continuation gate/schema selection is absent or stale")
    receipt_path = root / D43_RECEIPT_PATH
    expected_sha = spec.get("continuation_receipt_sha256")
    if (not receipt_path.is_file() or sha256_file(receipt_path) != expected_sha):
        raise SystemExit("D43 continuation receipt is missing or hash-stale")
    gate_path = root / "scripts" / "d43_measurement_amendment_gate.py"
    module_spec = importlib.util.spec_from_file_location(
        "_nextlat_d43_measurement_amendment_gate", gate_path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit("D43 continuation validator could not be loaded")
    gate = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(gate)
        receipt = gate.validate_receipt(
            root,
            root / D43_PREDECESSOR_ARCHIVE_PATH,
            root / D43_D41_BASELINE_ARCHIVE_PATH,
            root / D43_SUCCESSOR_ARCHIVE_PATH,
        )
    except Exception as exc:
        raise SystemExit("D43 continuation validation failed: %s" % exc)
    expected_lifecycle = {
        "training_started": True,
        "completed_hmm_training_cells": 10,
        "total_hmm_training_cells": 30,
        "scientific_evaluations_started": False,
        "scientific_evaluations_inspected": False,
    }
    successor = receipt.get("archives", {}).get("d43_measurement_successor", {})
    checkpoint_lineage = receipt.get("exact_ten_checkpoint_lineage", {})
    jobs = checkpoint_lineage.get("predecessor_to_successor_provenance")
    continuation_binding = receipt.get("outcome_blind_atomic_continuation", {})
    state_binding = continuation_binding.get("atomic_continuation_state")
    state_path = root / str(state_binding.get("path", "")) if isinstance(
        state_binding, dict) else root / "__missing_d43_state__"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("D43 atomic continuation state is missing or invalid") from exc
    pending = [job for job in gate.ALL_HMM_JOB_IDS if job not in gate.EXACT_TEN_JOB_IDS]
    if (receipt.get("schema") != D43_CONTINUATION_SCHEMA or receipt.get("status") != "PASS" or
            receipt.get("authorization") != "MEASUREMENT_AMENDMENT_GO" or
            receipt.get("confirmatory_lifecycle") != expected_lifecycle or
            successor.get("sha256") != source_sha256 or
            spec.get("predecessor_source_sha256") != gate.ORIGINAL_PREDECESSOR_SHA256 or
            not isinstance(jobs, list) or [item.get("job_id") for item in jobs] !=
            list(gate.EXACT_TEN_JOB_IDS) or
            any(item.get("created_under_predecessor_source_sha256") !=
                gate.ORIGINAL_PREDECESSOR_SHA256 or
                item.get("consumed_read_only_by_successor_source_sha256") != source_sha256
                for item in jobs) or
            state.get("training_started") is not True or
            state.get("completed_job_ids") != list(gate.EXACT_TEN_JOB_IDS) or
            state.get("pending_job_ids") != pending or state.get("evaluated_job_ids") != [] or
            state.get("scientific_evaluations_started") is not False or
            state.get("scientific_metrics_inspected") is not False or
            state.get("evaluator_invocations") != 0):
        raise SystemExit("D43 continuation lifecycle/partition/provenance binding failed")
    return {
        "schema": D43_LAUNCH_BINDING_SCHEMA,
        "gate": D43_CONTINUATION_GATE,
        "gate_schema": D43_CONTINUATION_SCHEMA,
        "receipt_path": D43_RECEIPT_PATH,
        "receipt_sha256": expected_sha,
        "source_sha256": source_sha256,
        "predecessor_source_sha256": gate.ORIGINAL_PREDECESSOR_SHA256,
        "completed_job_ids": list(gate.EXACT_TEN_JOB_IDS),
        "pending_job_ids": pending,
        "scientific_evaluations_started": False,
        "scientific_metrics_inspected": False,
    }


def validate_confirmatory_clearance(project_root, spec, source_sha256,
                                     clearance_path=None):
    """Fail closed unless one reviewed receipt binds the exact launch snapshot."""
    validate_confirmatory_job_spec(spec)
    root = pathlib.Path(project_root).resolve()
    clearance_path = pathlib.Path(clearance_path or
                                   root / ".agent_state" / "confirmatory-clearance.json")
    if not clearance_path.is_absolute():
        clearance_path = root / clearance_path
    clearance = _load_bound_receipt(
        clearance_path, schema=CONFIRMATORY_CLEARANCE_SCHEMA,
        label="confirmatory clearance")
    if clearance.get("authorization") != "GO":
        raise SystemExit("confirmatory clearance is not GO")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source_sha256)):
        raise SystemExit("computed source snapshot hash is invalid")
    if clearance.get("source_sha256") != source_sha256:
        raise SystemExit("confirmatory clearance source binding mismatch")
    if clearance.get("job_spec_sha256") != canonical_json_sha256(spec):
        raise SystemExit("confirmatory clearance job-spec binding mismatch")

    input_bundle = clearance.get("input_bundle")
    verified_input_bundle = validate_input_bundle_receipt(root)
    if input_bundle != verified_input_bundle:
        raise SystemExit("confirmatory clearance input-bundle binding mismatch")

    bindings = clearance.get("protocol_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(CONFIRMATORY_PROTOCOL_PATHS):
        raise SystemExit("confirmatory clearance protocol binding set mismatch")
    for relative in CONFIRMATORY_PROTOCOL_PATHS:
        path = root / relative
        if not path.is_file() or bindings.get(relative) != sha256_file(path):
            raise SystemExit("confirmatory protocol binding mismatch: %s" % relative)

    preregistration = clearance.get("preregistration")
    preregistration_keys = {
        "receipt_path", "receipt_sha256", "receipt_schema", "evidence_sha256",
        "validator_sha256", "source_archive_sha256",
    }
    if not isinstance(preregistration, dict) or set(preregistration) != preregistration_keys:
        raise SystemExit("confirmatory clearance preregistration binding set mismatch")
    d43_selected = spec.get("continuation_gate") == D43_CONTINUATION_GATE
    if ((root / D43_RECEIPT_PATH).is_file() and
            spec.get("predecessor_source_sha256") is not None and not d43_selected):
        raise SystemExit("D43 continuation receipt exists; refusing fallback to D41")
    if d43_selected:
        verified_continuation = validate_d43_continuation_bundle(
            root, spec, source_sha256)
        if clearance.get("continuation") != verified_continuation:
            raise SystemExit("confirmatory D43 continuation binding mismatch")
        gate41_path = root / "scripts" / "d41_continuation_gate.py"
        gate41_spec = importlib.util.spec_from_file_location(
            "_nextlat_d41_reference_for_d43", gate41_path)
        if gate41_spec is None or gate41_spec.loader is None:
            raise SystemExit("D43 predecessor reference validator could not be loaded")
        gate41 = importlib.util.module_from_spec(gate41_spec)
        gate41_spec.loader.exec_module(gate41)
        reference = json.loads(
            (root / gate41.PREDECESSOR_REFERENCE_PATH).read_text(encoding="utf-8"))
        predecessor_preregistration = reference.get("issued_clearance", {}).get(
            "preregistration")
        if preregistration != predecessor_preregistration:
            raise SystemExit("confirmatory D43 predecessor preregistration binding mismatch")
    elif spec.get("predecessor_source_sha256") is not None:
        gate_path = root / "scripts" / "d41_continuation_gate.py"
        module_spec = importlib.util.spec_from_file_location(
            "_nextlat_d41_continuation_gate", gate_path)
        if module_spec is None or module_spec.loader is None:
            raise SystemExit("D41 continuation validator could not be loaded")
        gate = importlib.util.module_from_spec(module_spec)
        try:
            module_spec.loader.exec_module(gate)
            verified_continuation = gate.validate_d41_continuation_bundle(
                root, spec, source_sha256)
            reference = json.loads(
                (root / gate.PREDECESSOR_REFERENCE_PATH).read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit("D41 continuation validation failed: %s" % exc)
        if clearance.get("continuation") != verified_continuation:
            raise SystemExit("confirmatory D41 continuation binding mismatch")
        predecessor_preregistration = reference.get("issued_clearance", {}).get(
            "preregistration")
        if preregistration != predecessor_preregistration:
            raise SystemExit("confirmatory predecessor preregistration binding mismatch")
    else:
        if "continuation" in clearance:
            raise SystemExit("fresh confirmatory clearance unexpectedly contains continuation")
        verified_preregistration = validate_preregistration_pass_receipt(
            root, source_sha256, preregistration.get("receipt_path"))
        if preregistration != verified_preregistration:
            raise SystemExit("confirmatory clearance preregistration binding mismatch")

    test_path = root / ".agent_state" / "confirmatory-test-receipt.json"
    review_path = root / ".agent_state" / "confirmatory-review-receipt.json"
    if clearance.get("test_receipt_sha256") != (sha256_file(test_path)
                                                   if test_path.is_file() else None):
        raise SystemExit("confirmatory test receipt binding mismatch")
    if clearance.get("review_receipt_sha256") != (sha256_file(review_path)
                                                     if review_path.is_file() else None):
        raise SystemExit("confirmatory review receipt binding mismatch")
    test_receipt = _load_bound_receipt(
        test_path, schema=CONFIRMATORY_TEST_SCHEMA, label="confirmatory test receipt")
    review_receipt = _load_bound_receipt(
        review_path, schema=CONFIRMATORY_REVIEW_SCHEMA, label="confirmatory review receipt")
    if (test_receipt.get("outcome") != "PASS" or test_receipt.get("exit_code") != 0 or
            test_receipt.get("source_sha256") != source_sha256 or
            int(test_receipt.get("tests_passed", 0)) <= 0):
        raise SystemExit("confirmatory test receipt is not a passing source-bound receipt")
    if (review_receipt.get("verdict") != "PASS" or
            review_receipt.get("source_sha256") != source_sha256 or
            not review_receipt.get("reviewer")):
        raise SystemExit("confirmatory review receipt is not a passing source-bound receipt")
    report_relative = review_receipt.get("report_path")
    if not isinstance(report_relative, str) or pathlib.Path(report_relative).is_absolute():
        raise SystemExit("confirmatory review report path is invalid")
    report_path = (root / report_relative).resolve()
    try:
        report_path.relative_to(root)
    except ValueError:
        raise SystemExit("confirmatory review report escapes the project root")
    if (not report_path.is_file() or
            review_receipt.get("report_sha256") != sha256_file(report_path)):
        raise SystemExit("confirmatory review report binding mismatch")
    print("CONFIRMATORY_CLEARANCE=GO SOURCE_SHA256=%s" % source_sha256, flush=True)
    return clearance


def checkpoint_step(path):
    """Read upstream's step from names such as ``recovery_ckpt_iter_250.pt``."""
    match = re.search(r"(?:^|_)iter_(\d+)(?:_|\.|$)", os.path.basename(path))
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", os.path.basename(path))
    return int(numbers[-1]) if numbers else 0


def secure_adc(path):
    """Refuse to run without the uploaded ADC and restrict it before authentication."""
    if not os.path.isfile(path):
        raise SystemExit("uploaded ADC is missing; refusing runtime GCS access")
    os.chmod(path, 0o600)


def verify_runtime_gpu(requested_gpu, expected_torch_version=None):
    """Fail before training if Colab did not provide the requested CUDA capability."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable on the provisioned Colab runtime")
    name = torch.cuda.get_device_name(0)
    bf16 = bool(torch.cuda.is_bf16_supported())
    version = str(torch.__version__)
    requested = str(requested_gpu or "").upper()
    expected_name = {"A100": "A100", "L4": "L4", "T4": "T4"}.get(requested)
    if expected_name and expected_name not in name.upper():
        raise SystemExit("requested %s but Colab assigned %s" % (requested, name))
    if requested == "A100" and not bf16:
        raise SystemExit("A100 confirmatory runtime does not report BF16 support")
    if expected_torch_version is not None and version != expected_torch_version:
        raise SystemExit("dependency bootstrap replaced Colab torch %s with %s" %
                         (expected_torch_version, version))
    print("RUNTIME_GPU=%s TORCH=%s CUDA=%s BF16=%s" %
          (name, version, torch.version.cuda, bf16), flush=True)
    return {"name": name, "torch_version": version, "cuda": str(torch.version.cuda),
            "bf16": bf16}


def download_runtime_recovery_receipt(bucket, spec, destination):
    """Read the clearance-bound D41 receipt from its exact immutable GCS generation."""
    if spec.get("predecessor_source_sha256") is None:
        return None
    name = spec.get("recovery_receipt_object")
    generation = spec.get("recovery_receipt_generation")
    try:
        blob = bucket.blob(name, generation=int(generation))
    except TypeError:
        blob = bucket.blob(name)
    destination = pathlib.Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    blob.download_to_filename(str(partial))
    blob.reload()
    if (str(getattr(blob, "generation", None)) != str(generation) or
            sha256_file(partial) != spec.get("recovery_receipt_sha256")):
        partial.unlink(missing_ok=True)
        raise SystemExit("D41 recovery receipt exact-generation readback failed")
    os.replace(partial, destination)
    try:
        receipt = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("D41 recovery receipt is invalid: %s" % exc)
    if (receipt.get("schema") != D41_RECOVERY_RECEIPT_SCHEMA or
            receipt.get("status") != "PASS" or
            receipt.get("scientific_metrics_inspected") is not False or
            tuple(item.get("job_id") for item in receipt.get("jobs", [])
                  if isinstance(item, dict)) != tuple(spec.get("recovery_job_ids", ()))):
        raise SystemExit("D41 recovery receipt semantic binding failed")
    return receipt


def verify_d41_runtime_equivalence(receipt, actual, *, project_root=None,
                                   patch_receipt_path=None,
                                   audit_patch_receipt_path=None):
    """Fail before state restore when the successor runtime differs from launch evidence."""
    if receipt is None:
        return actual
    expected_predecessor = {
        "device_name": "NVIDIA A100-SXM4-40GB",
        "torch_version": "2.11.0+cu128",
        "cuda_version": "12.8",
        "bf16_supported": True,
        "pinned_upstream_commit": PINNED,
    }
    contract = receipt.get("runtime_equivalence")
    if not isinstance(contract, dict):
        raise SystemExit("D41 receipt lacks runtime-equivalence evidence")
    predecessor = contract.get("predecessor_runtime_evidence")
    observed = predecessor.get("observed_preflight") if isinstance(predecessor, dict) else None
    if observed != expected_predecessor:
        raise SystemExit("D41 predecessor runtime fingerprint is absent or changed")
    required = contract.get("expected_successor_contract")
    if not isinstance(required, dict) or set(required) != set(expected_predecessor):
        raise SystemExit("D41 successor runtime contract field set mismatch")
    mismatches = {
        key: (required.get(key), actual.get(key)) for key in required
        if required.get(key) != actual.get(key)
    }
    if mismatches:
        raise SystemExit("D41 successor runtime fingerprint mismatch: %s" % mismatches)
    patch = contract.get("runtime_patch")
    source = pathlib.Path(project_root) / "scripts" / "runtime_bootstrap.py"
    patch_receipt_path = pathlib.Path(patch_receipt_path)
    if (not isinstance(patch, dict) or not source.is_file() or
            sha256_file(source) != patch.get("source_sha256") or
            not patch_receipt_path.is_file()):
        raise SystemExit("D41 runtime-patch source binding mismatch")
    applied_receipt_sha = sha256_file(patch_receipt_path)
    audit_receipt_sha = None
    if audit_patch_receipt_path is not None:
        audit_patch_receipt_path = pathlib.Path(audit_patch_receipt_path)
        if (not audit_patch_receipt_path.is_file() or
                audit_patch_receipt_path.read_bytes() != patch_receipt_path.read_bytes()):
            raise SystemExit(
                "D41 applied runtime-patch receipt differs from its runtime audit copy")
        audit_receipt_sha = sha256_file(audit_patch_receipt_path)
    patch_receipt = json.loads(patch_receipt_path.read_text(encoding="utf-8"))
    semantic_keys = (
        "schema", "patch_version", "upstream_commit", "before_sha256", "after_sha256",
        "helper_sha256", "adaptation_trainer_sha256", "adaptation_contract",
        "bst_parameter_count", "optimizer_update_rule", "optimizer_fusion_rule",
        "amp_scaler_checkpoint_rule", "deterministic_runtime_rule",
    )
    projection = {key: patch_receipt.get(key) for key in semantic_keys}
    unified_diff = patch_receipt.get("unified_diff")
    if not isinstance(unified_diff, str):
        raise SystemExit("D41 successor runtime-patch receipt lacks its applied diff")
    projection["unified_diff_sha256"] = hashlib.sha256(unified_diff.encode()).hexdigest()
    projection_sha = canonical_json_sha256(projection)
    if (projection != patch.get("expected_receipt_projection") or
            projection_sha != patch.get("expected_receipt_projection_sha256")):
        raise SystemExit("D41 successor runtime-patch receipt projection mismatch")
    result = dict(actual)
    result.update({
        "runtime_patch_source_sha256": sha256_file(source),
        "runtime_patch_receipt_sha256": applied_receipt_sha,
        "runtime_patch_applied_receipt_sha256": applied_receipt_sha,
        "runtime_patch_receipt_projection_sha256": projection_sha,
    })
    if audit_receipt_sha is not None:
        result["runtime_patch_audit_receipt_sha256"] = audit_receipt_sha
    return result


def archive_predecessor_hmm_ledger(durability, ledger_path, predecessor_source):
    """Archive the stale predecessor ledger and atomically create an empty successor ledger."""
    ledger = pathlib.Path(ledger_path)
    archived = None
    if ledger.is_file():
        digest = sha256_file(ledger)
        archived = ledger.with_name(
            "hmm_run_ledger.predecessor-%s-%s.json" % (predecessor_source[:12], digest))
        os.replace(ledger, archived)
        durability._upload_file(
            archived,
            "%s/recovery_audit/predecessor_ledgers/%s.json" %
            (durability.prefix, digest),
        )
    payload = {"schema": 1, "entries": []}
    partial = ledger.with_name(ledger.name + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, ledger)
    return str(archived) if archived is not None else None


def write_runtime_recovery_barrier(root, restored, spec):
    """Commit the all-ten restore barrier consumed by the matrix before any launcher."""
    expected = tuple(spec.get("recovery_job_ids", ()))
    if expected != D41_RECOVERY_JOB_IDS:
        raise SystemExit("runtime recovery barrier is not the clearance-bound exact ten")
    records = []
    for job_id in expected:
        state = restored.get(job_id)
        if not isinstance(state, dict):
            raise SystemExit("runtime recovery barrier is missing %s" % job_id)
        provenance = state.get("recovery_provenance")
        if (state.get("restored_step") != 3000 or
                not isinstance(provenance, dict) or
                provenance.get("checkpoint_sha256") !=
                sha256_file(state["restored_checkpoint"]) or
                provenance.get("recovery_receipt_sha256") !=
                spec.get("recovery_receipt_sha256")):
            raise SystemExit("runtime recovery barrier verification failed for %s" % job_id)
        authoritative_artifacts = {
            item["local_path"]: item["sha256"]
            for item in state.get("artifacts", [])
            if (isinstance(item, dict) and item.get("local_path") and item.get("sha256")
                and (str(item["local_path"]).endswith("materialized_config.yaml") or
                     str(item["local_path"]).endswith("metrics.csv") or
                     str(item["local_path"]).endswith("step_0_contract.json")))
        }
        if not authoritative_artifacts:
            raise SystemExit(
                "runtime recovery barrier has no authoritative completion artifacts for %s" %
                job_id)
        records.append({
            "job_id": job_id,
            "target_step": 3000,
            "checkpoint_path": state["restored_checkpoint"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "recovery_provenance": provenance,
            "authoritative_artifacts": authoritative_artifacts,
        })
    document = {
        "schema": "nextlat_forgetting/runtime_recovery_barrier/1",
        "status": "PASS",
        "source_snapshot_sha256": spec["source_sha256"],
        "predecessor_source_sha256": spec["predecessor_source_sha256"],
        "recovery_receipt_sha256": spec["recovery_receipt_sha256"],
        "job_ids": list(expected),
        "jobs": records,
    }
    path = pathlib.Path(root) / "runtime_recovery_barrier.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)
    return path


def durable_progress_advanced(before, after):
    """Whether any authoritative GCS state advanced to a newer checkpoint step."""
    for run_id, current in after.items():
        previous = before.get(run_id, {})
        if int(current.get("step", -1)) > int(previous.get("step", -1)):
            return True
        if (current.get("checkpoint_sha256") and
                current.get("checkpoint_sha256") != previous.get("checkpoint_sha256")):
            return True
    return False


def host_durable_progress():
    """Read committed per-run state through the host CLI for circuit-breaker decisions.

    This function is LOOP-only. Runtime code never invokes gcloud; Colab uses the Python
    storage client exclusively.
    """
    state_glob = GCS + "/runs/*/state.json"
    rc, listing = sh("gcloud storage ls %s" % shlex.quote(state_glob),
                     check=False, quiet=True, silent=True, max_lines=None)
    if rc != 0:
        return {}
    progress = {}
    for uri in listing.splitlines():
        uri = uri.strip()
        if not uri.startswith("gs://") or not uri.endswith("/state.json"):
            continue
        state_rc, raw = sh("gcloud storage cat %s" % shlex.quote(uri),
                           check=False, quiet=True, silent=True, max_lines=None)
        if state_rc != 0:
            continue
        try:
            state = json.loads(raw)
            checkpoint = state.get("checkpoint", {})
            progress[state["run_id"]] = {
                "step": int(state.get("step", -1)),
                "checkpoint_sha256": checkpoint.get("sha256"),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return progress


def host_runtime_activity():
    """Read verified telemetry receipts used only to avoid killing useful live work."""
    pattern = GCS + "/runs/*/telemetry/latest.json"
    rc, listing = sh("gcloud storage ls %s" % shlex.quote(pattern),
                     check=False, quiet=True, silent=True, max_lines=None)
    if rc != 0:
        return {}
    activity = {}
    for uri in listing.splitlines():
        uri = uri.strip()
        if not uri.startswith("gs://") or not uri.endswith("/telemetry/latest.json"):
            continue
        read_rc, raw = sh("gcloud storage cat %s" % shlex.quote(uri),
                          check=False, quiet=True, silent=True, max_lines=None)
        if read_rc != 0:
            continue
        try:
            receipt = json.loads(raw)
            if receipt.get("schema") != "nextlat_forgetting/runtime_telemetry/1":
                continue
            activity[str(receipt["run_id"])] = {
                "synced_at": float(receipt["synced_at"]),
                "artifact_hashes": sorted(item["sha256"] for item in receipt["artifacts"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return activity


def host_terminal_marker(session_id, expected_source_sha256):
    """Read a verified session-scoped completion marker through the host CLI."""
    uri = "%s/control/%s/terminal.json" % (GCS, session_id)
    rc, raw = sh("gcloud storage cat %s" % shlex.quote(uri),
                 check=False, quiet=True, silent=True, max_lines=None)
    if rc != 0:
        return None
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if (marker.get("schema") != "nextlat_forgetting/runtime_terminal/1" or
            marker.get("session_id") != str(session_id) or
            marker.get("source_snapshot_sha256") != expected_source_sha256):
        return None
    return marker


def monitor_owned_runtime(
    session_id,
    source_sha256,
    baseline,
    *,
    status_reader=colab_status_pair,
    progress_reader=host_durable_progress,
    terminal_reader=host_terminal_marker,
    activity_reader=None,
    stall_limit=MAX_STALLED_SYNC_WINDOWS,
):
    """Preserve live work after an exec-stream timeout instead of stopping the GPU.

    A timeout is a transport event.  The runtime is stopped only after a verified terminal marker,
    two agreeing gone reads, or several full sync windows with no committed checkpoint advance.
    """
    current = dict(baseline)
    activity = activity_reader() if activity_reader is not None else {}
    stalled = 0
    while True:
        first, second = status_reader()
        runtime_state = agreed_runtime_state(first, second)
        if runtime_state == "gone":
            return {"reason": "gone", "progress": current, "marker": None}
        if runtime_state == "uncertain":
            continue
        try:
            require_owned_session(first, second, session_id)
        except OwnershipUncertain:
            return {"reason": "ownership_uncertain", "progress": current, "marker": None,
                    "status_first": first, "status_second": second}
        marker = terminal_reader(session_id, source_sha256)
        if marker is not None:
            terminal_progress = progress_reader()
            return {"reason": "terminal", "progress": terminal_progress or current,
                    "marker": marker}
        after = progress_reader()
        if durable_progress_advanced(current, after):
            print("owned runtime still advancing durable checkpoints; keeping it alive", flush=True)
            current = after
            stalled = 0
            continue
        if activity_reader is not None:
            after_activity = activity_reader()
            if after_activity != activity and after_activity:
                print("owned runtime telemetry still advancing; waiting for next checkpoint",
                      flush=True)
                activity = after_activity
                stalled = 0
                continue
        stalled += 1
        print("owned runtime active but no durable advance %d/%d" % (stalled, stall_limit),
              flush=True)
        if stalled >= stall_limit:
            return {"reason": "stalled", "progress": after, "marker": None}


def stop_owned_runtime(reason, session_file, *, stopper=None, status_reader=colab_status_pair):
    """Stop only a terminal/stalled owned runtime and require a two-read gone verdict."""
    if reason not in {"terminal", "stalled", "input-upload-failed"}:
        raise ValueError("refusing to stop a runtime for non-terminal reason: %s" % reason)
    if stopper is None:
        sid = pathlib.Path(session_file).read_text().strip()
        if not sid:
            raise RuntimeError("owned session file is empty during teardown")
        stopper = lambda: sh("colab stop --session %s" % shlex.quote(sid), check=False)
    stopper()
    first, second = status_reader()
    if agreed_runtime_state(first, second) != "gone":
        stopper()
        first, second = status_reader()
    if agreed_runtime_state(first, second) != "gone":
        raise SystemExit("Colab runtime did not reach two-read stopped state")
    pathlib.Path(session_file).unlink(missing_ok=True)


class RuntimeDurability:
    """During-run verified GCS sync and exact-path restore for Colab runtimes.

    A job's files are mirrored below ``{prefix}/runs/{job_id}/``.  ``state.json`` is
    published last and is the commit record: readers never treat an uploaded checkpoint as
    durable until this object contains its exact runtime path, step, size and SHA-256.
    """

    def __init__(self, bucket, root, prefix=PREFIX, logger=print, source_sha256=None,
                 checkpoint_loader=None, predecessor_source_sha256=None,
                 recovery_receipt=None, recovery_receipt_sha256=None,
                 runtime_fingerprint=None):
        self.bucket = bucket
        self.root = pathlib.Path(root).resolve()
        self.prefix = prefix.strip("/")
        self.log = logger
        self.source_sha256 = source_sha256
        self.predecessor_source_sha256 = predecessor_source_sha256
        self.recovery_receipt = recovery_receipt
        self.recovery_receipt_sha256 = recovery_receipt_sha256
        self.runtime_fingerprint = dict(runtime_fingerprint or {})
        self._recovery_jobs = {}
        self._restored_provenance = {}
        for label, digest in (("source", source_sha256),
                              ("predecessor source", predecessor_source_sha256)):
            if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise ValueError("%s hash must be exact lowercase SHA-256" % label)
        if predecessor_source_sha256 is not None:
            if source_sha256 is None:
                raise ValueError("predecessor source requires a current source hash")
            if predecessor_source_sha256 == source_sha256:
                raise ValueError("predecessor source must differ from current source")
            if (not isinstance(recovery_receipt, dict) or
                    recovery_receipt.get("schema") != D41_RECOVERY_RECEIPT_SCHEMA or
                    recovery_receipt.get("status") != "PASS" or
                    recovery_receipt.get("scientific_metrics_inspected") is not False):
                raise ValueError("predecessor recovery requires the exact PASS D41 receipt")
            if not re.fullmatch(r"[0-9a-f]{64}", str(recovery_receipt_sha256)):
                raise ValueError("predecessor recovery requires an exact receipt SHA-256")
            jobs = recovery_receipt.get("jobs")
            if (not isinstance(jobs, list) or
                    tuple(item.get("job_id") for item in jobs if isinstance(item, dict)) !=
                    D41_RECOVERY_JOB_IDS):
                raise ValueError("D41 receipt does not contain the exact canonical ten jobs")
            self._recovery_jobs = {item["job_id"]: item for item in jobs}
        self.checkpoint_loader = checkpoint_loader
        self._lock = threading.Lock()
        self._uploaded = set()
        self._upload_records = {}
        self._deep_verified = set()
        self._sidecar_normalizations = {}

    def _accept_restore_source(self, actual, label):
        """Accept current state or one explicitly authorized predecessor generation."""
        if self.source_sha256 is None:
            return False
        accepted = {self.source_sha256}
        if self.predecessor_source_sha256 is not None:
            accepted.add(self.predecessor_source_sha256)
        if actual not in accepted:
            raise RuntimeError("%s source snapshot does not match this runtime: %s" %
                               (label, actual))
        migrated = actual != self.source_sha256
        if migrated:
            self.log("[restore] %s source migration %s -> %s authorized" %
                     (label, actual, self.source_sha256))
        return migrated

    def _remote(self, run_id, relative):
        return "%s/runs/%s/%s" % (self.prefix, run_id, relative.lstrip("/"))

    def _binding_name(self, binding, label):
        if not isinstance(binding, dict):
            raise RuntimeError("D41 %s binding is absent" % label)
        uri = str(binding.get("uri", ""))
        prefix = "gs://%s/" % self.bucket.name
        if not uri.startswith(prefix):
            raise RuntimeError("D41 %s URI is outside the authoritative bucket" % label)
        name = uri[len(prefix):]
        if (not name or not str(binding.get("generation", "")).isdigit() or
                not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))) or
                isinstance(binding.get("size_bytes"), bool) or
                not isinstance(binding.get("size_bytes"), int)):
            raise RuntimeError("D41 %s lacks exact generation/SHA/size" % label)
        return name

    def _generation_blob(self, name, generation):
        try:
            return self.bucket.blob(name, generation=int(generation))
        except TypeError:
            # Test doubles and older clients may not accept the keyword.  The generation is
            # still checked after reload, so this never becomes a latest-version bypass.
            return self.bucket.blob(name)

    def _download_exact_binding(self, binding, local, label):
        name = self._binding_name(binding, label)
        local = pathlib.Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".partial")
        blob = self._generation_blob(name, binding["generation"])
        blob.download_to_filename(str(partial))
        blob.reload()
        actual_generation = getattr(blob, "generation", None)
        if (str(actual_generation) != str(binding["generation"]) or
                partial.stat().st_size != int(binding["size_bytes"]) or
                sha256_file(partial) != binding["sha256"]):
            partial.unlink(missing_ok=True)
            raise RuntimeError("D41 %s exact-generation verification failed" % label)
        os.replace(partial, local)
        return local

    def download_file(self, remote, local, required=True):
        """Download one object atomically through the Python storage client."""
        local = pathlib.Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".partial")
        blob = self.bucket.blob(remote)
        try:
            blob.download_to_filename(str(partial))
        except Exception:
            partial.unlink(missing_ok=True)
            if required:
                raise
            return False
        os.replace(partial, local)
        return True

    def download_prefix(self, remote_prefix, local_root):
        """Mirror a flat/prefixed GCS tree without invoking the runtime gcloud CLI."""
        count = 0
        prefix = remote_prefix.rstrip("/") + "/"
        for blob in self.bucket.list_blobs(prefix=prefix):
            relative = blob.name[len(prefix):]
            if not relative or relative.endswith("/"):
                continue
            local = pathlib.Path(local_root) / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            partial = local.with_name(local.name + ".partial")
            blob.download_to_filename(str(partial))
            os.replace(partial, local)
            count += 1
        return count

    def _local_path(self, value):
        path = pathlib.Path(value)
        if not path.is_absolute():
            raise RuntimeError("durable paths must be absolute: %s" % path)
        resolved = path.resolve()
        runs = (self.root / "runs").resolve()
        if os.path.commonpath((str(resolved), str(runs))) != str(runs):
            raise RuntimeError("refusing durable path outside %s: %s" % (runs, resolved))
        return resolved

    def _upload_file(self, local, remote):
        local = pathlib.Path(local)
        digest = sha256_file(local)
        size = local.stat().st_size
        cache_key = (remote, digest, size)
        if cache_key in self._uploaded:
            return dict(self._upload_records[cache_key])
        blob = self.bucket.blob(remote)
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        blob.upload_from_filename(str(local))
        if sha256_file(local) != digest or local.stat().st_size != size:
            raise RuntimeError("file changed while being synchronized: %s" % local)
        blob.reload()
        remote_size = int(blob.size) if blob.size is not None else -1
        remote_sha = (blob.metadata or {}).get("sha256")
        if remote_size != size or remote_sha != digest:
            raise RuntimeError("GCS verification failed for gs://%s/%s" %
                               (self.bucket.name, remote))
        self._uploaded.add(cache_key)
        record = {"local_path": str(local), "remote": remote,
                  "sha256": digest, "size_bytes": size}
        if getattr(blob, "generation", None) is not None:
            record["generation"] = str(blob.generation)
        self._upload_records[cache_key] = dict(record)
        return record

    def _publish_immutable_file(self, local, remote):
        """Create or reuse one content-addressed object without generation churn.

        The D41 predecessor receipt names exact GCS generations.  A successor durability
        transaction therefore must never upload restored bytes back to those names.  Its
        own recovery copy lives at a source/content-addressed name and is create-only; a
        retry verifies and reuses that same generation.
        """
        local = pathlib.Path(local)
        digest = sha256_file(local)
        size = local.stat().st_size
        cache_key = (remote, digest, size)
        if cache_key in self._uploaded:
            return dict(self._upload_records[cache_key])
        blob = self.bucket.blob(remote)
        exists = False
        try:
            blob.reload()
            exists = blob.size is not None
        except Exception as exc:
            if exc.__class__.__name__ != "NotFound":
                raise
        if exists:
            payload = blob.download_as_bytes()
            if (len(payload) != size or hashlib.sha256(payload).hexdigest() != digest or
                    (blob.metadata or {}).get("sha256") != digest):
                raise RuntimeError(
                    "immutable successor recovery object conflicts: gs://%s/%s" %
                    (self.bucket.name, remote))
        else:
            blob.metadata = dict(blob.metadata or {}, sha256=digest,
                                 immutable_recovery="true",
                                 source_snapshot_sha256=str(self.source_sha256))
            try:
                blob.upload_from_filename(str(local), if_generation_match=0)
            except TypeError:
                # Small test doubles may not model preconditions.  Production GCS does, and
                # the exact readback below remains mandatory in both environments.
                blob.upload_from_filename(str(local))
            blob.reload()
        if (int(blob.size or -1) != size or
                (blob.metadata or {}).get("sha256") != digest or
                hashlib.sha256(blob.download_as_bytes()).hexdigest() != digest or
                getattr(blob, "generation", None) is None):
            raise RuntimeError("immutable successor object verification failed: gs://%s/%s" %
                               (self.bucket.name, remote))
        self._uploaded.add(cache_key)
        record = {
            "local_path": str(local), "remote": remote, "sha256": digest,
            "size_bytes": size, "generation": str(blob.generation),
        }
        self._upload_records[cache_key] = dict(record)
        return record

    def _successor_recovery_remote(self, run_id, kind, digest, filename):
        if (not re.fullmatch(r"[0-9a-f]{64}", str(self.source_sha256)) or
                not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or
                pathlib.PurePosixPath(filename).name != filename or
                kind not in {"checkpoint", "sidecar"}):
            raise RuntimeError("invalid immutable successor recovery identity")
        return "%s/runs/%s/successor_recovery/%s/%s/%s/%s" % (
            self.prefix, run_id, self.source_sha256, kind, digest, filename)

    def _artifact_binding(self, artifact):
        if not isinstance(artifact, dict) or artifact.get("generation") is None:
            raise RuntimeError("successor recovery artifact lacks exact generation")
        return {
            "uri": "gs://%s/%s" % (self.bucket.name, artifact["remote"]),
            "generation": str(artifact["generation"]),
            "sha256": artifact["sha256"],
            "size_bytes": int(artifact["size_bytes"]),
        }

    def _upload_snapshot_file(self, local, remote):
        """Upload one coherent read of a live telemetry file without racing its writer."""
        local = pathlib.Path(local)
        return self._upload_snapshot_payload(local, local.read_bytes(), remote)

    def _upload_snapshot_payload(self, local, payload, remote):
        """Upload caller-frozen bytes while retaining their absolute restore path."""
        local = pathlib.Path(local)
        digest = hashlib.sha256(payload).hexdigest()
        size = len(payload)
        cache_key = (remote, digest, size)
        if cache_key in self._uploaded:
            return dict(self._upload_records[cache_key])
        blob = self.bucket.blob(remote)
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        blob.upload_from_string(payload, content_type="application/octet-stream")
        blob.reload()
        if (blob.download_as_bytes() != payload or int(blob.size or -1) != size or
                (blob.metadata or {}).get("sha256") != digest):
            raise RuntimeError("GCS telemetry verification failed for gs://%s/%s" %
                               (self.bucket.name, remote))
        self._uploaded.add(cache_key)
        record = {"local_path": str(local), "remote": remote,
                  "sha256": digest, "size_bytes": size}
        if getattr(blob, "generation", None) is not None:
            record["generation"] = str(blob.generation)
        self._upload_records[cache_key] = dict(record)
        return record

    def _upload_state(self, run_id, state):
        remote = self._remote(run_id, "state.json")
        payload = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        blob = self.bucket.blob(remote)
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        # A GCS object replacement is atomic. Publishing this single object last commits
        # the already verified artifacts above without exposing a partial JSON document.
        blob.upload_from_string(payload, content_type="application/json")
        blob.reload()
        if blob.download_as_bytes() != payload or (blob.metadata or {}).get("sha256") != digest:
            raise RuntimeError("state.json verification failed for %s" % run_id)

    def publish_terminal(self, session_id, *, returncode, training_complete):
        """Publish a session-scoped marker only after the final durability transaction."""
        payload = (json.dumps({
            "schema": "nextlat_forgetting/runtime_terminal/1",
            "session_id": str(session_id),
            "source_snapshot_sha256": self.source_sha256,
            "returncode": int(returncode),
            "training_complete": bool(training_complete),
            "published_at": time.time(),
        }, indent=2, sort_keys=True) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        remote = "%s/control/%s/terminal.json" % (self.prefix, session_id)
        blob = self.bucket.blob(remote)
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        blob.upload_from_string(payload, content_type="application/json")
        blob.reload()
        if blob.download_as_bytes() != payload or (blob.metadata or {}).get("sha256") != digest:
            raise RuntimeError("terminal marker verification failed for %s" % session_id)
        return json.loads(payload)

    def _pointer_targets(self, out_root):
        found = []
        for name in ("recovery_ckpt", "latest_ckpt"):
            pointer = out_root / name
            if not pointer.is_file():
                continue
            target_text = pointer.read_text().strip()
            if not target_text:
                continue
            target = self._local_path(target_text)
            if target.is_file() and target.suffix != ".partial":
                found.append((pointer, target))
        return found

    def _verify_checkpoint_for_sync(self, checkpoint, *, expected_step=None,
                                    expected_run_id=None,
                                    legacy_sidecar_binding=None):
        """Require filename, sidecar, and loaded optimizer steps to agree exactly."""
        checkpoint = pathlib.Path(checkpoint).resolve()
        filename_step = checkpoint_step(checkpoint)
        if filename_step < 0:
            raise RuntimeError("checkpoint filename has no exact training step: %s" % checkpoint)
        if expected_step is not None and filename_step != int(expected_step):
            raise RuntimeError(
                "checkpoint filename step %d != exact expected step %d: %s" %
                (filename_step, int(expected_step), checkpoint))
        sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
        if not sidecar.is_file():
            raise RuntimeError("checkpoint lacks verification metadata: %s" % checkpoint)
        metadata = json.loads(sidecar.read_text())
        size = checkpoint.stat().st_size
        digest = sha256_file(checkpoint)
        original_sidecar_sha = sha256_file(sidecar)
        original_sidecar_size = sidecar.stat().st_size
        if legacy_sidecar_binding is not None:
            self._binding_name(legacy_sidecar_binding, "legacy checkpoint sidecar")
            if (original_sidecar_sha != legacy_sidecar_binding.get("sha256") or
                    original_sidecar_size !=
                    int(legacy_sidecar_binding.get("size_bytes", -1))):
                raise RuntimeError(
                    "legacy checkpoint sidecar differs from receipt-pinned bytes: %s" %
                    checkpoint)
        try:
            from lurestar.durable_checkpoint import exact_sidecar_step
            sidecar_step = exact_sidecar_step(metadata)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "checkpoint metadata has invalid step identity: %s: %s" %
                (checkpoint, exc)) from exc
        if legacy_sidecar_binding is None and "training_steps" not in metadata:
            raise RuntimeError(
                "current-source checkpoint sidecar lacks canonical training_steps: %s" %
                checkpoint)
        if (int(metadata.get("size_bytes", -1)) != size or
                metadata.get("sha256") != digest or
                metadata.get("path") != str(checkpoint) or
                (expected_run_id is not None and
                 metadata.get("run_id") != str(expected_run_id)) or
                int(sidecar_step) != filename_step):
            raise RuntimeError("checkpoint metadata does not match payload: %s" % checkpoint)
        cache_key = (str(checkpoint), digest, size)
        if cache_key not in self._deep_verified:
            if self.checkpoint_loader is None:
                import torch
                state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            else:
                state = self.checkpoint_loader(str(checkpoint))
            if not isinstance(state, dict):
                raise RuntimeError("checkpoint did not deserialize to a mapping: %s" % checkpoint)
            loaded_step = state.get("training_steps")
            if (isinstance(loaded_step, bool) or not isinstance(loaded_step, int) or
                    int(loaded_step) != filename_step):
                raise RuntimeError(
                    "loaded checkpoint training_steps %r != filename/sidecar step %d: %s" %
                    (loaded_step, filename_step, checkpoint))
            self._deep_verified.add(cache_key)
        if legacy_sidecar_binding is not None and "training_steps" not in metadata:
            normalized = dict(metadata)
            normalized["training_steps"] = int(sidecar_step)
            payload = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode()
            partial = sidecar.with_name(sidecar.name + ".partial")
            with open(partial, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, sidecar)
            normalized_sha = sha256_file(sidecar)
            self._sidecar_normalizations[str(checkpoint)] = {
                "schema": "nextlat_forgetting/recovery_sidecar_normalization/1",
                "from_sha256": original_sidecar_sha,
                "from_size_bytes": original_sidecar_size,
                "from_generation": str(legacy_sidecar_binding["generation"]),
                "to_sha256": normalized_sha,
                "to_size_bytes": sidecar.stat().st_size,
                "training_steps": int(sidecar_step),
                "preserved_step": int(metadata["step"]),
                "path": str(checkpoint),
                "run_id": str(expected_run_id),
            }
        return sidecar

    def _stage_runtime_patch_audit(self, out_root):
        """Mirror the exact applied diff/receipt under each run's durable namespace."""
        source = self.root / "source_snapshot" / "runtime_patch"
        staged = out_root / "runtime_patch"
        paths = []
        for name in ("runtime_patch.diff", "runtime_patch_receipt.json"):
            original = source / name
            if not original.is_file():
                continue
            staged.mkdir(parents=True, exist_ok=True)
            destination = staged / name
            payload = original.read_bytes()
            partial = destination.with_name(destination.name + ".partial")
            with open(partial, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, destination)
            paths.append(destination)
        return paths

    def _ledger_artifact_paths(self, entry, out_root):
        paths = set()
        artifacts = entry.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            raise RuntimeError("ledger artifacts must be a relative-path to SHA mapping")
        for relative, expected_sha in artifacts.items():
            if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                                  str(expected_sha)):
                raise RuntimeError("ledger artifact has invalid path/SHA: %r" % relative)
            path = (out_root / relative).resolve()
            if os.path.commonpath((str(path), str(out_root))) != str(out_root):
                raise RuntimeError("ledger artifact escapes out_root: %s" % relative)
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise RuntimeError("ledger artifact is missing or hash-invalid: %s" % relative)
            paths.add(path)
        return paths

    def _d44_receipt_pinned_pointer_targets(self, out_root, pointer_targets,
                                             recovery_record, recovery_provenance,
                                             run_id):
        """Preflight a D41 terminal sync before any artifact can be uploaded.

        Generic jobs deliberately retain two recovery generations.  The D41 migration is
        different: its receipt authorizes exactly one predecessor checkpoint/sidecar at step
        3000.  In particular, a stale ``recovery_ckpt`` pointer must fail closed rather than
        allowing the generic two-generation scan to upload, normalize, or advertise it.
        """
        if not isinstance(recovery_provenance, dict):
            raise RuntimeError("D41 successor sync lacks predecessor recovery provenance")
        expected_checkpoint = recovery_record["checkpoint_object"]
        expected_sidecar = recovery_record["sidecar_object"]
        if (recovery_provenance.get("checkpoint_generation") !=
                str(expected_checkpoint["generation"]) or
                recovery_provenance.get("checkpoint_sha256") !=
                expected_checkpoint["sha256"] or
                recovery_provenance.get("sidecar_generation") !=
                str(expected_sidecar["generation"]) or
                recovery_provenance.get("sidecar_sha256") !=
                expected_sidecar["sha256"]):
            raise RuntimeError("D41 successor sync lost predecessor generation provenance")
        normalization = recovery_provenance.get("sidecar_normalization")
        if normalization is not None:
            target_paths = {str(target) for _, target in pointer_targets}
            if len(target_paths) != 1:
                raise RuntimeError("D41 receipt-bound pointers do not name one exact target")
            required_normalization = {
                "schema": "nextlat_forgetting/recovery_sidecar_normalization/1",
                "from_sha256": expected_sidecar["sha256"],
                "from_size_bytes": int(expected_sidecar["size_bytes"]),
                "from_generation": str(expected_sidecar["generation"]),
                "training_steps": int(recovery_record["target_step"]),
                "preserved_step": int(recovery_record["target_step"]),
                "path": next(iter(target_paths)),
                "run_id": run_id,
            }
            if (not isinstance(normalization, dict) or
                    any(normalization.get(key) != value
                        for key, value in required_normalization.items()) or
                    not re.fullmatch(r"[0-9a-f]{64}",
                                     str(normalization.get("to_sha256", ""))) or
                    not isinstance(normalization.get("to_size_bytes"), int) or
                    normalization["to_size_bytes"] <= 0):
                raise RuntimeError("D41 sidecar normalization provenance is invalid")
            expected_successor_sidecar_sha = normalization["to_sha256"]
            expected_successor_sidecar_size = normalization["to_size_bytes"]
        else:
            expected_successor_sidecar_sha = expected_sidecar["sha256"]
            expected_successor_sidecar_size = int(expected_sidecar["size_bytes"])
        if not pointer_targets:
            raise RuntimeError("D41 receipt-bound sync lacks an exact terminal pointer")
        target_paths = {str(target) for _, target in pointer_targets}
        if len(target_paths) != 1:
            raise RuntimeError("D41 receipt-bound pointers do not name one exact target")
        checkpoint = pointer_targets[0][1]
        expected_step = int(recovery_record["target_step"])
        # Do this before any normalization or upload.  A legacy 2750/2500 pointer is not a
        # recoverable candidate under the D41 receipt, even if an exact 3000 file also exists.
        if (checkpoint_step(checkpoint) != expected_step or
                sha256_file(checkpoint) != expected_checkpoint["sha256"]):
            raise RuntimeError("D41 receipt-bound pointer target is not the exact checkpoint")
        if normalization is None:
            self._verify_checkpoint_for_sync(
                checkpoint, expected_step=expected_step, expected_run_id=run_id,
                legacy_sidecar_binding=expected_sidecar)
        else:
            sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
            if (not sidecar.is_file() or
                    sha256_file(sidecar) != expected_successor_sidecar_sha or
                    sidecar.stat().st_size != expected_successor_sidecar_size):
                raise RuntimeError("D41 normalized sidecar differs from receipt provenance")
            self._verify_checkpoint_for_sync(
                checkpoint, expected_step=expected_step, expected_run_id=run_id)
        return {
            "expected_checkpoint": expected_checkpoint,
            "expected_sidecar": expected_sidecar,
            "normalization": normalization,
            "expected_successor_sidecar_sha": expected_successor_sidecar_sha,
            "expected_successor_sidecar_size": expected_successor_sidecar_size,
            "allowed_sidecar_sha256": {
                expected_sidecar["sha256"], expected_successor_sidecar_sha,
            },
        }, pointer_targets, checkpoint

    def _artifact_paths(self, out_root, pointer_targets, entry,
                        receipt_pinned_checkpoint=None):
        paths = set()
        for pointer, target in pointer_targets:
            paths.update((pointer, target))
        if receipt_pinned_checkpoint is None:
            # The pointer exposes the newest checkpoint, but generic jobs deliberately retain
            # two verified recovery generations.  Both cross their durability boundary.
            for checkpoint in out_root.glob("*/recovery_ckpt_iter_*.pt"):
                if checkpoint.is_file() and not checkpoint.name.endswith(".partial"):
                    paths.add(checkpoint)
            for checkpoint in tuple(path for path in paths if path.suffix == ".pt"):
                paths.add(self._verify_checkpoint_for_sync(checkpoint))
        else:
            # The D41 preflight already deeply verified and, if necessary, normalized only the
            # receipt-pinned checkpoint.  Do not even enumerate stale recovery generations.
            receipt_pinned_checkpoint = pathlib.Path(receipt_pinned_checkpoint).resolve()
            if {path.resolve() for _, path in pointer_targets} != {receipt_pinned_checkpoint}:
                raise RuntimeError("D41 receipt-bound pointers changed after preflight")
            sidecar = receipt_pinned_checkpoint.with_name(
                receipt_pinned_checkpoint.name + ".meta.json")
            if not sidecar.is_file():
                raise RuntimeError("D41 receipt-pinned checkpoint sidecar is missing")
            paths.update((receipt_pinned_checkpoint, sidecar))
        paths.update(self._stage_runtime_patch_audit(out_root))
        for name in ("durable_index.json", "training_completion.json", "final_summary.json"):
            path = out_root / name
            if path.is_file():
                paths.add(path)
        # Only immutable/atomic files enter resumable state. Live CSV/evaluation progress is
        # snapshotted separately so it can never block a checkpoint transaction.
        for pattern in ("*/materialized_config.yaml", "metrics/*.json"):
            paths.update(path for path in out_root.glob(pattern) if path.is_file())
        paths.update(self._ledger_artifact_paths(entry, out_root))
        if receipt_pinned_checkpoint is not None:
            expected_sidecar = receipt_pinned_checkpoint.with_name(
                receipt_pinned_checkpoint.name + ".meta.json")
            unexpected = [path for path in paths if (
                (path.suffix == ".pt" and path.resolve() != receipt_pinned_checkpoint) or
                (path.name.endswith(".pt.meta.json") and path.resolve() != expected_sidecar))]
            if unexpected:
                raise RuntimeError(
                    "D41 receipt-bound sync includes a non-target recovery artifact: %s" %
                    unexpected[0])
        return paths

    def _telemetry_paths(self, out_root):
        """Return useful non-resume artifacts that may exist before the first checkpoint."""
        paths = set()
        for pattern in ("*/materialized_config.yaml", "*/version_*/metrics.csv",
                        "metrics/*.json", "evaluation/*.json", "evaluation/*.sha256"):
            paths.update(path for path in out_root.glob(pattern) if path.is_file())
        return paths

    def _representation_cache_snapshot(self, out_root):
        """Freeze and verify one state-last HMM extraction-cache generation.

        ``progress.json`` is the cache commit pointer.  Its bytes are captured before any
        upload, every immutable chunk and sidecar named by those bytes is verified, and the
        exact captured pointer is uploaded only after those dependencies.  A concurrent
        evaluator may advance to a later generation, but can never make this transaction
        advertise a chunk that was not durably uploaded.
        """
        cache_root = (out_root / "evaluation" / "representation_cache").resolve()
        progress_path = cache_root / "progress.json"
        if not progress_path.is_file():
            return None
        progress_payload = progress_path.read_bytes()
        try:
            progress = json.loads(progress_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("representation cache progress is invalid JSON") from exc
        if (progress.get("schema") != "nextlat_forgetting/hmm_representation_cache/1" or
                not isinstance(progress.get("chunks"), dict)):
            raise RuntimeError("representation cache progress has an invalid schema")
        identity_sha = progress.get("identity_sha256")
        if not isinstance(identity_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", identity_sha):
            raise RuntimeError("representation cache progress lacks a valid identity hash")

        paths = set()
        for key, record in progress["chunks"].items():
            if not isinstance(key, str) or not isinstance(record, dict):
                raise RuntimeError("representation cache contains an invalid chunk record")
            expected_sha = record.get("sha256")
            expected_sidecar_sha = record.get("sidecar_sha256")
            if (not isinstance(expected_sha, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or
                    not isinstance(expected_sidecar_sha, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", expected_sidecar_sha)):
                raise RuntimeError("representation cache chunk lacks valid hashes: %s" % key)
            chunk = self._local_path(record.get("path", ""))
            sidecar = self._local_path(record.get("sidecar", ""))
            for path in (chunk, sidecar):
                if os.path.commonpath((str(path), str(cache_root))) != str(cache_root):
                    raise RuntimeError("representation cache artifact escapes cache root: %s" % path)
                if not path.is_file():
                    raise RuntimeError("representation cache artifact is missing: %s" % path)
            if sha256_file(chunk) != expected_sha or sha256_file(sidecar) != expected_sidecar_sha:
                raise RuntimeError("representation cache chunk failed hash verification: %s" % key)
            sidecar_fields = sidecar.read_text(encoding="utf-8").strip().split()
            if not sidecar_fields or sidecar_fields[0].lower() != expected_sha:
                raise RuntimeError("representation cache sidecar disagrees with chunk: %s" % key)
            paths.update((chunk, sidecar))
        return {
            "progress_path": progress_path,
            "progress_payload": progress_payload,
            "identity_sha256": identity_sha,
            "n_chunks": len(progress["chunks"]),
            "paths": paths,
        }

    def _sync_telemetry(self, run_id, out_root):
        """Commit observed metrics and a coherent resumable evaluation cache."""
        artifacts = []
        for path in sorted(self._telemetry_paths(out_root)):
            relative = str(path.relative_to(out_root))
            artifacts.append(self._upload_snapshot_file(
                path, self._remote(run_id, "telemetry/" + relative)))
        cache = self._representation_cache_snapshot(out_root)
        cache_record = None
        if cache is not None:
            # Immutable chunks and sidecars cross the durability boundary first.
            for path in sorted(cache["paths"]):
                relative = str(path.relative_to(out_root))
                artifacts.append(self._upload_file(
                    path, self._remote(run_id, "telemetry/" + relative)))
            # The exact progress bytes observed above are the cache's state-last commit.
            progress_path = cache["progress_path"]
            relative = str(progress_path.relative_to(out_root))
            progress_artifact = self._upload_snapshot_payload(
                progress_path, cache["progress_payload"],
                self._remote(run_id, "telemetry/" + relative))
            artifacts.append(progress_artifact)
            cache_record = {
                "schema": "nextlat_forgetting/hmm_representation_cache/1",
                "identity_sha256": cache["identity_sha256"],
                "n_chunks": cache["n_chunks"],
                "progress": progress_artifact,
            }
        if not artifacts:
            return None
        payload = (json.dumps({
            "schema": "nextlat_forgetting/runtime_telemetry/1",
            "run_id": run_id,
            "source_snapshot_sha256": self.source_sha256,
            "resumable": cache_record is not None,
            "evaluation_cache": cache_record,
            "artifacts": artifacts,
            "synced_at": time.time(),
        }, indent=2, sort_keys=True) + "\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        blob = self.bucket.blob(self._remote(run_id, "telemetry/latest.json"))
        blob.metadata = dict(blob.metadata or {}, sha256=digest)
        blob.upload_from_string(payload, content_type="application/json")
        blob.reload()
        if blob.download_as_bytes() != payload or (blob.metadata or {}).get("sha256") != digest:
            raise RuntimeError("telemetry receipt verification failed for %s" % run_id)
        return json.loads(payload)

    def sync_job(self, entry):
        run_id = str(entry["job_id"])
        recovery_record = self._recovery_jobs.get(run_id)
        recovery_provenance = entry.get("recovery_provenance")
        if recovery_provenance is None:
            recovery_provenance = self._restored_provenance.get(run_id)
        out_root = self._local_path(entry["out_root"])
        pointer_targets = self._pointer_targets(out_root)
        if (not pointer_targets and entry.get("status") in {"TRAINED", "DONE"}
                and entry.get("final_checkpoint")):
            # The runner clears recovery_ckpt on completion. Upstream normally leaves
            # latest_ckpt, but the ledger's verified final checkpoint is the authoritative
            # fallback if it did not. Recreate the standard pointer atomically so a restored
            # DONE job still has a complete, self-contained durability receipt.
            target = self._local_path(entry["final_checkpoint"])
            expected_sha = entry.get("final_checkpoint_sha256")
            if (not target.is_file() or not expected_sha or
                    sha256_file(target) != expected_sha):
                raise RuntimeError("DONE ledger checkpoint does not verify: %s" % run_id)
            pointer = out_root / "latest_ckpt"
            partial = pointer.with_name(pointer.name + ".partial")
            partial.write_text(str(target))
            os.replace(partial, pointer)
            pointer_targets = [(pointer, target)]
        if not pointer_targets:
            if recovery_record is not None:
                raise RuntimeError("D41 receipt-bound sync lacks an exact terminal pointer")
            try:
                self._sync_telemetry(run_id, out_root)
            except Exception as exc:
                self.log("[telemetry] %s skipped: %s" % (run_id, exc))
            return None
        recovery_sync_contract = None
        if recovery_record is not None:
            recovery_sync_contract, pointer_targets, checkpoint = (
                self._d44_receipt_pinned_pointer_targets(
                    out_root, pointer_targets, recovery_record, recovery_provenance, run_id))
        else:
            # recovery_ckpt is authoritative upstream whenever it exists; otherwise latest_ckpt.
            authoritative_pointer, checkpoint = pointer_targets[0]
        authoritative_pointer = pointer_targets[0][0]
        artifacts = []
        paths = self._artifact_paths(
            out_root, pointer_targets, entry,
            receipt_pinned_checkpoint=(checkpoint if recovery_record is not None else None))
        path_digests = {path: sha256_file(path) for path in paths}
        if recovery_sync_contract is not None:
            allowed_sidecar_sha256 = recovery_sync_contract["allowed_sidecar_sha256"]
            for path, digest in path_digests.items():
                if (path.name.endswith(".pt.meta.json") and
                        digest not in allowed_sidecar_sha256):
                    raise RuntimeError(
                        "D41 receipt-bound sidecar lacks an immutable verified route: %s" %
                        path)
        # Upload checkpoint payloads before their pointers, and every artifact before state.
        for path in sorted(paths, key=lambda p: (p.name in ("recovery_ckpt", "latest_ckpt"),
                                                  str(p))):
            relative = str(path.relative_to(out_root))
            digest = path_digests[path]
            immutable_kind = None
            if recovery_record is not None:
                if (path.suffix == ".pt" and
                        digest == recovery_record["checkpoint_object"]["sha256"]):
                    immutable_kind = "checkpoint"
                elif path.name.endswith(".pt.meta.json"):
                    # The preflight above proved this digest is either the exact predecessor
                    # sidecar or its provenance-bound canonical successor. Never route a
                    # receipt-bound sidecar through the mutable object namespace.
                    immutable_kind = "sidecar"
            if immutable_kind is None:
                artifact = self._upload_file(path, self._remote(run_id, relative))
            else:
                remote = self._successor_recovery_remote(
                    run_id, immutable_kind, digest, path.name)
                artifact = self._publish_immutable_file(path, remote)
            artifacts.append(artifact)
        by_local = {artifact["local_path"]: artifact for artifact in artifacts}
        checkpoint_artifact = by_local[str(checkpoint)]
        checkpoint_sha = checkpoint_artifact["sha256"]
        checkpoint_size = checkpoint_artifact["size_bytes"]
        if (checkpoint.stat().st_size != checkpoint_size or
                sha256_file(checkpoint) != checkpoint_sha):
            raise RuntimeError("checkpoint changed before state commit: %s" % checkpoint)
        for pointer, target in pointer_targets:
            if pointer.read_text().strip() != str(target):
                raise RuntimeError("checkpoint pointer changed before state commit: %s" % pointer)
        recovery_candidates = []
        for candidate in sorted(
                (path for path in paths if path.suffix == ".pt"),
                key=checkpoint_step, reverse=True):
            artifact = by_local[str(candidate)]
            recovery_candidates.append({
                "path": str(candidate),
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
                "step": checkpoint_step(candidate),
                "metadata_path": str(candidate.with_name(candidate.name + ".meta.json")),
                "metadata_sha256": by_local[
                    str(candidate.with_name(candidate.name + ".meta.json"))]["sha256"],
                "generation": artifact.get("generation"),
                "metadata_generation": by_local[
                    str(candidate.with_name(candidate.name + ".meta.json"))].get("generation"),
            })
        state = {
            "schema": STATE_SCHEMA,
            "run_id": run_id,
            "status": entry.get("status"),
            "step": checkpoint_step(checkpoint),
            "out_root": str(out_root),
            "pointer": str(authoritative_pointer),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "size_bytes": checkpoint_size,
                "generation": checkpoint_artifact.get("generation"),
            },
            "recovery_candidates": recovery_candidates,
            "artifacts": artifacts,
            "source_snapshot_sha256": self.source_sha256,
            "runtime_fingerprint": self.runtime_fingerprint,
            "synced_at": time.time(),
        }
        if recovery_record is not None:
            expected_checkpoint = recovery_sync_contract["expected_checkpoint"]
            expected_sidecar = recovery_sync_contract["expected_sidecar"]
            normalization = recovery_sync_contract["normalization"]
            expected_successor_sidecar_sha = recovery_sync_contract[
                "expected_successor_sidecar_sha"]
            expected_successor_sidecar_size = recovery_sync_contract[
                "expected_successor_sidecar_size"]
            checkpoint_sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
            sidecar_artifact = by_local.get(str(checkpoint_sidecar))
            if (checkpoint_sha != expected_checkpoint["sha256"] or
                    sidecar_artifact is None or
                    sidecar_artifact.get("sha256") != expected_successor_sidecar_sha or
                    int(sidecar_artifact.get("size_bytes", -1)) !=
                    expected_successor_sidecar_size):
                raise RuntimeError("D41 successor sync changed receipt-bound checkpoint bytes")
            immutable_checkpoint = self._artifact_binding(checkpoint_artifact)
            immutable_sidecar = self._artifact_binding(sidecar_artifact)
            for key, current in (
                ("successor_checkpoint_object", immutable_checkpoint),
                ("successor_sidecar_object", immutable_sidecar),
            ):
                prior = recovery_provenance.get(key)
                if prior is not None and prior != current:
                    raise RuntimeError(
                        "D41 immutable successor generation changed across retry: %s" % key)
            recovery_provenance = dict(recovery_provenance)
            recovery_provenance.update({
                "predecessor_checkpoint_object": dict(expected_checkpoint),
                "predecessor_sidecar_object": dict(expected_sidecar),
                "successor_checkpoint_object": immutable_checkpoint,
                "successor_sidecar_object": immutable_sidecar,
            })
            self._restored_provenance[run_id] = recovery_provenance
        if recovery_provenance is not None:
            state["recovery_provenance"] = recovery_provenance
        self._upload_state(run_id, state)
        self.log("[sync] %s step=%d sha=%s durable" %
                 (run_id, state["step"], checkpoint_sha[:12]))
        # Telemetry is valuable but optional. It runs only after the authoritative checkpoint
        # commit and cannot prevent that state from advancing.
        try:
            self._sync_telemetry(run_id, out_root)
        except Exception as exc:
            self.log("[telemetry] %s skipped after checkpoint commit: %s" % (run_id, exc))
        return state

    def sync_once(self, ledger_path, ledger_object="run_ledger.json"):
        """Synchronize every started job plus the append-only ledger."""
        with self._lock:
            ledger_path = pathlib.Path(ledger_path)
            if not ledger_path.is_file():
                return {}
            document = json.loads(ledger_path.read_text())
            states = {}
            for run_id, entry in latest_ledger_entries(document).items():
                if entry.get("status") != "PENDING" and entry.get("out_root"):
                    state = self.sync_job(entry)
                    if state is not None:
                        states[run_id] = state
            if pathlib.PurePosixPath(ledger_object).name != ledger_object:
                raise RuntimeError("ledger object must be one filename")
            self._upload_file(ledger_path, "%s/%s" % (self.prefix, ledger_object))
            return states

    def restore(self):
        """Restore every committed run state to its recorded absolute runtime paths."""
        restored = {}
        state_prefix = "%s/runs/" % self.prefix
        state_blobs = [blob for blob in self.bucket.list_blobs(prefix=state_prefix)
                       if blob.name.endswith("/state.json")]
        for state_blob in state_blobs:
            state_payload = state_blob.download_as_bytes()
            state_blob.reload()
            state_sha = hashlib.sha256(state_payload).hexdigest()
            if (state_blob.metadata or {}).get("sha256") != state_sha:
                raise RuntimeError("state.json hash metadata mismatch: %s" % state_blob.name)
            state = json.loads(state_payload)
            if state.get("schema") != STATE_SCHEMA:
                continue
            run_id = state["run_id"]
            recovery_record = self._recovery_jobs.get(run_id)
            if recovery_record is not None:
                if (recovery_record.get("target_step") != 3000 or
                        recovery_record.get("regime") != "persistent_moderate" or
                        recovery_record.get("predecessor_source_sha256") !=
                        self.predecessor_source_sha256):
                    raise RuntimeError("D41 recovery identity mismatch for %s" % run_id)
                checks = recovery_record.get("verification")
                required_checks = (
                    "state_trained_exact_target", "checkpoint_bytes_sha256_verified",
                    "sidecar_bytes_sha256_verified", "sidecar_binds_checkpoint",
                    "source_identity_verified", "payload_training_steps_verified",
                )
                if (not isinstance(checks, dict) or set(checks) != set(required_checks) or
                        any(checks.get(key) is not True for key in required_checks)):
                    raise RuntimeError("D41 recovery verification receipt failed for %s" % run_id)
            restored_source = state.get("source_snapshot_sha256")
            source_migrated = self._accept_restore_source(
                restored_source, "run state %s" % state.get("run_id", state_blob.name))
            if source_migrated and recovery_record is None:
                raise RuntimeError(
                    "predecessor state outside the clearance-bound exact ten: %s" % run_id)
            if recovery_record is not None and (
                    state.get("status") not in {"TRAINED", "DONE"} or
                    int(state.get("step", -1)) != int(recovery_record["target_step"])):
                raise RuntimeError("D41 state is not terminal at the exact target for %s" % run_id)
            if recovery_record is not None and source_migrated:
                state_binding = recovery_record.get("state_object")
                state_name = self._binding_name(state_binding, "%s state" % run_id)
                if (state_blob.name != state_name or
                        str(getattr(state_blob, "generation", None)) !=
                        str(state_binding["generation"]) or
                        len(state_payload) != int(state_binding["size_bytes"]) or
                        state_sha != state_binding["sha256"]):
                    raise RuntimeError("D41 predecessor state generation mismatch for %s" % run_id)
            elif recovery_record is not None:
                prior_provenance = state.get("recovery_provenance")
                if (not isinstance(prior_provenance, dict) or
                        prior_provenance.get("checkpoint_creation_source_sha256") !=
                        self.predecessor_source_sha256 or
                        prior_provenance.get("successor_terminalization_source_sha256") !=
                        self.source_sha256 or
                        prior_provenance.get("recovery_receipt_sha256") !=
                        self.recovery_receipt_sha256):
                    raise RuntimeError("current-source D41 state lost recovery provenance")
            artifacts = list(state.get("artifacts", []))
            by_local = {item["local_path"]: item for item in artifacts}
            candidates = list(state.get("recovery_candidates") or [state["checkpoint"]])
            recovery_checkpoint_binding = None
            recovery_sidecar_binding = None
            if recovery_record is not None:
                def generation_matches(observed, expected):
                    # Frozen a962 state predates generation fields.  Only a migrated state may
                    # omit them; any value it does carry, and every successor value, is exact.
                    return ((source_migrated and observed is None) or
                            str(observed) == str(expected))

                prior_provenance = state.get("recovery_provenance")
                if source_migrated:
                    recovery_checkpoint_binding = recovery_record.get("checkpoint_object")
                    recovery_sidecar_binding = recovery_record.get("sidecar_object")
                else:
                    recovery_checkpoint_binding = prior_provenance.get(
                        "successor_checkpoint_object")
                    recovery_sidecar_binding = prior_provenance.get(
                        "successor_sidecar_object")
                    for label, successor, predecessor in (
                        ("checkpoint", recovery_checkpoint_binding,
                         recovery_record.get("checkpoint_object")),
                        ("sidecar", recovery_sidecar_binding,
                         recovery_record.get("sidecar_object")),
                    ):
                        successor_name = self._binding_name(
                            successor, "%s successor %s" % (run_id, label))
                        expected_prefix = (
                            "%s/runs/%s/successor_recovery/%s/%s/%s/" %
                            (self.prefix, run_id, self.source_sha256, label,
                             successor.get("sha256")))
                        expected_successor_sha = predecessor.get("sha256")
                        expected_successor_size = int(predecessor.get("size_bytes", -2))
                        if label == "sidecar":
                            normalization = prior_provenance.get("sidecar_normalization")
                            if normalization is not None:
                                if (not isinstance(normalization, dict) or
                                        normalization.get("from_sha256") !=
                                        predecessor.get("sha256") or
                                        int(normalization.get("from_size_bytes", -1)) !=
                                        int(predecessor.get("size_bytes", -2)) or
                                        normalization.get("from_generation") !=
                                        str(predecessor.get("generation")) or
                                        normalization.get("training_steps") !=
                                        int(recovery_record["target_step"]) or
                                        normalization.get("preserved_step") !=
                                        int(recovery_record["target_step"])):
                                    raise RuntimeError(
                                        "D41 successor sidecar normalization is invalid for %s" %
                                        run_id)
                                expected_successor_sha = normalization.get("to_sha256")
                                expected_successor_size = int(
                                    normalization.get("to_size_bytes", -1))
                        if (successor.get("sha256") != expected_successor_sha or
                                int(successor.get("size_bytes", -1)) !=
                                expected_successor_size or
                                not successor_name.startswith(expected_prefix)):
                            raise RuntimeError(
                                "D41 successor %s binding differs from predecessor receipt for %s" %
                                (label, run_id))
                checkpoint_binding = recovery_checkpoint_binding
                checkpoint_name = self._binding_name(
                    checkpoint_binding, "%s checkpoint" % run_id)
                candidates = [candidate for candidate in candidates if (
                    candidate.get("sha256") == checkpoint_binding["sha256"] and
                    int(candidate.get("step", -1)) == int(recovery_record["target_step"]) and
                    by_local.get(candidate.get("path"), {}).get("remote") == checkpoint_name and
                    generation_matches(
                        candidate.get("generation"), checkpoint_binding["generation"]) and
                    generation_matches(
                        by_local.get(candidate.get("path"), {}).get("generation"),
                        checkpoint_binding["generation"])
                )]
                if len(candidates) != 1:
                    raise RuntimeError("D41 exact checkpoint is absent or ambiguous for %s" % run_id)
                committed_checkpoint = state.get("checkpoint")
                if (isinstance(committed_checkpoint, dict) and
                        committed_checkpoint.get("path") == candidates[0].get("path") and
                        not generation_matches(
                            committed_checkpoint.get("generation"),
                            checkpoint_binding["generation"])):
                    raise RuntimeError("D41 committed checkpoint generation mismatch for %s" %
                                       run_id)
            candidate_paths = {item["path"] for item in candidates}
            candidate_meta_paths = {
                item.get("metadata_path", item["path"] + ".meta.json") for item in candidates
            }

            def download_artifact(artifact):
                local = self._local_path(artifact["local_path"])
                local.parent.mkdir(parents=True, exist_ok=True)
                partial = local.with_name(local.name + ".partial")
                blob = self.bucket.blob(artifact["remote"])
                blob.download_to_filename(str(partial))
                if (partial.stat().st_size != int(artifact["size_bytes"]) or
                        sha256_file(partial) != artifact["sha256"]):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("restored artifact failed verification: %s" % local)
                os.replace(partial, local)

            # Restore non-checkpoint artifacts strictly. Pointers and checkpoint generations are
            # handled below so a corrupt newest generation can fall back to the older verified one.
            skipped_regenerable = []
            predecessor_durable_index = (
                self._local_path(state["out_root"]) / "durable_index.json").resolve()
            for artifact in artifacts:
                local_name = artifact["local_path"]
                if (local_name in candidate_paths or local_name in candidate_meta_paths or
                        pathlib.Path(local_name).name in ("recovery_ckpt", "latest_ckpt")):
                    continue
                if (recovery_record is not None and source_migrated and
                        self._local_path(local_name) == predecessor_durable_index and
                        artifact.get("remote") == self._remote(run_id, "durable_index.json")):
                    # The a962 state captured this live operational index immediately before a
                    # later retry rewrote the same remote name.  It is neither scientific output
                    # nor checkpoint evidence.  Leave it absent: the successor checkpointer
                    # deterministically rebuilds it by adopting the exact receipt-pinned target.
                    skipped_regenerable.append({
                        "local_path": local_name,
                        "remote": artifact.get("remote"),
                        "recorded_sha256": artifact.get("sha256"),
                        "recorded_size_bytes": artifact.get("size_bytes"),
                        "reason": "mutable predecessor durability index; regenerate",
                    })
                    predecessor_durable_index.unlink(missing_ok=True)
                    continue
                download_artifact(artifact)

            checkpoint_path = None
            recovery_errors = []
            for candidate in sorted(candidates, key=lambda item: int(item.get("step", 0)),
                                    reverse=True):
                local = self._local_path(candidate["path"])
                meta_local = candidate.get("metadata_path", candidate["path"] + ".meta.json")
                try:
                    if recovery_record is not None:
                        sidecar_binding = recovery_sidecar_binding
                        sidecar_name = self._binding_name(
                            sidecar_binding, "%s sidecar" % run_id)
                        sidecar_artifact = by_local.get(meta_local, {})
                        if (sidecar_artifact.get("remote") != sidecar_name or
                                not generation_matches(
                                    candidate.get("metadata_generation"),
                                    sidecar_binding["generation"]) or
                                not generation_matches(
                                    sidecar_artifact.get("generation"),
                                    sidecar_binding["generation"])):
                            raise RuntimeError("D41 sidecar object disagrees with state")
                        self._download_exact_binding(
                            sidecar_binding, meta_local, "%s sidecar" % run_id)
                        self._download_exact_binding(
                            recovery_checkpoint_binding, local,
                            "%s checkpoint" % run_id)
                    else:
                        if meta_local in by_local:
                            download_artifact(by_local[meta_local])
                        download_artifact(by_local[candidate["path"]])
                    if (local.stat().st_size != int(candidate["size_bytes"]) or
                            sha256_file(local) != candidate["sha256"]):
                        raise RuntimeError("candidate disagrees with committed state")
                    self._verify_checkpoint_for_sync(
                        local, expected_step=(recovery_record or {}).get("target_step"),
                        expected_run_id=(run_id if recovery_record is not None else None),
                        legacy_sidecar_binding=(
                            recovery_sidecar_binding if source_migrated else None))
                    if checkpoint_path is None:
                        checkpoint_path = local
                except Exception as exc:
                    local.unlink(missing_ok=True)
                    local.with_name(local.name + ".meta.json").unlink(missing_ok=True)
                    recovery_errors.append("%s: %s" % (local, exc))
            if checkpoint_path is None:
                raise RuntimeError("no verified recovery generation for %s (%s)" %
                                   (run_id, "; ".join(recovery_errors)))
            pointer = self._local_path(state["pointer"])
            pointer.parent.mkdir(parents=True, exist_ok=True)
            partial = pointer.with_name(pointer.name + ".partial")
            partial.write_text(str(checkpoint_path))
            os.replace(partial, pointer)
            restored_state = dict(state)
            restored_state["restored_checkpoint"] = str(checkpoint_path)
            restored_state["restored_step"] = checkpoint_step(checkpoint_path)
            restored_state["restore_provenance"] = {
                "from_source_snapshot_sha256": restored_source,
                "to_source_snapshot_sha256": self.source_sha256,
                "migrated": source_migrated,
            }
            if skipped_regenerable:
                restored_state["restore_provenance"][
                    "skipped_regenerable_predecessor_artifacts"] = skipped_regenerable
            if recovery_record is not None:
                if source_migrated:
                    provenance = {
                        "checkpoint_creation_source_sha256": self.predecessor_source_sha256,
                        "successor_terminalization_source_sha256": self.source_sha256,
                        "checkpoint_generation": str(
                            recovery_record["checkpoint_object"]["generation"]),
                        "checkpoint_sha256": recovery_record["checkpoint_object"]["sha256"],
                        "sidecar_generation": str(
                            recovery_record["sidecar_object"]["generation"]),
                        "sidecar_sha256": recovery_record["sidecar_object"]["sha256"],
                        "predecessor_checkpoint_object": dict(
                            recovery_record["checkpoint_object"]),
                        "predecessor_sidecar_object": dict(recovery_record["sidecar_object"]),
                        "recovery_receipt_sha256": self.recovery_receipt_sha256,
                        "runtime_fingerprint": self.runtime_fingerprint,
                    }
                    normalization = self._sidecar_normalizations.get(str(checkpoint_path))
                    if normalization is not None:
                        provenance["sidecar_normalization"] = dict(normalization)
                else:
                    # A later disconnect restores the exact immutable successor generation,
                    # while retaining the predecessor creation generations byte-for-byte.
                    provenance = dict(state["recovery_provenance"])
                restored_state["recovery_provenance"] = provenance
                self._restored_provenance[run_id] = provenance
            restored[run_id] = restored_state
            self.log("[restore] %s step=%s sha=%s verified" %
                     (run_id, restored_state["restored_step"], sha256_file(checkpoint_path)[:12]))
        # Evaluation progress and live metrics are committed independently of resume state.
        # Restore only the latest verified receipt for this exact source snapshot.
        telemetry_blobs = [blob for blob in self.bucket.list_blobs(prefix=state_prefix)
                           if blob.name.endswith("/telemetry/latest.json")]
        for receipt_blob in telemetry_blobs:
            payload = receipt_blob.download_as_bytes()
            receipt_blob.reload()
            digest = hashlib.sha256(payload).hexdigest()
            if (receipt_blob.metadata or {}).get("sha256") != digest:
                raise RuntimeError("telemetry receipt hash mismatch: %s" % receipt_blob.name)
            receipt = json.loads(payload)
            if receipt.get("schema") != "nextlat_forgetting/runtime_telemetry/1":
                continue
            telemetry_source = receipt.get("source_snapshot_sha256")
            # The exact terminal state, not a later retry logger, owns recovery completion.
            # In particular, the invalid target+1 retry's ``version_1/metrics.csv`` must remain
            # incident telemetry and can never enter successor completion artifacts.
            if receipt.get("run_id") in self._recovery_jobs:
                continue
            telemetry_migrated = self._accept_restore_source(
                telemetry_source, "telemetry %s" % receipt.get("run_id", receipt_blob.name))
            artifacts = list(receipt.get("artifacts", []))
            cache_record = receipt.get("evaluation_cache")
            progress_record = cache_record.get("progress") if isinstance(cache_record, dict) else None
            if progress_record is not None and progress_record not in artifacts:
                raise RuntimeError("evaluation cache progress is absent from telemetry receipt")
            # Restore the cache commit pointer last.  If any dependency download fails, an
            # older local progress generation remains authoritative and no partial generation
            # is advertised to the resumed evaluator.
            ordered_artifacts = sorted(
                artifacts, key=lambda artifact: artifact == progress_record)
            for artifact in ordered_artifacts:
                local = self._local_path(artifact["local_path"])
                local.parent.mkdir(parents=True, exist_ok=True)
                partial = local.with_name(local.name + ".partial")
                blob = self.bucket.blob(artifact["remote"])
                blob.download_to_filename(str(partial))
                if (partial.stat().st_size != int(artifact["size_bytes"]) or
                        sha256_file(partial) != artifact["sha256"]):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("restored telemetry failed verification: %s" % local)
                os.replace(partial, local)
            if progress_record is not None:
                progress_path = self._local_path(progress_record["local_path"])
                if progress_path.name != "progress.json":
                    raise RuntimeError("evaluation cache commit pointer has an invalid path")
                out_root = progress_path.parent.parent.parent
                verified = self._representation_cache_snapshot(out_root)
                if (verified is None or
                        hashlib.sha256(verified["progress_payload"]).hexdigest() !=
                        progress_record["sha256"] or
                        verified["identity_sha256"] != cache_record.get("identity_sha256") or
                        verified["n_chunks"] != int(cache_record.get("n_chunks", -1))):
                    raise RuntimeError("restored evaluation cache failed state-last verification")
            if telemetry_migrated and receipt.get("run_id") in restored:
                restored[receipt["run_id"]]["telemetry_restore_provenance"] = {
                    "from_source_snapshot_sha256": telemetry_source,
                    "to_source_snapshot_sha256": self.source_sha256,
                    "migrated": True,
                }
        if self._recovery_jobs:
            missing = sorted(set(self._recovery_jobs) - set(restored))
            if missing:
                raise RuntimeError(
                    "D41 atomic recovery barrier missing jobs: %s" % ", ".join(missing))
        return restored


def apply_runtime_bootstrap(project_root, upstream_root):
    """Apply the required packaged project-local integration to the fresh upstream clone.

    Runtime-only integrations must live outside the pinned ``upstream/`` checkout.  The
    bootstrap installs the checkpoint and optimizer compatibility patches on which safe resume
    depends, so a source package without it is invalid and must fail before training.
    """
    bootstrap = os.path.join(project_root, "scripts", "runtime_bootstrap.py")
    if not os.path.isfile(bootstrap):
        raise SystemExit("required scripts/runtime_bootstrap.py is absent; refusing to train")
    applied_receipt = pathlib.Path(upstream_root) / ".lurestar_runtime_patch_receipt.json"
    audit_receipt = (pathlib.Path(project_root) / "source_snapshot" / "runtime_patch" /
                     "runtime_patch_receipt.json")
    if applied_receipt.exists():
        raise SystemExit(
            "fresh pinned upstream unexpectedly has a runtime-patch receipt before bootstrap")
    started_at = time.time()
    cmd = "%s %s --project-root %s --upstream %s" % tuple(
        shlex.quote(os.fspath(value))
        for value in (sys.executable, bootstrap, project_root, upstream_root)
    )
    sh(cmd, cwd=project_root)
    completed_at = time.time()
    if not applied_receipt.is_file() or not audit_receipt.is_file():
        raise SystemExit("runtime bootstrap did not emit both applied and audit receipts")
    applied_payload = applied_receipt.read_bytes()
    if audit_receipt.read_bytes() != applied_payload:
        raise SystemExit(
            "runtime bootstrap audit receipt differs from the receipt emitted by this runtime")
    try:
        document = json.loads(applied_payload)
    except json.JSONDecodeError as exc:
        raise SystemExit("runtime bootstrap emitted an invalid receipt: %s" % exc)
    generated_at = document.get("generated_at_unix")
    if (document.get("schema") != "nextlat_forgetting/runtime_patch/1" or
            document.get("patch_version") != 5 or
            document.get("upstream_commit") != PINNED or
            isinstance(generated_at, bool) or not isinstance(generated_at, (int, float)) or
            not (started_at - 1.0 <= float(generated_at) <= completed_at + 1.0)):
        raise SystemExit(
            "runtime bootstrap receipt was not freshly emitted for this pinned invocation")
    digest = hashlib.sha256(applied_payload).hexdigest()
    return {
        "applied_receipt_path": str(applied_receipt),
        "audit_receipt_path": str(audit_receipt),
        "applied_receipt_sha256": digest,
        "audit_receipt_sha256": digest,
        "generated_at_unix": float(generated_at),
    }


# --------------------------------------------------------------------------- DRIVER

def download_required_input_corpora(durability, data_root, input_bundle_prefix, dispatch):
    """Download only the input-bundle corpus required by validated dispatch semantics.

    The confirmatory HMM runner is family-only.  The superseded single-HMM corpus is not a
    fallback and must never become a hidden runtime dependency.  Lure-Star base training uses
    the separately frozen StarGraph corpus, so it must not download either HMM corpus.
    """
    if dispatch.get("runner") == "lurestar":
        only = dispatch.get("only")
        if (dispatch.get("phase") != "base" or not isinstance(only, tuple) or
                any(job_id not in FROZEN_BASE_JOB_IDS for job_id in only)):
            raise SystemExit("runtime corpus selection received an unvalidated dispatch")
        return {}
    if (dispatch.get("runner") != "hmm" or dispatch.get("family") is not True or
            dispatch.get("phase") not in {"train", "evaluate"}):
        raise SystemExit("runtime corpus selection received an unvalidated dispatch")

    data_root = pathlib.Path(data_root)
    remote_prefix = input_bundle_prefix + "/corpus/hmm_family"
    local_root = data_root / "hmm_family"
    local_root.mkdir(parents=True, exist_ok=True)
    if durability.download_prefix(remote_prefix, str(local_root)) == 0:
        raise SystemExit("frozen HMM-family arrays are absent from GCS")
    return {remote_prefix: str(local_root)}


def verify_runtime_input_subset(inventory, runtime_root, dispatch):
    """Verify every locally required inventory object for the validated runner.

    The host already verifies *all* receipt objects at their exact generations before paid
    provisioning.  At runtime we additionally verify the manifests shared by both runners and
    exactly the corpus downloaded for this dispatch.  Irrelevant corpus domains remain bound by
    the inventory hash and host receipt without becoming runtime download requirements.
    """
    inventory = pathlib.Path(inventory)
    runtime_root = pathlib.Path(runtime_root).resolve()
    if not inventory.is_file():
        raise SystemExit("manifest inventory is absent from GCS")
    runner = dispatch.get("runner")
    if runner not in {"lurestar", "hmm"}:
        raise SystemExit("runtime inventory verification received an unvalidated dispatch")
    if runner == "lurestar":
        only = dispatch.get("only")
        if (dispatch.get("phase") != "base" or not isinstance(only, tuple) or
                any(job_id not in FROZEN_BASE_JOB_IDS for job_id in only)):
            raise SystemExit("runtime inventory verification received an unvalidated dispatch")
    required_prefixes = ("manifests/",)
    if runner == "hmm":
        if (dispatch.get("family") is not True or
                dispatch.get("phase") not in {"train", "evaluate"}):
            raise SystemExit("runtime inventory verification received an unvalidated dispatch")
        required_prefixes += ("data/hmm_family/",)

    line_re = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
    verified = {prefix: 0 for prefix in required_prefixes}
    for number, line in enumerate(inventory.read_text(encoding="utf-8").splitlines(), 1):
        match = line_re.fullmatch(line)
        if match is None:
            raise SystemExit("malformed manifest inventory line %d" % number)
        digest, relative = match.groups()
        pure = pathlib.PurePosixPath(relative)
        if (pure.is_absolute() or str(pure) != relative or "\\" in relative or
                any(part in ("", ".", "..") for part in pure.parts)):
            raise SystemExit("unsafe manifest inventory path: %s" % relative)
        if not relative.startswith(("manifests/", "data/hmm/", "data/hmm_family/")):
            raise SystemExit("unexpected manifest inventory domain: %s" % relative)
        required_prefix = next(
            (prefix for prefix in required_prefixes if relative.startswith(prefix)), None)
        if required_prefix is None:
            continue
        local = runtime_root.joinpath(*pure.parts)
        try:
            resolved = local.resolve(strict=True)
            resolved.relative_to(runtime_root)
        except (OSError, ValueError):
            raise SystemExit("required runtime input is absent or escapes root: %s" % relative)
        if not local.is_file() or local.is_symlink() or sha256_file(local) != digest:
            raise SystemExit("required runtime input hash mismatch: %s" % relative)
        verified[required_prefix] += 1
    if verified["manifests/"] == 0:
        raise SystemExit("frozen manifest inventory contains no manifest objects")
    if runner == "hmm" and verified["data/hmm_family/"] == 0:
        raise SystemExit("frozen manifest inventory contains no HMM-family arrays")
    return verified


def driver():
    print("=== DRIVER role ===", flush=True)
    spec = json.load(open(SPEC_PATH)) if os.path.exists(SPEC_PATH) else {}
    dispatch = validate_confirmatory_job_spec(
        spec, runtime_overlay=True, require_session=True)
    print("job spec: %s" % json.dumps(spec), flush=True)
    session_id = spec.get("session_id")
    if not isinstance(session_id, str) or not session_id.startswith("gpu-"):
        raise SystemExit("job sidecar has no owned Colab session identity")

    stop = threading.Event()

    def heartbeat_loop():
        started = time.time()
        while not stop.wait(HEARTBEAT_SECONDS):
            print("[heartbeat] runtime alive elapsed_s=%d" % (time.time() - started), flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    adc_path = "/content/adc.json"
    secure_adc(adc_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path
    os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT
    requested_gpu = spec.get("gpu", "a100")
    initial_gpu = verify_runtime_gpu(requested_gpu)
    original_torch = initial_gpu["torch_version"]
    sh("pip -q install google-cloud-storage google-auth", quiet=True)
    from google.cloud import storage

    root = "/content/lurestar"
    proj = "/content/project"
    os.makedirs(root, exist_ok=True)
    source_object = spec.get("source_object")
    source_sha256 = spec.get("source_sha256")
    if not source_object or not re.fullmatch(r"lurestar/source/project-[0-9a-f]{64}\.tar\.gz",
                                              str(source_object)):
        raise SystemExit("job sidecar has no immutable source object")
    if source_sha256 not in str(source_object):
        raise SystemExit("job sidecar source object/hash mismatch")
    input_bundle_sha256 = spec.get("input_bundle_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(input_bundle_sha256)):
        raise SystemExit("job sidecar has no content-addressed input bundle")
    input_bundle_prefix = "%s/input_bundles/%s" % (PREFIX, input_bundle_sha256)
    if spec.get("input_bundle_prefix") != input_bundle_prefix:
        raise SystemExit("job sidecar input bundle prefix/hash mismatch")
    bucket = storage.Client(project=GCP_PROJECT).bucket(BUCKET)
    recovery_receipt = download_runtime_recovery_receipt(
        bucket, spec, "/content/d41-exact-ten-recovery-receipt.json")
    durability = RuntimeDurability(
        bucket, root, PREFIX,
        source_sha256=source_sha256,
        predecessor_source_sha256=spec.get("predecessor_source_sha256"),
        recovery_receipt=recovery_receipt,
        recovery_receipt_sha256=spec.get("recovery_receipt_sha256"))

    print("=== pull project ===", flush=True)
    durability.download_file(source_object, "/content/project.tar.gz")
    if sha256_file("/content/project.tar.gz") != source_sha256:
        raise SystemExit("immutable source snapshot hash mismatch")
    install_source_snapshot("/content/project.tar.gz", proj)
    runtime_src = os.path.join(proj, "src")
    if runtime_src not in sys.path:
        sys.path.insert(0, runtime_src)
    packaged_inventory = os.path.join(proj, "manifests", "manifest_inventory.sha256")
    if (not os.path.isfile(packaged_inventory) or
            sha256_file(packaged_inventory) != input_bundle_sha256):
        raise SystemExit("source snapshot/input inventory binding mismatch")

    print("=== pull upstream at the pinned commit ===", flush=True)
    up = os.path.join(proj, "upstream", "NextLat")
    if not os.path.isdir(os.path.join(up, ".git")):
        sh("git clone -q %s %s" % (UPSTREAM_URL, up))
    sh("cd %s && git checkout -q %s && git rev-parse HEAD" % (up, PINNED))
    # The pinned trainer imports every datamodule at module load. Install its dependency surface,
    # but retain Colab's known-good CUDA torch rather than letting pip replace it.
    sh("pip -q install -r <(grep -v '^torch' requirements.txt)", cwd=up, quiet=True)
    verify_runtime_gpu(requested_gpu, expected_torch_version=original_torch)
    applied_patch = apply_runtime_bootstrap(proj, up)
    final_gpu = verify_runtime_gpu(requested_gpu, expected_torch_version=original_torch)
    runtime_contract = {
        "device_name": final_gpu["name"],
        "torch_version": final_gpu["torch_version"],
        "cuda_version": final_gpu["cuda"],
        "bf16_supported": final_gpu["bf16"],
        "pinned_upstream_commit": PINNED,
    }
    runtime_fingerprint = verify_d41_runtime_equivalence(
        recovery_receipt, runtime_contract, project_root=proj,
        patch_receipt_path=applied_patch["applied_receipt_path"],
        audit_patch_receipt_path=applied_patch["audit_receipt_path"])
    if recovery_receipt is not None and (
            runtime_fingerprint.get("runtime_patch_applied_receipt_sha256") !=
            applied_patch["applied_receipt_sha256"] or
            runtime_fingerprint.get("runtime_patch_audit_receipt_sha256") !=
            applied_patch["audit_receipt_sha256"]):
        raise SystemExit("D41 runtime fingerprint lost the current bootstrap receipt binding")
    durability.runtime_fingerprint = runtime_fingerprint
    print("D41_RUNTIME_FINGERPRINT=%s" % json.dumps(
        runtime_fingerprint, sort_keys=True), flush=True)

    print("=== pull frozen manifests and runner-required corpus ===", flush=True)
    manifest_dir = os.path.join(root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    if durability.download_prefix(input_bundle_prefix + "/manifests", manifest_dir) == 0:
        raise SystemExit("frozen manifest set is absent from GCS")
    inventory = os.path.join(manifest_dir, "manifest_inventory.sha256")
    if not os.path.isfile(inventory):
        raise SystemExit("manifest inventory is absent from GCS")
    if sha256_file(inventory) != input_bundle_sha256:
        raise SystemExit("downloaded manifest inventory/input bundle mismatch")

    data_dir = os.path.join(root, "data", "stargraph")
    if dispatch["runner"] == "lurestar":
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(os.path.join(data_dir, "graph_5_5_sample_200000.txt")):
            count = durability.download_prefix(PREFIX + "/corpus/stargraph", data_dir)
            if count == 0:
                raise SystemExit("immutable stargraph corpus is absent from GCS")
        corpus_hashes = os.path.join(manifest_dir, "corpus.sha256")
        if not os.path.isfile(corpus_hashes):
            raise SystemExit("frozen StarGraph corpus hash manifest is absent")
        # The frozen manifest's third column records byte size; sha256sum consumes the first
        # two columns only.  Preserve the historical normalization while sourcing the manifest
        # from the immutable input bundle rather than a mutable convenience object.
        rc, _ = sh("sha256sum -c --ignore-missing <(awk '{print $1\"  \"$2}' %s)" %
                   shlex.quote(corpus_hashes), cwd=data_dir, check=False)
        print("CORPUS_HASH_VERIFIED=%s" % (rc == 0), flush=True)
        if rc != 0:
            raise SystemExit("corpus hash mismatch -- refusing to train on unverified data")
    download_required_input_corpora(
        durability, os.path.join(root, "data"), input_bundle_prefix, dispatch)
    verified_inputs = verify_runtime_input_subset(inventory, root, dispatch)
    print("RUNTIME_INPUT_SUBSET_VERIFIED=%s" % json.dumps(
        verified_inputs, sort_keys=True), flush=True)

    print("=== pull prior run state (this is what makes a re-exec a resume) ===", flush=True)
    restored = durability.restore()
    runner = dispatch["runner"]
    ledger_object = "hmm_run_ledger.json" if runner == "hmm" else "run_ledger.json"
    ledger = os.path.join(root, ledger_object)
    durability.download_file(PREFIX + "/" + ledger_object, ledger, required=False)
    recovery_barrier = None
    if spec.get("predecessor_source_sha256") is not None:
        if runner != "hmm" or dispatch["phase"] != "train":
            raise SystemExit("predecessor migration reached a non-HMM-training driver")
        archived = archive_predecessor_hmm_ledger(
            durability, ledger, spec["predecessor_source_sha256"])
        print("ARCHIVED_PREDECESSOR_HMM_LEDGER=%s" % archived, flush=True)
        recovery_barrier = write_runtime_recovery_barrier(root, restored, spec)
        print("D41_ATOMIC_RESTORE_BARRIER=PASS JOBS=10", flush=True)

    sync_abort = threading.Event()
    sync_diagnostic = os.path.join(root, SYNC_FAILURE_DIAGNOSTIC)
    sync_thread = threading.Thread(
        target=durable_sync_loop,
        args=(stop, sync_abort, durability, ledger, ledger_object, sync_diagnostic),
        kwargs={
            "source_sha256": source_sha256,
            "predecessor_source_sha256": spec.get("predecessor_source_sha256"),
        },
        daemon=True,
    )
    sync_thread.start()

    if runner == "hmm":
        phase = dispatch["phase"]
        args = ["--phase", phase]
        cmd_parts = [
            sys.executable, os.path.join(proj, "scripts", "run_hmm_matrix.py"),
            "--root", root, "--project-root", proj, "--snapshot-root", root,
            "--data-root", root, "--upstream", up, "--ledger", ledger,
            "--phase", phase, "--family", "--driver-managed-durability",
        ]
        cmd = " ".join(shlex.quote(str(value)) for value in cmd_parts)
    else:
        args = spec.get("run_matrix_args", ["--phase", "base"])
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            raise SystemExit("run_matrix_args must be a JSON string list")
        cmd_parts = [
            sys.executable, os.path.join(proj, "scripts", "run_matrix.py"),
            "--root", root, "--ledger", ledger, "--upstream", up, *args,
        ]
        cmd = " ".join(shlex.quote(str(value)) for value in cmd_parts)
    phase = requested_phase(args)
    if runner == "lurestar" and phase == "base":
        rc = run_lurestar_base_stages(
            project=proj, root=root, ledger=ledger, upstream=up, args=args,
            data_dir=data_dir, manifest_dir=manifest_dir, durability=durability,
            abort_event=sync_abort)
    else:
        if runner == "hmm" and recovery_barrier is not None:
            try:
                run_hmm_recovery_terminalization_stage(
                    cmd_parts, recovery_barrier, spec["recovery_job_ids"],
                    durability, ledger,
                    command_runner=lambda command, **kwargs: sh(
                        command, abort_event=sync_abort, **kwargs))
            except SystemExit:
                if not sync_abort.is_set():
                    raise
        if sync_abort.is_set():
            rc = 74
        else:
            rc, _ = sh(cmd, check=False, quiet=False, abort_event=sync_abort)
    if sync_abort.is_set() and rc == 0:
        rc = 74
    print("RUN_MATRIX_RC=%d" % rc, flush=True)

    # Base checkpoints are not scientifically usable merely because training returned.  Bind
    # deterministic greedy competence receipts and promote TRAINED -> DONE before terminal.
    # Base evaluation is stage-local above. A completed checkpoint is never followed by another
    # paid training stage until its deterministic competence receipt is DONE and durable.

    stop.set()
    sync_thread.join(timeout=30)
    heartbeat_thread.join(timeout=5)
    print("=== final durable sync ===", flush=True)
    final_sync_ok = True
    training_complete = False
    final_states = {}
    try:
        final_states = durability.sync_once(ledger, ledger_object=ledger_object)
    except Exception as exc:
        final_sync_ok = False
        print("FINAL_SYNC_FAILED=%s" % exc, flush=True)
    if os.path.isfile(sync_diagnostic):
        try:
            durability._upload_file(
                sync_diagnostic,
                "%s/control/%s/%s" % (PREFIX, session_id, SYNC_FAILURE_DIAGNOSTIC))
            print("SYNC_FAILURE_DIAGNOSTIC_DURABLE=True", flush=True)
        except Exception as exc:
            final_sync_ok = False
            print("SYNC_FAILURE_DIAGNOSTIC_UPLOAD_FAILED=%s" % exc, flush=True)
    if os.path.exists(ledger):
        st = json.load(open(ledger))
        terminal_statuses = ({"DONE"} if (runner == "lurestar" and phase == "base") or
                             (runner == "hmm" and phase == "evaluate")
                             else {"TRAINED", "DONE"})
        terminal, total = ledger_progress(st, terminal_statuses=terminal_statuses)
        expected_states = state_required_jobs(st)
        if not expected_states.issubset(final_states):
            final_sync_ok = False
            missing_states = sorted(expected_states - set(final_states))
            print("FINAL_SYNC_MISSING_STATES=%s" % ",".join(missing_states), flush=True)
        print("LEDGER_TRAINING_TERMINAL=%d LEDGER_TOTAL=%d" % (terminal, total), flush=True)
        training_complete = (not sync_abort.is_set() and rc == 0 and final_sync_ok and
                             terminal == total and total > 0)
        print("TRAINING_COMPLETE=%s" % training_complete, flush=True)
    if final_sync_ok:
        durability.publish_terminal(
            session_id, returncode=rc, training_complete=training_complete)
    print("=== DRIVER DONE ===", flush=True)
    return int(rc)


# ----------------------------------------------------------------------------- LOOP

def package(project_root):
    out = os.path.join(project_root, ".agent_state", "project.tar.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    skip_dirs = {".venv", ".git", "__pycache__", "data", "output", "results", ".secrets",
                 ".agent_state", "report", "upstream"}
    skip_files = {"HANDOFF.md"}  # operational status changes must not invalidate a live run

    def flt(ti):
        parts = ti.name.split("/")
        if any(p in skip_dirs for p in parts):
            return None
        # Runtime source archives are not a credential transport.  A top-level dotfile is
        # already skipped below, but nested dotenv/ADC files must be rejected as well.
        basename = parts[-1]
        if (basename == ".env" or basename.startswith(".env.") or
                basename in {"adc.json", "application_default_credentials.json"}):
            return None
        # Links and special files make extraction depend on host filesystem state and can
        # escape the destination even when the member name itself is harmless.
        if not (ti.isfile() or ti.isdir()):
            return None
        if ti.name.endswith((".pt", ".ckpt", ".tar.gz")):
            return None
        # Source identity must survive host restarts.  Normalize every archive field that
        # otherwise depends on the local user, filesystem, or wall clock.
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        ti.mtime = 0
        ti.pax_headers = {}
        return ti

    # tarfile's ``w:gz`` embeds the current time in the gzip header.  Supplying the gzip
    # layer ourselves makes two identical trees byte-identical as well as content-equivalent.
    with open(out, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for name in sorted(os.listdir(project_root)):
                    if name in skip_dirs or name in skip_files or name.startswith("."):
                        continue
                    tf.add(os.path.join(project_root, name), arcname=name, filter=flt)
    return out


def safe_extract_tar(archive, destination):
    """Extract the immutable source snapshot without traversal or link semantics."""
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for member in members:
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError("source archive contains a link or special file: %s" %
                                   member.name)
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError("source archive contains path traversal: %s" % member.name)
        # Members were explicitly restricted to contained regular files/directories above.
        # Avoid tarfile's newer ``filter=`` keyword so the host-side verifier also runs on
        # Python 3.11; Colab's newer interpreter follows the identical checked path.
        tf.extractall(destination, members=members)


def install_source_snapshot(archive, destination):
    """Extract into a new empty directory and atomically install it at a fresh path."""
    destination = pathlib.Path(destination)
    if destination.exists():
        try:
            next(destination.iterdir())
        except StopIteration:
            destination.rmdir()
        else:
            raise RuntimeError("refusing to overlay nonempty source destination: %s" %
                               destination)
    temporary = pathlib.Path(tempfile.mkdtemp(
        prefix=destination.name + "-extract-", dir=str(destination.parent)))
    try:
        safe_extract_tar(archive, temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def prepare_runtime_spec(project_root, base_spec, archive):
    """Bind a sidecar to one content-addressed source archive and write it atomically."""
    validate_confirmatory_job_spec(base_spec)
    digest = sha256_file(archive)
    document = dict(base_spec)
    document["source_sha256"] = digest
    document["source_object"] = "%s/source/project-%s.tar.gz" % (PREFIX, digest)
    inventory = pathlib.Path(project_root) / "manifests" / "manifest_inventory.sha256"
    if not inventory.is_file():
        raise SystemExit("manifest inventory is absent; refusing to prepare a runtime spec")
    input_digest = sha256_file(inventory)
    document["input_bundle_sha256"] = input_digest
    document["input_bundle_prefix"] = "%s/input_bundles/%s" % (PREFIX, input_digest)
    validate_confirmatory_job_spec(document, runtime_overlay=True)
    path = pathlib.Path(project_root) / ".agent_state" / "job_spec.runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return path, document


def prepare_session_spec(path, runtime_spec, session_id):
    """Bind one uploaded sidecar to the already-owned runtime without changing source identity."""
    document = dict(runtime_spec)
    document["session_id"] = str(session_id)
    validate_confirmatory_job_spec(
        document, runtime_overlay=True, require_session=True)
    path = pathlib.Path(path)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return document


def upload_with_retry(session_id, local_path, remote_path, attempts=3, runner=None,
                      sleeper=time.sleep, retry_delays=(2, 5)):
    """Retry transient kernel writes on the already-paid runtime before giving up."""
    if attempts < 1:
        raise ValueError("upload attempts must be positive")
    runner = sh if runner is None else runner
    command = "colab upload --session %s %s %s" % (
        shlex.quote(str(session_id)), shlex.quote(str(local_path)),
        shlex.quote(str(remote_path)))
    last_output = ""
    for attempt in range(1, attempts + 1):
        rc, last_output = runner(command, check=False)
        if rc == 0:
            return attempt
        print("runtime input upload failed attempt %d/%d; retrying same session" %
              (attempt, attempts), flush=True)
        if attempt < attempts:
            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            sleeper(delay)
    raise SystemExit("runtime input upload failed %d times on the same session: %s" %
                     (attempts, last_output[-2000:]))


def upload_session_inputs(session_id, adc, runtime_spec_file, runtime_spec):
    """Bind session identity before either runtime input is uploaded."""
    prepare_session_spec(runtime_spec_file, runtime_spec, session_id)
    adc_attempts = upload_with_retry(session_id, adc, "/content/adc.json")
    spec_attempts = upload_with_retry(session_id, runtime_spec_file, SPEC_PATH)
    print("RUNTIME_INPUTS_UPLOADED adc_attempts=%d spec_attempts=%d" %
          (adc_attempts, spec_attempts), flush=True)
    return {"adc_attempts": adc_attempts, "spec_attempts": spec_attempts}


def selected_base_jobs(args):
    """Expand a frozen base request into deterministic per-cell train/evaluate stages."""
    args = list(args)

    def values(flag, defaults):
        if flag not in args:
            return list(defaults)
        start = args.index(flag) + 1
        selected = []
        for value in args[start:]:
            if value.startswith("--"):
                break
            selected.append(value)
        if not selected:
            raise SystemExit("%s requires at least one value" % flag)
        return selected

    models = values("--models", ("gpt", "nextlat", "bst"))
    seeds = [int(value) for value in values("--seeds", (1234, 1235, 1236, 1237, 1238))]
    if any(model not in {"gpt", "nextlat", "bst"} for model in models):
        raise SystemExit("base stage request contains an unknown model")
    jobs = [(model, seed, "%s-s%d-base" % (model, seed))
            for model in models for seed in seeds]
    if "--only" in args:
        only = set(values("--only", ()))
        jobs = [cell for cell in jobs if cell[2] in only]
        if {cell[2] for cell in jobs} != only:
            raise SystemExit("base --only contains an unknown or non-base job id")
    if not jobs:
        raise SystemExit("base stage request selects no jobs")
    return jobs


def run_lurestar_base_stages(*, project, root, ledger, upstream, args, data_dir,
                             manifest_dir, durability, abort_event=None):
    """Train, evaluate, and durably commit each base cell before spending on the next."""
    runner_script = os.path.join(project, "scripts", "run_matrix.py")
    evaluator = os.path.join(project, "scripts", "evaluate_trained_bases.py")
    for model, seed, job_id in selected_base_jobs(args):
        print("=== BASE STAGE %s: train/resume ===" % job_id, flush=True)
        stage_args = list(args)
        if "--only" in stage_args:
            index = stage_args.index("--only")
            end = index + 1
            while end < len(stage_args) and not stage_args[end].startswith("--"):
                end += 1
            del stage_args[index:end]
        stage_args.extend(("--only", job_id))
        command = [sys.executable, runner_script, "--root", root, "--ledger", ledger,
                   "--upstream", upstream, *stage_args]
        rc, _ = sh(" ".join(shlex.quote(str(value)) for value in command),
                   check=False, quiet=False, abort_event=abort_event)
        print("BASE_STAGE_TRAIN_RC=%d JOB=%s" % (rc, job_id), flush=True)
        if rc != 0:
            return rc
        print("=== BASE STAGE %s: evaluate/promote ===" % job_id, flush=True)
        eval_command = [
            sys.executable, evaluator, "--ledger", ledger, "--upstream", upstream,
            "--dataset", os.path.join(data_dir, "graph_5_5_test_20000.txt"),
            "--manifest", os.path.join(manifest_dir, "corpus.sha256"),
            "--precision", "bf16-mixed", "--devices", "1",
            "--models", model, "--seeds", str(seed),
        ]
        eval_rc, _ = sh(" ".join(shlex.quote(str(value)) for value in eval_command),
                        check=False, quiet=False, abort_event=abort_event)
        print("BASE_STAGE_EVAL_RC=%d JOB=%s" % (eval_rc, job_id), flush=True)
        if eval_rc != 0:
            return eval_rc
        states = durability.sync_once(ledger, ledger_object="run_ledger.json")
        if job_id not in states:
            raise SystemExit("evaluated base stage did not commit a resumable durable state")
        print("BASE_STAGE_DURABLE=True JOB=%s" % job_id, flush=True)
    return 0


def run_hmm_recovery_terminalization_stage(
        command_parts, recovery_barrier, recovery_job_ids, durability, ledger,
        *, command_runner=sh):
    """Terminalize and durably commit the exact ten before the first remaining launcher."""
    stage = [*command_parts, "--only", *recovery_job_ids,
             "--recovery-barrier", str(recovery_barrier)]
    rc, _ = command_runner(
        " ".join(shlex.quote(str(value)) for value in stage),
        check=False, quiet=False)
    if rc != 0:
        raise SystemExit("D41 exact-ten terminalization stage failed before any launcher")
    committed = durability.sync_once(ledger, ledger_object="hmm_run_ledger.json")
    document = json.loads(pathlib.Path(ledger).read_text(encoding="utf-8"))
    latest = latest_ledger_entries(document)
    for job_id in tuple(recovery_job_ids):
        entry = latest.get(job_id)
        state = committed.get(job_id)
        if (not isinstance(entry, dict) or entry.get("status") not in {"TRAINED", "DONE"} or
                entry.get("step") != 3000 or
                not isinstance(entry.get("recovery_provenance"), dict) or
                not isinstance(state, dict) or state.get("status") not in {"TRAINED", "DONE"} or
                state.get("step") != 3000 or
                not isinstance(state.get("recovery_provenance"), dict)):
            raise SystemExit(
                "D41 exact-ten successor state/ledger commit failed for %s" % job_id)
    print("D41_EXACT_TEN_SUCCESSOR_COMMIT=DURABLE JOBS=10", flush=True)
    return committed


def write_sync_failure_diagnostic(path, *, failures, errors, source_sha256,
                                  predecessor_source_sha256, ledger_object):
    """Atomically retain the fail-stop reason without claiming scientific completion."""
    document = {
        "schema": "nextlat_forgetting/sync_failure_circuit_breaker/1",
        "status": "ABORTED_BEFORE_FURTHER_PAID_WORK",
        "consecutive_failures": failures,
        "max_consecutive_failures": MAX_CONSECUTIVE_SYNC_FAILURES,
        "errors": list(errors),
        "source_sha256": source_sha256,
        "predecessor_source_sha256": predecessor_source_sha256,
        "ledger_object": ledger_object,
        "training_complete": False,
        "recorded_at_unix": time.time(),
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return document


def durable_sync_loop(stop_event, abort_event, durability, ledger, ledger_object,
                      diagnostic_path, *, source_sha256,
                      predecessor_source_sha256=None, interval=SYNC_SECONDS):
    """Sync until stopped; three consecutive failures fail-stop the paid subprocess."""
    consecutive = 0
    errors = collections.deque(maxlen=MAX_CONSECUTIVE_SYNC_FAILURES)
    while not stop_event.wait(interval):
        try:
            durability.sync_once(ledger, ledger_object=ledger_object)
        except Exception as exc:
            consecutive += 1
            errors.append("%s: %s" % (type(exc).__name__, exc))
            print("[sync] failed consecutive=%d/%d: %s" % (
                consecutive, MAX_CONSECUTIVE_SYNC_FAILURES, exc), flush=True)
            if consecutive >= MAX_CONSECUTIVE_SYNC_FAILURES:
                document = write_sync_failure_diagnostic(
                    diagnostic_path, failures=consecutive, errors=errors,
                    source_sha256=source_sha256,
                    predecessor_source_sha256=predecessor_source_sha256,
                    ledger_object=ledger_object)
                print("SYNC_FAILURE_CIRCUIT_BREAKER=%s" % json.dumps(
                    document, sort_keys=True), flush=True)
                abort_event.set()
                return document
        else:
            if consecutive:
                print("[sync] recovered; consecutive failure counter reset", flush=True)
            consecutive = 0
            errors.clear()
    return None


def _owned_loop(project_root):
    spec_file = os.path.join(project_root, ".agent_state", "job_spec.json")
    spec = json.load(open(spec_file)) if os.path.exists(spec_file) else {
        "run_matrix_args": ["--phase", "base"], "gpu": "a100", "max_attempts": 20,
        "hard_stop_balance_cu": 1188.61}
    validate_confirmatory_job_spec(spec)
    gpu = spec.get("gpu", "a100")

    # Packaging is local and free. Bind the exact source bytes before any quota read,
    # upload, or paid runtime action, so stale reviews can never launch accidentally.
    print("=== LOOP role: packaging project for clearance verification ===", flush=True)
    tar = package(project_root)
    runtime_spec_file, runtime_spec = prepare_runtime_spec(project_root, spec, tar)
    clearance = validate_confirmatory_clearance(
        project_root, spec, runtime_spec["source_sha256"],
        clearance_path=spec.get("confirmatory_clearance_path"))
    runtime_spec = attach_recovery_receipt_transport(
        project_root, runtime_spec_file, runtime_spec)
    # This is a read-only remote check and happens before quota/status/provisioning.  The paid
    # runtime must never be the first place a missing frozen input is discovered.
    verify_remote_input_bundle(project_root, clearance["input_bundle"])

    hard_floor = float(spec.get("hard_stop_balance_cu", 1188.61))

    # Establish runtime ownership before any host-side upload or Colab mutation.  The loop reads
    # status again on every attempt to close races, but an ambiguous preflight must remain wholly
    # read-only rather than publishing a source and only discovering uncertainty afterwards.
    session_file = os.path.join(project_root, ".colab_session")
    preflight_first, preflight_second = colab_status_pair()
    preflight_state = agreed_runtime_state(preflight_first, preflight_second)
    if preflight_state != "gone":
        expected_sid = (pathlib.Path(session_file).read_text().strip()
                        if os.path.isfile(session_file) else "<missing-owned-session>")
        if preflight_state == "uncertain":
            diagnostic = write_ownership_uncertain_diagnostic(
                project_root, expected_session_id=expected_sid,
                status_first=preflight_first, status_second=preflight_second,
                stage="pre_upload_status_disagreement")
            print("COLAB_OWNERSHIP_UNCERTAIN=%s; read-only exit for reconciliation" %
                  diagnostic, flush=True)
            return 4
        try:
            require_owned_session(preflight_first, preflight_second, expected_sid)
        except OwnershipUncertain:
            diagnostic = write_ownership_uncertain_diagnostic(
                project_root, expected_session_id=expected_sid,
                status_first=preflight_first, status_second=preflight_second,
                stage="pre_upload_active_runtime")
            print("COLAB_OWNERSHIP_UNCERTAIN=%s; read-only exit for reconciliation" %
                  diagnostic, flush=True)
            return 4

    print("=== LOOP role: uploading cleared project ===", flush=True)
    publish_immutable_host_file(
        project_root, tar, runtime_spec["source_object"])
    runtime_spec_sha = sha256_file(runtime_spec_file)
    publish_immutable_host_file(
        project_root, runtime_spec_file,
        "%s/source/job_spec-%s.json" % (PREFIX, runtime_spec_sha))

    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if not os.path.isfile(adc):
        raise SystemExit("local ADC missing; refusing Colab credential upload")
    fast_exits = 0
    owned_sid = None
    try:
      for attempt in range(1, int(spec.get("max_attempts", 20)) + 1):
        print("\n=== attempt %d ===" % attempt, flush=True)
        progress_before = host_durable_progress()
        status_first, status_second = colab_status_pair()
        runtime_state = agreed_runtime_state(status_first, status_second)
        if runtime_state == "uncertain":
            print("Colab status reads disagree; refusing to start a possible duplicate", flush=True)
            continue
        if runtime_state == "active":
            if not os.path.isfile(session_file):
                raise SystemExit("an active Colab runtime is not owned by this loop")
            sid = pathlib.Path(session_file).read_text().strip()
            if not sid:
                raise SystemExit("owned Colab session file is empty")
            try:
                require_owned_session(status_first, status_second, sid)
            except OwnershipUncertain:
                diagnostic = write_ownership_uncertain_diagnostic(
                    project_root, expected_session_id=sid,
                    status_first=status_first, status_second=status_second,
                    stage="active_runtime_reuse")
                print("COLAB_OWNERSHIP_UNCERTAIN=%s; read-only exit for reconciliation" %
                      diagnostic, flush=True)
                return 4
            print("reusing owned active session=%s" % sid, flush=True)
            # An active owned runtime may still be training after the exec transport returned.
            # Observe its durable progress; re-executing the driver could duplicate work.
            owned_sid = sid
            t0 = time.time()
            out = ""
            outcome = monitor_owned_runtime(
                sid, runtime_spec["source_sha256"], progress_before,
                activity_reader=host_runtime_activity)
            elapsed = time.time() - t0
        else:
            # Re-authorize every paid provisioning attempt. The quota pair happens after
            # the previous runtime is known gone, and the final status pair closes the
            # race window immediately before `colab start`.
            authorize_provisioning(hard_floor)
            rc, out = sh("colab start --gpu %s --json" % shlex.quote(str(gpu)), check=False)
            try:
                sid = parse_cli_json(out)["session"]
            except Exception:
                print("could not start a runtime; backing off", flush=True)
                time.sleep(60)
                continue
            pathlib.Path(session_file).write_text(str(sid) + "\n")
            owned_sid = sid
            print("session=%s" % sid, flush=True)
            try:
                upload_session_inputs(sid, adc, runtime_spec_file, runtime_spec)
            except BaseException:
                # This runtime is known-owned and training never began. Do not strand paid idle
                # compute or feed it into the training monitor without inputs.
                stop_owned_runtime("input-upload-failed", session_file)
                owned_sid = None
                raise

            t0 = time.time()
            driver_path = os.path.join(HERE, "colab_train_loop.py")
            rc, out = sh("colab exec --session %s --timeout 240m %s" % (sid, driver_path),
                         check=False)
            elapsed = time.time() - t0
            # EOF and CLI timeout describe the transport, not the runtime.  Keep observing an
            # active, advancing job until a durable terminal marker, true disconnect, or stall.
            outcome = monitor_owned_runtime(
                sid, runtime_spec["source_sha256"], progress_before,
                activity_reader=host_runtime_activity)

        elapsed = time.time() - t0
        reason = outcome["reason"]
        marker = outcome.get("marker") or {}
        progress_after = outcome.get("progress") or host_durable_progress()
        advanced = durable_progress_advanced(progress_before, progress_after)
        if reason == "ownership_uncertain":
            diagnostic = write_ownership_uncertain_diagnostic(
                project_root, expected_session_id=sid,
                status_first=outcome.get("status_first"),
                status_second=outcome.get("status_second"),
                stage="active_runtime_monitor")
            print("COLAB_OWNERSHIP_UNCERTAIN=%s; read-only exit for reconciliation" %
                  diagnostic, flush=True)
            return 4
        if reason in {"terminal", "stalled"}:
            stop_owned_runtime(reason, session_file)
            owned_sid = None
        elif reason == "gone":
            pathlib.Path(session_file).unlink(missing_ok=True)
            owned_sid = None

        rc, settled_quota = sh("colab quota --json", check=False, quiet=True, silent=True,
                               max_lines=None)
        time.sleep(30)
        rc2, settled_quota2 = sh("colab quota --json", check=False, quiet=True, silent=True,
                                  max_lines=None)
        if rc == 0 and rc2 == 0:
            settled = parse_cli_json(settled_quota2)
            print("SETTLED_BALANCE_CU=%s ACTIVE_RUNTIMES=%s BURN_RATE=%s" %
                  (settled.get("paid_balance"), settled.get("active_runtimes"),
                   settled.get("burn_rate_hourly")), flush=True)
        # Exec stdout is an untrusted/partial transport and may contain stale child text.  Only
        # the source/session-bound terminal object written after final sync can close training.
        if marker.get("training_complete") is True:
            print("\nTRAINING MATRIX COMPLETE after %d attempt(s); evaluation remains separate" %
                  attempt, flush=True)
            return 0
        # A driver that dies instantly provisions a fresh GPU for nothing. Two in a row
        # means the failure is deterministic and another runtime will not fix it.
        if elapsed < FAST_EXIT_SECONDS and not advanced:
            fast_exits += 1
            print("fast exit with no durable step advance %d/%d (%.0fs)" %
                  (fast_exits, MAX_FAST_EXITS, elapsed), flush=True)
            if fast_exits >= MAX_FAST_EXITS:
                print("ABORTING: two consecutive fast exits without durable progress; "
                      "the failure is deterministic", flush=True)
                return 2
        else:
            fast_exits = 0
            if advanced:
                print("durable state advanced; circuit-breaker strike reset", flush=True)
        print("runtime ended after %.0fs; resuming on a fresh one" % elapsed, flush=True)
      print("exhausted attempts", flush=True)
      return 3
    finally:
        if owned_sid is not None:
            print("host loop exited while session %s may still be active; preserving ownership "
                  "for the next monitor invocation" % owned_sid, flush=True)


def loop():
    project_root = os.path.dirname(HERE)
    with controller_lock(project_root):
        return _owned_loop(project_root)


if __name__ == "__main__" or "get_ipython" in dir():
    sys.exit(driver() if os.path.isdir("/content") else loop())
