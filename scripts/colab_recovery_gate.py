#!/usr/bin/env python3
"""Preregistered real-trainer T4 interruption/recovery gate.

The host role packages one content-addressed project snapshot, provisions one T4 only
after two agreeing status reads, and uploads a fixed sidecar.  The runtime role uses the
guarded patch in :mod:`scripts.runtime_bootstrap` to create one step-150 lineage, kills and
durably restores it, then compares two continuations from that hash-identical lineage at
step 300.  This is an engineering gate, never a confirmatory experiment.

There are intentionally no command-line controls for the seed, data, update counts,
precision, or tolerances.  Changing one requires a reviewed source change before a run.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid


try:
    HERE = pathlib.Path(__file__).resolve().parent
except NameError:  # colab exec executes this file as a notebook cell
    HERE = pathlib.Path.cwd()

PROJECT_ROOT = HERE.parent
REMOTE_SPEC = "/content/recovery_gate_job.json"
ADC_PATH = "/content/adc.json"
BUCKET = "nextlat-lurestar-project-flash-490419"
GCS_PREFIX = "lurestar/recovery-gates"
GCP_PROJECT = "project-flash-490419"
PINNED_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"
SCHEMA = "nextlat_forgetting/recovery_gate/2"
SEED = 910_241
CONFIRMATORY_SEEDS = frozenset({1234, 1235, 1236, 1237, 1238})
TRAIN_STEPS = 300
INTERRUPT_STEP = 150
CHECKPOINT_EVERY = 50
TRAIN_SAMPLES = 4096
TEST_SAMPLES = 512
DATA_FIRST_SEED = 1_000_000
HEARTBEAT_SECONDS = 30
DURABLE_PROGRESS_SECONDS = 60
STATUS_DELAY_SECONDS = 30
HARD_STOP_BALANCE_CU = 1188.61
AMP_SCALER_KEY = "lurestar_amp_scaler_state_v1"

# Frozen before execution.  They are deliberately not CLI parameters.
TOLERANCES = {
    "weights_atol": 2e-6,
    "weights_rtol": 2e-5,
    "optimizer_atol": 2e-6,
    "optimizer_rtol": 2e-5,
    "logits_atol": 5e-5,
    "logits_rtol": 5e-5,
    "metrics_atol": 5e-5,
    "metrics_rtol": 5e-5,
    "scheduler_atol": 0.0,
    "scheduler_rtol": 0.0,
    "rng_exact": True,
}


class GateError(RuntimeError):
    """A fail-closed recovery-gate refusal."""


def sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


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


def append_receipt(path: pathlib.Path, event: dict) -> None:
    """Append and fsync one event; an earlier gate record is never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record.setdefault("recorded_at_unix", time.time())
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def package_project(project_root: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    """Create a byte-reproducible source archive, excluding results, data, and secrets."""
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
                for child in sorted(project_root.iterdir(), key=lambda p: p.name):
                    if child.name.startswith(".") or child.name in excluded:
                        continue
                    archive.add(child, arcname=child.name, filter=archive_filter)
    return destination


def build_spec(source_sha256: str, gate_id: str) -> dict:
    if SEED in CONFIRMATORY_SEEDS:
        raise GateError("recovery-gate seed overlaps the confirmatory seed set")
    document = {
        "schema": SCHEMA,
        "gate_id": gate_id,
        "gpu": "t4",
        "precision": "16-mixed",
        "seed": SEED,
        "train_steps": TRAIN_STEPS,
        "interrupt_step": INTERRUPT_STEP,
        "checkpoint_every": CHECKPOINT_EVERY,
        "data": {
            "kind": "nonconfirmatory_upstream_generator",
            "first_generator_seed": DATA_FIRST_SEED,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
        },
        "tolerances": dict(TOLERANCES),
        "source_sha256": source_sha256,
        "source_object": "lurestar/source/project-%s.tar.gz" % source_sha256,
        "result_object": "%s/%s/result.json" % (GCS_PREFIX, gate_id),
        "event_prefix": "%s/%s/events" % (GCS_PREFIX, gate_id),
        "resume_prefix": "%s/%s/resume" % (GCS_PREFIX, gate_id),
        "nonconfirmatory": True,
    }
    validate_spec(document)
    document["preregistration_sha256"] = canonical_sha256(document)
    return document


def validate_spec(spec: dict) -> None:
    """Reject a sidecar that changes any preregistered science-facing field."""
    required = {
        "schema": SCHEMA,
        "gpu": "t4",
        "precision": "16-mixed",
        "seed": SEED,
        "train_steps": TRAIN_STEPS,
        "interrupt_step": INTERRUPT_STEP,
        "checkpoint_every": CHECKPOINT_EVERY,
        "tolerances": TOLERANCES,
        "nonconfirmatory": True,
    }
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise GateError("sidecar drift in %s" % key)
    if spec["seed"] in CONFIRMATORY_SEEDS:
        raise GateError("confirmatory seed forbidden in recovery gate")
    data = spec.get("data", {})
    if data != {
        "kind": "nonconfirmatory_upstream_generator",
        "first_generator_seed": DATA_FIRST_SEED,
        "train_samples": TRAIN_SAMPLES,
        "test_samples": TEST_SAMPLES,
    }:
        raise GateError("sidecar data contract drift")
    digest = str(spec.get("source_sha256", ""))
    source = str(spec.get("source_object", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise GateError("invalid source digest")
    if source != "lurestar/source/project-%s.tar.gz" % digest:
        raise GateError("source object is not content-addressed")
    gate_id = str(spec.get("gate_id", ""))
    if not re.fullmatch(r"rg-[0-9a-f]{12}-[0-9]{10,20}-[0-9a-f]{8}", gate_id):
        raise GateError("invalid gate id")
    expected_root = "%s/%s/" % (GCS_PREFIX, gate_id)
    for field in ("result_object", "event_prefix", "resume_prefix"):
        if not str(spec.get(field, "")).startswith(expected_root):
            raise GateError("%s escapes the gate object root" % field)
    prereg = spec.get("preregistration_sha256")
    if prereg is not None:
        unsigned = dict(spec)
        unsigned.pop("preregistration_sha256", None)
        if prereg != canonical_sha256(unsigned):
            raise GateError("sidecar preregistration hash mismatch")


def parse_cli_json(text: str) -> dict:
    text = str(text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise GateError("CLI output contained no JSON object")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise GateError("CLI JSON was not an object")
    return value


def run_argv(argv: list[str], *, check: bool = True, relay: bool = True,
             max_lines: int | None = 200) -> tuple[int, str]:
    """Run a fixed argv vector, relaying output and preserving the child's real status."""
    print("+ " + " ".join(argv), flush=True)
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    tail = [] if max_lines is None else collections.deque(maxlen=max_lines)
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line.rstrip())
        if relay:
            print("  | " + line, end="", flush=True)
    rc = process.wait()
    output = "\n".join(tail)
    if check and rc:
        raise GateError("command failed rc=%d: %s\n%s" % (rc, argv[0], output))
    return rc, output


def status_pair(delay: int = STATUS_DELAY_SECONDS, sleeper=time.sleep) -> tuple[dict, dict]:
    first = parse_cli_json(run_argv(["colab", "status", "--json"], relay=False)[1])
    sleeper(delay)
    second = parse_cli_json(run_argv(["colab", "status", "--json"], relay=False)[1])
    return first, second


def quota_pair(delay: int = STATUS_DELAY_SECONDS, sleeper=time.sleep) -> tuple[dict, dict]:
    first = parse_cli_json(run_argv(["colab", "quota", "--json"], relay=False)[1])
    sleeper(delay)
    second = parse_cli_json(run_argv(["colab", "quota", "--json"], relay=False)[1])
    return first, second


def agreed_runtime_state(first: dict, second: dict) -> str:
    gone = [doc.get("status") == "no_runtime" for doc in (first, second)]
    if all(gone):
        return "gone"
    if not any(gone):
        return "active"
    return "uncertain"


def upload_with_retry(
    session: str,
    local_path: pathlib.Path,
    remote_path: str,
    *,
    attempts: int = 3,
    runner=None,
) -> int:
    """Retry transient kernel writes on the same paid runtime before reprovisioning."""
    if attempts < 1:
        raise GateError("upload attempts must be positive")
    runner = run_argv if runner is None else runner
    last_tail = ""
    for attempt in range(1, attempts + 1):
        rc, last_tail = runner(
            ["colab", "upload", "--session", session, str(local_path), remote_path],
            check=False,
        )
        if rc == 0:
            return attempt
    raise GateError(
        "Colab kernel upload failed %d times on the same runtime: %s"
        % (attempts, last_tail[-2000:])
    )


def _safe_extract(archive_path: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise GateError("source archive contains a path traversal")
            if member.issym() or member.islnk():
                raise GateError("source archive contains a link member")
            if not (member.isdir() or member.isfile()):
                raise GateError("source archive contains a special member")
        for member in archive.getmembers():
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise GateError("regular archive member has no payload")
            with source, open(target, "wb") as stream:
                while True:
                    block = source.read(1 << 20)
                    if not block:
                        break
                    stream.write(block)
            os.chmod(target, member.mode & 0o777)


def _heartbeat(stop: threading.Event) -> None:
    started = time.time()
    while not stop.wait(HEARTBEAT_SECONDS):
        print("[recovery-gate heartbeat] elapsed_s=%d" % (time.time() - started), flush=True)


def _runtime_event(bucket, spec: dict, sequence: int, event: str, **details) -> None:
    record = {
        "schema": SCHEMA,
        "gate_id": spec["gate_id"],
        "sequence": sequence,
        "event": event,
        "recorded_at_unix": time.time(),
        **details,
    }
    name = "%s/%04d-%s.json" % (spec["event_prefix"], sequence, uuid.uuid4().hex)
    bucket.blob(name).upload_from_string(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
    )


def _publish_progress_snapshot(bucket, spec: dict, runtime_root: pathlib.Path,
                               sequence: int) -> dict:
    """Commit an append-only metrics snapshot; its state object is written last."""
    root = "%s/%s/progress/%06d" % (GCS_PREFIX, spec["gate_id"], sequence)
    artifacts = []
    for path in sorted(runtime_root.glob("**/metrics.csv")):
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            # A completed arm can be atomically renamed while this scan is in flight.
            continue
        relative = path.relative_to(runtime_root).as_posix()
        name = "%s/artifacts/%s" % (root, relative)
        bucket.blob(name).upload_from_string(
            payload, content_type="text/csv", if_generation_match=0
        )
        artifacts.append({
            "path": relative, "object": name,
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
        })
    pointers = {}
    for pointer in sorted(runtime_root.glob("**/*_ckpt")):
        if pointer.is_file():
            try:
                pointers[pointer.relative_to(runtime_root).as_posix()] = \
                    pointer.read_text().strip()
            except FileNotFoundError:
                continue
    state = {
        "schema": SCHEMA, "gate_id": spec["gate_id"], "sequence": sequence,
        "source_sha256": spec["source_sha256"], "recorded_at_unix": time.time(),
        "artifacts": artifacts, "checkpoint_pointers": pointers,
    }
    bucket.blob("%s/state.json" % root).upload_from_string(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        content_type="application/json", if_generation_match=0,
    )
    _runtime_event(
        bucket, spec, 1000 + sequence, "progress_snapshot_committed",
        progress_sequence=sequence, metric_artifact_count=len(artifacts),
        checkpoint_pointer_count=len(pointers),
    )
    return state


def _durable_progress_loop(bucket, spec: dict, runtime_root: pathlib.Path,
                            stop: threading.Event, errors: list[str]) -> None:
    sequence = 1
    while not stop.wait(DURABLE_PROGRESS_SECONDS):
        try:
            _publish_progress_snapshot(bucket, spec, runtime_root, sequence)
        except Exception as exc:  # preserved in the terminal result/diagnostic
            errors.append("%s: %s" % (type(exc).__name__, exc))
        sequence += 1


def _verify_t4(expected_torch: str | None = None) -> str:
    import torch

    if not torch.cuda.is_available():
        raise GateError("CUDA unavailable")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise GateError("requested T4 but runtime device is %s" % name)
    if expected_torch is not None and torch.__version__ != expected_torch:
        raise GateError("dependency install replaced torch: %s != %s" %
                        (torch.__version__, expected_torch))
    print("RUNTIME_GPU=%s TORCH=%s CUDA=%s" %
          (name, torch.__version__, torch.version.cuda), flush=True)
    return torch.__version__


def _generate_gate_data(upstream: pathlib.Path, destination: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate a disjoint fixed corpus with the pinned upstream generator itself."""
    module_spec = importlib.util.spec_from_file_location(
        "_rg_prepare", upstream / "data" / "stargraph" / "prepare.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise GateError("cannot load pinned stargraph generator")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    destination.mkdir(parents=True, exist_ok=True)
    train = destination / "graph_5_5_sample_4096.txt"
    test = destination / "graph_5_5_test_512.txt"

    def render(sample_seed: int) -> str:
        random.seed(sample_seed)
        edges, path, source, goal = module.star_or_sink_graph_maker(5, 5, 100, True, False)
        return ("|".join("%d,%d" % edge for edge in edges) +
                "/%d,%d=%s" % (source, goal, ",".join(map(str, path))))

    train.write_text("\n".join(render(DATA_FIRST_SEED + i) for i in range(TRAIN_SAMPLES)) + "\n")
    test.write_text("\n".join(
        render(DATA_FIRST_SEED + TRAIN_SAMPLES + i) for i in range(TEST_SAMPLES)
    ) + "\n")
    return train, test


def _training_command(project: pathlib.Path, upstream: pathlib.Path, branch_root: pathlib.Path,
                      train: pathlib.Path, test: pathlib.Path, *, resume: bool) -> tuple[list[str], dict]:
    command = [
        str(project / "scripts" / "launch_train.sh"), "gpt_lurestar.yaml", str(SEED),
        "trainer.train_batches=%d" % TRAIN_STEPS,
        "trainer.save_recovery_checkpoint=%d" % CHECKPOINT_EVERY,
        "trainer.init_from=%s" % ("resume" if resume else "scratch"),
        "trainer.val_interval=1000000", "trainer.test_interval=1000000",
        "trainer.val_batches=1", "trainer.test_batches=1",
        "trainer.save_best_checkpoint=false", "trainer.always_save_checkpoint=true",
        "data.stargraph_train_data_path=%s" % train,
        "data.stargraph_test_data_path=%s" % test,
    ]
    env = dict(os.environ)
    for name in ("DRY_RUN", "LURESTAR_ENTRY", "LURESTAR_MODEL",
                 "LURESTAR_PARENT_CKPT", "PROFILE_PROBE_JSON"):
        env.pop(name, None)
    env.update({
        "NEXTLAT_REPO": str(upstream), "LURESTAR_ROOT": str(branch_root),
        "LURESTAR_PRECISION": "16-mixed", "LURESTAR_STRATEGY": "ddp",
        "LURESTAR_ALLOW_ANY_SEED": "1", "LURESTAR_NONCONFIRMATORY": "1",
        "LURESTAR_DETERMINISTIC_RUNTIME": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
    })
    return command, env


class TrainingObservationLatch:
    """Latch structured trainer events independently of the bounded diagnostic tail."""

    SCHEMA = "nextlat_forgetting/recovery_gate/training_observations/1"
    _FAST_FORWARD = re.compile(r"(?:^|\s)Fast forwarded data to step ([0-9]+)(?:\s|$)")

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._line_count = 0
        self._events: list[dict] = []

    def observe_line(self, line: str) -> None:
        """Record a recognized event before the line can be evicted from diagnostics."""
        with self._lock:
            self._line_count += 1
            match = self._FAST_FORWARD.search(line)
            if match is None:
                return
            event = {
                "kind": "data_fast_forward",
                "step": int(match.group(1)),
                "line_number": self._line_count,
            }
            if event not in self._events:
                self._events.append(event)
            if self.path is not None:
                atomic_json(self.path, self._snapshot_unlocked())

    def _snapshot_unlocked(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "line_count": self._line_count,
            "events": list(self._events),
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()


def _observed_fast_forward(observations: dict, step: int) -> bool:
    if observations.get("schema") != TrainingObservationLatch.SCHEMA:
        return False
    events = observations.get("events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, dict) and event.get("kind") == "data_fast_forward" and
        event.get("step") == step
        for event in events
    )


def _start_training(
    command: list[str], env: dict, *, observation_path: pathlib.Path | None = None
) -> tuple[subprocess.Popen, collections.deque, threading.Thread, TrainingObservationLatch]:
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    # Keep a generous diagnostic tail, but never use it as proof: recognized events are
    # latched structurally before the corresponding text can be evicted.
    tail: collections.deque[str] = collections.deque(maxlen=5000)
    observations = TrainingObservationLatch(observation_path)

    def relay() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            observations.observe_line(line)
            tail.append(line.rstrip())
            print("  | " + line, end="", flush=True)

    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    return process, tail, thread, observations


def _finish_training(
    command: list[str], env: dict, *, observation_path: pathlib.Path | None = None
) -> tuple[str, dict]:
    process, tail, relay, observations = _start_training(
        command, env, observation_path=observation_path
    )
    rc = process.wait()
    relay.join(timeout=30)
    if relay.is_alive():
        raise GateError("trainer output relay did not stop within 30 seconds")
    if rc:
        raise GateError("trainer failed rc=%d\n%s" % (rc, "\n".join(tail)))
    return "\n".join(tail), observations.snapshot()


def _process_table() -> dict[int, dict]:
    """Return stable process identities without depending on process-group ancestry.

    ``launch_train.sh`` can ultimately create workers in process groups other than the
    launcher's.  Parent PID ancestry still identifies those workers before the parent is
    killed.  The start timestamp prevents signaling a recycled PID.
    """
    proc_root = pathlib.Path("/proc")
    if proc_root.is_dir():
        table = {}
        for stat_path in proc_root.glob("[0-9]*/stat"):
            try:
                raw = stat_path.read_text()
                # comm is parenthesized and may contain spaces or parentheses; fields
                # after its final ')' start at proc-stat field 3.
                fields = raw[raw.rfind(")") + 2:].split()
                pid = int(stat_path.parent.name)
                table[pid] = {
                    "pid": pid, "state": fields[0], "ppid": int(fields[1]),
                    "pgid": int(fields[2]), "started": "proc:%s" % fields[19],
                }
            except (FileNotFoundError, IndexError, PermissionError, ValueError):
                # Processes can disappear while /proc is enumerated.
                continue
        return table

    # Portable local-test fallback. Colab uses the /proc branch above, whose start
    # tick has finer identity resolution than ps(1)'s wall-clock timestamp.
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,lstart=,stat="],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise GateError("cannot inspect trainer process tree: %s" % completed.stderr[-2000:])
    table = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        # pid, ppid, pgid, five lstart fields, stat
        if len(fields) < 9:
            continue
        try:
            pid, ppid, pgid = map(int, fields[:3])
        except ValueError:
            continue
        table[pid] = {
            "pid": pid, "ppid": ppid, "pgid": pgid,
            "started": " ".join(fields[3:8]), "state": fields[8],
        }
    return table


def _descendant_pids(table: dict[int, dict], root_pid: int) -> set[int]:
    children: dict[int, set[int]] = {}
    for pid, record in table.items():
        children.setdefault(int(record["ppid"]), set()).add(pid)
    descendants = {root_pid} if root_pid in table else set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in descendants:
                descendants.add(child)
                frontier.append(child)
    return descendants


def _same_process(record: dict, table: dict[int, dict]) -> bool:
    current = table.get(int(record["pid"]))
    return current is not None and current["started"] == record["started"]


def _kill_process_tree(process: subprocess.Popen, *, timeout: float = 60.0,
                       poll_seconds: float = 0.02) -> dict:
    """Freeze, capture, SIGKILL, and verify every trainer descendant is gone.

    A process-group kill is insufficient because Fabric/torch workers can call
    ``setsid`` or otherwise escape the launcher's group.  We first freeze the current
    parent-PID tree, repeat discovery until it is stable, and only then kill every
    captured identity.  No checkpoint snapshot may begin until all captured PIDs (with
    their original start timestamps) have disappeared.
    """
    deadline = time.time() + timeout
    captured: dict[int, dict] = {}
    stable_rounds = 0
    while time.time() < deadline and stable_rounds < 2:
        table = _process_table()
        discovered = _descendant_pids(table, process.pid)
        # Once a parent is stopped, follow every already captured live identity too.
        for record in tuple(captured.values()):
            if _same_process(record, table):
                discovered.update(_descendant_pids(table, int(record["pid"])))
        before = set(captured)
        for pid in discovered:
            record = table.get(pid)
            if record is not None:
                captured[pid] = record
        # Stop each identity individually; this includes escaped process groups.
        for record in tuple(captured.values()):
            if _same_process(record, table) and not str(table[record["pid"]]["state"]).startswith("Z"):
                try:
                    os.kill(int(record["pid"]), signal.SIGSTOP)
                except ProcessLookupError:
                    pass
        time.sleep(poll_seconds)
        after = _process_table()
        live = [record for record in captured.values() if _same_process(record, after)]
        all_quiescent = all(str(after[record["pid"]]["state"]).startswith(("T", "Z"))
                            for record in live)
        stable_rounds = stable_rounds + 1 if set(captured) == before and all_quiescent else 0
    if not captured or process.pid not in captured or stable_rounds < 2:
        raise GateError("could not freeze a stable trainer process tree before interruption")

    # Retain process groups as audit evidence, but signal captured identities rather
    # than trusting group membership. Descendants first, launcher last.
    table = _process_table()
    for record in sorted(captured.values(), key=lambda item: item["pid"] == process.pid):
        if _same_process(record, table):
            try:
                os.kill(int(record["pid"]), signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=max(0.1, deadline - time.time()))
    except subprocess.TimeoutExpired as exc:
        raise GateError("trainer launcher did not die after tree SIGKILL") from exc

    survivors = []
    while time.time() < deadline:
        table = _process_table()
        survivors = [record for record in captured.values() if _same_process(record, table)]
        if not survivors:
            break
        time.sleep(poll_seconds)
    if survivors:
        raise GateError("captured trainer descendants survived SIGKILL: %s" %
                        sorted(record["pid"] for record in survivors))
    return {
        "captured_pids": sorted(captured),
        "captured_process_groups": sorted({record["pgid"] for record in captured.values()}),
        "all_captured_identities_gone": True,
    }


def _kill_at_checkpoint(command: list[str], env: dict, out_dir: pathlib.Path) -> tuple[pathlib.Path, str]:
    process, tail, relay, _observations = _start_training(command, env)
    pointer = out_dir / "recovery_ckpt"
    deadline = time.time() + 3600
    target = None
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                raise GateError("trainer exited before intentional interruption\n%s" %
                                "\n".join(tail))
            if pointer.is_file():
                candidate = pathlib.Path(pointer.read_text().strip())
                if candidate.name == "recovery_ckpt_iter_%d.pt" % INTERRUPT_STEP:
                    target = candidate
                    break
            time.sleep(0.01)
        if target is None:
            raise GateError("step-%d recovery pointer did not appear" % INTERRUPT_STEP)
        termination = _kill_process_tree(process)
        rc = process.returncode
        relay.join(timeout=30)
        if rc >= 0:
            raise GateError("intentional interruption was not a signal termination")
        sys.path.insert(0, str(env["NEXTLAT_REPO"]))
        from lurestar_runtime import verify_checkpoint
        metadata, state = verify_checkpoint(target, deserialize=True, require_metadata=True)
        if int(state.get("training_steps", -1)) != INTERRUPT_STEP:
            raise GateError("interruption checkpoint has wrong optimizer step")
        if pathlib.Path(pointer.read_text().strip()).resolve() != target.resolve():
            raise GateError("recovery pointer changed away from step 150 before kill")
        return target, json.dumps(termination, sort_keys=True) + "\n" + "\n".join(tail)
    finally:
        if process.poll() is None:
            _kill_process_tree(process)


def _upload_resume_snapshot(bucket, spec: dict, out_dir: pathlib.Path,
                            target: pathlib.Path) -> dict:
    from lurestar_runtime import verify_checkpoint

    pointer = out_dir / "recovery_ckpt"
    checkpoints = sorted(target.parent.glob("recovery_ckpt_iter_*.pt"))
    if len(checkpoints) < 2:
        raise GateError("hardened runtime retained fewer than two recovery checkpoints")
    artifacts = []
    artifact_paths = []
    for path in checkpoints[-2:]:
        metadata, state = verify_checkpoint(path, deserialize=True, require_metadata=True)
        if "lurestar_rng_state_v1" not in state:
            raise GateError("recovery checkpoint lacks RNG state")
        artifact_paths.extend((path, path.with_name(path.name + ".meta.json")))
    artifact_paths.append(pointer)
    # Restore the metric history before relaunch. CSVLogger will then select version_1,
    # preserving version_0 instead of silently replacing the pre-kill trajectory.
    receipt_candidates = (
        out_dir / ".lurestar_job_identity.json",
        out_dir / "metrics" / "step_0_contract.json",
        target.parent / "materialized_config.yaml",
    )
    artifact_paths.extend(path for path in receipt_candidates if path.is_file())
    metric_paths = sorted(target.parent.glob("version_*/metrics.csv"))
    if not metric_paths:
        raise GateError("interruption snapshot has no step metrics")
    artifact_paths.extend(metric_paths)
    seen = set()
    for local in artifact_paths:
        local = local.resolve()
        if str(local) in seen:
            continue
        seen.add(str(local))
        relative = local.relative_to(out_dir.resolve()).as_posix()
        name = "%s/artifacts/%s" % (spec["resume_prefix"], relative)
        # The state record below is immutable, so every byte object it names must be
        # immutable too.  A rerun of the same gate ID must fail rather than replace a
        # blob already committed by state.json.
        bucket.blob(name).upload_from_filename(str(local), if_generation_match=0)
        artifacts.append({
            "local_path": str(local), "object": name,
            "sha256": sha256_file(local), "size_bytes": local.stat().st_size,
            "kind": "metrics" if local.name == "metrics.csv" else
                    ("receipt" if local.suffix in {".json", ".yaml"} else "checkpoint"),
        })
    state = {
        "schema": SCHEMA, "gate_id": spec["gate_id"], "step": INTERRUPT_STEP,
        "source_sha256": spec["source_sha256"], "checkpoint": str(target.resolve()),
        "checkpoint_sha256": sha256_file(target), "artifacts": artifacts,
    }
    # Commit record is always uploaded last.
    bucket.blob("%s/state.json" % spec["resume_prefix"]).upload_from_string(
        json.dumps(state, indent=2, sort_keys=True) + "\n", content_type="application/json",
        if_generation_match=0,
    )
    return state


def _restore_resume_snapshot(bucket, spec: dict, state: dict, *,
                             runtime_root: pathlib.Path | None = None) -> pathlib.Path:
    if state.get("source_sha256") != spec["source_sha256"] or state.get("step") != INTERRUPT_STEP:
        raise GateError("durable resume state identity mismatch")
    expected_root = (runtime_root or
                     (pathlib.Path("/content/rgate") / spec["gate_id"])).resolve()
    for artifact in state["artifacts"]:
        local = pathlib.Path(artifact["local_path"]).resolve()
        if not local.is_relative_to(expected_root):
            raise GateError("resume artifact escapes exact runtime root")
        local.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(artifact["object"]).download_to_filename(str(local))
        if local.stat().st_size != artifact["size_bytes"] or sha256_file(local) != artifact["sha256"]:
            raise GateError("restored artifact failed size/hash verification")
    target = pathlib.Path(state["checkpoint"])
    from lurestar_runtime import verify_checkpoint
    metadata, payload = verify_checkpoint(target, deserialize=True, require_metadata=True)
    if metadata["sha256"] != state["checkpoint_sha256"]:
        raise GateError("restored checkpoint disagrees with committed state")
    pointer = target.parents[1] / "recovery_ckpt"
    if pathlib.Path(pointer.read_text().strip()).resolve() != target.resolve():
        raise GateError("restored pointer does not name the verified checkpoint")
    return target


def _assert_committed_lineage(checkpoint: pathlib.Path, state: dict) -> str:
    """Prove a continuation uses the exact committed step-150 parent bytes/path."""
    resolved = checkpoint.resolve()
    if state.get("step") != INTERRUPT_STEP:
        raise GateError("shared lineage state is not the step-150 commit")
    if pathlib.Path(str(state.get("checkpoint", ""))).resolve() != resolved:
        raise GateError("continuation checkpoint path differs from committed lineage")
    digest = sha256_file(resolved)
    if state.get("checkpoint_sha256") != digest:
        raise GateError("continuation checkpoint bytes differ from committed lineage")
    matching = [
        artifact for artifact in state.get("artifacts", [])
        if pathlib.Path(str(artifact.get("local_path", ""))).resolve() == resolved
    ]
    if len(matching) != 1 or matching[0].get("sha256") != digest:
        raise GateError("committed state does not uniquely bind the lineage artifact")
    return digest


def _publish_final_checkpoint(bucket, spec: dict, label: str, checkpoint: pathlib.Path) -> dict:
    """Publish a deeply verified final checkpoint, with its state commit last."""
    if label not in {"reference", "recovered"}:
        raise GateError("unknown final checkpoint label")
    from lurestar_runtime import verify_checkpoint

    metadata, payload = verify_checkpoint(checkpoint, deserialize=True, require_metadata=True)
    if int(payload.get("training_steps", -1)) != TRAIN_STEPS:
        raise GateError("%s final artifact is not exact step %d" % (label, TRAIN_STEPS))
    root = "%s/%s/final/%s" % (GCS_PREFIX, spec["gate_id"], label)
    artifact_records = []
    for local in (checkpoint, checkpoint.with_name(checkpoint.name + ".meta.json")):
        name = "%s/%s" % (root, local.name)
        # Final state.json is an immutable commit record.  Keep its referents equally
        # immutable so replaying a gate ID cannot silently change committed bytes.
        bucket.blob(name).upload_from_filename(str(local), if_generation_match=0)
        artifact_records.append({
            "object": name, "sha256": sha256_file(local),
            "size_bytes": local.stat().st_size,
        })
    state = {
        "schema": SCHEMA, "gate_id": spec["gate_id"], "label": label,
        "step": TRAIN_STEPS, "source_sha256": spec["source_sha256"],
        "checkpoint_sha256": metadata["sha256"], "artifacts": artifact_records,
    }
    bucket.blob("%s/state.json" % root).upload_from_string(
        json.dumps(state, indent=2, sort_keys=True) + "\n", content_type="application/json",
        if_generation_match=0,
    )
    return state


def _tree_metrics(left, right, *, atol: float, rtol: float, path: str = "root") -> dict:
    import numpy as np

    result = {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "mismatch": None}

    def fail(message: str) -> None:
        if result["ok"]:
            result["mismatch"] = message
        result["ok"] = False

    def visit(a, b, location: str) -> None:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                fail(location + ": ndarray type mismatch")
                return
            if a.shape != b.shape or a.dtype != b.dtype:
                fail(location + ": ndarray shape/dtype mismatch")
                return
            if a.size == 0:
                return
            if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
                aa, bb = a.astype(np.float64), b.astype(np.float64)
                if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
                    fail(location + ": non-finite ndarray")
                    return
                diff = np.abs(aa - bb)
                max_abs = float(np.max(diff))
                denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1e-30)
                max_rel = float(np.max(diff / denom))
                result["max_abs"] = max(result["max_abs"], max_abs)
                result["max_rel"] = max(result["max_rel"], max_rel)
                if not np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True):
                    fail(location + ": numeric tolerance exceeded")
            elif not np.array_equal(a, b):
                fail(location + ": exact ndarray mismatch")
            return
        try:
            import torch
        except ImportError:  # NumPy-only metadata checks do not require a local torch install.
            torch = None
        if torch is not None and (torch.is_tensor(a) or torch.is_tensor(b)):
            if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape:
                fail(location + ": tensor type/shape mismatch")
                return
            if a.dtype != b.dtype:
                fail(location + ": tensor dtype mismatch")
                return
            if a.numel() == 0:
                return
            if a.is_floating_point() or a.is_complex():
                aa, bb = a.detach().cpu().to(torch.float64), b.detach().cpu().to(torch.float64)
                if not bool(torch.isfinite(aa).all()) or not bool(torch.isfinite(bb).all()):
                    fail(location + ": non-finite tensor")
                    return
                diff = (aa - bb).abs()
                max_abs = float(diff.max())
                denom = torch.maximum(aa.abs(), bb.abs()).clamp_min(1e-30)
                max_rel = float((diff / denom).max())
                result["max_abs"] = max(result["max_abs"], max_abs)
                result["max_rel"] = max(result["max_rel"], max_rel)
                if not torch.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=False):
                    fail(location + ": numeric tolerance exceeded")
            elif not torch.equal(a.cpu(), b.cpu()):
                fail(location + ": exact tensor mismatch")
            return
        if isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
                fail(location + ": mapping keys mismatch")
                return
            for key in sorted(a, key=str):
                visit(a[key], b[key], "%s.%s" % (location, key))
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(location + ": sequence mismatch")
                return
            for index, (aa, bb) in enumerate(zip(a, b)):
                visit(aa, bb, "%s[%d]" % (location, index))
            return
        if isinstance(a, float) or isinstance(b, float):
            try:
                aa, bb = float(a), float(b)
                if not math.isfinite(aa) or not math.isfinite(bb):
                    fail(location + ": non-finite scalar")
                    return
                delta = abs(aa - bb)
                scale = max(abs(aa), abs(bb), 1e-30)
            except (TypeError, ValueError):
                fail(location + ": scalar type mismatch")
                return
            result["max_abs"] = max(result["max_abs"], delta)
            result["max_rel"] = max(result["max_rel"], delta / scale)
            if delta > atol + rtol * abs(float(b)):
                fail(location + ": scalar tolerance exceeded")
        elif a != b:
            fail(location + ": exact value mismatch")

    visit(left, right, path)
    return result


_METRIC_TELEMETRY = frozenset({"steps_per_sec", "tokens_per_sec"})


def _normalized_metrics(out_dir: pathlib.Path) -> tuple[dict[object, dict[str, float]], dict]:
    """Normalize CSVLogger output without inventing optimizer-step identity.

    Automatic optimization emits a useful, unique ``step`` column, so those files are
    folded into one optimizer-step series.  This project uses manual optimization for
    the recovery rehearsal, for which Lightning emits the constant value zero.  In that
    documented case the stable identity is the immutable logger segment plus row ordinal.
    Mixed/partly duplicated step histories remain ambiguous and fail closed.
    """
    paths = sorted(out_dir.glob("*/version_*/metrics.csv"))
    if not paths:
        raise GateError("no CSV metric history under %s" % out_dir)
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.DictReader(stream), start=2):
                raw_step = row.get("step", "")
                if raw_step in (None, ""):
                    continue
                step = int(float(raw_step))
                normalized = {}
                for key, value in row.items():
                    if key == "step" or key in _METRIC_TELEMETRY or value in (None, ""):
                        continue
                    try:
                        normalized[key] = float(value)
                    except ValueError as exc:
                        raise GateError("nonnumeric metric %s at %s:%d" %
                                        (key, path, line_number)) from exc
                    if not math.isfinite(normalized[key]):
                        raise GateError("non-finite metric %s at %s:%d" %
                                        (key, path, line_number))
                if not normalized:
                    raise GateError("metric step %d contains no stable metrics" % step)
                rows.append((path, line_number, step, normalized))
    if not rows:
        raise GateError("CSV metric history contains no optimizer steps")

    raw_steps = [row[2] for row in rows]
    if len(set(raw_steps)) == len(raw_steps):
        ordered = sorted(raw_steps)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            raise GateError("metric history has an optimizer-step gap")
        values = {step: normalized for _path, _line, step, normalized in rows}
        sources = {step: "%s:%d" % (path, line) for path, line, step, _row in rows}
        return values, {
            "mode": "optimizer_step", "paths": [str(path) for path in paths],
            "first_step": ordered[0], "last_step": ordered[-1],
            "step_count": len(ordered), "sources": sources,
        }

    if len(set(raw_steps)) != 1:
        duplicate = next(step for step in raw_steps if raw_steps.count(step) > 1)
        raise GateError("duplicate metric step %d across CSVLogger histories" % duplicate)

    # Lightning's manual-optimization CSVLogger leaves ``step`` constant.  Preserve
    # logger-version boundaries so a missing or duplicated resume row cannot be hidden.
    values = {}
    sources = {}
    segment_counts = {}
    seen_versions = {}
    for path in paths:
        try:
            version = int(path.parent.name.removeprefix("version_"))
        except ValueError as exc:
            raise GateError("invalid CSVLogger version path %s" % path) from exc
        if version in seen_versions:
            raise GateError(
                "duplicate CSVLogger version %d under %s and %s" %
                (version, seen_versions[version], path)
            )
        seen_versions[version] = path
        segment_rows = [row for row in rows if row[0] == path]
        segment_counts[str(version)] = len(segment_rows)
        for ordinal, (_path, line, _step, normalized) in enumerate(segment_rows, start=1):
            key = (version, ordinal)
            values[key] = normalized
            sources["%d:%d" % key] = "%s:%d" % (path, line)
    return values, {
        "mode": "logger_segment_row", "constant_raw_step": raw_steps[0],
        "paths": [str(path) for path in paths], "segments": segment_counts,
        "step_count": len(values), "sources": sources,
    }


def compare_metric_histories(clean_out: pathlib.Path, resumed_out: pathlib.Path) -> dict:
    clean, clean_info = _normalized_metrics(clean_out)
    resumed, resumed_info = _normalized_metrics(resumed_out)
    if set(clean) != set(resumed):
        return {
            "ok": False, "mismatch": "optimizer-step sets differ",
            "missing_steps": sorted(set(clean) - set(resumed)),
            "extra_steps": sorted(set(resumed) - set(clean)),
            "clean": clean_info, "resumed": resumed_info,
        }
    worst_abs = 0.0
    worst_rel = 0.0
    mismatch = None
    atol, rtol = TOLERANCES["metrics_atol"], TOLERANCES["metrics_rtol"]
    for step in sorted(clean):
        if set(clean[step]) != set(resumed[step]):
            mismatch = "metric fields differ at identity %s" % (step,)
            break
        for key in sorted(clean[step]):
            left, right = clean[step][key], resumed[step][key]
            delta = abs(left - right)
            relative = delta / max(abs(left), abs(right), 1e-30)
            worst_abs, worst_rel = max(worst_abs, delta), max(worst_rel, relative)
            if delta > atol + rtol * abs(right):
                mismatch = "metric %s differs at identity %s" % (key, step)
                break
        if mismatch:
            break
    return {
        "ok": mismatch is None, "mismatch": mismatch,
        "max_abs": worst_abs, "max_rel": worst_rel,
        "ignored_telemetry": sorted(_METRIC_TELEMETRY),
        "clean": clean_info, "resumed": resumed_info,
    }


def _probe_logits(checkpoint: dict):
    import torch
    from models.model_gpt import GPTConfig, Transformer

    config = GPTConfig(block_size=69, vocab_size=106, n_layer=12, n_head=6, n_embd=384,
                       dropout=0.0, bias=False, context_length=62, eos_token_id=104)
    model = Transformer(config)
    state = checkpoint["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        normalized = {}
        for key, value in state.items():
            for prefix in ("_forward_module.", "module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            normalized[key] = value
        model.load_state_dict(normalized, strict=True)
    model.eval()
    probe = torch.tensor([[0, 1, 100, 2, 3, 101, 4, 5, 104]], dtype=torch.long)
    with torch.inference_mode():
        return model(probe).detach().cpu()


def compare_final_checkpoints(reference_path: pathlib.Path, recovered_path: pathlib.Path,
                              *, reference_out: pathlib.Path, recovered_out: pathlib.Path,
                              reference_observations: dict, recovered_observations: dict,
                              lineage_sha: str) -> dict:
    from lurestar_runtime import verify_checkpoint

    reference_meta, reference = verify_checkpoint(
        reference_path, deserialize=True, require_metadata=True
    )
    recovered_meta, recovered = verify_checkpoint(
        recovered_path, deserialize=True, require_metadata=True
    )
    if (reference.get("training_steps") != TRAIN_STEPS or
            recovered.get("training_steps") != TRAIN_STEPS):
        raise GateError("a final checkpoint is not exactly step 300")
    if lineage_sha == recovered_meta["sha256"]:
        raise GateError("recovered final checkpoint did not advance beyond its parent")
    weights = _tree_metrics(reference["model"], recovered["model"],
                            atol=TOLERANCES["weights_atol"], rtol=TOLERANCES["weights_rtol"],
                            path="model")
    optimizer = _tree_metrics(reference["optimizer"], recovered["optimizer"],
                              atol=TOLERANCES["optimizer_atol"],
                              rtol=TOLERANCES["optimizer_rtol"], path="optimizer")
    scheduler = _tree_metrics(reference.get("lr_scheduler_state"),
                              recovered.get("lr_scheduler_state"),
                              atol=0.0, rtol=0.0, path="scheduler")
    rng = _tree_metrics(reference.get("lurestar_rng_state_v1"),
                        recovered.get("lurestar_rng_state_v1"),
                        atol=0.0, rtol=0.0, path="rng")
    if AMP_SCALER_KEY not in reference or AMP_SCALER_KEY not in recovered:
        missing = [label for label, payload in (
            ("reference", reference), ("recovered", recovered)
        ) if AMP_SCALER_KEY not in payload]
        raise GateError("FP16 final checkpoint lacks AMP GradScaler state: %s" %
                        ", ".join(missing))
    amp_scaler = _tree_metrics(
        reference[AMP_SCALER_KEY], recovered[AMP_SCALER_KEY],
        atol=0.0, rtol=0.0, path="amp_grad_scaler",
    )
    reference_logits, recovered_logits = _probe_logits(reference), _probe_logits(recovered)
    logits = _tree_metrics(reference_logits, recovered_logits,
                           atol=TOLERANCES["logits_atol"],
                           rtol=TOLERANCES["logits_rtol"], path="logits")
    metrics = compare_metric_histories(reference_out, recovered_out)
    batches_per_epoch = TRAIN_SAMPLES // 512
    data_position = {
        "ok": (TRAIN_STEPS // batches_per_epoch, TRAIN_STEPS % batches_per_epoch) ==
              (int(recovered["training_steps"]) // batches_per_epoch,
               int(recovered["training_steps"]) % batches_per_epoch),
        "batches_per_epoch": batches_per_epoch,
        "final_epoch": TRAIN_STEPS // batches_per_epoch,
        "final_cursor": TRAIN_STEPS % batches_per_epoch,
        "resume_fast_forward_step": INTERRUPT_STEP,
        "reference_fast_forward_observed":
            _observed_fast_forward(reference_observations, INTERRUPT_STEP),
        "recovered_fast_forward_observed":
            _observed_fast_forward(recovered_observations, INTERRUPT_STEP),
        "observation_schema": TrainingObservationLatch.SCHEMA,
        "reference_observation": reference_observations,
        "recovered_observation": recovered_observations,
    }
    data_position["ok"] = (
        data_position["ok"] and data_position["reference_fast_forward_observed"] and
        data_position["recovered_fast_forward_observed"]
    )
    checks = {
        "final_step": True, "weights": weights, "optimizer": optimizer,
        "scheduler": scheduler, "rng": rng, "amp_grad_scaler": amp_scaler,
        "logits": logits, "metrics": metrics,
        "data_position": data_position,
        "checkpoint_lineage": {
            "ok": lineage_sha != recovered_meta["sha256"],
            "shared_parent_sha256": lineage_sha,
            "recovered_final_sha256": recovered_meta["sha256"],
        },
    }
    passed = all(value is True or (isinstance(value, dict) and value.get("ok"))
                 for value in checks.values())
    return {
        "passed": passed, "checks": checks, "tolerances": dict(TOLERANCES),
        "reference_checkpoint": reference_meta, "recovered_checkpoint": recovered_meta,
    }


def _latest_checkpoint(out_dir: pathlib.Path) -> pathlib.Path:
    pointer = out_dir / "latest_ckpt"
    if not pointer.is_file():
        raise GateError("final latest_ckpt pointer missing")
    target = pathlib.Path(pointer.read_text().strip())
    if not target.is_file():
        raise GateError("final latest_ckpt target missing")
    return target


def runtime_main() -> int:
    spec = json.loads(pathlib.Path(REMOTE_SPEC).read_text())
    validate_spec(spec)
    if not spec.get("preregistration_sha256"):
        raise GateError("unsigned recovery-gate sidecar")
    adc = pathlib.Path(ADC_PATH)
    if not adc.is_file():
        raise GateError("uploaded ADC missing")
    os.chmod(adc, 0o600)
    if adc.stat().st_mode & 0o077:
        raise GateError("ADC mode is not 0600")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ADC_PATH
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(stop,), daemon=True)
    heartbeat.start()
    original_torch = _verify_t4()
    run_argv([sys.executable, "-m", "pip", "install", "-q",
              "google-cloud-storage", "google-auth"])
    _verify_t4(original_torch)
    from google.cloud import storage
    bucket = storage.Client(project=GCP_PROJECT).bucket(BUCKET)
    _runtime_event(bucket, spec, 1, "runtime_verified", torch=original_torch)

    runtime_root = pathlib.Path("/content/rgate") / spec["gate_id"]
    project = runtime_root / "project"
    upstream = runtime_root / "nextlat"
    source_archive = runtime_root / "project.tar.gz"
    runtime_root.mkdir(parents=True, exist_ok=True)
    progress_errors: list[str] = []
    progress = threading.Thread(
        target=_durable_progress_loop,
        args=(bucket, spec, runtime_root, stop, progress_errors), daemon=True,
    )
    progress.start()
    bucket.blob(spec["source_object"]).download_to_filename(str(source_archive))
    if sha256_file(source_archive) != spec["source_sha256"]:
        raise GateError("downloaded source snapshot hash mismatch")
    _safe_extract(source_archive, project)
    run_argv(["git", "clone", "-q", UPSTREAM_URL, str(upstream)])
    run_argv(["git", "-C", str(upstream), "checkout", "-q", PINNED_COMMIT])
    head = run_argv(["git", "-C", str(upstream), "rev-parse", "HEAD"], relay=False)[1].strip()
    if head != PINNED_COMMIT:
        raise GateError("pinned upstream checkout drift")
    requirements = runtime_root / "requirements-no-torch.txt"
    requirements.write_text("".join(
        line for line in (upstream / "requirements.txt").read_text().splitlines(True)
        if not line.lstrip().startswith("torch")
    ))
    run_argv([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    _verify_t4(original_torch)
    run_argv([sys.executable, str(project / "scripts" / "runtime_bootstrap.py"),
              "--project-root", str(project), "--upstream", str(upstream)])
    sys.path.insert(0, str(upstream))
    train, test = _generate_gate_data(upstream, runtime_root / "data")
    _runtime_event(bucket, spec, 2, "preregistered",
                   preregistration_sha256=spec["preregistration_sha256"],
                   train_sha256=sha256_file(train), test_sha256=sha256_file(test),
                   tolerances=spec["tolerances"])

    # One scratch lineage is the parent of both comparison arms.  Its process tree is
    # intentionally killed at step 150 before any snapshot reads begin.
    active_root = runtime_root / "active-lineage"
    active_out = active_root / "runs" / "gpt" / ("seed%d" % SEED) / "base"
    scratch_cmd, scratch_env = _training_command(
        project, upstream, active_root, train, test, resume=False
    )
    target, termination_log = _kill_at_checkpoint(
        scratch_cmd, scratch_env, active_out
    )
    durable = _upload_resume_snapshot(bucket, spec, active_out, target)
    _assert_committed_lineage(target, durable)
    _runtime_event(
        bucket, spec, 3, "intentional_sigkill_tree_verified",
        step=INTERRUPT_STEP, checkpoint_sha256=durable["checkpoint_sha256"],
        termination=json.loads(termination_log.splitlines()[0]),
    )

    # First continue the retained local snapshot. This becomes the reference arm.
    # The active path is then vacated, allowing the GCS copy to be restored to the exact
    # same absolute paths for the recovered arm.
    reference_cmd, reference_env = _training_command(
        project, upstream, active_root, train, test, resume=True
    )
    _reference_log, reference_observations = _finish_training(
        reference_cmd, reference_env,
        observation_path=runtime_root / "observations" / "reference.json",
    )
    reference_final_active = _latest_checkpoint(active_out)
    reference_relative = reference_final_active.relative_to(active_root)
    reference_state = _publish_final_checkpoint(
        bucket, spec, "reference", reference_final_active
    )
    reference_finished_root = runtime_root / "reference-finished"
    if reference_finished_root.exists():
        raise GateError("reference-finished target unexpectedly exists")
    os.replace(active_root, reference_finished_root)
    reference_out = reference_finished_root / "runs" / "gpt" / ("seed%d" % SEED) / "base"
    reference_final = reference_finished_root / reference_relative
    _runtime_event(
        bucket, spec, 4, "reference_complete",
        checkpoint_sha256=reference_state["checkpoint_sha256"],
        shared_parent_sha256=durable["checkpoint_sha256"],
    )

    # Restore only from the committed state record. A failed/partial artifact upload has
    # no state.json and therefore cannot enter this path.
    state = json.loads(bucket.blob("%s/state.json" % spec["resume_prefix"]).download_as_text())
    restored = _restore_resume_snapshot(bucket, spec, state)
    if state != durable:
        raise GateError("downloaded lineage commit differs from the retained commit")
    _assert_committed_lineage(restored, durable)
    if restored.resolve() != target.resolve():
        raise GateError("restored lineage did not return to its original absolute path")
    _runtime_event(
        bucket, spec, 5, "gcs_restore_verified",
        checkpoint_sha256=sha256_file(restored),
        reference_parent_sha256=durable["checkpoint_sha256"],
    )

    resume_cmd, resume_env = _training_command(
        project, upstream, active_root, train, test, resume=True
    )
    _resume_log, recovered_observations = _finish_training(
        resume_cmd, resume_env,
        observation_path=runtime_root / "observations" / "recovered.json",
    )
    recovered_out = active_root / "runs" / "gpt" / ("seed%d" % SEED) / "base"
    recovered_final = _latest_checkpoint(recovered_out)
    recovered_state = _publish_final_checkpoint(
        bucket, spec, "recovered", recovered_final
    )
    comparison = compare_final_checkpoints(
        reference_final, recovered_final,
        reference_out=reference_out, recovered_out=recovered_out,
        reference_observations=reference_observations,
        recovered_observations=recovered_observations,
        lineage_sha=durable["checkpoint_sha256"],
    )
    try:
        _publish_progress_snapshot(bucket, spec, runtime_root, 999999)
    except Exception as exc:
        progress_errors.append("%s: %s" % (type(exc).__name__, exc))
    stop.set()
    progress.join(timeout=5)
    heartbeat.join(timeout=5)
    if progress.is_alive():
        progress_errors.append("durable progress publisher did not stop within 5 seconds")
    comparison["checks"]["durable_progress"] = {
        "ok": not progress_errors,
        "cadence_seconds": DURABLE_PROGRESS_SECONDS,
        "errors": progress_errors,
    }
    comparison["passed"] = comparison["passed"] and not progress_errors
    result = {
        "schema": SCHEMA, "gate_id": spec["gate_id"],
        "preregistration_sha256": spec["preregistration_sha256"],
        "source_sha256": spec["source_sha256"], "seed": SEED,
        "data_first_generator_seed": DATA_FIRST_SEED,
        "confirmatory_data_used": False, "confirmatory_seed_used": False,
        "intentional_termination": (
            "stable PPID tree freeze and per-identity SIGKILL after atomic "
            "step-150 pointer publication"
        ),
        "comparison_design": (
            "retained-local reference versus GCS-restored continuation from one "
            "hash-identical step-150 lineage at identical absolute paths"
        ),
        "shared_lineage_checkpoint_sha256": durable["checkpoint_sha256"],
        "durable_reference_final": reference_state,
        "durable_recovered_final": recovered_state,
        **comparison,
    }
    bucket.blob(spec["result_object"]).upload_from_string(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        content_type="application/json", if_generation_match=0,
    )
    _runtime_event(
        bucket, spec, 6, "gate_complete", passed=result["passed"],
        reference_checkpoint_sha256=reference_state["checkpoint_sha256"],
        recovered_checkpoint_sha256=recovered_state["checkpoint_sha256"],
    )
    print("RECOVERY_GATE_PASSED=%s" % result["passed"], flush=True)
    return 0 if result["passed"] else 3


def host_main(*, run: bool, receipt_path: pathlib.Path) -> int:
    archive = PROJECT_ROOT / ".agent_state" / "recovery-gate-project.tar.gz"
    package_project(PROJECT_ROOT, archive)
    source_sha = sha256_file(archive)
    gate_id = "rg-%s-%d-%s" % (source_sha[:12], time.time_ns(), uuid.uuid4().hex[:8])
    spec = build_spec(source_sha, gate_id)
    sidecar = PROJECT_ROOT / ".agent_state" / ("recovery-gate-%s.json" % gate_id)
    atomic_json(sidecar, spec)
    append_receipt(receipt_path, {
        "schema": SCHEMA, "gate_id": gate_id, "event": "PREREGISTERED",
        "source_sha256": source_sha,
        "preregistration_sha256": spec["preregistration_sha256"],
        "tolerances": TOLERANCES, "seed": SEED,
        "data_first_generator_seed": DATA_FIRST_SEED,
    })
    print("RECOVERY_GATE_ID=%s" % gate_id)
    print("SOURCE_SHA256=%s" % source_sha)
    print("PREREGISTRATION_SHA256=%s" % spec["preregistration_sha256"])
    print("SIDECAR=%s" % sidecar)
    if not run:
        print("PREPARED_ONLY=True (no Colab runtime provisioned)")
        return 0

    quota = parse_cli_json(run_argv(["colab", "quota", "--json"], relay=False)[1])
    if float(quota.get("paid_balance", 0.0)) <= HARD_STOP_BALANCE_CU:
        raise GateError("compute hard-stop balance reached")
    first, second = status_pair()
    state = agreed_runtime_state(first, second)
    if state != "gone":
        raise GateError("refusing recovery gate unless two status reads agree no runtime: %s" % state)
    adc = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if not adc.is_file() or adc.stat().st_mode & 0o077:
        raise GateError("local ADC missing or not mode 0600")

    source_uri = "gs://%s/%s" % (BUCKET, spec["source_object"])
    sidecar_uri = "gs://%s/%s/%s/job-%s.json" % (
        BUCKET, GCS_PREFIX, gate_id, spec["preregistration_sha256"]
    )
    run_argv(["gcloud", "storage", "cp", str(archive), source_uri])
    run_argv(["gcloud", "storage", "cp", str(sidecar), sidecar_uri])
    session = None
    start_attempted = False
    try:
        start_attempted = True
        started = parse_cli_json(run_argv(["colab", "start", "--gpu", "t4", "--json"],
                                         relay=False)[1])
        session = str(started.get("session", ""))
        if not session:
            raise GateError("Colab start returned no session id")
        append_receipt(receipt_path, {"schema": SCHEMA, "gate_id": gate_id,
                                      "event": "SESSION_STARTED", "session": session})
        adc_attempts = upload_with_retry(session, adc, ADC_PATH)
        spec_attempts = upload_with_retry(session, sidecar, REMOTE_SPEC)
        append_receipt(receipt_path, {
            "schema": SCHEMA, "gate_id": gate_id, "event": "INPUTS_UPLOADED",
            "adc_attempts": adc_attempts, "sidecar_attempts": spec_attempts,
        })
        exec_started = time.time()
        rc, tail = run_argv(["colab", "exec", "--session", session, "--timeout", "120m",
                             str(pathlib.Path(__file__).resolve())], check=False)
        append_receipt(receipt_path, {"schema": SCHEMA, "gate_id": gate_id,
                                      "event": "EXEC_RETURNED", "returncode": rc,
                                      "tail": tail[-12000:]})
        status_rc, status_text = run_argv(
            ["colab", "status", "--json"], check=False, relay=False, max_lines=None
        )
        diagnostic = {
            "schema": SCHEMA, "gate_id": gate_id, "event": "HOST_EXEC_DIAGNOSTIC",
            "recorded_at_unix": time.time(), "elapsed_seconds": time.time() - exec_started,
            "exec_returncode": rc, "exec_tail": tail[-12000:],
            "status_returncode": status_rc, "status_output": status_text,
            "timeout_possible": rc != 0,
        }
        diagnostic_path = PROJECT_ROOT / ".agent_state" / (
            "recovery-gate-diagnostic-%s.json" % gate_id
        )
        atomic_json(diagnostic_path, diagnostic)
        diagnostic_uri = "gs://%s/%s/%s/host-diagnostics/%s.json" % (
            BUCKET, GCS_PREFIX, gate_id, uuid.uuid4().hex
        )
        diagnostic_rc, diagnostic_tail = 1, "not attempted"
        diagnostic_attempts = 0
        for diagnostic_attempts in range(1, 4):
            diagnostic_rc, diagnostic_tail = run_argv(
                ["gcloud", "storage", "cp", str(diagnostic_path), diagnostic_uri],
                check=False,
            )
            if diagnostic_rc == 0:
                break
        append_receipt(receipt_path, {
            "schema": SCHEMA, "gate_id": gate_id, "event": "HOST_DIAGNOSTIC",
            "object": diagnostic_uri, "upload_returncode": diagnostic_rc,
            "upload_attempts": diagnostic_attempts, "upload_tail": diagnostic_tail[-2000:],
        })
        if diagnostic_rc:
            raise GateError("host terminal diagnostic was not durably uploaded after 3 attempts")
    finally:
        if session:
            stopped = False
            for _ in range(2):
                run_argv(["colab", "stop", "--session", session], check=False)
                first, second = status_pair()
                if agreed_runtime_state(first, second) == "gone":
                    stopped = True
                    break
            if not stopped:
                raise GateError("runtime did not reach a two-read stopped state")
            quota_first, quota_second = quota_pair()
            for settled in (quota_first, quota_second):
                if (int(settled.get("active_runtimes", -1)) != 0 or
                        float(settled.get("burn_rate_hourly", -1)) != 0.0):
                    raise GateError("post-stop quota reads did not both settle to zero burn")
            append_receipt(receipt_path, {"schema": SCHEMA, "gate_id": gate_id,
                                          "event": "SESSION_STOP_VERIFIED",
                                          "settled_balance_cu":
                                              quota_second.get("paid_balance")})
        elif start_attempted:
            first, second = status_pair()
            if agreed_runtime_state(first, second) != "gone":
                raise GateError(
                    "Colab start established no owned session id; refusing an unscoped stop"
                )

    result_uri = "gs://%s/%s" % (BUCKET, spec["result_object"])
    rc, result_text = run_argv(["gcloud", "storage", "cat", result_uri],
                               check=False, relay=False, max_lines=None)
    if rc:
        raise GateError("durable gate result is absent after exec return")
    result = json.loads(result_text)
    if result.get("gate_id") != gate_id or result.get("source_sha256") != source_sha:
        raise GateError("durable result identity mismatch")
    append_receipt(receipt_path, {"schema": SCHEMA, "gate_id": gate_id,
                                  "event": "RESULT", "result": result})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 3


def main(argv: list[str] | None = None) -> int:
    if pathlib.Path(REMOTE_SPEC).is_file():
        return runtime_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="provision one paid T4 and execute the preregistered gate")
    parser.add_argument("--receipt", type=pathlib.Path,
                        default=PROJECT_ROOT / "results" / "recovery_gate_receipts.jsonl")
    args = parser.parse_args(argv)
    try:
        return host_main(run=args.run, receipt_path=args.receipt)
    except GateError as exc:
        print("RECOVERY_GATE_REFUSED: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
