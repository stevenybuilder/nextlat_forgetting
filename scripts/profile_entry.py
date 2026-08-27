#!/usr/bin/env python
"""Instrumented entry point for the spec section 11 profiling gate.

`fabric run` launches THIS file instead of train.py; it installs three read-only probes and
then executes the pinned `train.py` unchanged via runpy, so the training code path, the
config merge and the CLI dotlist all behave exactly as in a confirmatory run.

    fabric run --devices 1 --precision bf16-mixed scripts/profile_entry.py \
        --config configs/nextlat_lurestar.yaml trainer.train_batches=500 ...

Why an entry wrapper rather than a driver-side measurement: docs/RUNLOG.md records that the
first profiling attempt read `torch.cuda.max_memory_allocated()` in the process that shelled
out to `fabric run`, not in the process that actually trained, and therefore reported
0.00 GB. Peak VRAM is only observable from inside the training process. The probes here run
in that process and write their record with `atexit` plus a SIGTERM handler, so an
interrupted profile still leaves its numbers behind.

What is measured, and what is deliberately not:
  * peak allocated / reserved VRAM  - torch.cuda high-water marks for this process.
  * host-input wait                 - wall time spent inside `next()` on a DataLoader
                                      iterator. core_train.py:481 starts its own step timer
                                      AFTER the batch is yielded, so `steps_per_sec` in
                                      metrics.csv is pure compute and this is the missing
                                      half of the wall clock.
  * checkpoint write                - duration and bytes of every `Fabric.save`.
None of the probes mutate a tensor, a gradient or a config value. NextLat's backward is a
manual two-stage graph split (model_nextlat.py:503-525) that an in-place hook would corrupt,
so nothing here touches model internals at all.
"""

from __future__ import annotations

import atexit
import json
import os
import runpy
import signal
import sys
import time

import torch
import lightning as L

PROBE_PATH = os.environ.get("PROFILE_PROBE_JSON")
if not PROBE_PATH:
    raise SystemExit("profile_entry.py: PROFILE_PROBE_JSON must name the output file")
PROBE_PATH = PROBE_PATH.replace("{pid}", str(os.getpid()))

_probe = {
    "pid": os.getpid(),
    "profile_attempt": int(os.environ.get("PROFILE_ATTEMPT", "0")),
    "argv": list(sys.argv),
    "process_start_unix": time.time(),
    "process_start_perf": time.perf_counter(),
    "dataloader_wait_s": 0.0,
    "dataloader_batches": 0,
    "checkpoint_writes": [],  # {path, seconds, bytes}
    "cuda": None,
    "peak_allocated_bytes": None,
    "peak_reserved_bytes": None,
    "wall_seconds": None,
    "exit": "incomplete",
}

if torch.cuda.is_available():
    _probe["cuda"] = {
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    torch.cuda.reset_peak_memory_stats()


# --- probe 1: host-input wait -----------------------------------------------------------
_orig_dataloader_iter = torch.utils.data.DataLoader.__iter__


def _timed_dataloader_iter(self):
    inner = _orig_dataloader_iter(self)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(inner)
        except StopIteration:
            _probe["dataloader_wait_s"] += time.perf_counter() - t0
            return
        _probe["dataloader_wait_s"] += time.perf_counter() - t0
        _probe["dataloader_batches"] += 1
        yield batch


torch.utils.data.DataLoader.__iter__ = _timed_dataloader_iter

# --- probe 2: checkpoint write duration and bytes ---------------------------------------
_orig_fabric_save = L.Fabric.save


def _timed_fabric_save(self, path, *args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_fabric_save(self, path, *args, **kwargs)
    elapsed = time.perf_counter() - t0
    try:
        size = os.path.getsize(path) if os.path.isfile(path) else _dir_bytes(path)
    except OSError:
        size = None
    _probe["checkpoint_writes"].append(
        {"path": str(path), "seconds": elapsed, "bytes": size}
    )
    return result


def _dir_bytes(path) -> int:
    """Fabric writes a directory for distributed checkpoints; sum it."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


L.Fabric.save = _timed_fabric_save


# --- probe 3: peak VRAM, written no matter how the process ends --------------------------
_written = False


def _flush(reason: str = "atexit") -> None:
    global _written
    if _written:
        return
    _written = True
    if torch.cuda.is_available():
        _probe["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        _probe["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    _probe["wall_seconds"] = time.perf_counter() - _probe["process_start_perf"]
    if _probe["exit"] == "incomplete":
        _probe["exit"] = reason
    os.makedirs(os.path.dirname(os.path.abspath(PROBE_PATH)) or ".", exist_ok=True)
    tmp = PROBE_PATH + ".partial"
    with open(tmp, "w") as fh:
        json.dump(_probe, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, PROBE_PATH)
    print(f"[profile_entry] probe written to {PROBE_PATH} ({reason})", flush=True)


atexit.register(_flush)


def _on_signal(signum, _frame):
    _flush(f"signal-{signum}")
    raise SystemExit(128 + signum)


for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _on_signal)
    except (ValueError, OSError):  # not the main thread / unsupported platform
        pass


# --- run the pinned trainer unchanged ----------------------------------------------------
TRAIN_PY = os.environ.get("PROFILE_TRAIN_PY") or os.path.join(os.getcwd(), "train.py")
if not os.path.isfile(TRAIN_PY):
    raise SystemExit(
        f"profile_entry.py: {TRAIN_PY} not found. train.py also loads defaults.yaml "
        f"relative to the CWD (train.py:348), so run this with the repo root as CWD."
    )

# Unlike executing ``python /path/to/train.py``, runpy does not put the target script's
# directory on sys.path.  Fabric/torchrun starts this wrapper as the real script, so without
# this insertion the pinned trainer's sibling imports (notably ``import core_train``) resolve
# against scripts/ and fail even when the working directory is the upstream repository.
TRAIN_DIR = os.path.dirname(os.path.abspath(TRAIN_PY))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

sys.argv[0] = TRAIN_PY
try:
    runpy.run_path(TRAIN_PY, run_name="__main__")
except SystemExit as exc:
    # Some valid trainer entry points use ``raise SystemExit(main())``.  Exit 0/None is a
    # successful process, not a failed probe; preserve nonzero exits verbatim for diagnosis.
    _probe["exit"] = "ok" if exc.code in (0, None) else f"SystemExit({exc.code})"
    raise
except BaseException as exc:  # noqa: BLE001 - record the failure, then re-raise
    _probe["exit"] = f"{type(exc).__name__}: {exc}"
    raise
else:
    _probe["exit"] = "ok"
finally:
    _flush(_probe["exit"])
