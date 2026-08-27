#!/usr/bin/env python3
"""Fail-closed attempt planning for resumable profiling jobs.

This module does not discover orphaned checkpoint objects.  It considers only a checkpoint
named by a restored, committed pointer and verifies its runtime-bootstrap metadata before
requesting ``trainer.init_from=resume``.  The attempt ledger lets profile_summarize discard
telemetry after a lost attempt's last durable checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import re


class ResumeError(RuntimeError):
    pass


def probe_succeeded(probe: dict) -> bool:
    """Accept normalized success plus the legacy success spelling already durable in GCS."""
    return probe.get("exit") in {"ok", "SystemExit(0)", "SystemExit(None)"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as stream:
        stream.write((json.dumps(document, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def verified_resume_step(out_dir: pathlib.Path, experiment: str, target_steps: int) -> int:
    """Return the committed optimizer step, or zero when no pointer exists."""
    out_dir = out_dir.resolve()
    pointers = (out_dir / "recovery_ckpt", out_dir / "latest_ckpt")
    pointer = next((path for path in pointers if path.is_file()), None)
    if pointer is None:
        return 0
    raw = pointer.read_text().strip()
    if not raw:
        raise ResumeError("profile resume pointer is empty: %s" % pointer)
    checkpoint = pathlib.Path(raw).resolve()
    expected_dir = (out_dir / experiment).resolve()
    if checkpoint.parent != expected_dir or not re.fullmatch(
            r"(?:recovery_)?ckpt_iter_[0-9]+(?:_[0-9.]+)?\.pt", checkpoint.name):
        raise ResumeError("profile resume pointer escapes the exact experiment: %s" % checkpoint)
    metadata_path = checkpoint.with_name(checkpoint.name + ".meta.json")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise ResumeError("profile resume checkpoint/metadata is incomplete: %s" % checkpoint)
    metadata = json.loads(metadata_path.read_text())
    if (int(metadata.get("size_bytes", -1)) != checkpoint.stat().st_size or
            metadata.get("sha256") != _sha256(checkpoint)):
        raise ResumeError("profile resume checkpoint failed hash/size verification")
    step = int(metadata.get("training_steps", -1))
    if not 0 < step <= target_steps:
        raise ResumeError("profile resume checkpoint step is outside (0, %d]: %d" %
                          (target_steps, step))
    filename_step = re.search(r"ckpt_iter_([0-9]+)", checkpoint.name)
    if filename_step is None or int(filename_step.group(1)) != step:
        raise ResumeError("checkpoint filename and training_steps disagree")
    return step


def completed_job(manifest_path: pathlib.Path, *, job: str, steps: int,
                  warmup_steps: int) -> bool:
    """A receipt skips compute only with complete output and final checkpoint evidence."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("job") != job or int(manifest.get("returncode", -1)) != 0 or
                int(manifest.get("steps", -1)) != steps or
                int(manifest.get("warmup_steps", -1)) != warmup_steps):
            return False
        log = pathlib.Path(manifest["log"])
        probes = [pathlib.Path(path) for path in glob.glob(manifest["probe_glob"])]
        if not log.is_file() or log.stat().st_size == 0 or not probes:
            return False
        if not any(probe_succeeded(json.loads(path.read_text())) for path in probes):
            return False
        return verified_resume_step(
            pathlib.Path(manifest["out_dir"]), manifest["experiment_name"], steps
        ) == steps
    except (KeyError, OSError, ValueError, json.JSONDecodeError, ResumeError):
        return False


def plan_attempt(*, jobs_dir: pathlib.Path, job: str, out_dir: pathlib.Path,
                 experiment: str, steps: int, warmup_steps: int) -> dict:
    manifest = jobs_dir / (job + ".job.json")
    ledger_path = jobs_dir / (job + ".attempts.json")
    if completed_job(manifest, job=job, steps=steps, warmup_steps=warmup_steps):
        return {"action": "skip", "resume_step": steps, "ledger": str(ledger_path)}

    resume_step = verified_resume_step(out_dir, experiment, steps)
    exp_dir = out_dir / experiment
    version_count = len(list(exp_dir.glob("version_*"))) if exp_dir.is_dir() else 0
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
    else:
        ledger = {"schema": "nextlat_forgetting/profile_attempts/1", "job": job,
                  "target_steps": steps, "warmup_steps": warmup_steps, "attempts": []}
    if (ledger.get("job") != job or int(ledger.get("target_steps", -1)) != steps or
            int(ledger.get("warmup_steps", -1)) != warmup_steps):
        raise ResumeError("profile attempt ledger differs from the frozen job contract")
    attempts = ledger.setdefault("attempts", [])
    successful_probes = []
    for path in glob.glob(str(jobs_dir / (job + ".probe.*.json"))):
        try:
            probe = json.loads(pathlib.Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if probe_succeeded(probe):
            successful_probes.append(probe)
    log_path = jobs_dir / (job + ".log")
    final_evidence_ready = log_path.is_file() and log_path.stat().st_size > 0
    if resume_step == steps and successful_probes and final_evidence_ready:
        if not attempts:
            if version_count != 1:
                raise ResumeError("target-step receipt repair needs exactly one logger version")
            attempts.append({"attempt": 0, "resume_step": 0, "version_start_index": 0})
            _atomic_json(ledger_path, ledger)
        final_attempt = len(attempts) - 1
        matching = [probe for probe in successful_probes
                    if int(probe.get("profile_attempt", -1)) == final_attempt]
        if len(matching) != 1:
            raise ResumeError(
                "target-step receipt repair needs exactly one final-attempt probe")
        return {"action": "finalize", "resume_step": steps, "ledger": str(ledger_path),
                "attempt": final_attempt}
    candidate = {"attempt": len(attempts), "resume_step": resume_step,
                 "version_start_index": version_count}
    # Idempotence when a shell dies after writing the ledger but before Fabric creates a logger.
    if attempts and {k: attempts[-1].get(k) for k in candidate if k != "attempt"} == {
            k: candidate[k] for k in candidate if k != "attempt"}:
        candidate = attempts[-1]
    else:
        attempts.append(candidate)
        _atomic_json(ledger_path, ledger)
    return {"action": "run", "resume_step": resume_step, "ledger": str(ledger_path),
            "attempt": int(candidate["attempt"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", required=True, type=pathlib.Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(plan_attempt(
        jobs_dir=args.jobs_dir, job=args.job, out_dir=args.out_dir,
        experiment=args.experiment, steps=args.steps, warmup_steps=args.warmup,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
