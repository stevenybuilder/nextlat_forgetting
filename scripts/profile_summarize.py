#!/usr/bin/env python
"""Turn the raw output of scripts/profile.sh into the spec section 11 profiling record.

Reads, per job: the job manifest written by profile.sh, the CSVLogger metrics.csv the run
produced, the `materialized_config.yaml` the trainer dumped (train.py:192-194, so the
RESOLVED values -- block_size 69, vocab_size 106 -- not the YAML placeholders), the
in-process probe from scripts/profile_entry.py, and the nvidia-smi sample stream.

Emits `<results>/profile_summary.json` and a markdown table on stdout covering every
quantity spec section 11 names:

    median and p95 seconds per step; examples and tokens per second; peak allocated and
    reserved VRAM; GPU utilization and host-input wait; checkpoint-write duration and bytes;
    GPT-vs-NextLat throughput and memory overhead; validation accuracy; projected
    end-to-end runtime.

Timing convention, stated because it changes what the numbers mean: core_train.py:481 starts
its per-step timer AFTER the dataloader has yielded the batch and stops it after
`optimizer_step`, so `steps_per_sec` in metrics.csv is COMPUTE time per optimizer update.
Wall time per step is derived separately from the process clock, and the gap between the two
is the host-input and validation overhead. Both are reported.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_lib import load_yaml_as_trainer_sees_it  # noqa: E402

# Confirmatory matrix sizes, from spec section 11's budget formulas.
N_SEEDS = 3
N_MODELS = 2
BASE_STEPS = 20000
ADAPT_STEPS = 500
ADAPT_BRANCHES_PER_MODEL = N_SEEDS * 2  # near + far, per seed
HMM_STEPS = 3000
INTERRUPTION_MARGIN = 0.20  # spec section 11: "a 20% interruption margin"


def _f(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_metrics(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolation percentile; no numpy dependency in the profiling path."""
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def step_seconds(rows: List[Dict[str, Any]], warmup: int) -> List[float]:
    """Seconds per optimizer update for every logged step at or beyond `warmup`.

    core_train.py:518-521 logs `steps_per_sec` as steps/elapsed over the window since the
    previous log; with log_interval: 1 that window is one step.
    """
    out = []
    for row in rows:
        step = _f(row.get("step"))
        rate = _f(row.get("steps_per_sec"))
        if step is None or rate is None or rate <= 0:
            continue
        if step < warmup:
            continue
        out.append(1.0 / rate)
    return out


def last_value(rows: List[Dict[str, Any]], pattern: str) -> Optional[Dict[str, Any]]:
    rx = re.compile(pattern)
    hit = None
    for row in rows:
        for key, value in row.items():
            if key and rx.search(key):
                val = _f(value)
                if val is not None:
                    hit = {"metric": key, "value": val, "step": _f(row.get("step"))}
    return hit


def read_gpu_samples(path: str) -> Dict[str, Any]:
    """nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used
    --format=csv,noheader,nounits"""
    util, mem_util, mem_used = [], [], []
    if not os.path.isfile(path):
        return {"samples": 0}
    with open(path, newline="") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            u, mu, mb = _f(parts[1]), _f(parts[2]), _f(parts[3])
            if u is None:
                continue
            util.append(u)
            if mu is not None:
                mem_util.append(mu)
            if mb is not None:
                mem_used.append(mb)
    if not util:
        return {"samples": 0}
    return {
        "samples": len(util),
        "gpu_util_pct_median": statistics.median(util),
        "gpu_util_pct_mean": statistics.fmean(util),
        "gpu_util_pct_p05": percentile(util, 0.05),
        "mem_util_pct_median": statistics.median(mem_util) if mem_util else None,
        "mem_used_mib_max": max(mem_used) if mem_used else None,
    }


def read_probes(pattern: str) -> Optional[Dict[str, Any]]:
    """profile_entry.py writes one probe per training process; keep the one that saw the
    most VRAM (with --devices 1 there is exactly one)."""
    files = sorted(glob.glob(pattern))
    probes = []
    for path in files:
        try:
            with open(path) as fh:
                probes.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    if not probes:
        return None
    return max(probes, key=lambda p: (p.get("peak_allocated_bytes") or 0))


GB = 1024 ** 3
MB = 1024 ** 2


def summarize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = job["out_dir"]
    exp = job["experiment_name"]
    exp_dir = os.path.join(out_dir, exp)

    versions = sorted(glob.glob(os.path.join(exp_dir, "version_*")))
    metrics_path = os.path.join(versions[-1], "metrics.csv") if versions else None
    rows = read_metrics(metrics_path) if metrics_path and os.path.isfile(metrics_path) else []

    resolved_path = os.path.join(exp_dir, "materialized_config.yaml")
    resolved = load_yaml_as_trainer_sees_it(resolved_path) if os.path.isfile(resolved_path) else {}

    probe = read_probes(job["probe_glob"])
    gpu = read_gpu_samples(job["gpu_samples_csv"])

    warmup = job["warmup_steps"]
    per_step = step_seconds(rows, warmup)

    eff_batch = resolved.get("data", {}).get("effective_batch_size")
    block_size = resolved.get("model", {}).get("block_size")

    rec: Dict[str, Any] = {
        "job": job["job"],
        "task": job["task"],
        "model": job["model"],
        "config": job["config"],
        "returncode": job.get("returncode"),
        "steps_requested": job["steps"],
        "warmup_steps": warmup,
        "steps_summarized": len(per_step),
        "metrics_csv": metrics_path,
        "resolved": {
            "effective_batch_size": eff_batch,
            "gradient_accum_steps": resolved.get("data", {}).get("gradient_accum_steps"),
            "device_batch_size": resolved.get("data", {}).get("device_batch_size"),
            "micro_batch_size": resolved.get("data", {}).get("micro_batch_size"),
            "block_size": block_size,
            "vocab_size": resolved.get("model", {}).get("vocab_size"),
            "n_layer": resolved.get("model", {}).get("n_layer"),
            "n_embd": resolved.get("model", {}).get("n_embd"),
            "proj_factor": resolved.get("model", {}).get("proj_factor"),
            "compile": resolved.get("trainer", {}).get("compile"),
        },
        "wall_seconds": job.get("wall_seconds"),
        "gpu": gpu,
    }

    if per_step:
        med = statistics.median(per_step)
        rec["seconds_per_step_median"] = med
        rec["seconds_per_step_p95"] = percentile(per_step, 0.95)
        rec["seconds_per_step_mean"] = statistics.fmean(per_step)
        if eff_batch:
            rec["examples_per_second"] = eff_batch / med
            if block_size:
                rec["tokens_per_second"] = eff_batch * block_size / med
    else:
        rec["seconds_per_step_median"] = None
        rec["error"] = (
            "no steps_per_sec rows at or beyond the warmup step; check that "
            "trainer.log_interval is 1 and that the run reached the warmup point"
        )

    if job.get("wall_seconds") and job["steps"]:
        rec["wall_seconds_per_step"] = job["wall_seconds"] / job["steps"]

    if probe:
        rec["peak_allocated_gb"] = (
            probe["peak_allocated_bytes"] / GB if probe.get("peak_allocated_bytes") else None
        )
        rec["peak_reserved_gb"] = (
            probe["peak_reserved_bytes"] / GB if probe.get("peak_reserved_bytes") else None
        )
        rec["host_input_wait_seconds"] = probe.get("dataloader_wait_s")
        rec["dataloader_batches"] = probe.get("dataloader_batches")
        if job.get("wall_seconds") and probe.get("dataloader_wait_s") is not None:
            rec["host_input_wait_fraction"] = probe["dataloader_wait_s"] / job["wall_seconds"]
        writes = probe.get("checkpoint_writes") or []
        if writes:
            secs = [w["seconds"] for w in writes]
            sizes = [w["bytes"] for w in writes if w.get("bytes")]
            rec["checkpoint_writes"] = len(writes)
            rec["checkpoint_write_seconds_median"] = statistics.median(secs)
            rec["checkpoint_write_seconds_max"] = max(secs)
            rec["checkpoint_bytes_median"] = statistics.median(sizes) if sizes else None
            rec["checkpoint_mb_median"] = (statistics.median(sizes) / MB) if sizes else None
        else:
            rec["checkpoint_writes"] = 0
        rec["cuda"] = probe.get("cuda")
        rec["probe_exit"] = probe.get("exit")
        cuda = probe.get("cuda") or {}
        total = cuda.get("total_memory_bytes")
        if total and rec.get("peak_reserved_gb") is not None:
            rec["vram_headroom_fraction"] = 1.0 - (probe["peak_reserved_bytes"] / total)
            rec["physical_batch_fits"] = bool(
                job.get("returncode") == 0 and probe["peak_reserved_bytes"] < total
            )
    else:
        rec["probe_missing"] = job["probe_glob"]

    acc = last_value(rows, r"test_accuracy$")
    if acc:
        rec["validation_accuracy"] = acc
    vloss = last_value(rows, r"^val/loss$")
    if vloss:
        rec["validation_loss"] = vloss
    tloss = last_value(rows, r"^loss$")
    if tloss:
        rec["train_loss_last"] = tloss
    return rec


def gpu_hours(seconds_per_step: float, steps: int, runs: int, ckpt_seconds: float = 0.0,
              ckpt_writes: int = 0) -> float:
    return (seconds_per_step * steps + ckpt_seconds * ckpt_writes) * runs / 3600.0


def project(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Spec section 11's budget arithmetic, per model rather than pooled.

        base GPU-hours       = s/step * 20,000 * 6 base runs / 3600
        adaptation GPU-hours = s/step * adaptation_steps * 12 branches / 3600
        HMM GPU-hours        = s/step * 3,000 * 6 runs / 3600
        + checkpoint overhead + a 20% interruption margin

    The spec writes one `s/step`; GPT and NextLat differ, so each term is split into its two
    model halves (3 seeds each) and summed. That is the same arithmetic with a tighter
    estimate, and the pooled form is reported alongside it.
    """
    out: Dict[str, Any] = {"assumptions": {
        "seeds": N_SEEDS, "models": N_MODELS,
        "base_steps": BASE_STEPS, "adaptation_steps": ADAPT_STEPS,
        "adaptation_branches_total": N_MODELS * ADAPT_BRANCHES_PER_MODEL,
        "hmm_steps": HMM_STEPS, "interruption_margin": INTERRUPTION_MARGIN,
    }}
    missing = []
    total = 0.0
    for task, steps, runs_per_model, recovery_every in [
        ("lurestar", BASE_STEPS, N_SEEDS, 250),
        ("adapt", ADAPT_STEPS, ADAPT_BRANCHES_PER_MODEL, 100),
        ("hmm", HMM_STEPS, N_SEEDS, 250),
    ]:
        # adaptation throughput is estimated from the Lure-Star base profile: the NextLat
        # adaptation branch still runs the full mtp_horizon=3 rollout (mtp_horizon is on the
        # frozen surface) and only multiplies its losses by zero, so per-step cost is the
        # base cost.
        src_task = "lurestar" if task in ("lurestar", "adapt") else "hmm"
        term = {}
        for model in ("gpt", "nextlat"):
            rec = records.get(f"{src_task}-{model}")
            if not rec or not rec.get("seconds_per_step_median"):
                missing.append(f"{src_task}-{model}")
                continue
            ckpt = rec.get("checkpoint_write_seconds_median") or 0.0
            writes = steps // recovery_every
            hours = gpu_hours(rec["seconds_per_step_median"], steps, runs_per_model,
                              ckpt, writes)
            term[model] = hours
            total += hours
        out[f"{task}_gpu_hours"] = term
        out[f"{task}_gpu_hours_total"] = sum(term.values()) if term else None
    if missing:
        out["incomplete_for"] = sorted(set(missing))
    out["subtotal_gpu_hours"] = total
    out["with_interruption_margin_gpu_hours"] = total * (1 + INTERRUPTION_MARGIN)
    return out


def contrast(records: Dict[str, Dict[str, Any]], task: str) -> Optional[Dict[str, Any]]:
    gpt, nl = records.get(f"{task}-gpt"), records.get(f"{task}-nextlat")
    if not gpt or not nl:
        return None
    out: Dict[str, Any] = {}
    if gpt.get("seconds_per_step_median") and nl.get("seconds_per_step_median"):
        out["nextlat_step_time_overhead"] = (
            nl["seconds_per_step_median"] / gpt["seconds_per_step_median"]
        )
        out["nextlat_throughput_ratio"] = 1.0 / out["nextlat_step_time_overhead"]
    for field, label in [("peak_allocated_gb", "allocated"), ("peak_reserved_gb", "reserved")]:
        if gpt.get(field) and nl.get(field):
            out[f"nextlat_peak_{label}_overhead"] = nl[field] / gpt[field]
    if gpt.get("checkpoint_bytes_median") and nl.get("checkpoint_bytes_median"):
        out["nextlat_checkpoint_bytes_overhead"] = (
            nl["checkpoint_bytes_median"] / gpt["checkpoint_bytes_median"]
        )
    return out or None


def _fmt(value: Any, spec: str = ".4g") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


ROWS = [
    ("median s/step (compute)", "seconds_per_step_median", ".4f"),
    ("p95 s/step (compute)", "seconds_per_step_p95", ".4f"),
    ("wall s/step", "wall_seconds_per_step", ".4f"),
    ("examples/s", "examples_per_second", ".1f"),
    ("tokens/s", "tokens_per_second", ".0f"),
    ("peak allocated VRAM (GB)", "peak_allocated_gb", ".2f"),
    ("peak reserved VRAM (GB)", "peak_reserved_gb", ".2f"),
    ("VRAM headroom", "vram_headroom_fraction", ".1%"),
    ("physical batch fits", "physical_batch_fits", ""),
    ("GPU util % (median)", ("gpu", "gpu_util_pct_median"), ".1f"),
    ("host-input wait (s)", "host_input_wait_seconds", ".2f"),
    ("host-input wait (frac wall)", "host_input_wait_fraction", ".1%"),
    ("checkpoint writes", "checkpoint_writes", "d"),
    ("checkpoint write (s, median)", "checkpoint_write_seconds_median", ".2f"),
    ("checkpoint size (MB)", "checkpoint_mb_median", ".1f"),
    ("steps summarized", "steps_summarized", "d"),
]


def render_table(records: Dict[str, Dict[str, Any]], task: str) -> List[str]:
    keys = [k for k in (f"{task}-gpt", f"{task}-nextlat") if k in records]
    if not keys:
        return [f"### {task}: no runs", ""]
    lines = [f"### {task}", "", "| metric | " + " | ".join(keys) + " |",
             "|---|" + "---|" * len(keys)]
    for label, field, spec in ROWS:
        cells = []
        for key in keys:
            rec = records[key]
            value = rec.get(field[0], {}).get(field[1]) if isinstance(field, tuple) else rec.get(field)
            cells.append(_fmt(value, spec) if spec else _fmt(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key in keys:
        acc = records[key].get("validation_accuracy")
        if acc:
            lines.append(f"| {acc['metric']} @ step {int(acc['step'] or 0)} ({key}) | "
                         + " | ".join(_fmt(acc["value"], ".4f") if k == key else "-" for k in keys)
                         + " |")
    con = contrast(records, task)
    if con:
        lines += ["", "NextLat vs GPT: " + ", ".join(
            f"{k} = {v:.3f}x" for k, v in con.items())]
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs-dir", required=True,
                    help="directory of job manifests written by scripts/profile.sh")
    ap.add_argument("--out", required=True, help="path for profile_summary.json")
    args = ap.parse_args()

    manifests = sorted(glob.glob(os.path.join(args.jobs_dir, "*.job.json")))
    if not manifests:
        print(f"no job manifests under {args.jobs_dir}", file=sys.stderr)
        return 2

    records: Dict[str, Dict[str, Any]] = {}
    for path in manifests:
        with open(path) as fh:
            job = json.load(fh)
        records[job["job"]] = summarize_job(job)

    summary = {
        "records": records,
        "contrast": {t: contrast(records, t) for t in ("lurestar", "hmm")},
        "projection": project(records),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)

    lines = ["# Profiling gate (spec section 11)", ""]
    for task in ("lurestar", "hmm"):
        lines += render_table(records, task)
    proj = summary["projection"]
    lines += ["### projected end-to-end runtime", ""]
    for task in ("lurestar", "adapt", "hmm"):
        term = proj.get(f"{task}_gpu_hours") or {}
        detail = ", ".join(f"{m} {h:.2f}" for m, h in sorted(term.items())) or "-"
        lines.append(f"- {task}: {_fmt(proj.get(f'{task}_gpu_hours_total'), '.2f')} GPU-h "
                     f"({detail})")
    lines += [
        f"- subtotal: {_fmt(proj.get('subtotal_gpu_hours'), '.2f')} GPU-h",
        f"- with {int(INTERRUPTION_MARGIN * 100)}% interruption margin: "
        f"{_fmt(proj.get('with_interruption_margin_gpu_hours'), '.2f')} GPU-h",
    ]
    if proj.get("incomplete_for"):
        lines.append(f"- INCOMPLETE, missing profiles for: {proj['incomplete_for']}")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as fh:
        fh.write(text)

    failed = [k for k, r in records.items()
              if r.get("returncode") not in (0, None) or r.get("seconds_per_step_median") is None]
    if failed:
        print(f"INCOMPLETE: jobs without a usable profile: {failed}", file=sys.stderr)
        return 1
    if proj.get("incomplete_for"):
        # A half-run gate is not a gate. PROGRAM.md invariant 2 requires a measured
        # seconds-per-step and a projected cost before any job launches, and the projection
        # above is missing terms.
        print(f"INCOMPLETE: no profile for {proj['incomplete_for']}, so the projected "
              f"end-to-end runtime is a lower bound, not a budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
