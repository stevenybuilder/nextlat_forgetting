"""The profiling gate is measurement code, so it is tested against inputs whose right answer
is known by construction rather than against a real run.

Two things are exercised:

  1. `scripts/profile_summarize.py` -- fed a synthetic job whose per-step times, checkpoint
     writes and GPU samples are chosen so that the median, the p95, the examples/s, the
     tokens/s and the projected GPU-hours all have exact closed-form values.

  2. `scripts/profile_entry.py` -- executed for real against stub `torch` and `lightning`
     modules and a stub `train.py`, which proves that the three probes actually fire: the
     DataLoader patch counts batches and accumulates wait time, the Fabric.save patch records
     duration and bytes, and the peak-VRAM record is written from inside the process that ran
     the training script. docs/RUNLOG.md records the profiling bug this last check exists to
     prevent: peak VRAM read in the driver process instead of the `fabric run` child, which
     silently reported 0.00 GB.

Negative controls at the bottom show the checks are not vacuous: a wrong warmup boundary
changes the summary, and a probe written by a driver process that never trained yields no
peak-VRAM number at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import profile_summarize as ps  # noqa: E402

PYTHON = str(REPO / ".venv" / "bin" / "python")


# --------------------------------------------------------------------------------------
# a synthetic job with known answers
# --------------------------------------------------------------------------------------

# 500 logged steps. Steps 0..99 are the warmup and run at 1.0 s/step; steps 100..499 are the
# steady state and run at exactly 0.20 s/step except for twenty stalls at 0.50 s/step, placed
# so that the p95 of the summarized 400 samples lands on the stall value.
STEADY = 0.20
STALL = 0.50
N_STALLS = 20


def _write_job(tmp_path: Path, *, model: str, task: str = "lurestar",
               eff_batch: int = 512, block_size: int = 69,
               steps: int = 500, warmup: int = 100,
               step_scale: float = 1.0) -> dict:
    out_dir = tmp_path / "root" / "runs" / model / "seed1234" / "base"
    exp = f"{model}-seed1234-base"
    version = out_dir / exp / "version_0"
    version.mkdir(parents=True)

    # The stargraph accuracy column name contains a comma -- `val_(5, 5)/test_accuracy` --
    # so csv.DictWriter quotes it, and anything reading metrics.csv must go through a real
    # CSV parser rather than str.split(",").
    header = 'step,steps_per_sec,loss,val/loss,"val_(5, 5)/test_accuracy"\n' 
    rows = [header]
    for step in range(steps):
        if step < warmup:
            sec = 1.0
        else:
            steady_index = step - warmup
            sec = STALL if steady_index >= (steps - warmup) - N_STALLS else STEADY
        sec *= step_scale
        acc = "0.9312" if step == steps - 1 else ""
        vloss = "0.1234" if step == steps - 1 else ""
        rows.append(f"{step},{1.0 / sec},0.44,{vloss},{acc}\n")
    (version / "metrics.csv").write_text("".join(rows))

    (out_dir / exp / "materialized_config.yaml").write_text(textwrap.dedent(f"""\
        data:
          effective_batch_size: {eff_batch}
          gradient_accum_steps: 1
          device_batch_size: {eff_batch}
          micro_batch_size: {eff_batch}
        model:
          block_size: {block_size}
          vocab_size: 106
          n_layer: 12
          n_embd: 384
          proj_factor: 0.5
        trainer:
          compile: false
          learning_rate: 5e-4
        """))

    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    probe = {
        "pid": 4242,
        "dataloader_wait_s": 8.0,
        "dataloader_batches": steps,
        "checkpoint_writes": [
            {"path": "a.pt", "seconds": 2.0, "bytes": 256 * 1024 * 1024},
            {"path": "b.pt", "seconds": 4.0, "bytes": 256 * 1024 * 1024},
            {"path": "c.pt", "seconds": 3.0, "bytes": 256 * 1024 * 1024},
        ],
        "cuda": {"device_name": "NVIDIA A100-SXM4-40GB", "capability": [8, 0],
                 "bf16_supported": True, "total_memory_bytes": 40 * 1024 ** 3,
                 "torch": "2.11.0+cu128", "cuda_version": "12.8"},
        "peak_allocated_bytes": 10 * 1024 ** 3,
        "peak_reserved_bytes": 12 * 1024 ** 3,
        "wall_seconds": 200.0,
        "exit": "ok",
    }
    (jobs / f"{task}-{model}.probe.4242.json").write_text(json.dumps(probe))
    gpu_csv = jobs / f"{task}-{model}.gpu.csv"
    gpu_csv.write_text("".join(
        f"2026/08/23 10:00:{i:02d}.000, {90 + (i % 3)}, {40 + (i % 3)}, {12000 + i}\n"
        for i in range(20)))

    job = {
        "job": f"{task}-{model}", "task": task, "model": model,
        "config": f"{model}_lurestar.yaml", "seed": 1234,
        "steps": steps, "warmup_steps": warmup,
        "out_dir": str(out_dir), "experiment_name": exp,
        "probe_glob": str(jobs / f"{task}-{model}.probe.*.json"),
        "gpu_samples_csv": str(gpu_csv),
        "log": "x.log", "returncode": 0,
        "wall_seconds": 200.0,
    }
    (jobs / f"{task}-{model}.job.json").write_text(json.dumps(job))
    return job


def test_percentile_matches_linear_interpolation() -> None:
    assert ps.percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert ps.percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert ps.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert ps.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        ps.percentile([], 0.5)


def test_summary_numbers_are_exact(tmp_path: Path) -> None:
    job = _write_job(tmp_path, model="gpt")
    rec = ps.summarize_job(job)

    # 400 summarized samples: 380 at 0.20 s and 20 at 0.50 s
    assert rec["steps_summarized"] == 400
    assert rec["seconds_per_step_median"] == pytest.approx(STEADY)
    # 380 samples at 0.20 then 20 at 0.50: the 95th percentile index is 399*0.95 = 379.05,
    # which interpolates 5% of the way from the last steady sample into the first stall.
    assert rec["seconds_per_step_p95"] == pytest.approx(0.20 + 0.05 * (STALL - STEADY))
    assert rec["seconds_per_step_mean"] == pytest.approx((380 * 0.2 + 20 * 0.5) / 400)

    assert rec["examples_per_second"] == pytest.approx(512 / 0.20)
    assert rec["tokens_per_second"] == pytest.approx(512 * 69 / 0.20)
    assert rec["wall_seconds_per_step"] == pytest.approx(200.0 / 500)

    assert rec["peak_allocated_gb"] == pytest.approx(10.0)
    assert rec["peak_reserved_gb"] == pytest.approx(12.0)
    assert rec["vram_headroom_fraction"] == pytest.approx(1 - 12 / 40)
    assert rec["physical_batch_fits"] is True

    assert rec["host_input_wait_seconds"] == pytest.approx(8.0)
    assert rec["host_input_wait_fraction"] == pytest.approx(8.0 / 200.0)

    assert rec["checkpoint_writes"] == 3
    assert rec["checkpoint_write_seconds_median"] == pytest.approx(3.0)
    assert rec["checkpoint_write_seconds_max"] == pytest.approx(4.0)
    assert rec["checkpoint_mb_median"] == pytest.approx(256.0)

    assert rec["gpu"]["samples"] == 20
    assert rec["gpu"]["gpu_util_pct_median"] == pytest.approx(91.0)

    assert rec["validation_accuracy"]["value"] == pytest.approx(0.9312)
    assert rec["validation_accuracy"]["metric"] == "val_(5, 5)/test_accuracy"
    assert rec["resolved"]["block_size"] == 69
    assert rec["resolved"]["effective_batch_size"] == 512
    assert rec["resolved"]["compile"] is False


def test_warmup_boundary_actually_bites(tmp_path: Path) -> None:
    """Negative control: including the warmup changes the answer, so the discard is real."""
    job = _write_job(tmp_path, model="gpt")
    kept = ps.summarize_job(job)
    job["warmup_steps"] = 0
    everything = ps.summarize_job(job)
    assert kept["steps_summarized"] == 400 and everything["steps_summarized"] == 500
    # the 100 warmup steps run at 1.0 s and are the slowest samples in the file, so keeping
    # them inflates both the mean and the tail
    assert everything["seconds_per_step_mean"] > kept["seconds_per_step_mean"]
    assert everything["seconds_per_step_p95"] == pytest.approx(1.0)
    assert kept["seconds_per_step_p95"] < 0.3


def test_contrast_and_projection(tmp_path: Path) -> None:
    gpt = _write_job(tmp_path, model="gpt")
    # NextLat is 1.25x slower per step and holds 1.2x the memory
    nl = _write_job(tmp_path, model="nextlat", step_scale=1.25)
    records = {"lurestar-gpt": ps.summarize_job(gpt),
               "lurestar-nextlat": ps.summarize_job(nl)}

    con = ps.contrast(records, "lurestar")
    assert con["nextlat_step_time_overhead"] == pytest.approx(1.25)
    assert con["nextlat_throughput_ratio"] == pytest.approx(0.8)
    assert con["nextlat_peak_allocated_overhead"] == pytest.approx(1.0)

    proj = ps.project(records)
    ckpt = 3.0  # median checkpoint write from the synthetic probe
    expected_gpt = (0.20 * 20000 + ckpt * (20000 // 250)) * 3 / 3600
    assert proj["lurestar_gpu_hours"]["gpt"] == pytest.approx(expected_gpt)
    expected_nl = (0.25 * 20000 + ckpt * (20000 // 250)) * 3 / 3600
    assert proj["lurestar_gpu_hours"]["nextlat"] == pytest.approx(expected_nl)

    # adaptation: 500 updates, 6 branches per model, recovery every 100 steps
    expected_adapt_gpt = (0.20 * 500 + ckpt * 5) * 6 / 3600
    assert proj["adapt_gpu_hours"]["gpt"] == pytest.approx(expected_adapt_gpt)

    # the HMM profile is absent, so the gate must say so rather than quietly under-budget
    assert "hmm-gpt" in proj["incomplete_for"] and "hmm-nextlat" in proj["incomplete_for"]
    assert proj["with_interruption_margin_gpu_hours"] == pytest.approx(
        proj["subtotal_gpu_hours"] * 1.20)


def test_missing_probe_is_reported_not_silently_zero(tmp_path: Path) -> None:
    """The RUNLOG bug class: a profile with no in-process probe must not report 0.00 GB."""
    job = _write_job(tmp_path, model="gpt")
    for path in Path(job["probe_glob"]).parent.glob("*.probe.*.json"):
        path.unlink()
    rec = ps.summarize_job(job)
    assert "peak_allocated_gb" not in rec
    assert rec["probe_missing"] == job["probe_glob"]


def test_cli_writes_summary_and_markdown(tmp_path: Path) -> None:
    _write_job(tmp_path, model="gpt")
    _write_job(tmp_path, model="nextlat", step_scale=1.25)
    out = tmp_path / "profile_summary.json"
    proc = subprocess.run(
        [PYTHON, str(REPO / "scripts" / "profile_summarize.py"),
         "--jobs-dir", str(tmp_path / "jobs"), "--out", str(out)],
        capture_output=True, text=True)
    # non-zero because the HMM half of the gate is absent: an incomplete gate must never
    # look like a pass
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "INCOMPLETE" in proc.stdout + proc.stderr
    assert out.is_file() and (tmp_path / "profile_summary.md").is_file()
    summary = json.loads(out.read_text())
    assert set(summary["records"]) == {"lurestar-gpt", "lurestar-nextlat"}
    text = (tmp_path / "profile_summary.md").read_text()
    for label in ["median s/step", "p95 s/step", "examples/s", "tokens/s",
                  "peak allocated VRAM", "peak reserved VRAM", "GPU util",
                  "host-input wait", "checkpoint write", "projected end-to-end runtime"]:
        assert label in text, label


# --------------------------------------------------------------------------------------
# profile_entry.py, run for real against stub torch / lightning
# --------------------------------------------------------------------------------------

_STUB_TORCH = '''
import types, sys
__version__ = "0.0.0-stub"
class _Version: cuda = None
version = _Version()
class _CudaStub:
    @staticmethod
    def is_available(): return False
cuda = _CudaStub()
class DataLoader:
    def __init__(self, data): self.data = list(data)
    def __iter__(self): return iter(self.data)
_data_mod = types.ModuleType("torch.utils.data")
_data_mod.DataLoader = DataLoader
_utils = types.ModuleType("torch.utils")
_utils.data = _data_mod
utils = _utils
sys.modules["torch.utils"] = _utils
sys.modules["torch.utils.data"] = _data_mod
'''

_STUB_LIGHTNING = '''
class Fabric:
    def save(self, path, state):
        with open(path, "wb") as fh:
            fh.write(b"x" * 1024)
        return "saved"
'''

_STUB_TRAIN = '''
import torch, lightning
loader = torch.utils.data.DataLoader([1, 2, 3, 4, 5])
seen = [b for b in loader]
assert seen == [1, 2, 3, 4, 5], seen
lightning.Fabric().save("ckpt.pt", {"model": None})
print("STUB TRAIN OK", len(seen))
'''


def _stub_env(tmp_path: Path) -> tuple:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text(_STUB_TORCH)
    (stubs / "lightning.py").write_text(_STUB_LIGHTNING)
    work = tmp_path / "work"
    work.mkdir()
    (work / "train.py").write_text(_STUB_TRAIN)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(stubs)
    return work, env


def test_profile_entry_probes_fire(tmp_path: Path) -> None:
    work, env = _stub_env(tmp_path)
    probe_path = tmp_path / "probe.{pid}.json"
    env["PROFILE_PROBE_JSON"] = str(probe_path)
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "profile_entry.py"),
                           "--config", "whatever.yaml", "seed=1234"],
                          cwd=str(work), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "STUB TRAIN OK 5" in proc.stdout

    written = list(tmp_path.glob("probe.*.json"))
    assert len(written) == 1, written
    probe = json.loads(written[0].read_text())

    # probe 1: the DataLoader patch counted every batch and accumulated wait time
    assert probe["dataloader_batches"] == 5
    assert probe["dataloader_wait_s"] >= 0.0
    # probe 2: the Fabric.save patch timed the write and measured the bytes
    assert len(probe["checkpoint_writes"]) == 1
    assert probe["checkpoint_writes"][0]["bytes"] == 1024
    assert probe["checkpoint_writes"][0]["seconds"] > 0.0
    # probe 3: no CUDA in the stub, so the VRAM fields are explicitly None, not 0
    assert probe["cuda"] is None
    assert probe["peak_allocated_bytes"] is None
    assert probe["exit"] == "ok"
    assert probe["wall_seconds"] > 0.0
    # the training script ran with the repo as CWD, as train.py:348 requires
    assert probe["argv"][0].endswith("profile_entry.py")
    assert (work / "ckpt.pt").is_file()


def test_profile_entry_records_a_crash_instead_of_swallowing_it(tmp_path: Path) -> None:
    work, env = _stub_env(tmp_path)
    (work / "train.py").write_text("raise RuntimeError('boom')\n")
    env["PROFILE_PROBE_JSON"] = str(tmp_path / "probe.{pid}.json")
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "profile_entry.py")],
                          cwd=str(work), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    probe = json.loads(next(tmp_path.glob("probe.*.json")).read_text())
    assert probe["exit"] == "RuntimeError: boom"


def test_profile_entry_refuses_without_a_probe_path(tmp_path: Path) -> None:
    work, env = _stub_env(tmp_path)
    env.pop("PROFILE_PROBE_JSON", None)
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "profile_entry.py")],
                          cwd=str(work), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "PROFILE_PROBE_JSON" in proc.stderr


def test_profile_entry_refuses_without_train_py(tmp_path: Path) -> None:
    work, env = _stub_env(tmp_path)
    (work / "train.py").unlink()
    env["PROFILE_PROBE_JSON"] = str(tmp_path / "probe.{pid}.json")
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "profile_entry.py")],
                          cwd=str(work), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "train.py" in proc.stderr


# --------------------------------------------------------------------------------------
# profile.sh
# --------------------------------------------------------------------------------------


def test_profile_sh_dry_run_emits_the_spec_step_counts(tmp_path: Path) -> None:
    repo = tmp_path / "fakerepo"
    repo.mkdir()
    (repo / "train.py").write_text("DATAMODULES = {'stargraph': None}\n")
    (repo / "defaults.yaml").write_text("seed: 1234\n")
    env = dict(os.environ, NEXTLAT_REPO=str(repo))
    proc = subprocess.run(["bash", str(REPO / "scripts" / "profile.sh"),
                           "--dry-run", "--out", str(tmp_path / "prof")],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # spec section 11: 500 Lure-Star steps for each model, through profile_entry.py
    assert out.count("trainer.train_batches=500") == 2
    assert out.count("profile_entry.py") == 2
    assert "--devices 1" in out and "--precision bf16-mixed" in out
    assert "configs/gpt_lurestar.yaml" in out and "configs/nextlat_lurestar.yaml" in out
    # the HMM half must announce that it is skipped, not vanish
    assert "HMM skipped" in proc.stderr
    # and the profile must not write into a confirmatory output root
    assert "/content/lurestar/runs" not in out


def test_profile_sh_reports_an_incomplete_gate(tmp_path: Path) -> None:
    """Skipping the HMM half is a non-zero exit, so a caller cannot mistake it for a pass."""
    repo = tmp_path / "fakerepo"
    repo.mkdir()
    (repo / "train.py").write_text("DATAMODULES = {'stargraph': None}\n")
    (repo / "defaults.yaml").write_text("seed: 1234\n")
    env = dict(os.environ, NEXTLAT_REPO=str(repo))
    # no jobs will have run, so profile_summarize also fails; the exit code must be non-zero
    proc = subprocess.run(["bash", str(REPO / "scripts" / "profile.sh"),
                           "--hmm-only", "--out", str(tmp_path / "prof2")],
                          capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "HMM skipped" in proc.stderr
