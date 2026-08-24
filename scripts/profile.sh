#!/usr/bin/env bash
# The spec section 11 profiling gate.
#
#   scripts/profile.sh [--lurestar-only] [--hmm-only] [--out DIR] [--dry-run]
#
# "Profile Lure-Star GPT/NextLat for 500 steps and HMM GPT/NextLat for 300 steps. For the
#  500-step Lure-Star profile, treat the first 100 steps as warmup and summarize the final
#  400. Record: median and p95 seconds per step; examples and tokens per second; peak
#  allocated and reserved VRAM; GPU utilization and host-input wait; checkpoint-write
#  duration and bytes; GPT versus NextLat throughput and memory overhead; validation accuracy
#  and projected end-to-end runtime."
#
# Every one of those is produced by scripts/profile_summarize.py from three sources this
# script collects per job: the CSVLogger metrics.csv the run writes, an nvidia-smi sample
# stream, and the in-process probe from scripts/profile_entry.py (peak VRAM, host-input wait,
# checkpoint write duration and bytes).
#
# PROGRAM.md loop invariant 2: "Profile before sweeping. No job launches without a measured
# seconds-per-step and a projected compute-unit cost." This script is that gate.
#
# The runs use the REAL confirmatory configs. The only overrides are the step count and the
# validation/recovery cadence needed to observe a checkpoint write inside a short run, plus a
# profile-only output root so nothing here can be mistaken for, or collide with, a
# confirmatory run. Spec section 11: "Do not change width, depth, sequence construction,
# precision, loss, or training examples to improve throughput." Nothing here does.
#
# Environment: NEXTLAT_REPO, LURESTAR_PRECISION and LURESTAR_STRATEGY are read by
# scripts/launch_train.sh; see that file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${LURESTAR_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

OUT_ROOT="${PROFILE_OUT:-$PROJECT_DIR/results/profile}"
RUN_LURESTAR=1
RUN_HMM=1
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lurestar-only) RUN_HMM=0; shift ;;
    --hmm-only)      RUN_LURESTAR=0; shift ;;
    --out)           OUT_ROOT="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)       sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "profile.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

JOBS_DIR="$OUT_ROOT/jobs"
# Passed to launch_train.sh as LURESTAR_ROOT, which appends "runs/<model>/seed<n>/...".
PROFILE_LURESTAR_ROOT="$OUT_ROOT/root"
mkdir -p "$JOBS_DIR" "$PROFILE_LURESTAR_ROOT"

SEED=1234

echo "profiling gate -> $OUT_ROOT"
echo "python         $PYTHON"
echo "repo           ${NEXTLAT_REPO:-/content/nextlat}"

HAVE_NVIDIA_SMI=0
if command -v nvidia-smi >/dev/null 2>&1; then HAVE_NVIDIA_SMI=1; fi
if [[ $HAVE_NVIDIA_SMI == 0 ]]; then
  echo "note: nvidia-smi not found; GPU utilization will be reported as 0 samples" >&2
fi

SKIPPED=""

run_job() {
  # run_job <job-id> <task> <model> <config> <steps> <warmup>
  local job="$1" task="$2" model="$3" config="$4" steps="$5" warmup="$6"
  local val_interval=$(( steps / 2 ))
  local recovery=$(( steps / 4 ))
  local out_dir exp probe_glob gpu_csv log t0 t1 rc

  case "$task" in
    lurestar) out_dir="$PROFILE_LURESTAR_ROOT/runs/$model/seed$SEED/base"; exp="$model-seed$SEED-base" ;;
    hmm)      out_dir="$PROFILE_LURESTAR_ROOT/runs/hmm/$model/seed$SEED/base"; exp="$model-seed$SEED-hmm" ;;
    *) echo "profile.sh: unknown task $task" >&2; return 2 ;;
  esac

  probe_glob="$JOBS_DIR/$job.probe.*.json"
  gpu_csv="$JOBS_DIR/$job.gpu.csv"
  log="$JOBS_DIR/$job.log"
  rm -f $probe_glob "$gpu_csv"

  echo
  echo "=== $job: $steps steps, warmup $warmup, config $config ==="

  local -a cmd=(
    env
    "LURESTAR_ROOT=$PROFILE_LURESTAR_ROOT"
    "LURESTAR_ENTRY=$SCRIPT_DIR/profile_entry.py"
    "PROFILE_PROBE_JSON=$JOBS_DIR/$job.probe.{pid}.json"
    "$SCRIPT_DIR/launch_train.sh" "$config" "$SEED"
    "trainer.train_batches=$steps"
    "trainer.val_interval=$val_interval"
    "trainer.test_interval=$val_interval"
    "trainer.save_recovery_checkpoint=$recovery"
  )

  if [[ $DRY == 1 ]]; then
    DRY_RUN=1 "${cmd[@]}"
    return 0
  fi

  local smi_pid=""
  if [[ $HAVE_NVIDIA_SMI == 1 ]]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
               --format=csv,noheader,nounits -lms 250 > "$gpu_csv" 2>/dev/null &
    smi_pid=$!
  fi

  t0="$($PYTHON -c 'import time;print(repr(time.time()))')"
  set +e
  "${cmd[@]}" 2>&1 | tee "$log"
  # NEVER read $? after a pipeline here: it is tee's status. PIPESTATUS[0] is the launcher's.
  rc="${PIPESTATUS[0]}"
  set -e
  t1="$($PYTHON -c 'import time;print(repr(time.time()))')"

  if [[ -n "$smi_pid" ]]; then kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true; fi

  "$PYTHON" - "$JOBS_DIR/$job.job.json" <<PY
import json, sys
record = {
    "job": "$job", "task": "$task", "model": "$model",
    "config": "$config", "seed": $SEED,
    "steps": $steps, "warmup_steps": $warmup,
    "out_dir": "$out_dir", "experiment_name": "$exp",
    "probe_glob": "$probe_glob", "gpu_samples_csv": "$gpu_csv",
    "log": "$log", "returncode": $rc,
    "wall_seconds": $t1 - $t0,
}
with open(sys.argv[1], "w") as fh:
    json.dump(record, fh, indent=2)
PY
  echo "--- $job finished rc=$rc"
  return 0
}

hmm_prerequisites_ok() {
  local repo="${NEXTLAT_REPO:-/content/nextlat}"
  if [[ ! -f "$repo/train.py" ]]; then
    echo "HMM skipped: no train.py under NEXTLAT_REPO=$repo" >&2; return 1
  fi
  if ! grep -q "hmm_belief" "$repo/train.py"; then
    echo "HMM skipped: 'hmm_belief' is not registered in DATAMODULES (train.py:34-42). \
Apply the datamodule registration to the working copy first." >&2
    return 1
  fi
  local train_npz
  train_npz="$("$PYTHON" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from config_lib import load_yaml_as_trainer_sees_it
print(load_yaml_as_trainer_sees_it('$PROJECT_DIR/configs/gpt_hmm.yaml')['data']['hmm']['hmm_train_data_path'])")"
  if [[ ! -f "$train_npz" ]]; then
    echo "HMM skipped: training sequences missing at $train_npz" >&2; return 1
  fi
  return 0
}

if [[ $RUN_LURESTAR == 1 ]]; then
  # spec section 11: 500 steps, first 100 discarded as warmup, final 400 summarized.
  run_job lurestar-gpt     lurestar gpt     gpt_lurestar.yaml     500 100
  run_job lurestar-nextlat lurestar nextlat nextlat_lurestar.yaml 500 100
fi

if [[ $RUN_HMM == 1 ]]; then
  if hmm_prerequisites_ok; then
    # spec section 11 fixes 300 steps for HMM and states no warmup rule; the same 20%
    # proportion as Lure-Star is used, i.e. the first 60 steps are discarded.
    run_job hmm-gpt     hmm gpt     gpt_hmm.yaml     300 60
    run_job hmm-nextlat hmm nextlat nextlat_hmm.yaml 300 60
  else
    SKIPPED="$SKIPPED hmm"
  fi
fi

if [[ $DRY == 1 ]]; then echo "dry run: no jobs executed"; exit 0; fi

echo
SUMMARY_RC=0
"$PYTHON" "$SCRIPT_DIR/profile_summarize.py" \
  --jobs-dir "$JOBS_DIR" --out "$OUT_ROOT/profile_summary.json" || SUMMARY_RC=$?

if [[ -n "${SKIPPED// /}" ]]; then
  echo "GATE INCOMPLETE: skipped$SKIPPED" >&2
  exit 3
fi
exit "$SUMMARY_RC"
