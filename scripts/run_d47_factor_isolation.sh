#!/usr/bin/env bash

# Follow-up to D47: isolate the source of the one-GPU trajectory reversal.
# Runs only after the clean one-GPU/two-GPU topology block has completed.
# All cells stop immediately after the 2,000-update validation checkpoint.

set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_FACTOR_ROOT:-/workspace/d47-factors}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
CLEAN="$PROJECT/upstream/NextLat"
PATCHED="$ROOT/patched-nextlat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
RUNS="$ROOT/runs"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_FACTOR_GCS_PREFIX:?D47_FACTOR_GCS_PREFIX is required}"
ADC="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"

export GOOGLE_APPLICATION_CREDENTIALS="$ADC"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT" "$RUNS"

[[ -z "$(git -C "$CLEAN" status --porcelain)" ]] || {
  echo "clean upstream is dirty" >&2
  exit 2
}

if [[ ! -f "$PATCHED/.lurestar_runtime_patch_receipt.json" ]]; then
  cp -a "$CLEAN" "$PATCHED"
  python3 "$PROJECT/scripts/runtime_bootstrap.py" \
    --project-root "$PROJECT" --upstream "$PATCHED"
fi

cat > "$ROOT/sync_results.py" <<'PY'
from google.cloud import storage
import hashlib, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
bucket = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]).bucket(sys.argv[2])
prefix = sys.argv[3].strip("/")
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.suffix == ".pt" or path.name == "sync_results.py":
        continue
    rel = path.relative_to(root).as_posix()
    parts = pathlib.PurePosixPath(rel).parts
    if any(
        part == ".git"
        or part == "__pycache__"
        or part.startswith("patched-nextlat")
        for part in parts
    ):
        continue
    blob = bucket.blob(f"{prefix}/{rel}")
    blob.metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    blob.upload_from_filename(path, timeout=600)
PY

run_cell() {
  local name="$1" upstream="$2" gpu="$3" strategy="$4"
  local cell="$RUNS/$name"
  local out="$cell/nextlat/seed1235/output"
  mkdir -p "$cell" "$out"
  local cmd=(fabric run --precision bf16-mixed --devices 1 --main-port "$((35000 + gpu))")
  [[ -n "$strategy" ]] && cmd+=(--strategy "$strategy")
  cmd+=(train.py --config "$CONFIG"
    seed=1235
    "trainer.out_dir=$out"
    "trainer.experiment_name=nextlat-seed1235-$name"
    trainer.train_batches=2100
    trainer.compile=false
    trainer.log_to_wandb=false
    trainer.save_last_checkpoint=false
    trainer.save_best_checkpoint=false
    trainer.always_save_checkpoint=false
    trainer.save_recovery_checkpoint=-1
    "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
    "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu" > "$cell/command.txt"
  printf '%q ' "${cmd[@]}" >> "$cell/command.txt"
  printf '\n' >> "$cell/command.txt"
  local start end rc
  start="$(date +%s)"
  set +e
  (
    cd "$upstream"
    CUDA_VISIBLE_DEVICES="$gpu" LURESTAR_NONCONFIRMATORY=1 "${cmd[@]}"
  ) > "$cell/train.log" 2>&1
  rc=$?
  set -e
  end="$(date +%s)"
  printf '{"returncode":%d,"started_at_unix":%d,"ended_at_unix":%d,"elapsed_seconds":%d}\n' \
    "$rc" "$start" "$end" "$((end-start))" > "$cell/terminal.json"
  python3 "$ROOT/sync_results.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX" || true
  return "$rc"
}

# A: exact clean repeat tests within-host repeatability.
# B: DDP-at-world-size-1 changes the sampler without multi-GPU gradient reduction.
run_cell clean-plain-repeat "$CLEAN" 0 "" &
pid_a=$!
run_cell clean-ddp1-sampler "$CLEAN" 1 ddp &
pid_b=$!
set +e
wait "$pid_a"; rc_a=$?
wait "$pid_b"; rc_b=$?
set -e
[[ "$rc_a" == 0 && "$rc_b" == 0 ]] || exit 2

# C: exact old launcher class on GPU 0: DDP-at-world-size-1 plus the project
# runtime patch. The separate GPU-1 crossover cell is required to isolate the
# runtime patch because B and C otherwise alias patch state with physical GPU.
run_cell patched-ddp1 "$PATCHED" 0 ddp
python3 "$ROOT/sync_results.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX"
printf '{"status":"COMPLETE"}\n' > "$ROOT/COMPLETE.json"
python3 "$ROOT/sync_results.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX"
