#!/usr/bin/env bash
set -euo pipefail

# Same physical GPU and DDP1 launch as clean-ddp1-sampler; only the guarded
# project runtime patch differs. This closes the GPU-vs-patch alias in D47.
source /venv/main/bin/activate

ROOT="${D47_FACTOR_ROOT:-/workspace/d47-factors}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
UPSTREAM="$ROOT/patched-nextlat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
CELL="$ROOT/runs/patched-ddp1-gpu1-crosscheck"
OUT="$CELL/nextlat/seed1235/output"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_FACTOR_GCS_PREFIX:?D47_FACTOR_GCS_PREFIX is required}/patched-ddp1-gpu1-crosscheck"

[[ -f "$UPSTREAM/.lurestar_runtime_patch_receipt.json" ]] || {
  echo "patched upstream receipt is missing" >&2
  exit 2
}
[[ ! -e "$CELL" ]] || {
  echo "refusing to overwrite $CELL" >&2
  exit 2
}
mkdir -p "$OUT"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cmd=(fabric run --precision bf16-mixed --devices 1 --main-port 35101 --strategy ddp
  train.py --config "$CONFIG"
  seed=1235
  "trainer.out_dir=$OUT"
  trainer.experiment_name=nextlat-seed1235-patched-ddp1-gpu1-crosscheck
  trainer.train_batches=2100
  trainer.compile=false
  trainer.log_to_wandb=false
  trainer.save_last_checkpoint=false
  trainer.save_best_checkpoint=false
  trainer.always_save_checkpoint=false
  trainer.save_recovery_checkpoint=-1
  "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
  "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")
printf 'CUDA_VISIBLE_DEVICES=1 ' > "$CELL/command.txt"
printf '%q ' "${cmd[@]}" >> "$CELL/command.txt"
printf '\n' >> "$CELL/command.txt"

start="$(date +%s)"
set +e
(
  cd "$UPSTREAM"
  CUDA_VISIBLE_DEVICES=1 LURESTAR_NONCONFIRMATORY=1 "${cmd[@]}"
) > "$CELL/train.log" 2>&1
rc=$?
set -e
end="$(date +%s)"
printf '{"returncode":%d,"started_at_unix":%d,"ended_at_unix":%d,"elapsed_seconds":%d}\n' \
  "$rc" "$start" "$end" "$((end-start))" > "$CELL/terminal.json"
python3 "$ROOT/sync_results.py" "$CELL" "$GCS_BUCKET" "$GCS_PREFIX" || true
exit "$rc"
