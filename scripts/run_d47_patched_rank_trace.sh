#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_ROOT:-/workspace/d47-clean}"
PROJECT="${D47_PROJECT:-$ROOT/project}"
UPSTREAM="${D47_FACTOR_ROOT:-/workspace/d47-factors}/patched-nextlat"
TRACE_ROOT="${D47_TRACE_ROOT:-$ROOT/rank-parity}"
LABEL=patched1
OUT="$ROOT/runs/rank-parity/$LABEL/nextlat/seed1235"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
TRAIN="$PROJECT/data/stargraph/graph_5_5_sample_200000.txt"
TEST="$PROJECT/data/stargraph/graph_5_5_test_20000.txt"
ENTRY="$PROJECT/scripts/d47_rank_parity_entry.py"

[[ -f "$UPSTREAM/.lurestar_runtime_patch_receipt.json" ]] || exit 2
[[ ! -e "$TRACE_ROOT/$LABEL" && ! -e "$OUT" ]] || {
  echo "refusing to overwrite patched rank trace" >&2
  exit 2
}
mkdir -p "$TRACE_ROOT"
cd "$UPSTREAM"
CUDA_VISIBLE_DEVICES=0 \
D47_TRACE_UPSTREAM="$UPSTREAM" \
D47_TRACE_DIR="$TRACE_ROOT" \
D47_TRACE_LABEL="$LABEL" \
LURESTAR_NONCONFIRMATORY=1 \
WANDB_MODE=disabled \
fabric run --precision bf16-mixed --devices 1 --main-port 36104 --strategy ddp \
  "$ENTRY" --config "$CONFIG" \
  seed=1235 \
  "trainer.out_dir=$OUT" \
  trainer.experiment_name=nextlat-seed1235-d47-rank-patched1 \
  trainer.train_batches=2 \
  trainer.compile=false \
  trainer.log_to_wandb=false \
  trainer.log_to_file=false \
  trainer.save_last_checkpoint=false \
  trainer.save_best_checkpoint=false \
  trainer.always_save_checkpoint=false \
  trainer.save_recovery_checkpoint=-1 \
  "data.stargraph_train_data_path=$TRAIN" \
  "data.stargraph_test_data_path=$TEST" \
  >"$TRACE_ROOT/$LABEL.log" 2>&1
