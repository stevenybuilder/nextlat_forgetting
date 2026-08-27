#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_FULL_ROOT:-/workspace/d47-full-baseline/seed1235}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
UPSTREAM="$PROJECT/upstream/NextLat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
OUT="$ROOT/output"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_FULL_GCS_PREFIX:?D47_FULL_GCS_PREFIX is required}"

[[ ! -e "$ROOT" ]] || {
  echo "refusing to overwrite $ROOT" >&2
  exit 2
}
[[ -z "$(git -C "$UPSTREAM" status --porcelain)" ]] || exit 2
mkdir -p "$OUT"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cmd=(fabric run --precision bf16-mixed --devices 2 --main-port 36235 --strategy ddp
  train.py --config "$CONFIG"
  seed=1235
  "trainer.out_dir=$OUT"
  trainer.experiment_name=d47-official-ddp2-seed1235-full
  trainer.train_batches=20000
  trainer.compile=false
  trainer.log_to_wandb=false
  trainer.log_to_file=false
  trainer.save_last_checkpoint=false
  trainer.save_best_checkpoint=false
  trainer.always_save_checkpoint=false
  trainer.save_recovery_checkpoint=-1
  'trainer.keep_checkpoint_steps=[10000,20000]'
  "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
  "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")
printf 'CUDA_VISIBLE_DEVICES=0,1 ' > "$ROOT/command.txt"
printf '%q ' "${cmd[@]}" >> "$ROOT/command.txt"
printf '\n' >> "$ROOT/command.txt"
(
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  /venv/main/bin/python3 -c 'import lightning,torch; print("torch="+torch.__version__); print("lightning="+lightning.__version__); print("cuda="+str(torch.version.cuda))'
  git -C "$UPSTREAM" rev-parse HEAD
) > "$ROOT/runtime.txt"

start="$(date +%s)"
set +e
(
  cd "$UPSTREAM"
  CUDA_VISIBLE_DEVICES=0,1 "${cmd[@]}"
) > "$ROOT/train.log" 2>&1 &
train_pid=$!
set -e

# One off-box midpoint save; no rolling checkpoints or repeated full-tree sync.
midpoint_uploaded=0
while kill -0 "$train_pid" 2>/dev/null; do
  if [[ "$midpoint_uploaded" == 0 ]] && find "$OUT" -name 'ckpt_iter_10000.pt' -print -quit | grep -q .; then
    /venv/main/bin/python3 "$PROJECT/scripts/d47_upload_tree.py" \
      "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX/midpoint" || true
    midpoint_uploaded=1
  fi
  sleep 30
done
set +e
wait "$train_pid"
rc=$?
set -e
end="$(date +%s)"
printf '{"returncode":%d,"started_at_unix":%d,"ended_at_unix":%d,"elapsed_seconds":%d}\n' \
  "$rc" "$start" "$end" "$((end-start))" > "$ROOT/terminal.json"
/venv/main/bin/python3 "$PROJECT/scripts/d47_upload_tree.py" \
  "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX/final" || true
exit "$rc"
