#!/usr/bin/env bash

# D47 final trigger test: keep the clean public two-GPU recipe but require
# deterministic CUDA algorithms, including nn.Embedding backward.

set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_DETERMINISTIC_ROOT:-/workspace/d47-deterministic/seed1235}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
CLEAN="$PROJECT/upstream/NextLat"
SOURCE="${D47_DETERMINISTIC_SOURCE:-/workspace/d47-deterministic/nextlat-source}"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_DETERMINISTIC_GCS_PREFIX:?D47_DETERMINISTIC_GCS_PREFIX is required}"

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-project-flash-490419}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

[[ ! -e "$ROOT" && ! -e "$SOURCE" ]] || {
  echo "refusing to overwrite deterministic diagnostic paths" >&2
  exit 2
}
[[ -z "$(git -C "$CLEAN" status --porcelain)" ]] || exit 2
mkdir -p "$ROOT/output"
cp -a "$CLEAN" "$SOURCE"

git -C "$SOURCE" apply - <<'PATCH'
diff --git a/train.py b/train.py
index 19f8344..ce09a9d 100644
--- a/train.py
+++ b/train.py
@@ -168,6 +168,10 @@ def main(fabric, config):
     # Initialize PyTorch settings
     seed_offset = fabric.global_rank
     fabric.seed_everything(int(config.seed) + int(seed_offset))
+    torch.use_deterministic_algorithms(True)
+    torch.backends.cudnn.benchmark = False
+    torch.backends.cudnn.deterministic = True
+    fabric.print("D47_DETERMINISTIC_ALGORITHMS=True")
     torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
     torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
     torch._dynamo.config.cache_size_limit = 16  # allow more recompiles
PATCH

git -C "$SOURCE" diff --check
git -C "$SOURCE" diff > "$ROOT/deterministic.patch"
(
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  python -c 'import lightning,torch; print("torch="+torch.__version__); print("lightning="+lightning.__version__); print("cuda="+str(torch.version.cuda)); print("deterministic="+str(torch.are_deterministic_algorithms_enabled()))'
  git -C "$CLEAN" rev-parse HEAD
) > "$ROOT/runtime.txt"

cmd=(fabric run --precision bf16-mixed --devices 2 --strategy ddp --main-port 36337
  train.py --config "$CONFIG"
  seed=1235
  "trainer.out_dir=$ROOT/output"
  trainer.experiment_name=d47-deterministic-ddp2-seed1235
  trainer.train_batches=2100
  trainer.compile=false
  trainer.log_to_wandb=false
  trainer.log_to_file=false
  trainer.save_last_checkpoint=false
  trainer.save_best_checkpoint=false
  trainer.always_save_checkpoint=false
  trainer.save_recovery_checkpoint=-1
  'trainer.keep_checkpoint_steps=[2000]'
  "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
  "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")

printf 'CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0,1 ' > "$ROOT/command.txt"
printf '%q ' "${cmd[@]}" >> "$ROOT/command.txt"
printf '\n' >> "$ROOT/command.txt"

start="$(date +%s)"
set +e
(
  cd "$SOURCE"
  CUDA_VISIBLE_DEVICES=0,1 "${cmd[@]}"
) > "$ROOT/train.log" 2>&1
rc=$?
set -e
end="$(date +%s)"
printf '{"returncode":%d,"started_at_unix":%d,"ended_at_unix":%d,"elapsed_seconds":%d}\n' \
  "$rc" "$start" "$end" "$((end-start))" > "$ROOT/terminal.json"
python "$PROJECT/scripts/d47_upload_tree.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX" || true
exit "$rc"
