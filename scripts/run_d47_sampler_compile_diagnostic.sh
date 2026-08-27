#!/usr/bin/env bash

# D47 follow-up: isolate the two remaining execution-path differences for seed 1235.
# Both cells use the pinned public source, two same-host GPUs, DDP, BF16, the
# frozen Path-Star corpus, and stop after the validation at update 2,000.

set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_TRIGGER_ROOT:-/workspace/d47-triggers/seed1235}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
CLEAN="$PROJECT/upstream/NextLat"
SHARED="${D47_SHARED_SOURCE:-/workspace/d47-triggers/shared-sampler-nextlat-seed1235}"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_TRIGGER_GCS_PREFIX:?D47_TRIGGER_GCS_PREFIX is required}"

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-project-flash-490419}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

[[ ! -e "$ROOT" ]] || {
  echo "refusing to overwrite $ROOT" >&2
  exit 2
}
[[ -z "$(git -C "$CLEAN" status --porcelain)" ]] || {
  echo "clean upstream is dirty" >&2
  exit 2
}
mkdir -p "$ROOT/runs"
cp -a "$CLEAN" "$SHARED"

# The public launcher seeds each rank with base_seed + global_rank. In the
# installed Lightning version, PL_GLOBAL_SEED is also the DistributedSampler
# seed. Thus rank 0 and rank 1 do not shard one common permutation. This
# diagnostic preserves rank-specific model/RNG seeding while forcing only the
# sampler seed back to the base seed before the datamodule is constructed.
git -C "$SHARED" apply - <<'PATCH'
diff --git a/train.py b/train.py
index 19f8344..1a09f08 100644
--- a/train.py
+++ b/train.py
@@ -168,6 +168,9 @@ def main(fabric, config):
     # Initialize PyTorch settings
     seed_offset = fabric.global_rank
     fabric.seed_everything(int(config.seed) + int(seed_offset))
+    if os.environ.get("NEXTLAT_SHARED_DISTRIBUTED_SAMPLER_SEED") == "1":
+        # D47 diagnostic only: keep compute RNG rank-specific, share sampler RNG.
+        os.environ["PL_GLOBAL_SEED"] = str(int(config.seed))
     torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
     torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
     torch._dynamo.config.cache_size_limit = 16  # allow more recompiles
PATCH

git -C "$SHARED" diff --check
git -C "$SHARED" diff > "$ROOT/shared-sampler.patch"
(
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
  python -c 'import lightning,torch; print("torch="+torch.__version__); print("lightning="+lightning.__version__); print("cuda="+str(torch.version.cuda))'
  git -C "$CLEAN" rev-parse HEAD
) > "$ROOT/runtime.txt"
sha256sum "$DATA/graph_5_5_sample_200000.txt" "$DATA/graph_5_5_test_20000.txt" > "$ROOT/corpus.sha256"

upload_results() {
  python "$PROJECT/scripts/d47_upload_tree.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX" || true
}

run_cell() {
  local name="$1" upstream="$2" compile="$3" shared_sampler="$4" port="$5"
  local cell="$ROOT/runs/$name"
  local out="$cell/output"
  mkdir -p "$out"

  local cmd=(fabric run --precision bf16-mixed --devices 2 --strategy ddp --main-port "$port"
    train.py --config "$CONFIG"
    seed=1235
    "trainer.out_dir=$out"
    "trainer.experiment_name=d47-$name"
    trainer.train_batches=2100
    "trainer.compile=$compile"
    trainer.log_to_wandb=false
    trainer.log_to_file=false
    trainer.save_last_checkpoint=false
    trainer.save_best_checkpoint=false
    trainer.always_save_checkpoint=false
    trainer.save_recovery_checkpoint=-1
    'trainer.keep_checkpoint_steps=[2000]'
    "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
    "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")

  printf 'NEXTLAT_SHARED_DISTRIBUTED_SAMPLER_SEED=%q CUDA_VISIBLE_DEVICES=0,1 ' "$shared_sampler" > "$cell/command.txt"
  printf '%q ' "${cmd[@]}" >> "$cell/command.txt"
  printf '\n' >> "$cell/command.txt"

  local start end rc
  start="$(date +%s)"
  set +e
  (
    cd "$upstream"
    NEXTLAT_SHARED_DISTRIBUTED_SAMPLER_SEED="$shared_sampler" \
      CUDA_VISIBLE_DEVICES=0,1 "${cmd[@]}"
  ) > "$cell/train.log" 2>&1
  rc=$?
  set -e
  end="$(date +%s)"
  printf '{"returncode":%d,"started_at_unix":%d,"ended_at_unix":%d,"elapsed_seconds":%d}\n' \
    "$rc" "$start" "$end" "$((end-start))" > "$cell/terminal.json"
  upload_results
  return "$rc"
}

# Cell 1: clean source plus one surgical sampler-seed correction.
run_cell shared-sampler-ddp2 "$SHARED" false 1 36335

# Cell 2: untouched source with the public config's compile=true setting. The
# upstream README warns that this path is inconsistent on Path-Star; this cell
# measures it rather than assuming either outcome.
run_cell public-compile-ddp2 "$CLEAN" true 0 36336

printf '{"status":"COMPLETE"}\n' > "$ROOT/COMPLETE.json"
upload_results
