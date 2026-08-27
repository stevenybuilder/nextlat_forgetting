#!/usr/bin/env bash

# Two exact two-update repeats with the explicit corrected sampler.  This cell
# localizes which first-step gradients diverge before long-horizon amplification.

set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_GRADIENT_ROOT:-/workspace/d47-first-gradient-repeat}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
UPSTREAM="$PROJECT/upstream/NextLat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
ENTRY="$PROJECT/scripts/d47_sampler_factorial_entry.py"
DATA="$PROJECT/data/stargraph"
SYNC="/workspace/d47-sampler-factorial/sync_results.py"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_GRADIENT_GCS_PREFIX:?D47_GRADIENT_GCS_PREFIX is required}"

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT"

run_trace() {
  local label="$1" port="$2"
  local cell="$ROOT/$label"
  mkdir -p "$cell/output" "$cell/trace"
  (
    cd "$UPSTREAM"
    D47_UPSTREAM="$UPSTREAM" D47_TRACE_DIR="$cell/trace" \
      D47_INIT_SEED=1235 D47_DATA_SEED=1235 D47_EPOCH_MODE=reshuffle \
      fabric run --precision bf16-mixed --devices 2 --strategy ddp --main-port "$port" \
        "$ENTRY" --config "$CONFIG" \
        seed=1235 \
        "trainer.out_dir=$cell/output" \
        "trainer.experiment_name=d47-first-gradient-$label" \
        trainer.train_batches=2 \
        trainer.val_batches=1 \
        trainer.test_batches=1 \
        trainer.compile=false \
        trainer.log_to_file=false \
        trainer.log_to_wandb=false \
        trainer.save_last_checkpoint=false \
        trainer.save_best_checkpoint=false \
        trainer.always_save_checkpoint=false \
        trainer.save_recovery_checkpoint=-1 \
        "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt" \
        "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt"
  ) > "$cell/train.log" 2>&1
}

run_trace repeat-a 37201
run_trace repeat-b 37202

python3 - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {"schema": "nextlat_forgetting/d47_first_gradient_repeat/1", "ranks": {}}
for rank in (0, 1):
    a = json.loads((root / "repeat-a" / "trace" / f"first_update.rank{rank}.json").read_text())
    b = json.loads((root / "repeat-b" / "trace" / f"first_update.rank{rank}.json").read_text())
    names = sorted(set(a["gradient_parameters"]) | set(b["gradient_parameters"]))
    mismatches = []
    for name in names:
        left = a["gradient_parameters"].get(name)
        right = b["gradient_parameters"].get(name)
        left_hash = None if left is None else left["sha256"]
        right_hash = None if right is None else right["sha256"]
        if left_hash != right_hash:
            mismatches.append({"name": name, "repeat_a": left, "repeat_b": right})
    summary["ranks"][str(rank)] = {
        "initial_parameters_equal": a["initial_parameters_sha256"] == b["initial_parameters_sha256"],
        "local_batch_equal": a["local_batch_sha256"] == b["local_batch_sha256"],
        "aggregate_gradient_equal": a["gradients_sha256"] == b["gradients_sha256"],
        "gradient_parameter_mismatches": mismatches,
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

python3 "$SYNC" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX"
printf '{"status":"COMPLETE"}\n' > "$ROOT/COMPLETE.json"
python3 "$SYNC" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX"
