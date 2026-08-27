#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate

# Run after the longer D47 cells.  These are two-update traces, not outcome runs.
ROOT="${D47_ROOT:-/workspace/d47-clean}"
PROJECT="${D47_PROJECT:-$ROOT/project}"
UPSTREAM="${D47_UPSTREAM:-$PROJECT/upstream/NextLat}"
TRACE_ROOT="${D47_TRACE_ROOT:-$ROOT/rank-parity}"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_TRACE_GCS_PREFIX:-}"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
TRAIN="$PROJECT/data/stargraph/graph_5_5_sample_200000.txt"
TEST="$PROJECT/data/stargraph/graph_5_5_test_20000.txt"
ENTRY="$PROJECT/scripts/d47_rank_parity_entry.py"

mkdir -p "$TRACE_ROOT" "$ROOT/runs/rank-parity"

run_trace() {
  local label="$1" devices="$2" strategy="$3" visible="$4" port="$5"
  local out="$ROOT/runs/rank-parity/$label"
  [[ ! -e "$out" && ! -e "$TRACE_ROOT/$label" ]] || {
    echo "refusing to overwrite an existing rank-parity cell: $label" >&2
    return 2
  }
  local cmd=(fabric run --precision bf16-mixed --devices "$devices" --main-port "$port")
  [[ -n "$strategy" ]] && cmd+=(--strategy "$strategy")
  (
    cd "$UPSTREAM"
    CUDA_VISIBLE_DEVICES="$visible" \
    D47_TRACE_UPSTREAM="$UPSTREAM" \
    D47_TRACE_DIR="$TRACE_ROOT" \
    D47_TRACE_LABEL="$label" \
    WANDB_MODE=disabled \
      "${cmd[@]}" "$ENTRY" --config "$CONFIG" \
        seed=1235 \
        trainer.out_dir="$out" \
        trainer.experiment_name="d47-rank-$label" \
        trainer.train_batches=2 \
        trainer.compile=false \
        trainer.log_to_wandb=false \
        trainer.log_to_file=false \
        trainer.save_last_checkpoint=false \
        trainer.save_best_checkpoint=false \
        trainer.always_save_checkpoint=false \
        trainer.save_recovery_checkpoint=-1 \
        data.stargraph_train_data_path="$TRAIN" \
        data.stargraph_test_data_path="$TEST"
  ) >"$TRACE_ROOT/$label.log" 2>&1
}

run_trace plain1 1 "" 0 36101
run_trace ddp1 1 ddp 0 36102

# The GPU-1 patch crossover may still be finishing. Do not contend with it;
# the two-rank trace starts as soon as that one bounded cell releases GPU 1.
if command -v supervisorctl >/dev/null 2>&1; then
  while supervisorctl status d47-crosscheck 2>/dev/null | grep -q ' RUNNING '; do
    sleep 10
  done
fi
run_trace ddp2 2 ddp 0,1 36103

/venv/main/bin/python3 - "$TRACE_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cells = {}
for label in ("plain1", "ddp1", "ddp2"):
    backward = json.loads((root / label / "after_backward.json").read_text())
    optimizer = json.loads((root / label / "after_optimizer.json").read_text())
    cells[label] = {"backward": backward, "optimizer": optimizer}

def parity(records, field):
    names = records[0][field]
    mismatches = []
    for name in names:
        values = [rank[field][name] for rank in records]
        hashes = [None if value is None else value["sha256"] for value in values]
        if len(set(hashes)) != 1:
            mismatches.append(name)
    return mismatches

summary = {"schema": "nextlat_forgetting/d47_rank_parity_summary/1", "cells": {}}
for label, cell in cells.items():
    backward_ranks = cell["backward"]["ranks"]
    optimizer_ranks = cell["optimizer"]["ranks"]
    summary["cells"][label] = {
        "world_size": cell["backward"]["world_size"],
        "global_batch_multiset_sha256": cell["backward"]["global_batch_multiset_sha256"],
        "initial_parameter_rank_mismatches": parity(backward_ranks, "parameters"),
        "gradient_rank_mismatches": parity(backward_ranks, "gradients"),
        "post_optimizer_parameter_rank_mismatches": parity(optimizer_ranks, "parameters"),
        "missing_gradient_parameters": [
            name for name, value in backward_ranks[0]["gradients"].items() if value is None
        ],
    }

summary["comparisons"] = {
    "ddp1_vs_ddp2_same_global_batch_multiset": (
        summary["cells"]["ddp1"]["global_batch_multiset_sha256"]
        == summary["cells"]["ddp2"]["global_batch_multiset_sha256"]
    ),
    "plain1_vs_ddp1_same_global_batch_multiset": (
        summary["cells"]["plain1"]["global_batch_multiset_sha256"]
        == summary["cells"]["ddp1"]["global_batch_multiset_sha256"]
    ),
}
payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
(root / "summary.json").write_text(payload)
(root / "summary.json.sha256").write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n")
print(payload)
PY

if [[ -n "$GCS_PREFIX" && -f /workspace/d47-factors/sync_results.py ]]; then
  /venv/main/bin/python3 /workspace/d47-factors/sync_results.py \
    "$TRACE_ROOT" "$GCS_BUCKET" "$GCS_PREFIX"
fi
