#!/usr/bin/env bash

# D47 diagnostic executor. This script runs inside one same-host two-GPU Vast
# instance after the minimal project bundle and GCP ADC have been staged.
#
# Fixed order:
#   1. one-GPU seed 1234 and seed 1235, concurrently on distinct GPUs
#   2. two-GPU DDP seed 1235
#   3. two-GPU DDP seed 1234
#
# Each cell stops at 4,100 optimizer updates so the upstream validation trace at
# updates 1,000, 2,000, 3,000, and 4,000 is present. Diagnostic weights never
# replace the frozen confirmatory checkpoints.

set -euo pipefail

source /venv/main/bin/activate

ROOT="${D47_ROOT:-/workspace/d47}"
PROJECT="$ROOT/project"
UPSTREAM="$PROJECT/upstream/NextLat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
DATA="$PROJECT/data/stargraph"
RUNS="$ROOT/runs"
PROVENANCE="$ROOT/provenance"
ADC="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_GCS_PREFIX:?D47_GCS_PREFIX is required}"
EXPECTED_COMMIT="3770be6009cea2b3c455a9ce7f2ca88b504bb955"
EXPECTED_TRAIN_SHA="d13199b00c41d74325931cfecb15e3cf876d5e7d999c3257aaf4962e44827d76"
EXPECTED_TEST_SHA="f52fb14ef1c39aee187fd0ae40e781e7e716045d7b8270cddf9b7760afb74be9"

mkdir -p "$RUNS" "$PROVENANCE"
export GOOGLE_APPLICATION_CREDENTIALS="$ADC"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

die() {
  echo "D47 REFUSED: $*" >&2
  exit 2
}

[[ -f "$CONFIG" ]] || die "missing diagnostic config: $CONFIG"
[[ -f "$DATA/graph_5_5_sample_200000.txt" ]] || die "missing training corpus"
[[ -f "$DATA/graph_5_5_test_20000.txt" ]] || die "missing held-out corpus"
[[ -f "$ADC" ]] || die "missing GCP ADC: $ADC"
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "upstream commit drift"
[[ "$(sha256sum "$DATA/graph_5_5_sample_200000.txt" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA" ]] || die "training corpus drift"
[[ "$(sha256sum "$DATA/graph_5_5_test_20000.txt" | awk '{print $1}')" == "$EXPECTED_TEST_SHA" ]] || die "held-out corpus drift"

grep -vE '^[[:space:]]*torch([><=]|[[:space:]]|$)' "$UPSTREAM/requirements.txt" \
  > "$ROOT/requirements-no-torch.txt"
uv pip install --quiet -r "$ROOT/requirements-no-torch.txt"
uv pip install --quiet \
  'omegaconf==2.3.0' 'lightning==2.6.5' 'google-cloud-storage>=3.4,<4'

[[ -z "$(git -C "$UPSTREAM" status --porcelain)" ]] || \
  die "diagnostic requires an untouched pinned upstream checkout"

python3 - "$PROVENANCE/runtime.json" "$EXPECTED_COMMIT" <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import lightning
import numpy
import omegaconf
import torch

out = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
if torch.__version__ != "2.11.0+cu128":
    raise SystemExit(f"refusing runtime drift: torch={torch.__version__!r}")
if lightning.__version__ != "2.6.5":
    raise SystemExit(f"refusing runtime drift: lightning={lightning.__version__!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit(f"expected exactly two visible GPUs, found {torch.cuda.device_count()}")

payload = {
    "schema": "nextlat_forgetting/d47_runtime/1",
    "captured_at_unix": time.time(),
    "instance_id": os.environ.get("VAST_INSTANCE_ID"),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "lightning": lightning.__version__,
    "omegaconf": omegaconf.__version__,
    "numpy": numpy.__version__,
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "upstream_commit": expected_commit,
    "nvidia_smi": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).splitlines(),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

(
  cd "$PROJECT"
  sha256sum \
    configs/nextlat_lurestar.yaml \
    data/stargraph/graph_5_5_sample_200000.txt \
    data/stargraph/graph_5_5_test_20000.txt \
    scripts/run_d47_causal_diagnostic.sh \
    upstream/NextLat/train.py \
    upstream/NextLat/core_train.py \
    upstream/NextLat/models/model_base.py \
    > "$PROVENANCE/input_hashes.sha256"
  git -C upstream/NextLat status --porcelain > "$PROVENANCE/upstream_status.txt"
)
python3 -m pip freeze > "$PROVENANCE/pip_freeze.txt"
cp "$CONFIG" "$PROVENANCE/base_config.yaml"

cat > "$ROOT/sync_once.py" <<'PY'
from __future__ import annotations

import hashlib
import os
import pathlib
import sys

from google.cloud import storage

root = pathlib.Path(sys.argv[1]).resolve()
bucket_name = sys.argv[2]
prefix = sys.argv[3].strip("/")
client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
bucket = client.bucket(bucket_name)

for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(root)
    if rel.parts[0] in {"project", "data"} or rel.name == "sync_once.py":
        continue
    if path.suffix == ".pt":
        if len(rel.parts) < 3 or rel.parts[0] != "runs":
            continue
        if not (root / "runs" / rel.parts[1] / "terminal.json").is_file():
            continue
    blob = bucket.blob(f"{prefix}/{rel.as_posix()}")
    blob.metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    blob.upload_from_filename(path, timeout=600)
PY

sync_once() {
  python3 "$ROOT/sync_once.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX" || true
}

sync_loop() {
  while [[ ! -f "$ROOT/STOP_SYNC" ]]; do
    sync_once
    sleep 120
  done
  sync_once
}

sync_loop > "$ROOT/sync.log" 2>&1 &
SYNC_PID=$!
cleanup() {
  touch "$ROOT/STOP_SYNC"
  wait "$SYNC_PID" 2>/dev/null || true
}
trap cleanup EXIT

run_cell() {
  local topology="$1"
  local seed="$2"
  local visible="$3"
  local devices="$4"
  local strategy="$5"
  local name="${topology}-seed${seed}"
  local cell="$RUNS/$name"
  local out="$cell/output"
  local exp="d47-${topology}-seed${seed}"
  mkdir -p "$cell" "$out"

  local main_port
  if [[ "$topology" == "onegpu" ]]; then
    main_port=$((30000 + seed))
  else
    main_port=$((32000 + seed))
  fi
  local cmd=(fabric run --precision bf16-mixed --devices "$devices" --main-port "$main_port")
  if [[ -n "$strategy" ]]; then
    cmd+=(--strategy "$strategy")
  fi
  cmd+=(train.py --config "$CONFIG"
    "seed=$seed"
    "trainer.out_dir=$out"
    "trainer.experiment_name=$exp"
    "trainer.train_batches=4100"
    "trainer.compile=false"
    "trainer.log_to_wandb=false"
    "trainer.save_last_checkpoint=true"
    "trainer.save_best_checkpoint=false"
    "trainer.always_save_checkpoint=false"
    "trainer.save_recovery_checkpoint=-1"
    "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
    "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")

  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$visible"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$cell/command.txt"

  local started ended rc
  started="$(date +%s)"
  set +e
  (
    cd "$UPSTREAM"
    CUDA_VISIBLE_DEVICES="$visible" "${cmd[@]}"
  ) > "$cell/train.log" 2>&1
  rc=$?
  set -e
  ended="$(date +%s)"
  RC="$rc" STARTED="$started" ENDED="$ended" TOPOLOGY="$topology" SEED="$seed" \
    VISIBLE="$visible" DEVICES="$devices" python3 - "$cell/terminal.json" <<'PY'
import json
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema": "nextlat_forgetting/d47_cell_terminal/1",
    "returncode": int(os.environ["RC"]),
    "started_at_unix": int(os.environ["STARTED"]),
    "ended_at_unix": int(os.environ["ENDED"]),
    "elapsed_seconds": int(os.environ["ENDED"]) - int(os.environ["STARTED"]),
    "topology": os.environ["TOPOLOGY"],
    "seed": int(os.environ["SEED"]),
    "visible_devices": os.environ["VISIBLE"],
    "fabric_devices": int(os.environ["DEVICES"]),
}, indent=2, sort_keys=True) + "\n")
PY
  sync_once
  return "$rc"
}

echo "D47 phase 1: two independent one-GPU cells"
set +e
run_cell onegpu 1234 0 1 "" &
P1234=$!
run_cell onegpu 1235 1 1 "" &
P1235=$!
wait "$P1234"; RC_ONE_1234=$?
wait "$P1235"; RC_ONE_1235=$?
set -e
if [[ "$RC_ONE_1234" -ne 0 || "$RC_ONE_1235" -ne 0 ]]; then
  die "one-GPU phase failed: seed1234=$RC_ONE_1234 seed1235=$RC_ONE_1235"
fi

echo "D47 phase 2: two-GPU DDP seed 1235"
run_cell twogpu-ddp 1235 0,1 2 ddp

echo "D47 phase 3: two-GPU DDP seed 1234 control"
run_cell twogpu-ddp 1234 0,1 2 ddp

python3 - "$ROOT/COMPLETE.json" <<'PY'
import json
import pathlib
import sys
import time

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema": "nextlat_forgetting/d47_complete/1",
    "status": "COMPLETE",
    "completed_at_unix": time.time(),
    "cells": [
        "onegpu-seed1234",
        "onegpu-seed1235",
        "twogpu-ddp-seed1235",
        "twogpu-ddp-seed1234",
    ],
}, indent=2, sort_keys=True) + "\n")
PY
sync_once
echo "D47 diagnostic complete"
