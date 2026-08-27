#!/usr/bin/env bash

# Outcome-blind diagnostic that separates initialization seed, sampler seed,
# epoch reshuffling, and residual repeatability in clean public NextLat DDP2.

set -euo pipefail
source /venv/main/bin/activate

ROOT="${D47_FACTORIAL_ROOT:-/workspace/d47-sampler-factorial}"
PROJECT="${D47_PROJECT:-/workspace/d47-clean/project}"
UPSTREAM="$PROJECT/upstream/NextLat"
CONFIG="$PROJECT/configs/nextlat_lurestar.yaml"
ENTRY="$PROJECT/scripts/d47_sampler_factorial_entry.py"
DATA="$PROJECT/data/stargraph"
RUNS="$ROOT/runs"
ADC="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.nextlat-secrets/gcp-adc.json}"
GCS_BUCKET="${D47_GCS_BUCKET:-nextlat-lurestar-project-flash-490419}"
GCS_PREFIX="${D47_FACTORIAL_GCS_PREFIX:?D47_FACTORIAL_GCS_PREFIX is required}"
EXPECTED_COMMIT="3770be6009cea2b3c455a9ce7f2ca88b504bb955"
EXPECTED_TRAIN_SHA="d13199b00c41d74325931cfecb15e3cf876d5e7d999c3257aaf4962e44827d76"
EXPECTED_TEST_SHA="f52fb14ef1c39aee187fd0ae40e781e7e716045d7b8270cddf9b7760afb74be9"

export GOOGLE_APPLICATION_CREDENTIALS="$ADC"
export GOOGLE_CLOUD_PROJECT="project-flash-490419"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
mkdir -p "$RUNS" "$ROOT/provenance"

die() { echo "D47 sampler factorial refused: $*" >&2; exit 2; }

[[ -f "$ENTRY" && -f "$CONFIG" && -f "$ADC" ]] || die "missing staged input"
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit drift"
[[ -z "$(git -C "$UPSTREAM" status --porcelain)" ]] || die "upstream tree is not clean"
[[ "$(sha256sum "$DATA/graph_5_5_sample_200000.txt" | awk '{print $1}')" == "$EXPECTED_TRAIN_SHA" ]] || die "training corpus drift"
[[ "$(sha256sum "$DATA/graph_5_5_test_20000.txt" | awk '{print $1}')" == "$EXPECTED_TEST_SHA" ]] || die "test corpus drift"

grep -vE '^[[:space:]]*torch([><=]|[[:space:]]|$)' "$UPSTREAM/requirements.txt" \
  > "$ROOT/requirements-no-torch.txt"
uv pip install --quiet -r "$ROOT/requirements-no-torch.txt"
uv pip install --quiet 'omegaconf==2.3.0' 'lightning==2.6.5' 'google-cloud-storage>=3.4,<4'

python3 - <<'PY' > "$ROOT/provenance/runtime.json"
import json, platform, subprocess
import lightning, torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert lightning.__version__ == "2.6.5", lightning.__version__
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
print(json.dumps({
    "schema": "nextlat_forgetting/d47_sampler_factorial_runtime/1",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "lightning": lightning.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    "nvidia_smi": subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader"], text=True).splitlines(),
}, indent=2, sort_keys=True))
PY

(cd "$PROJECT" && sha256sum \
  scripts/d47_sampler_factorial_entry.py \
  scripts/run_d47_sampler_factorial.sh \
  configs/nextlat_lurestar.yaml \
  data/stargraph/graph_5_5_sample_200000.txt \
  data/stargraph/graph_5_5_test_20000.txt \
  > "$ROOT/provenance/input_hashes.sha256")

cat > "$ROOT/sync_results.py" <<'PY'
from google.cloud import storage
import hashlib, os, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
bucket = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]).bucket(sys.argv[2])
prefix = sys.argv[3].strip("/")
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.suffix == ".pt" or path.name == "sync_results.py":
        continue
    rel = path.relative_to(root).as_posix()
    blob = bucket.blob(f"{prefix}/{rel}")
    blob.metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    blob.upload_from_filename(path, timeout=600)
PY

sync_results() {
  python3 "$ROOT/sync_results.py" "$ROOT" "$GCS_BUCKET" "$GCS_PREFIX" || true
}

run_cell() {
  local name="$1" init_seed="$2" data_seed="$3" epoch_mode="$4" port="$5"
  local cell="$RUNS/$name"
  local out="$cell/output"
  local trace="$cell/trace"
  mkdir -p "$out" "$trace"
  local cmd=(fabric run --precision bf16-mixed --devices 2 --strategy ddp --main-port "$port"
    "$ENTRY" --config "$CONFIG"
    "seed=$init_seed"
    "trainer.out_dir=$out"
    "trainer.experiment_name=d47-$name"
    trainer.train_batches=2100
    trainer.compile=false
    trainer.log_to_wandb=false
    trainer.save_last_checkpoint=false
    trainer.save_best_checkpoint=false
    trainer.always_save_checkpoint=false
    trainer.save_recovery_checkpoint=-1
    "data.stargraph_train_data_path=$DATA/graph_5_5_sample_200000.txt"
    "data.stargraph_test_data_path=$DATA/graph_5_5_test_20000.txt")
  printf 'D47_INIT_SEED=%q D47_DATA_SEED=%q D47_EPOCH_MODE=%q ' \
    "$init_seed" "$data_seed" "$epoch_mode" > "$cell/command.txt"
  printf '%q ' "${cmd[@]}" >> "$cell/command.txt"
  printf '\n' >> "$cell/command.txt"

  local started ended rc
  started="$(date +%s)"
  set +e
  (
    cd "$UPSTREAM"
    D47_UPSTREAM="$UPSTREAM" D47_TRACE_DIR="$trace" \
      D47_INIT_SEED="$init_seed" D47_DATA_SEED="$data_seed" \
      D47_EPOCH_MODE="$epoch_mode" "${cmd[@]}"
  ) > "$cell/train.log" 2>&1
  rc=$?
  set -e
  ended="$(date +%s)"

  RC="$rc" STARTED="$started" ENDED="$ended" NAME="$name" \
  INIT_SEED="$init_seed" DATA_SEED="$data_seed" EPOCH_MODE="$epoch_mode" \
    python3 - "$cell/terminal.json" "$cell/train.log" <<'PY'
import json, os, pathlib, re, sys
log = pathlib.Path(sys.argv[2]).read_text(errors="replace")
accuracies = [float(x) for x in re.findall(r"StarGraph Test Accuracy .*?: ([0-9.]+)%", log)]
payload = {
    "schema": "nextlat_forgetting/d47_sampler_factorial_terminal/1",
    "name": os.environ["NAME"],
    "returncode": int(os.environ["RC"]),
    "started_at_unix": int(os.environ["STARTED"]),
    "ended_at_unix": int(os.environ["ENDED"]),
    "elapsed_seconds": int(os.environ["ENDED"]) - int(os.environ["STARTED"]),
    "init_seed": int(os.environ["INIT_SEED"]),
    "data_seed": int(os.environ["DATA_SEED"]),
    "epoch_mode": os.environ["EPOCH_MODE"],
    "reported_exact_path_accuracy_percent": accuracies,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  sync_results
  return "$rc"
}

# Frozen order.  Do not stop or add cells in response to intermediate outcomes.
run_cell fixed-i1235-d1235 1235 1235 fixed 37101
run_cell reshuffle-i1235-d1235-a 1235 1235 reshuffle 37102
run_cell reshuffle-i1234-d1234 1234 1234 reshuffle 37103
run_cell reshuffle-i1234-d1235 1234 1235 reshuffle 37104
run_cell reshuffle-i1235-d1234 1235 1234 reshuffle 37105
run_cell reshuffle-i1235-d1235-b 1235 1235 reshuffle 37106

printf '{"status":"COMPLETE","cells":6}\n' > "$ROOT/COMPLETE.json"
sync_results
