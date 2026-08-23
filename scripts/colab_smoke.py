"""Colab smoke test: prove the transport, the environment, and a 20-step NextLat run.

Runs as the DRIVER on a Colab GPU runtime under `colab exec`. Everything is baked in:
`colab exec` runs the file in a Jupyter kernel that never sees argv, and `__file__` is
undefined there, so no parameter may arrive by command line and no path may be derived
from `__file__`.
"""
import collections
import os
import subprocess
import sys
import time

PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
REPO = "https://github.com/JaydenTeoh/NextLat.git"
WORK = "/content/nextlat"
GCS = "gs://nextlat-lurestar-project-flash-490419/lurestar/smoke"


def sh(cmd, check=True, timeout=1800):
    """Run a shell command, relaying every child line to OUR stdout as it arrives.

    Two transport facts drive this. First, a child process\'s stdout does NOT reach the
    `colab exec` stream on its own -- it must be read in-process and re-printed, or the
    command runs completely blind. Second, a fully captured (silent) long job starves the
    exec websocket and the connection drops, so lines are relayed as they arrive rather
    than collected and printed at the end. Never pipe a checked command into `tail`: the
    pipeline\'s exit status is tail\'s, so a crashed job reports success.
    """
    print("+ " + cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = collections.deque(maxlen=40)
    deadline = time.time() + timeout
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        print("  | " + line, flush=True)
        if time.time() > deadline:
            proc.kill()
            raise SystemExit("TIMEOUT after %ds: %s" % (timeout, cmd))
    rc = proc.wait()
    print("  rc=%d" % rc, flush=True)
    if check and rc != 0:
        raise SystemExit("FAILED (%d): %s" % (rc, cmd))
    return rc


print("=== 1. runtime identity ===", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      flush=True)
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print("capability", cap, "bf16_supported", torch.cuda.is_bf16_supported(), flush=True)

print("=== 2. durable auth ===", flush=True)
# The local credential is an `authorized_user` ADC (refresh token), not a service-account
# key -- the org policy constraints/iam.disableServiceAccountKeyCreation forbids issuing
# one. `gcloud auth activate-service-account` therefore cannot consume it. Two things do:
# the python google-cloud-storage client reads it directly from
# GOOGLE_APPLICATION_CREDENTIALS, and the gcloud CLI accepts a short-lived access token
# minted from it via CLOUDSDK_AUTH_ACCESS_TOKEN. The refresh token itself is long-lived,
# so the access token can be re-minted in-process for the length of any run.
sh("pip -q install google-cloud-storage google-auth", check=False)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "project-flash-490419"

BUCKET = "nextlat-lurestar-project-flash-490419"
PREFIX = "lurestar/smoke"
gcs_ok = False
try:
    from google.cloud import storage
    client = storage.Client(project="project-flash-490419")
    blob = client.bucket(BUCKET).blob(PREFIX + "/probe_python.txt")
    blob.upload_from_string("smoke %s\n" % time.time())
    print("PY_CLIENT_WRITE=True", flush=True)
    print("PY_CLIENT_READ=%r" % blob.download_as_text().strip(), flush=True)
    gcs_ok = True
except Exception as exc:
    print("PY_CLIENT_WRITE=False %s: %s" % (type(exc).__name__, exc), flush=True)

try:
    import google.auth
    import google.auth.transport.requests as gart
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(gart.Request())
    os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = creds.token
    print("MINTED_ACCESS_TOKEN=True", flush=True)
except Exception as exc:
    print("MINTED_ACCESS_TOKEN=False %s: %s" % (type(exc).__name__, exc), flush=True)

rc = sh("echo smoke-$(date -u +%%s) > /tmp/probe.txt && "
        "gcloud storage cp /tmp/probe.txt gs://%s/%s/probe_cli.txt" % (BUCKET, PREFIX),
        check=False)
print("GCS_CLI_WRITABLE=%s" % (rc == 0), flush=True)
print("GCS_ANY_WRITABLE=%s" % (gcs_ok or rc == 0), flush=True)

print("=== 3. pinned repo ===", flush=True)
if not os.path.isdir(WORK):
    sh("git clone -q %s %s" % (REPO, WORK))
sh("cd %s && git checkout -q %s && git rev-parse HEAD" % (WORK, PINNED))

print("=== 4. deps ===", flush=True)
sh("pip -q install omegaconf lightning", check=False)
sh("python -c \"import lightning, omegaconf; print('lightning', lightning.__version__)\"",
   check=False)

print("=== 5. tiny stargraph data (pipeline test only, NOT scientific) ===", flush=True)
sh("cd %s && python data/stargraph/prepare.py --num_samples 2000 --num_test_samples 200 "
   "--num_paths 5 --path_len 5 --max_nodes 100 --generate_test_data "
   "--data_dir data/stargraph" % WORK, check=False)
sh("ls -la %s/data/stargraph/ | head" % WORK, check=False)
sh("head -c 300 %s/data/stargraph/graph_5_5_sample_2000.txt" % WORK, check=False)

print("\n=== 6. 20-step NextLat run ===", flush=True)
cfg = os.path.join(WORK, "config/stargraph/5_5/nextlat_smoke.yaml")
with open(cfg, "w") as f:
    f.write("""use_nextlat: true
trainer:
  epochs: -1
  train_batches: 20
  val_batches: 2
  test_batches: 2
  log_interval: 1
  val_interval: 10
  test_interval: 1000
  out_dir: /content/out_smoke
  init_from: scratch
  compile: false
  experiment_name: smoke
  wandb_project: null
data:
  dataset: stargraph
  effective_batch_size: 64
  gradient_accum_steps: 1
  num_workers: 0
  stargraph_train_data_path: "data/stargraph/graph_5_5_sample_2000.txt"
  stargraph_test_data_path: "data/stargraph/graph_5_5_test_200.txt"
  stargraph_max_nodes: 100
model:
  n_layer: 12
  n_head: 6
  n_embd: 384
  dropout: 0.0
  bias: false
  block_size: 1024
  gpt_mode: next_token
  lambda_kl: 1.0
  lambda_mse: 1.0
  mtp_horizon: 3
  proj_factor: 0.5
optimizer:
  optimizer_type: adam
  learning_rate: 5e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 100
lr_scheduler:
  warmup_iters: 0
  warmdown_iters: 0
""")
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
t0 = time.time()
rc = sh("cd %s && WANDB_MODE=disabled fabric run --devices 1 --precision bf16-mixed "
        "train.py --config config/stargraph/5_5/nextlat_smoke.yaml"
        % WORK, check=False, timeout=2400)
print("\nSMOKE_TRAIN_RC=%d elapsed=%.1fs" % (rc, time.time() - t0), flush=True)
sh("find /content/out_smoke -type f 2>/dev/null | head -20", check=False)
print("=== SMOKE DONE ===", flush=True)
