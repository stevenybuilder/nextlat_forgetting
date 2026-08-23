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
# prepare.py prints a 50-character progress bar per sample with a carriage return; relayed
# line-by-line that is one log line per graph and it floods the exec stream. Import the
# generator and call it directly with showLoadingBar=False instead of shelling out.
train_txt = os.path.join(WORK, "data/stargraph/graph_5_5_sample_2000.txt")
if not os.path.exists(train_txt):
    sys.path.insert(0, WORK)
    from data.stargraph.prepare import generate_and_save_star_or_sink_graph_data as gen
    t0 = time.time()
    gen(numOfSamples=2000, numOfTestSamples=200, numOfPathsFromSource=5, lenOfEachPath=5,
        maxNodes=100, data_dir=os.path.join(WORK, "data/stargraph"),
        generate_test_data=True, showLoadingBar=False)
    print("GEN_2200_SECONDS=%.1f" % (time.time() - t0), flush=True)
else:
    print("data already present, skipping generation", flush=True)
sh("ls -la %s/data/stargraph/" % WORK, check=False)
sh("head -c 300 %s/data/stargraph/graph_5_5_sample_2000.txt" % WORK, check=False)

print("\n=== 6. 20-step NextLat run ===", flush=True)
# Derive the smoke config from the OFFICIAL yaml rather than writing one by hand. A
# reconstructed config silently drops keys the trainer requires -- the first attempt at
# this died on `Missing key test_generalization` -- which is exactly why the spec says to
# copy the official configuration and override only what is permitted.
import yaml  # noqa: E402

OFFICIAL = os.path.join(WORK, "config/stargraph/5_5/nextlat_stargraph_5_5.yaml")
cfg_path = os.path.join(WORK, "config/stargraph/5_5/nextlat_smoke.yaml")
with open(OFFICIAL) as f:
    conf = yaml.safe_load(f)
conf["trainer"].update({
    "train_batches": 20, "val_batches": 2, "test_batches": 2, "val_interval": 10,
    "test_interval": 1000, "out_dir": "/content/out_smoke", "compile": False,
    "experiment_name": "smoke", "wandb_project": "stargraph",
})
conf["data"].update({
    "effective_batch_size": 64,
    "stargraph_train_data_path": "data/stargraph/graph_5_5_sample_2000.txt",
    "stargraph_test_data_path": "data/stargraph/graph_5_5_test_200.txt",
})
conf.pop("sweep", None)  # a sweep would launch five seeds; the smoke test wants one
with open(cfg_path, "w") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
print("smoke config derived from official; overridden keys only", flush=True)
sh("cat " + cfg_path, check=False)

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
t0 = time.time()
rc = sh("cd %s && WANDB_MODE=disabled fabric run --devices 1 --precision bf16-mixed "
        "train.py --config config/stargraph/5_5/nextlat_smoke.yaml"
        % WORK, check=False, timeout=2400)
print("\nSMOKE_TRAIN_RC=%d elapsed=%.1fs" % (rc, time.time() - t0), flush=True)
sh("find /content/out_smoke -type f 2>/dev/null | head -20", check=False)
print("=== SMOKE DONE ===", flush=True)
