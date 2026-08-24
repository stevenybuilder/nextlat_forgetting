"""Spec section 11 profiling gate: 500 steps of GPT and NextLat at exact paper scale.

DRIVER role, runs on a Colab GPU runtime under `colab exec`. Everything is baked in --
`colab exec` forwards no argv and leaves `__file__` undefined. Child process output is
relayed line by line, because a Colab runtime does not surface it otherwise and a silent
long job starves the exec websocket.

Nothing here is scientific output. It measures seconds per step, VRAM and checkpoint cost
on the real 200,000-graph corpus at the real architecture, so the sweep can be budgeted
before it is launched.
"""
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
REPO = "https://github.com/JaydenTeoh/NextLat.git"
WORK = "/content/nextlat"
BUCKET = "nextlat-lurestar-project-flash-490419"
PREFIX = "lurestar"
PROFILE_STEPS = 500
WARMUP_STEPS = 100


STEP_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) Step: (\d+)")


def sh(cmd, check=True, timeout=3600, quiet=False, collect_steps=None):
    """Relay child output. When collect_steps is a list, every per-step timestamp is
    appended to it as the line arrives -- a bounded tail buffer silently truncates the
    warmup-discarded window and yields no steady-state estimate at all."""
    print("+ " + cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = collections.deque(maxlen=60)
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        if collect_steps is not None:
            m = STEP_RE.search(line)
            if m:
                collect_steps.append((m.group(1), int(m.group(2))))
        if not quiet:
            print("  | " + line, flush=True)
    rc = proc.wait()
    if quiet:
        for line in tail:
            print("  | " + line, flush=True)
    print("  rc=%d" % rc, flush=True)
    if check and rc != 0:
        raise SystemExit("FAILED (%d): %s" % (rc, cmd))
    return rc, "\n".join(tail)


print("=== runtime ===", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
import torch  # noqa: E402
GPU = torch.cuda.get_device_name(0)
print("GPU=%s torch=%s cuda=%s bf16=%s" % (GPU, torch.__version__, torch.version.cuda,
                                           torch.cuda.is_bf16_supported()), flush=True)

print("=== auth ===", flush=True)
sh("pip -q install google-cloud-storage google-auth", check=False, quiet=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "project-flash-490419"
import google.auth  # noqa: E402
import google.auth.transport.requests as gart  # noqa: E402
_creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
_creds.refresh(gart.Request())
os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = _creds.token
print("auth ok", flush=True)

print("=== repo ===", flush=True)
if not os.path.isdir(WORK):
    sh("git clone -q %s %s" % (REPO, WORK))
sh("cd %s && git checkout -q %s && git rev-parse HEAD" % (WORK, PINNED))
sh("pip -q install omegaconf lightning", check=False, quiet=True)

print("=== corpus (pull the immutable one, never regenerate) ===", flush=True)
os.makedirs(WORK + "/data/stargraph", exist_ok=True)
if not os.path.exists(WORK + "/data/stargraph/graph_5_5_sample_200000.txt"):
    sh("gcloud storage cp gs://%s/%s/corpus/stargraph/*.txt %s/data/stargraph/"
       % (BUCKET, PREFIX, WORK))
sh("ls -la %s/data/stargraph/" % WORK, check=False)
sh("sha256sum %s/data/stargraph/*.txt" % WORK, check=False)

import yaml  # noqa: E402

results = {"gpu": GPU, "torch": torch.__version__, "steps": PROFILE_STEPS,
           "warmup": WARMUP_STEPS, "runs": {}}

for tag, official in (("gpt", "gpt_stargraph_5_5.yaml"),
                      ("nextlat", "nextlat_stargraph_5_5.yaml")):
    print("\n=== profile %s ===" % tag, flush=True)
    src = os.path.join(WORK, "config/stargraph/5_5", official)
    with open(src) as f:
        conf = yaml.safe_load(f)
    # Override ONLY: step count, output, compile, and the paths to the immutable corpus.
    # Width, depth, optimizer, schedule, loss coefficients and effective batch size are
    # untouched -- the point of a profile is to measure the configuration that will run.
    conf["trainer"].update({
        "train_batches": PROFILE_STEPS, "val_batches": 20, "test_batches": 20,
        "val_interval": PROFILE_STEPS, "test_interval": PROFILE_STEPS * 10,
        "out_dir": "/content/out_profile_" + tag, "compile": False,
        "experiment_name": "profile_" + tag, "save_recovery_checkpoint": 250,
    })
    conf["data"].update({
        "stargraph_train_data_path": "data/stargraph/graph_5_5_sample_200000.txt",
        "stargraph_test_data_path": "data/stargraph/graph_5_5_test_20000.txt",
    })
    conf.pop("sweep", None)
    path = os.path.join(WORK, "config/stargraph/5_5", "profile_%s.yaml" % tag)
    with open(path, "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)
    print("effective_batch_size=%s n_layer=%s n_embd=%s compile=%s" % (
        conf["data"]["effective_batch_size"], conf["model"]["n_layer"],
        conf["model"]["n_embd"], conf["trainer"]["compile"]), flush=True)

    # Peak VRAM belongs to the `fabric run` CHILD process; torch.cuda stats in THIS
    # process report zero for it. Poll nvidia-smi in a background thread instead.
    peak_mib = [0]
    stop = threading.Event()

    def _poll():
        while not stop.wait(0.25):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], text=True)
                peak_mib[0] = max(peak_mib[0], int(out.strip().split("\n")[0]))
            except Exception:
                pass

    watcher = threading.Thread(target=_poll, daemon=True)
    watcher.start()
    stamps = []
    t0 = time.time()
    rc, log = sh("cd %s && WANDB_MODE=disabled fabric run --devices 1 --precision bf16-mixed "
                 "train.py --config config/stargraph/5_5/profile_%s.yaml" % (WORK, tag),
                 check=False, timeout=5400, quiet=True, collect_steps=stamps)
    wall = time.time() - t0
    stop.set()
    watcher.join(timeout=2)

    # Steady-state timing from the trainer's own per-step timestamps: discard warmup.
    per_step = None
    if len(stamps) > WARMUP_STEPS + 10:
        import datetime
        def _p(s):
            return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        pts = [(_p(a), int(b)) for a, b in stamps]
        pts = [p for p in pts if p[1] >= WARMUP_STEPS]
        if len(pts) > 10:
            span = (pts[-1][0] - pts[0][0]).total_seconds()
            nsteps = pts[-1][1] - pts[0][1]
            per_step = span / max(nsteps, 1)
    peak_alloc = peak_mib[0] / 1024.0
    peak_res = peak_alloc
    ckpt_bytes = 0
    for root, _, files in os.walk("/content/out_profile_" + tag):
        for fn in files:
            if fn.endswith(".pt"):
                ckpt_bytes = max(ckpt_bytes, os.path.getsize(os.path.join(root, fn)))
    results["runs"][tag] = {
        "rc": rc, "wall_seconds": round(wall, 1),
        "steady_seconds_per_step": round(per_step, 4) if per_step else None,
        "peak_device_memory_gb": round(peak_alloc, 2),
        "peak_measured_by": "nvidia-smi polling of the child process",
        "checkpoint_bytes": ckpt_bytes,
    }
    print("PROFILE %s: rc=%d wall=%.1fs steady_s_per_step=%s peak_device_mem=%.2fGB ckpt=%.0fMB"
          % (tag, rc, wall, per_step, peak_alloc, ckpt_bytes / 1e6), flush=True)

# Project the full spec section 11 workload from what was just measured.
g = results["runs"].get("gpt", {}).get("steady_seconds_per_step")
n = results["runs"].get("nextlat", {}).get("steady_seconds_per_step")
if g and n:
    base_h = (g + n) * 3 * 20000 / 3600.0
    adapt_h = (g + n) * 3 * 2 * 500 / 3600.0
    results["projection"] = {
        "base_gpu_hours": round(base_h, 2),
        "adapt_gpu_hours": round(adapt_h, 2),
        "nextlat_overhead_x": round(n / g, 2),
        "total_plus_20pct_gpu_hours": round((base_h + adapt_h) * 1.2, 2),
    }
    print("\nPROJECTION %s" % json.dumps(results["projection"]), flush=True)

with open("/content/profile_results.json", "w") as f:
    json.dump(results, f, indent=2)
sh("gcloud storage cp /content/profile_results.json gs://%s/%s/results/profile_results.json"
   % (BUCKET, PREFIX), check=False)
print("PROFILE_JSON=" + json.dumps(results), flush=True)
print("=== PROFILE DONE ===", flush=True)
