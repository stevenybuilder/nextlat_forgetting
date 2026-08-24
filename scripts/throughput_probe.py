"""Measure the data path on the real runtime, before changing anything.

DRIVER role only: runs on a Colab GPU runtime under `colab exec`, which forwards no argv and
leaves `__file__` undefined, so every parameter is baked in and child output is relayed line by
line (docs/RUNLOG.md records what happens otherwise).

What it measures, in order, each section guarded so one failure does not lose the rest:

  0  region, GPU, host CPU, disk        -- transfer classification and host-CPU context
  1  loader-only s/batch, four variants -- upstream as shipped, num_workers 2 and 4, and the
                                           pre-tokenized matrix.  Upstream's DataLoader passes
                                           no num_workers, so loader time is strictly additive
                                           to GPU time and this IS the host-input tax.
  2  end-to-end trainer, 300 GPT steps  -- unpatched vs pre-tokenized.  The number that counts.
  3  checkpoint save cost               -- a blocking save at real size, plus the read-back and
                                           hash the durable layer adds.
  4  GCS upload rate from this region   -- what a cross-region runtime would cost per sync.

This process never initializes a CUDA context: every GPU-touching measurement runs in a child,
so the parent is not holding VRAM while the trainer tries to fit effective batch 512 on 40 GB.

Nothing here is scientific output and no model trained here is kept.
"""
import collections
import json
import os
import re
import subprocess
import sys
import time
import traceback

PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
REPO = "https://github.com/JaydenTeoh/NextLat.git"
WORK = "/content/nextlat"
BUCKET = "nextlat-lurestar-project-flash-490419"
PREFIX = "lurestar"
GCS = "gs://%s/%s" % (BUCKET, PREFIX)

WARMUP_BATCHES = 20
TIMED_BATCHES = 200
TRAIN_STEPS = 300

RESULTS = {}
STEP_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) Step: (\d+)")


def sh(cmd, check=True, quiet=False, collect_steps=None):
    print("+ " + cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = collections.deque(maxlen=40)
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
        for ln in tail:
            print("  | " + ln, flush=True)
    print("  rc=%d" % rc, flush=True)
    if check and rc != 0:
        raise SystemExit("FAILED (%d): %s" % (rc, cmd))
    return rc, "\n".join(tail)


def guarded(name, fn):
    print("\n=== %s ===" % name, flush=True)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        RESULTS.setdefault("errors", {})[name] = traceback.format_exc(limit=4)


# ------------------------------------------------------------------ 0  environment

print("=== environment ===", flush=True)
try:
    import urllib.request
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"})
    zone = urllib.request.urlopen(req, timeout=5).read().decode().split("/")[-1]
except Exception as exc:
    zone = "unknown (%r)" % exc
region = "-".join(zone.split("-")[:2]) if zone.count("-") >= 2 else zone
_, gpu_line = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
cpu_model = ""
try:
    for ln in open("/proc/cpuinfo"):
        if ln.startswith("model name"):
            cpu_model = ln.split(":", 1)[1].strip()
            break
except Exception:
    pass
sh("df -h /content | tail -1", check=False)
RESULTS["env"] = {
    "zone": zone, "region": region, "bucket_region": "us-central1",
    "same_region": region == "us-central1",
    "gpu": gpu_line.strip().splitlines()[0] if gpu_line.strip() else "unknown",
    "cpu": cpu_model, "cores": os.cpu_count(),
}
print("ZONE=%s REGION=%s SAME_REGION=%s CPU=%s cores=%s"
      % (zone, region, RESULTS["env"]["same_region"], cpu_model, os.cpu_count()), flush=True)

print("\n=== auth ===", flush=True)
sh("pip -q install google-cloud-storage google-auth omegaconf lightning",
   check=False, quiet=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "project-flash-490419"
import google.auth  # noqa: E402
import google.auth.transport.requests as gart  # noqa: E402
_c, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
_c.refresh(gart.Request())
os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = _c.token
print("auth ok", flush=True)

print("\n=== repo + corpus ===", flush=True)
if not os.path.isdir(WORK):
    sh("git clone -q %s %s" % (REPO, WORK))
sh("cd %s && git checkout -q %s && git rev-parse HEAD" % (WORK, PINNED))
os.makedirs(WORK + "/data/stargraph", exist_ok=True)
if not os.path.exists(WORK + "/data/stargraph/graph_5_5_sample_200000.txt"):
    sh("gcloud storage cp %s/corpus/stargraph/*.txt %s/data/stargraph/" % (GCS, WORK))
sh("sha256sum %s/data/stargraph/*.txt" % WORK, check=False)


# ------------------------------------------------------------------ shared child preamble

PREAMBLE = '''
import json, os, sys, time
sys.path.insert(0, "/content")
sys.path.insert(0, "%(work)s")
os.chdir("%(work)s")
from omegaconf import OmegaConf

def build_config(official="config/stargraph/5_5/gpt_stargraph_5_5.yaml"):
    cfg = OmegaConf.merge(OmegaConf.load("defaults.yaml"), OmegaConf.load(official))
    cfg.pop("sweep", None)
    cfg.data.stargraph_train_data_path = "data/stargraph/graph_5_5_sample_200000.txt"
    cfg.data.stargraph_test_data_path = "data/stargraph/graph_5_5_test_20000.txt"
    return cfg
''' % {"work": WORK}


LOADER_BENCH = PREAMBLE + '''
import torch, torch.utils.data as tud, lightning as L
import fast_stargraph
import data.stargraph as sg

WARMUP, TIMED = %(warmup)d, %(timed)d
fabric = L.Fabric(devices=1, accelerator="cuda", precision="bf16-mixed")
fabric.launch()
original = sg.StarGraphDataModule.train_dataloader
out = {}

def run(label, num_workers, pretokenized):
    fast_stargraph.uninstall()
    if pretokenized:
        os.environ["LURESTAR_FAST_LOADER"] = "1"
        fast_stargraph.install(cache_dir="/content/tokcache", verify=True)

    def patched(self):
        dl = tud.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=True, collate_fn=self.collate_fn,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
        return self.fabric.setup_dataloaders(dl, use_distributed_sampler=True)

    sg.StarGraphDataModule.train_dataloader = patched
    try:
        t0 = time.perf_counter()
        cfg = build_config()
        cfg.data.device_batch_size = cfg.data.effective_batch_size   # train.py:143, world_size 1
        dm = sg.StarGraphDataModule(fabric, cfg)
        t_init = time.perf_counter() - t0
        t0 = time.perf_counter()
        it = iter(dm.train_dataloader())
        first = next(it)
        t_first = time.perf_counter() - t0
        for _ in range(WARMUP):
            next(it)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            b = next(it)
        torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / TIMED
    finally:
        sg.StarGraphDataModule.train_dataloader = original
        fast_stargraph.uninstall()
    out[label] = {"seconds_per_batch": round(per, 5), "ms_per_batch": round(per*1e3, 2),
                  "datamodule_init_s": round(t_init, 2), "first_batch_s": round(t_first, 2),
                  "batch_shape": list(first.shape), "dtype": str(first.dtype),
                  "num_workers": num_workers, "pretokenized": pretokenized}
    print("LOADER %%-14s %%7.2f ms/batch (init %%.1fs, first %%.1fs, shape %%s)"
          %% (label, per*1e3, t_init, t_first, list(first.shape)), flush=True)

run("upstream_w0", 0, False)
run("workers2", 2, False)
run("workers4", 4, False)
run("pretokenized", 0, True)
json.dump(out, open("/content/loader_bench.json", "w"), indent=2)
print("LOADER_DONE", flush=True)
''' % {"warmup": WARMUP_BATCHES, "timed": TIMED_BATCHES}


TRAIN_SHIM = '''
import sys, os
sys.path.insert(0, "/content")
sys.path.insert(0, os.getcwd())
import train as upstream_train
from omegaconf import OmegaConf
from argparse import ArgumentParser
if os.environ.get("LURESTAR_FAST_LOADER", "1") != "0":
    import fast_stargraph
    fast_stargraph.install(cache_dir="/content/tokcache", verify=True)
p = ArgumentParser()
p.add_argument("-c", "--config", required=True)
p.add_argument("--no_pbar", action="store_true")
p.add_argument("--checkpoint_path", required=False)
a, rest = p.parse_known_args()
cfg = OmegaConf.merge(OmegaConf.load("defaults.yaml"), OmegaConf.load(a.config),
                      OmegaConf.from_dotlist(rest))
upstream_train.do_train(cfg, hide_progress_bar=True, checkpoint_path=a.checkpoint_path)
'''


CKPT_BENCH = '''
import hashlib, json, os, time, torch
N = 21
state = {k: {"w%d" % i: torch.randn(1_000_000) for i in range(N)}
         for k in ("model", "opt_m", "opt_v")}
state["training_steps"] = 250
p = "/content/probe_ckpt.pt"
t0 = time.perf_counter(); torch.save(state, p); t_save = time.perf_counter() - t0
os.sync()
t0 = time.perf_counter()
h = hashlib.sha256()
with open(p, "rb") as fh:
    for c in iter(lambda: fh.read(1 << 20), b""):
        h.update(c)
t_hash = time.perf_counter() - t0
t0 = time.perf_counter(); torch.load(p, map_location="cpu", weights_only=False)
t_load = time.perf_counter() - t0
size = os.path.getsize(p)
rec = {"bytes": size, "mb": round(size/1e6, 1), "torch_save_s": round(t_save, 3),
       "sha256_s": round(t_hash, 3), "torch_load_s": round(t_load, 3),
       "durable_verify_overhead_s": round(t_hash + t_load, 3)}
json.dump(rec, open("/content/ckpt_bench.json", "w"), indent=2)
print("CKPT %.0f MB save=%.2fs sha256=%.2fs load_back=%.2fs"
      % (size/1e6, t_save, t_hash, t_load), flush=True)
'''


# ------------------------------------------------------------------ 1  loader only

def _loader():
    with open("/content/loader_bench.py", "w") as f:
        f.write(LOADER_BENCH)
    sh("cd %s && python /content/loader_bench.py" % WORK, check=False)
    RESULTS["loader"] = json.load(open("/content/loader_bench.json"))


# ------------------------------------------------------------------ 2  end-to-end

def run_trainer(label, fast, recovery_every):
    import yaml
    from omegaconf import OmegaConf as OC
    cfg = OC.merge(OC.load(WORK + "/defaults.yaml"),
                   OC.load(WORK + "/config/stargraph/5_5/gpt_stargraph_5_5.yaml"))
    cfg.pop("sweep", None)
    d = OC.to_container(cfg, resolve=True)
    d["data"].update({
        "stargraph_train_data_path": "data/stargraph/graph_5_5_sample_200000.txt",
        "stargraph_test_data_path": "data/stargraph/graph_5_5_test_20000.txt",
    })
    d["trainer"].update({
        "train_batches": TRAIN_STEPS, "val_batches": 10, "test_batches": 10,
        "val_interval": TRAIN_STEPS * 10, "test_interval": TRAIN_STEPS * 10,
        "out_dir": "/content/out_" + label, "compile": False,
        "experiment_name": "probe_" + label, "log_to_wandb": False,
        "save_recovery_checkpoint": recovery_every, "log_interval": 1,
    })
    path = "/content/probe_%s.yaml" % label
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)

    stamps = []
    t0 = time.time()
    rc, _ = sh("cd %s && LURESTAR_FAST_LOADER=%d WANDB_MODE=disabled fabric run --devices 1 "
               "--precision bf16-mixed /content/train_shim.py --config %s"
               % (WORK, 1 if fast else 0, path), check=False, quiet=True, collect_steps=stamps)
    wall = time.time() - t0

    per_step = None
    if len(stamps) > 60:
        import datetime
        pts = [(datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S.%f"), b) for a, b in stamps]
        pts = [p for p in pts if p[1] >= 50]
        if len(pts) > 10:
            per_step = (pts[-1][0] - pts[0][0]).total_seconds() / max(pts[-1][1] - pts[0][1], 1)
    RESULTS.setdefault("trainer", {})[label] = {
        "rc": rc, "wall_seconds": round(wall, 1),
        "steady_seconds_per_step": round(per_step, 4) if per_step else None,
        "fast_loader": fast, "save_recovery_checkpoint": recovery_every,
        "steps_seen": len(stamps),
    }
    print("TRAINER %-16s rc=%d wall=%.1fs steady=%s s/step (fast=%s recovery=%s)"
          % (label, rc, wall, per_step, fast, recovery_every), flush=True)


def _trainer():
    with open("/content/train_shim.py", "w") as f:
        f.write(TRAIN_SHIM)
    run_trainer("baseline", False, -1)
    run_trainer("pretokenized", True, -1)
    run_trainer("pretok_ckpt100", True, 100)


# ------------------------------------------------------------------ 3  checkpoint

def _ckpt():
    with open("/content/ckpt_bench.py", "w") as f:
        f.write(CKPT_BENCH)
    sh("python /content/ckpt_bench.py", check=False)
    RESULTS["checkpoint"] = json.load(open("/content/ckpt_bench.json"))


# ------------------------------------------------------------------ 4  transfer

def _gcs():
    p = "/content/probe_ckpt.pt"
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(os.urandom(256 * 1024 * 1024))
    size = os.path.getsize(p)
    remote = "%s/_throughput_probe/probe_ckpt.pt" % GCS
    t0 = time.perf_counter()
    sh("gcloud storage cp %s %s" % (p, remote), check=False, quiet=True)
    t_up = time.perf_counter() - t0
    t0 = time.perf_counter()
    sh("gcloud storage cp %s /content/probe_ckpt_back.pt" % remote, check=False, quiet=True)
    t_down = time.perf_counter() - t0
    sh("gcloud storage rm %s" % remote, check=False, quiet=True)
    RESULTS["transfer"] = {
        "bytes": size, "upload_s": round(t_up, 2), "download_s": round(t_down, 2),
        "upload_mb_per_s": round(size / 1e6 / max(t_up, 1e-9), 1),
        "download_mb_per_s": round(size / 1e6 / max(t_down, 1e-9), 1),
        "region": RESULTS["env"]["region"],
    }
    print("GCS up=%.1fs (%.0f MB/s) down=%.1fs (%.0f MB/s) region=%s"
          % (t_up, size / 1e6 / t_up, t_down, size / 1e6 / t_down,
             RESULTS["env"]["region"]), flush=True)
    for junk in ("/content/probe_ckpt_back.pt", "/content/probe_ckpt.pt"):
        try:
            os.remove(junk)
        except OSError:
            pass


guarded("1  loader-only cost, four variants", _loader)
guarded("2  end-to-end trainer, 300 GPT steps, before vs after", _trainer)
guarded("3  checkpoint write, read-back and hash at real size", _ckpt)
guarded("4  GCS transfer rate from this region", _gcs)

with open("/content/throughput_probe.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
sh("gcloud storage cp /content/throughput_probe.json %s/results/throughput_probe.json" % GCS,
   check=False)
print("\nPROBE_JSON=" + json.dumps(RESULTS), flush=True)
print("=== PROBE DONE ===", flush=True)
