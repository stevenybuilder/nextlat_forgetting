#!/usr/bin/env python
"""Dual-role resumable driver for the confirmatory sweep on Colab.

Role is auto-detected by whether `/content` exists.

LOOP role (runs on the Mac)
    Package the project, push it to GCS, start a runtime, upload the credential, exec this
    same file on the runtime, and -- when the runtime drops, which it will -- start a fresh
    one and exec again. State lives in GCS, so a re-exec resumes rather than restarts. The
    loop ends when the ledger reports every job DONE.

DRIVER role (runs on the Colab GPU runtime)
    Pull the project, the pinned upstream repo and the immutable corpus from GCS, then hand
    control to scripts/run_matrix.py, which owns idempotency and checkpoint lineage. A
    background thread heartbeats and syncs checkpoints and metrics to GCS on a cadence, so a
    drop costs at most one sync interval rather than the whole run.

Three constraints from earlier failures in this project shape the design, and all three are
easy to reintroduce by accident:
  * `colab exec file.py -- args` does NOT forward argv. Every parameter arrives through the
    uploaded sidecar /content/job_spec.json, never the command line.
  * `__file__` is undefined under `colab exec`, and in DRIVER role the module executes top to
    bottom before any main() dispatch, so a top-level `__file__` reference crashes before the
    role is even chosen.
  * A child process's stdout does not reach the exec stream, and a silent long job starves the
    websocket. Child output is relayed line by line.
"""
import collections
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # colab exec: no __file__ in the kernel
    HERE = os.getcwd()

BUCKET = "nextlat-lurestar-project-flash-490419"
PREFIX = "lurestar"
GCS = "gs://%s/%s" % (BUCKET, PREFIX)
PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"
SPEC_PATH = "/content/job_spec.json"
SYNC_SECONDS = 300
FAST_EXIT_SECONDS = 120
MAX_FAST_EXITS = 2


def sh(cmd, check=True, timeout=None, cwd=None, quiet=False):
    print("+ " + cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = collections.deque(maxlen=80)
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        if not quiet:
            print("  | " + line, flush=True)
    rc = proc.wait()
    if quiet:
        for ln in tail:
            print("  | " + ln, flush=True)
    if check and rc != 0:
        raise SystemExit("FAILED (%d): %s" % (rc, cmd))
    return rc, "\n".join(tail)


# --------------------------------------------------------------------------- DRIVER

def driver():
    print("=== DRIVER role ===", flush=True)
    spec = json.load(open(SPEC_PATH)) if os.path.exists(SPEC_PATH) else {}
    print("job spec: %s" % json.dumps(spec), flush=True)

    sh("pip -q install google-cloud-storage google-auth omegaconf lightning",
       check=False, quiet=True)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "project-flash-490419"
    import google.auth
    import google.auth.transport.requests as gart

    def mint():
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(gart.Request())
        os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = creds.token

    mint()

    root = "/content/lurestar"
    proj = "/content/project"
    os.makedirs(root, exist_ok=True)
    os.makedirs(proj, exist_ok=True)

    print("=== pull project ===", flush=True)
    sh("gcloud storage cp %s/source/project.tar.gz /content/project.tar.gz" % GCS)
    with tarfile.open("/content/project.tar.gz") as tf:
        tf.extractall(proj)

    print("=== pull upstream at the pinned commit ===", flush=True)
    up = os.path.join(proj, "upstream", "NextLat")
    if not os.path.isdir(os.path.join(up, ".git")):
        sh("git clone -q %s %s" % (UPSTREAM_URL, up))
    sh("cd %s && git checkout -q %s && git rev-parse HEAD" % (up, PINNED))

    print("=== pull the immutable corpus and verify its hashes ===", flush=True)
    data_dir = os.path.join(root, "data", "stargraph")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(data_dir, "graph_5_5_sample_200000.txt")):
        sh("gcloud storage cp %s/corpus/stargraph/*.txt %s/" % (GCS, data_dir))
    sh("gcloud storage cp %s/manifests/corpus.sha256 /content/corpus.sha256" % GCS,
       check=False)
    rc, _ = sh("cd %s && sha256sum -c --ignore-missing "
               "<(awk '{print $1\"  \"$2}' /content/corpus.sha256)" % data_dir,
               check=False)
    print("CORPUS_HASH_VERIFIED=%s" % (rc == 0), flush=True)
    if rc != 0:
        raise SystemExit("corpus hash mismatch -- refusing to train on unverified data")

    print("=== pull prior run state (this is what makes a re-exec a resume) ===", flush=True)
    sh("gcloud storage rsync -r %s/runs %s/runs" % (GCS, root), check=False)
    sh("gcloud storage cp %s/run_ledger.json %s/run_ledger.json" % (GCS, root), check=False)

    stop = threading.Event()

    def sync_loop():
        """Heartbeat plus durable sync. A drop costs at most one interval of work."""
        while not stop.wait(SYNC_SECONDS):
            try:
                mint()
                subprocess.call(
                    "gcloud storage rsync -r %s/runs %s/runs" % (root, GCS),
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.call(
                    "gcloud storage cp %s/run_ledger.json %s/run_ledger.json" % (root, GCS),
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[sync] %s durable" % time.strftime("%H:%M:%S"), flush=True)
            except Exception as exc:                       # never let the sync kill the run
                print("[sync] failed: %s" % exc, flush=True)

    threading.Thread(target=sync_loop, daemon=True).start()

    args = spec.get("run_matrix_args", ["--phase", "base"])
    cmd = ("cd %s && PYTHONPATH=%s python scripts/run_matrix.py --root %s --ledger "
           "%s/run_ledger.json --upstream %s %s"
           % (proj, proj, root, root, up, " ".join(args)))
    rc, _ = sh(cmd, check=False, quiet=False)
    print("RUN_MATRIX_RC=%d" % rc, flush=True)

    stop.set()
    time.sleep(1)
    print("=== final durable sync ===", flush=True)
    mint()
    sh("gcloud storage rsync -r %s/runs %s/runs" % (root, GCS), check=False, quiet=True)
    sh("gcloud storage cp %s/run_ledger.json %s/run_ledger.json" % (root, GCS), check=False)
    ledger = os.path.join(root, "run_ledger.json")
    if os.path.exists(ledger):
        st = json.load(open(ledger))
        jobs = st.get("jobs", st)
        done = sum(1 for j in jobs.values() if str(j.get("state")) == "DONE")
        print("LEDGER_DONE=%d LEDGER_TOTAL=%d" % (done, len(jobs)), flush=True)
        print("ALL_DONE=%s" % (done == len(jobs) and len(jobs) > 0), flush=True)
    print("=== DRIVER DONE ===", flush=True)


# ----------------------------------------------------------------------------- LOOP

def package(project_root):
    out = os.path.join(project_root, ".agent_state", "project.tar.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    skip_dirs = {".venv", ".git", "__pycache__", "data", "output", ".secrets",
                 ".agent_state", "docs", "report", "upstream"}

    def flt(ti):
        parts = ti.name.split("/")
        if any(p in skip_dirs for p in parts):
            return None
        if ti.name.endswith((".pt", ".ckpt", ".tar.gz")):
            return None
        return ti

    with tarfile.open(out, "w:gz") as tf:
        for name in sorted(os.listdir(project_root)):
            if name in skip_dirs or name.startswith("."):
                continue
            tf.add(os.path.join(project_root, name), arcname=name, filter=flt)
    return out


def loop():
    project_root = os.path.dirname(HERE)
    spec_file = os.path.join(project_root, ".agent_state", "job_spec.json")
    spec = json.load(open(spec_file)) if os.path.exists(spec_file) else {
        "run_matrix_args": ["--phase", "base"], "gpu": "a100", "max_attempts": 20}
    gpu = spec.get("gpu", "a100")

    print("=== LOOP role: packaging and uploading project ===", flush=True)
    tar = package(project_root)
    sh("gcloud storage cp %s %s/source/project.tar.gz" % (tar, GCS))
    sh("gcloud storage cp %s %s/source/job_spec.json" % (spec_file, GCS), check=False)

    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    fast_exits = 0
    for attempt in range(1, int(spec.get("max_attempts", 20)) + 1):
        print("\n=== attempt %d ===" % attempt, flush=True)
        rc, out = sh("colab start --gpu %s --json" % gpu, check=False)
        try:
            sid = json.loads(out[out.index("{"):out.rindex("}") + 1])["session"]
        except Exception:
            print("could not start a runtime; backing off", flush=True)
            time.sleep(60)
            continue
        print("session=%s" % sid, flush=True)
        sh("colab upload --session %s %s /content/adc.json" % (sid, adc), check=False)
        sh("colab upload --session %s %s %s" % (sid, spec_file, SPEC_PATH), check=False)

        t0 = time.time()
        driver_path = os.path.join(HERE, "colab_train_loop.py")
        rc, out = sh("colab exec --session %s --timeout 240m %s" % (sid, driver_path),
                     check=False)
        elapsed = time.time() - t0
        sh("colab stop --session %s" % sid, check=False)

        if "ALL_DONE=True" in out:
            print("\nSWEEP COMPLETE after %d attempt(s)" % attempt, flush=True)
            return 0
        # A driver that dies instantly provisions a fresh GPU for nothing. Two in a row
        # means the failure is deterministic and another runtime will not fix it.
        if elapsed < FAST_EXIT_SECONDS:
            fast_exits += 1
            print("fast exit %d/%d (%.0fs)" % (fast_exits, MAX_FAST_EXITS, elapsed),
                  flush=True)
            if fast_exits >= MAX_FAST_EXITS:
                print("ABORTING: two consecutive fast exits, the failure is deterministic",
                      flush=True)
                return 2
        else:
            fast_exits = 0
        print("runtime ended after %.0fs; resuming on a fresh one" % elapsed, flush=True)
    print("exhausted attempts", flush=True)
    return 3


if __name__ == "__main__" or "get_ipython" in dir():
    sys.exit(driver() if os.path.isdir("/content") else loop())
