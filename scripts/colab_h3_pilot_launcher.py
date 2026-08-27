#!/usr/bin/env python3
"""Colab-cell launcher for the frozen, durable nonconfirmatory H3 pilot.

All parameters arrive through ``/content/h3-launch.json`` because ``colab exec`` does not
forward argv.  The launcher downloads a content-addressed source archive with the uploaded ADC,
installs it into a fresh directory, checks out the pinned upstream commit, and relays the durable
pilot driver's output with heartbeats.  Scientific scoring remains in the hash-frozen scorer.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


SPEC_PATH = "/content/h3-launch.json"
ADC_PATH = "/content/adc.json"
JOB_UPLOAD = "/content/h3-score-job.json"
PROJECT_ROOT = pathlib.Path("/content/h3-project")
ARCHIVE = pathlib.Path("/content/h3-project.tar.gz")
BUCKET = "nextlat-lurestar-project-flash-490419"
PROJECT = "project-flash-490419"
PINNED = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_URL = "https://github.com/JaydenTeoh/NextLat.git"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = pathlib.Path(destination)
    parent = destination.parent
    temporary = pathlib.Path(tempfile.mkdtemp(prefix="h3-source-", dir=parent))
    try:
        root = temporary.resolve()
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for member in members:
                target = (temporary / member.name).resolve()
                if (member.issym() or member.islnk() or
                        not (member.isfile() or member.isdir()) or
                        (target != root and root not in target.parents)):
                    raise RuntimeError("unsafe source archive member: %s" % member.name)
            tf.extractall(temporary, members=members)
        if destination.exists():
            raise RuntimeError("refusing to overlay an existing H3 project root")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def relay(command, cwd):
    stop = threading.Event()
    started = time.time()

    def heartbeat():
        while not stop.wait(30):
            print("[h3-heartbeat] elapsed_s=%d" % (time.time() - started), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    finally:
        stop.set()
        thread.join(timeout=2)


def main():
    spec = json.load(open(SPEC_PATH, encoding="utf-8"))
    if spec.get("schema") != "nextlat_forgetting/h3_pilot_launch/1":
        raise SystemExit("invalid H3 launch sidecar")
    source_sha = str(spec.get("source_sha256", ""))
    source_object = str(spec.get("source_object", ""))
    source_generation = str(spec.get("source_generation", ""))
    job_sha = str(spec.get("job_sha256", ""))
    launcher_sha = str(spec.get("launcher_sha256", ""))
    if (len(source_sha) != 64 or len(job_sha) != 64 or len(launcher_sha) != 64 or
            not source_generation.isdigit() or
            source_object != "lurestar/h3-pilot-source/project-%s.tar.gz" % source_sha):
        raise SystemExit("launch sidecar identity is incomplete")
    # Bind the executed cell through its explicit uploaded path.
    if sha256_file("/content/h3-pilot-launcher.py") != launcher_sha:
        raise SystemExit("uploaded launcher SHA-256 mismatch")
    if sha256_file(JOB_UPLOAD) != job_sha:
        raise SystemExit("uploaded H3 score job SHA-256 mismatch")
    if not os.path.isfile(ADC_PATH):
        raise SystemExit("uploaded ADC is absent")
    os.chmod(ADC_PATH, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ADC_PATH
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT

    try:
        from google.cloud import storage
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "google-cloud-storage>=2.16,<4"],
            check=True,
        )
        from google.cloud import storage
    bucket = storage.Client(project=PROJECT).bucket(BUCKET)
    blob = bucket.blob(source_object, generation=int(source_generation))
    blob.download_to_filename(str(ARCHIVE), if_generation_match=int(source_generation))
    blob.reload()
    if (str(blob.generation) != source_generation or
            (blob.metadata or {}).get("sha256") != source_sha):
        raise SystemExit("source object generation/metadata mismatch")
    if sha256_file(ARCHIVE) != source_sha:
        raise SystemExit("downloaded source archive SHA-256 mismatch")
    safe_extract(ARCHIVE, PROJECT_ROOT)

    upstream = PROJECT_ROOT / "upstream" / "NextLat"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", UPSTREAM_URL, str(upstream)], check=True)
    subprocess.run(["git", "checkout", "-q", PINNED], cwd=upstream, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != PINNED:
        raise SystemExit("upstream checkout identity mismatch")

    job_dir = PROJECT_ROOT / ".agent_state" / "pilot"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "h3-score-job.json"
    shutil.copyfile(JOB_UPLOAD, job_path)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_h3_pilot_durable.py"),
        "--mode", "run", "--job", str(job_path), "--adc", ADC_PATH, "--bootstrap",
        "--chunk-size", "1000", "--batch-size", "64",
    ]
    print("H3_PILOT_SOURCE_SHA256=%s JOB_SHA256=%s" % (source_sha, job_sha), flush=True)
    rc = relay(command, str(PROJECT_ROOT))
    print("H3_PILOT_DRIVER_RC=%d" % rc, flush=True)
    raise SystemExit(rc)


main()
