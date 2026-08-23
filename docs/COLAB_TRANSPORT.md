# Colab GPU transport, auth, and compute-unit budget

Recon date: 2026-08-23. Host: darwin 24.6.0, no local GPU, no local torch.
Upstream pinned at `3770be6009cea2b3c455a9ce7f2ca88b504bb955` ("Initial public release", 2026-05-25)
in `/Users/stevenyang/Documents/nextlat_forgetting/upstream/NextLat` (read-only).
Spec authority: `/Users/stevenyang/Documents/nextlat_forgetting/nextlat_v4_predictive_geometry_spec.md`.

Every command output quoted below was actually run on this host. Claims about the CLI's
internals come from `strings(1)` over the shipped Mach-O binary; claims about upstream code
carry `file:line` citations into the pinned checkout.

---

## 0. Incident to record first

`colab start --help` is **not** parsed as a help request. Running it **provisioned a real T4
runtime**:

```
$ colab start --help
Requesting T4 GPU runtime...
Runtime started: T4 (gpu-t4-s-kkb-usw4a0-3a5zwq759t14l)
Session: gpu-t4-s-kkb-usw4a0-3a5zwq759t14l
```

It was released ~40 s later:

```
$ colab stop
Looking for active runtime...
Releasing T4 (gpu-t4-s-kkb-usw4a0-3a5zwq759t14l)...
Released 1 runtime(s).
```

Paid balance was byte-identical before and after (`1788.7765366264678`), so nothing was
billed, but the lesson stands: **only `colab -h` / `colab --help` are safe. Never append
`--help` to `start` or `exec`.** Any subcommand flag probing must be done knowing it may
provision.

---

## 1. Verified environment snapshot

### `colab quota --json`

```json
{
  "tier": "Pro+",
  "paid_balance": 1788.7765366264678,
  "burn_rate_hourly": 0,
  "active_runtimes": 0,
  "free_remaining": 0,
  "free_refill_time": "0001-01-01T00:00:00Z",
  "eligible_gpus": [
    "H100",
    "G4",
    "A100",
    "L4",
    "T4"
  ],
  "eligible_tpus": [
    "V6E1",
    "V5E1"
  ]
}
```

### `colab status --json`

```json
{
  "message": "No active runtime",
  "status": "no_runtime"
}
```

### Other verified facts

| Item | Value | How verified |
|---|---|---|
| `colab` binary | `/Users/stevenyang/.local/bin/colab`, **v0.2.0**, Mach-O x86_64 (Go) | `colab -v`, `file` |
| Free tier balance | `free_remaining: 0` — **there is no free fallback**; every runtime-second is paid CU | quota JSON |
| GCS bucket | `gs://nextlat-lurestar-project-flash-490419` exists, **US-CENTRAL1**, currently empty | `gcloud storage buckets describe` → `nextlat-lurestar-project-flash-490419  US-CENTRAL1` |
| Local gcloud | Google Cloud SDK **562.0.0**, active account `<redacted-account>` | `gcloud version`, `gcloud auth list` |
| Local ADC | `/Users/stevenyang/.config/gcloud/application_default_credentials.json`, 397 bytes, mode `0600`, `type=authorized_user`, `quota_project_id=project-flash-490419` | direct JSON parse |

**Bucket region matters for the budget.** The bucket is `US-CENTRAL1`. Colab runtimes are
usually in `us-west*` / `us-central*`. If a runtime lands outside us-central1, every 256 MB
checkpoint upload crosses regions. Record the runtime's zone on first `start` and, if it is
persistently non-central, consider a multi-region or `US` bucket for `runs/` while keeping
`results/` where it is.

---

## 2. The `colab` CLI surface at v0.2.0 (complete)

From `colab --help` (the only safe help invocation):

```
colab auth                              # OAuth2 browser flow; token cached in ~/.config/colab-cli/
colab start   [--gpu t4|l4|a100]        # -> prints session id, e.g. gpu-t4-s-kkb-usw4a0-3a5zwq759t14l
colab exec    <file.py|file.ipynb> [--session <id>] [--gpu ...] [--timeout 30m]
colab exec    -c "code"                 # inline snippet
colab upload  <local> [remote] [--session <id>]
colab download <remote> [local] [--session <id>]
colab quota   [--json]
colab status  [--json]
colab stop                              # releases the active runtime; takes no --session in practice
```

Global flags: `--json`, `--gpu`, `--timeout` (default `30m`), `--session`, `-h`, `-v`.

### The H100 gap — decisive for GPU selection

`eligible_gpus` reports `["H100","G4","A100","L4","T4"]`, but the CLI's own help documents
only:

```
  --gpu t4|l4|a100      GPU type (default: t4)
```

`strings` over the binary finds `t4`, `l4`, `a100` only in that help string and in the two
usage examples; there is no other accelerator vocabulary in the binary. **Treat H100 and G4
as unreachable through `colab` v0.2.0 until proven otherwise.** Because probing costs a real
runtime (§0), the probe is a deliberate, budgeted step:

```bash
colab exec --gpu a100 -c "import torch;print(torch.cuda.get_device_name(0))"   # baseline, expect A100
# Only if H100 is worth chasing, and accepting it may provision:
colab exec --gpu h100 -c "import torch;print(torch.cuda.get_device_name(0))"
```

If the flag is rejected client-side the cost is zero; if it is accepted you have both the
device name and a live `burn_rate_hourly` reading for H100 in one shot. **Do not plan the
weekend around H100.**

### What `exec`, `upload`, and `download` actually do

The binary embeds a Jupyter kernel client (`/home/runner/work/colab-cli/colab-cli/kernel.go`)
speaking `execute_request` over `github.com/coder/websocket v1.8.14`, against a notebook
created as:

```json
{"kernel":{"name":"python3"},"name":"colab","path":"colab.ipynb","type":"notebook"}
```

`upload` is **not** a file transfer. It is generated Python executed in that kernel:

```python
import base64, os
os.makedirs(os.path.dirname(%q) or '.', exist_ok=True)
with open(%q, 'wb') as f:            # and an 'ab' chunked-append variant
    f.write(base64.b64decode(%q))
print('ok')
```

`download` is the mirror image:

```python
with open(%q, 'rb') as f:
    f.seek(%d)
    print(base64.b64encode(f.read(%d)).decode())
```

**Consequences, and they are load-bearing:**

1. Every uploaded and downloaded byte becomes base64 text inside websocket kernel messages —
   ~1.37× inflation plus JSON framing. `colab upload/download` is for **KB–single-MB
   sidecars only**: configs, manifests, `adc.json`, job JSON, small metrics.
2. A ~256 MB NextLat checkpoint (§7) must **never** move through `colab download`. All bulk
   traffic goes runtime ↔ GCS directly, over the runtime's own network.
3. `print('ok')` / the base64 payload are ordinary cell output, so a large transfer competes
   with your training log on the same websocket — another reason to keep it small.

---

## 3. The four ground-truth gotchas, and the countermeasure for each

These are treated as established facts from prior projects, and the transport mechanics in
§2 explain *why* each is true.

### (i) `colab exec script.py -- args` does not forward argv

The script is executed as a cell in a Jupyter kernel. That kernel's process was started by
Colab as `python -m ipykernel_launcher -f /root/.local/share/jupyter/runtime/kernel-*.json`,
so `sys.argv` inside your script is the *kernel's* argv, not yours.

This collides head-on with upstream: `train.py:259` declares

```python
parser.add_argument("-c", "--config", required=True, help="Path to config file")
```

so `train.py` **cannot** be the thing `colab exec` runs. It also takes `--no_pbar`,
`--shard`, `--checkpoint_path` (`train.py:260-264`) and calls `parse_known_args()`
(`train.py:265`), forwarding the remainder to OmegaConf as dotted overrides.

**Countermeasure — two layers, both required:**

* **Sidecar, not argv.** Upload a small `job.json` *before* exec; the driver reads it from a
  hard-coded absolute path. Nothing is passed on a command line to the driver.
* **Subprocess, not import.** The driver builds argv itself and launches upstream through
  `fabric run` (`train.py:9` documents this launcher), which restores a normal argv world for
  `train.py`. Do not `import train` and hand-poke `sys.argv`; upstream's sweep handling
  (`train.py:266+`) and Fabric's process setup both assume a real launch.

### (ii) `__file__` is undefined under `colab exec`

A cell has no `__file__`. Any `os.path.dirname(__file__)` raises `NameError` on line 1 and
the run dies before it costs anything — which is the *good* case; the bad case is a helper
imported later, twenty minutes in.

**Countermeasure:** a single constant at the top of every remote script, and a hard rule that
no remote code path may reference `__file__`:

```python
RUNTIME_ROOT = "/content/lurestar"     # never derived from __file__
SELF = f"{RUNTIME_ROOT}/driver.py"     # if a path to self is genuinely needed
```

Grep gate before every upload: `grep -n '__file__' <script>` must return nothing.

### (iii) Silent/captured stdout starves the exec websocket and drops it

`colab exec` holds a websocket for the cell's whole lifetime. A long stretch with no IOPub
stream message lets an intermediary time the connection out; the CLI surfaces this as
`failed to read frame header: EOF` — and it typically hits during a *trailing* CPU step
(final eval, checkpoint write, plot), i.e. right when the results exist and are not yet
durable.

**Countermeasure — three parts:**

* **Never capture the child.** Stream `fabric run` line by line with
  `stdout=PIPE, stderr=STDOUT, bufsize=1, text=True` and `print(line, end="", flush=True)`.
  Never `subprocess.run(..., capture_output=True)`.
* **Heartbeat regardless.** A daemon thread prints a one-line heartbeat every 30 s even when
  the child is silent, so a slow validation pass or a long checkpoint write still produces
  traffic.
* **Durability is independent of the websocket.** Every artifact is on GCS before the cell
  ends. A dropped websocket must degrade to "lost log tail", never "lost work". Upstream
  writes `save_recovery_checkpoint` every N steps (`core_train.py:575-578`), which is the hook
  the sync thread rides on.

### (iv) A crash-looping resume driver burns fresh runtimes

Each `colab exec --gpu ...` one-shot assigns, runs, and releases. If the driver dies at
import time — a missing pip, a bad path, `__file__` — the orchestrator sees exec return
"cleanly" in 40 s and dutifully starts another runtime. Repeat 30× and you have spent real CU
producing nothing.

**Countermeasure — a hard local circuit breaker, outside the runtime:**

Progress is defined **only** as the durable step counter in GCS
(`gs://.../runs/{run_id}/state.json`), never as exec's exit status.

```
if exec returned in < 120 s AND state.json step did not advance:
        consecutive_fast_failures += 1
else:
        consecutive_fast_failures = 0

if consecutive_fast_failures >= 2:
        ABORT. Do not start another runtime. Surface the last 200 log lines.
```

Two strikes, not three. The first fast return is plausibly a preemption; the second with no
durable progress is a bug in your code, and no amount of fresh hardware will fix it.

---

## 4. Exact command sequence for a long-running resumable training session

Local orchestrator: `scripts/run_matrix.py` (spec §9). Runtime driver: `scripts/colab_driver.py`.
`RUN_ID` is the deterministic job id from spec §9, e.g. `nextlat-s1234-base`.

### 4.0 One-time, per machine

```bash
colab auth                          # browser OAuth; token cached in ~/.config/colab-cli/
gcloud auth application-default login   # only if the ADC file is missing or its token is revoked
```

### 4.1 Pre-flight (free, no runtime)

```bash
colab quota  --json         # confirm paid_balance and that active_runtimes == 0
colab status --json         # must be {"status":"no_runtime"} before you start
gcloud storage ls gs://nextlat-lurestar-project-flash-490419/lurestar/runs/
```

Build the payload locally. **Nothing large is uploaded through the CLI**; the source snapshot
goes to GCS from the host and is pulled by the runtime.

```bash
GCS=gs://nextlat-lurestar-project-flash-490419/lurestar
RUN_ID=nextlat-s1234-base

# source snapshot: pinned tree + uncommitted diff (spec s9 "persist before training")
git -C upstream/NextLat bundle create /tmp/nextlat.bundle --all
tar czf /tmp/src_${RUN_ID}.tgz -C . scripts/ configs/ manifests/ /tmp/nextlat.bundle
gcloud storage cp /tmp/src_${RUN_ID}.tgz ${GCS}/source_snapshot/

# immutable data + manifests, hashed (spec s9)
gcloud storage cp -r data/stargraph/     ${GCS}/manifests/stargraph/
gcloud storage cp    manifests/*.jsonl   ${GCS}/manifests/
```

### 4.2 Start ONE persistent session and keep it

Persistent-session mode, not one-shot: a one-shot `colab exec --gpu a100 driver.py` re-provisions
and re-installs dependencies on **every** attempt, and pip + torch import is 3–6 minutes of
paid CU per attempt. With a held session you pay that once.

```bash
SESSION=$(colab start --gpu a100 | sed -n 's/^Session: //p')
echo "$SESSION" > .colab_session          # the orchestrator's only handle
colab status --json                        # record GPU name, RAM, zone into run_ledger.json
```

### 4.3 Upload the sidecars (small only)

```bash
# credentials — see §5 for the security posture
colab upload --session "$SESSION" ~/.config/gcloud/application_default_credentials.json /content/adc.json

# parameters — this is the argv substitute for gotcha (i)
cat > /tmp/job.json <<JSON
{
  "run_id":      "${RUN_ID}",
  "gcs_root":    "${GCS}",
  "config":      "config/stargraph/5_5/nextlat_stargraph_5_5.yaml",
  "out_dir":     "/content/lurestar/runs/${RUN_ID}",
  "seed":        1234,
  "overrides":   ["trainer.compile=false",
                  "trainer.save_recovery_checkpoint=250",
                  "trainer.init_from=resume",
                  "data.stargraph_max_nodes=100"],
  "precision":   "bf16-mixed",
  "sync_every_s": 120,
  "heartbeat_s":  30
}
JSON
colab upload --session "$SESSION" /tmp/job.json          /content/job.json
colab upload --session "$SESSION" scripts/colab_driver.py /content/driver.py
```

### 4.4 One-time runtime bootstrap (separate exec, so failures are cheap and legible)

Keep bootstrap out of the training cell: a pip failure inside the training exec looks
identical to a training crash to the circuit breaker.

```bash
colab exec --session "$SESSION" --timeout 20m scripts/colab_bootstrap.py
```

`colab_bootstrap.py` must:

* print `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_device_name(0)`, total VRAM;
* install upstream deps **without clobbering Colab's torch** — upstream pins `torch>=2.6.0`
  (`requirements.txt:1`), which pip will happily resolve into a CUDA-mismatched wheel:
  `pip install -r <(grep -v '^torch' requirements.txt)`;
* pull and unpack `${GCS}/source_snapshot/src_${RUN_ID}.tgz` into `/content/lurestar`;
* verify the ADC (§5) with a real GCS round-trip and exit non-zero if it fails;
* print the resolved `sys.executable` and `fabric --version`.

### 4.5 Launch training, streaming (long timeout)

```bash
# macOS has no timeout(1) — do NOT wrap colab. Detach and poll the log file instead.
mkdir -p logs
nohup colab exec --session "$SESSION" --timeout 12h /content/driver.py \
      > logs/${RUN_ID}.$(date +%s).log 2>&1 &
echo $! > .colab_exec.pid
```

Poll from the orchestrator loop (never `sleep` in the foreground):

```bash
tail -f logs/${RUN_ID}.*.log
gcloud storage cat ${GCS}/runs/${RUN_ID}/state.json     # authoritative progress
```

### 4.6 Drop detection and recovery

A drop shows up as one of: the log tail ending in `failed to read frame header: EOF`; the
backgrounded `colab exec` exiting; or `colab status --json` returning `no_runtime`. All three
are handled the same way, because **the exec exit code is never trusted** — `state.json` is.

```bash
recover() {
  colab status --json | grep -q '"status":"no_runtime"' && {
      SESSION=$(colab start --gpu a100 | sed -n 's/^Session: //p')
      echo "$SESSION" > .colab_session
      colab upload --session "$SESSION" ~/.config/gcloud/application_default_credentials.json /content/adc.json
      colab upload --session "$SESSION" /tmp/job.json           /content/job.json
      colab upload --session "$SESSION" scripts/colab_driver.py /content/driver.py
      colab exec   --session "$SESSION" --timeout 20m scripts/colab_bootstrap.py
  }
  nohup colab exec --session "$SESSION" --timeout 12h /content/driver.py \
        >> logs/${RUN_ID}.$(date +%s).log 2>&1 &
}
```

The driver is idempotent: it restores the newest verified checkpoint from GCS to the **exact
same absolute path** it had before (mandatory — see §6), then runs with
`trainer.init_from=resume`.

### 4.7 Teardown — always

```bash
colab stop
colab quota --json      # record paid_balance delta into run_ledger.json
```

`colab stop` is the only thing standing between you and an idle A100 burning CU overnight.
Put it in a shell `trap`:

```bash
trap 'colab stop >/dev/null 2>&1' EXIT INT TERM
```

### 4.8 The driver's own shape (satisfies i–iv)

```python
# /content/driver.py  — executed as a Jupyter cell. No argv. No __file__.
import json, os, subprocess, sys, threading, time, hashlib

RUNTIME_ROOT = "/content/lurestar"                 # (ii) never derived from __file__
JOB = json.load(open("/content/job.json"))         # (i) parameters via sidecar
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"   # (§5)

def heartbeat():                                   # (iii) keep the websocket fed
    t0 = time.time()
    while True:
        time.sleep(JOB["heartbeat_s"])
        print(f"[hb] t={time.time()-t0:8.0f}s alive", flush=True)
threading.Thread(target=heartbeat, daemon=True).start()

restore_from_gcs(JOB)          # exact-path restore of out_dir + pointer files (§6)

cmd = ["fabric", "run", "--devices", "1", "--precision", JOB["precision"],
       "train.py", "--config", JOB["config"],
       f"trainer.out_dir={JOB['out_dir']}", f"seed={JOB['seed']}", *JOB["overrides"]]
print("[cmd]", " ".join(cmd), flush=True)

p = subprocess.Popen(cmd, cwd=RUNTIME_ROOT, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, bufsize=1, text=True)   # (iii) never capture
threading.Thread(target=sync_checkpoints_to_gcs, args=(JOB,), daemon=True).start()
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()

final_sync(JOB)                # results durable BEFORE the cell can drop
print(f"[done] rc={rc}", flush=True)
sys.exit(rc)
```

---

## 5. GCS authentication from inside Colab

The proposed pattern — upload the local ADC to `/content/adc.json` and export
`GOOGLE_APPLICATION_CREDENTIALS` — was tested directly. **It is correct for the Python
client and wrong for the `gcloud` CLI.** Both halves were verified on this host.

### 5.1 Python `google-cloud-storage` / `google-auth` — VERIFIED WORKING

The local ADC is `type: authorized_user` (a user refresh token), not a service-account key.
`google-auth` supports that ADC type natively:

```
cred class : google.oauth2.credentials.Credentials
project    : project-flash-490419
valid      : True | expiry: 2026-08-24 00:53:20.943859
GCS list   : HTTP 200
```

Produced by loading `google.auth.default(scopes=[".../devstorage.read_write"])` with
`GOOGLE_APPLICATION_CREDENTIALS` set to the ADC path, calling `creds.refresh()`, and issuing a
real `storage.objects.list` against `nextlat-lurestar-project-flash-490419`.

Three things follow, and the third is why this pattern is right for a weekend-long job:

* the project is auto-detected from `quota_project_id`, so `storage.Client()` needs no args;
* the scope is sufficient for GCS reads *and* writes;
* the credential holds a **refresh token**, so `google-auth` mints new access tokens
  indefinitely — a 12-hour training run never hits the 1-hour access-token wall.

So the runtime pattern is exactly:

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
from google.cloud import storage
client = storage.Client()                      # project inferred from quota_project_id
bucket = client.bucket("nextlat-lurestar-project-flash-490419")
bucket.blob(f"lurestar/runs/{run_id}/ckpt.pt").upload_from_filename(local)
```

`google-cloud-storage` is preinstalled on Colab runtimes; the bootstrap step should still
assert it and assert the round-trip.

### 5.2 `gcloud storage` — VERIFIED **NOT** WORKING with `GOOGLE_APPLICATION_CREDENTIALS`

This is a real negative result, not an inference. With a fresh empty `CLOUDSDK_CONFIG` and
`GOOGLE_APPLICATION_CREDENTIALS` pointing at the ADC file:

```
$ gcloud storage ls gs://nextlat-lurestar-project-flash-490419/
ERROR: (gcloud.storage.ls) You do not currently have an active account selected.
Please run:
  $ gcloud auth login
```

`GOOGLE_APPLICATION_CREDENTIALS` configures **client libraries**, not the gcloud CLI, which
keeps its own credential store. Two follow-ups were also checked:

* `gcloud auth login --cred-file=` accepts only *workload-identity-federation config* or a
  *service-account key JSON* — an `authorized_user` ADC file is not an accepted input.
* `auth/credential_file_override` does not exist as a property in SDK 562.0.0; `gcloud config
  list --all` shows only `access_token_file`, `disable_credentials`,
  `impersonate_service_account`, `login_config_file`,
  `service_account_disable_id_token_refresh`, `service_account_use_self_signed_jwt`,
  `token_host` under `[auth]`.

**What does work for the CLI**, verified with the same empty config dir:

```bash
TOK=$(gcloud auth application-default print-access-token)     # token_len=256, "ya29.a…"
CLOUDSDK_AUTH_ACCESS_TOKEN="$TOK" gcloud storage ls gs://…/   # rc=0, no error
```

and the same bearer token returns `HTTP 200` from `storage.googleapis.com/storage/v1/b/…/o`.

But that token expires in ~1 hour. It is fine for a bootstrap check and useless for a 12-hour
run unless refreshed.

### 5.3 The rule for this project

> **All durable GCS I/O inside Colab goes through the Python client with
> `GOOGLE_APPLICATION_CREDENTIALS=/content/adc.json`. Do not use `gcloud storage` inside the
> runtime.**

If a shell-level copy is genuinely unavoidable, have the Python driver write a refreshed
token to a file every ~45 minutes and point gcloud at it with
`gcloud config set auth/access_token_file=<path>` — but prefer not to.

### 5.4 Security posture — flag this, do not wave it through

`/content/adc.json` is a **live OAuth refresh token for `<redacted-account>` with
cloud-platform scope**. It is not bucket-scoped; anything that account can reach in
`project-flash-490419`, that file can reach. Uploading it puts it on a Google-managed VM you
do not control the lifetime of.

Mandatory handling:

* `os.chmod("/content/adc.json", 0o600)` as the driver's first action;
* the file is uploaded per session and dies with `colab stop` — never persisted to Drive,
  never written into the source snapshot, never inside anything under `git`;
* add `application_default_credentials.json`, `adc.json`, `*.tgz`, `.colab_session` to
  `.gitignore` before the first commit;
* the driver must not echo the file, and `job.json` must not contain credentials.

**Preferred alternative if a few minutes can be spent:** create a dedicated service account
with `roles/storage.objectAdmin` on `nextlat-lurestar-project-flash-490419` *only*, download
its key, and upload that instead. It works with the identical
`GOOGLE_APPLICATION_CREDENTIALS` pattern (service-account keys are the ADC type that env var
was designed for), it is revocable independently of the human account, and its blast radius
is one bucket. The user-ADC route documented above is the working fallback, not the ideal.

---

## 6. Upstream resume mechanics — what the transport layer must preserve

Read before designing the GCS sync, because the transport has to honour upstream's exact
on-disk contract.

**Pointer files live at the output root.** `core_train.py:140-167` implements
`init_from: resume`: it reads `{out_dir}/recovery_ckpt`, falling back to `{out_dir}/latest_ckpt`.
Each is a **plain text file containing a checkpoint path**, and the path is `assert`ed to
exist (`core_train.py:148-150`, `core_train.py:157-159`). If neither pointer exists, upstream
**silently initialises from scratch** (`core_train.py:165-168`) — the single most dangerous
failure mode in this project, because a botched restore looks like a fresh run rather than an
error.

Three hard transport requirements follow:

1. **Restore to the identical absolute path.** The pointer stores the path that
   `_save_checkpoint` built from `out_dir` + `experiment_name` (`core_train.py:931-935`,
   `core_train.py:957-961`). A new runtime must recreate exactly
   `/content/lurestar/runs/{run_id}/…`. Do not let `out_dir` vary between sessions.
2. **Never share `out_dir` between branches.** The pointer is per-output-root, so a near and
   a far H3 branch sharing `out_dir` would resume from each other. This is spec §9's
   "must never cross branches", and `core_train.py:141-142` is why.
3. **Verify before declaring resume.** After restore, the driver asserts that
   `{out_dir}/recovery_ckpt` exists, that the path it names exists, and that its SHA-256
   matches `state.json`. Only then may it pass `init_from=resume`.

**Upstream's recovery rotation is unsafe as shipped.** `_save_recovery_checkpoint`
(`core_train.py:952-979`) writes the new checkpoint, overwrites the pointer non-atomically
(`core_train.py:970-974`), then **deletes the previous recovery checkpoint**
(`core_train.py:976-979`) with no verification of the new one. A runtime death mid-write
leaves a truncated new checkpoint, a pointer aimed at it, and the good one already gone. This
is precisely why spec §9 mandates `.partial` + atomic rename and *two* verified checkpoints.
Implement that as a minimal patch on top of the pinned tree, and record the diff in
`source_snapshot/`.

**Defaults that must be overridden** (`defaults.yaml`):

| Key | Upstream default | Required | Citation |
|---|---|---|---|
| `trainer.save_recovery_checkpoint` | `-1` (off) | `250` | `defaults.yaml:24`; gate at `core_train.py:575-578` |
| `trainer.init_from` | `scratch` | `resume` on every attempt after the first | `defaults.yaml:29-30` |
| `trainer.compile` | `true` (also `true` in both stargraph configs) | `false` | `defaults.yaml:40`; `config/stargraph/5_5/gpt_stargraph_5_5.yaml:17`; `config/stargraph/5_5/nextlat_stargraph_5_5.yaml:16`; spec §8 |
| `sweep.seed` | `[1234,1235,1236,1237,1238]` | `[1234,1235,1236]` | `config/stargraph/5_5/gpt_stargraph_5_5.yaml:54-55` |

Note the sweep: leaving five seeds in the YAML makes **one** `fabric run` invocation train
five models sequentially inside a single process. For resumability, run one seed per job with
one `out_dir` per seed, per spec §9's deterministic job ids.

---

## 7. Workload arithmetic (derived from the pinned configs, not assumed)

All of the following is computed from the checked-out tree; nothing is taken from the paper.

**Serialization** (`data/stargraph/prepare.py:68-72`): `u,v|u,v|…/source,goal=n1,…,n5`.
**Tokenizer** (`data/stargraph.py:9-57`): node ids and `| = / $` are single tokens and commas
are *skipped* (`data/stargraph.py:35-37`); an EOS is appended (`data/stargraph.py:53`).

Replaying that tokenizer on a real G(5,5) line generated with upstream's own algorithm:

| Quantity | Value |
|---|---|
| Sequence length `T` (= `block_size`, set at `data/stargraph.py:252`) | **69** |
| `graph_description_len` (prompt through `=`) | **62** |
| Target tokens | **5** (matches upstream's own assert, `data/stargraph.py:188-190`) |
| `vocab_size` (`data/stargraph.py:233`) | **106** |
| Model | 12L / 6H / 384d, bias-free (`config/stargraph/5_5/gpt_stargraph_5_5.yaml:33-38`) |
| Params, non-embedding | **21.24 M** |
| Params, total | **21.35 M** |
| Tokens per optimizer step (512 × 69) | **35,328** |
| FLOPs per step (6N·tokens, fwd+bwd) | **4.51 × 10¹²** |
| HMM model (spec §12: 4L/4H/128d, T=32, B=256) | **0.79 M** non-embedding, **3.87 × 10¹⁰** FLOP/step |
| Checkpoint size (`fabric.save` of weights + Adam m,v + steps, `models/model_base.py:404-417`) | **≈ 256 MB** |

**The single most important consequence: this model is tiny.** 21 M parameters at sequence
length 69 will *not* saturate a modern datacentre GPU. Step time will be dominated by kernel
launch overhead, the optimizer step, and the dataloader — not by FLOPs. That is why the MFU
assumptions below fall as the GPU gets bigger, and why the H100 gap in §2 costs less than it
appears to.

**VRAM is not a constraint.** Weights + fp32 Adam state ≈ 0.34 GB; activations at batch 512 ×
seq 69 × 384 in bf16 across 12 layers ≈ 2–4 GB. The paper batch fits on a 16 GB T4, so the
gradient-accumulation fallback in spec §11 should not be needed. Confirm at the profiling gate
and record peak allocated/reserved either way.

---

## 8. Compute-unit cost table

### 8.1 What the quota output does and does not give

`colab quota --json` returns `paid_balance` and `burn_rate_hourly` but **no rate card**.
With `active_runtimes: 0`, `burn_rate_hourly` reads `0`.

`burn_rate_hourly` is the authoritative per-GPU rate, and it is populated **only while a
runtime is live**. So the table below has a measured column that is deliberately empty, and a
one-command procedure to fill it:

```bash
for G in t4 l4 a100; do
  S=$(colab start --gpu $G | sed -n 's/^Session: //p')
  sleep 20
  echo -n "$G  "; colab quota --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["burn_rate_hourly"])'
  colab stop
done
```

Total cost of that calibration: under three minutes of runtime across three GPUs — round it
up to 1 CU. **Run it before committing the sweep.** Also capture `paid_balance` immediately
before and after each `colab stop` as an independent check.

### 8.2 Rate table

| GPU | `--gpu` value | CLI-selectable at v0.2.0 | CU/h — **prior, UNVERIFIED** | CU/h — measured | Peak bf16/fp16 dense | VRAM |
|---|---|---|---|---|---|---|
| T4 | `t4` | yes (default) | ~1.76 | *(pending)* | 65 TFLOP/s (fp16; **no bf16** — Turing) | 16 GB |
| L4 | `l4` | yes | ~4.82 | *(pending)* | 121 TFLOP/s | 24 GB |
| A100 | `a100` | yes | ~11.77 | *(pending)* | 312 TFLOP/s | 40/80 GB |
| H100 | — | **no** (help lists t4\|l4\|a100 only) | unknown | *(pending, §2)* | 990 TFLOP/s | 80 GB |
| G4 | — | **no** | unknown | — | — | — |

The CU/h priors are the commonly published Colab rates. **They are not derived from this
host's output** and must be replaced by the measured column before any spend decision. The
recommendation in §10 is deliberately built to survive being wrong about them by 3–4×.

The 40-second T4 from §0 produced **no measurable balance change**, so it yields no usable
rate. Sub-minute sessions are not a viable measurement instrument; use ≥20 s of *live* runtime
and read `burn_rate_hourly`, not the balance delta.

---

## 9. First-cut budget for spec §11

### 9.1 Workload

Straight from spec §11:

```
base       = 6 runs  × 20,000 steps  = 120,000 optimizer steps   (3 seeds × {GPT, NextLat})
adaptation = 12 branches × 500 steps =   6,000 optimizer steps   (2 models × 3 seeds × {near, far})
HMM        = 6 runs  ×  3,000 steps  =  18,000 optimizer steps   (3 seeds × {GPT, NextLat}, small model)
                                       -------
                                        144,000 optimizer steps
```

### 9.2 Model and its assumptions

`step_s = max(FLOP_per_step / (peak × MFU), 0.030 s)`, then ×1.6 for NextLat.

Assumptions, each of which the §11 profiling gate exists to replace:

* **MFU: T4 12%, L4 12%, A100 8%, H100 5%.** Falling with GPU size because a 21 M-param model
  at T=69 cannot keep a large GPU fed (§7).
* **Per-step floor 30 ms** — kernel launch + optimizer + dataloader; this floor, not FLOPs,
  governs the HMM runs entirely.
* **NextLat multiplier 1.6×** — latent-dynamics MLP, horizon-3 rollout, and KL term on top of
  the CE pass (`config/stargraph/5_5/nextlat_stargraph_5_5.yaml:41-43`, `:56-61`). Plausible
  band 1.4–1.9×.
* **Checkpoint overhead** at 250-step intervals (spec §9), 4 s per write including the GCS
  upload of ~256 MB: 576 writes total ≈ 0.64 h.
* **+20% interruption margin**, as spec §11 requires.

### 9.3 Result

| GPU | GPT s/step | NextLat s/step | Base h | Adapt h | HMM h | Ckpt h | **Total h** | **+20%** | **CU at prior rate** | % of 1788.78 |
|---|---|---|---|---|---|---|---|---|---|---|
| **T4** | 0.577 | 0.924 | 25.02 | 1.25 | 0.20 | 0.64 | 27.10 | **32.5** | ~57 | 3.2% |
| **L4** | 0.310 | 0.496 | 13.44 | 0.67 | 0.20 | 0.64 | 14.95 | **17.9** | ~86 | 4.8% |
| **A100** | 0.180 | 0.289 | 7.82 | 0.39 | 0.20 | 0.64 | 9.04 | **10.9** | ~128 | 7.1% |
| *H100 (not selectable)* | *0.091* | *0.146* | *3.94* | *0.20* | *0.20* | *0.64* | *4.97* | ***6.0*** | *unknown* | — |

### 9.4 Feasibility — CU is not the binding constraint; wall-clock is

**All three CLI-selectable GPUs fit inside 1788 CU with enormous headroom.** Even the most
expensive option consumes ~7% of the balance. Sensitivity on A100:

| If reality is | GPU-h | CU | % of balance |
|---|---|---|---|
| as modelled | 10.9 | 128 | 7.1% |
| 2× slower | 21.7 | 255 | 14.3% |
| 3× slower | 32.6 | 383 | 21.4% |
| 4× slower | 43.4 | 511 | 28.6% |

The budget survives being wrong by 4×. **So stop optimising for CU and optimise for
wall-clock and interruption exposure**, because those are what actually threaten the weekend:

* **T4 — INFEASIBLE for confirmatory runs.** 32.5 GPU-h against a weekend, spread across
  Colab session limits, means ~4+ forced resume cycles per base run and a real chance of not
  finishing. Turing has no bf16, so spec §8's `bf16-mixed` would have to become `16-mixed`,
  adding a precision deviation to the confirmatory path. Spec §11 already classifies T4 as
  smoke-test only. **Use for pipeline/recovery smoke tests only** — which is exactly the right
  place for the mandatory 300-step interruption test (spec §9), at negligible cost.
* **L4 — FEASIBLE BUT TIGHT.** 17.9 GPU-h. Supports bf16 (Ada), 24 GB is ample. Viable as an
  overflow lane if A100 assignment fails, but ~2× the wall-clock of A100 for ~40% of the CU
  cost — a bad trade when CU is not scarce.
* **A100 — FEASIBLE AND RECOMMENDED.** 10.9 GPU-h at ~128 CU. Fits the Saturday/Sunday
  schedule with slack, native bf16 as spec §8 requires, 40 GB removes any memory question,
  and it matches the class of hardware the paper used (A5000/H100 NVL/B200).
* **H100 — would halve wall-clock again, but is not reachable through `colab` v0.2.0.** The
  §2 probe is worth one budgeted attempt; do not plan around it.

---

## 10. Recommendation and gates

**Run everything on A100.** Budget **≈ 130 CU expected, 400 CU planning envelope**, against a
1788.78 CU balance. Hold T4 for smoke tests and the forced-interruption test.

Gate the spend in this order, and do not skip a gate:

1. **Calibrate rates** (§8.1). ~1 CU, three minutes. Replace every prior in §8.2 with a
   measured `burn_rate_hourly`.
2. **Probe A100 and confirm bf16** — `colab exec --gpu a100 -c "import torch;
   print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"`.
3. **Run the mandatory recovery test on T4** (spec §9): 300 steps clean, then 150 + kill +
   resume + finish at 300, verifying step, optimizer/scheduler state, data position, and final
   weights. Cheap hardware, and it is the transport layer that is under test, not the GPU.
4. **Profile 500 Lure-Star steps and 300 HMM steps on A100** (spec §11), warmup 100, summarise
   the last 400. Substitute the measured `seconds_per_step` into §9.2 and recompute. If
   measured step time exceeds the modelled A100 figure by more than ~2.5×, stop and
   investigate the dataloader (`data.num_workers: 0` in both stargraph configs, `:25` and
   `:24`, is an obvious first suspect for a 35k-token step) before spending on the sweep.
5. **Only then launch the sweep**, one seed per job, one `out_dir` per job, with the §4.6
   recovery loop and the §3(iv) two-strike circuit breaker armed.

Record into `run_ledger.json` for every job: session id, GPU name, zone, `paid_balance` before
and after, wall-clock, measured s/step, peak VRAM, checkpoint count and bytes, resume count,
and `parent_checkpoint_sha256`.

---

## Appendix — verification log

| Claim | Evidence |
|---|---|
| CLI version and surface | `colab -v` → `colab v0.2.0`; `colab --help` |
| `colab start --help` provisions a runtime | ran it; T4 `gpu-t4-s-kkb-usw4a0-3a5zwq759t14l` started, then `colab stop` released it |
| quota / status JSON | quoted verbatim in §1 |
| `--gpu` accepts only t4\|l4\|a100 | `colab --help`; `strings` over the binary finds no other accelerator vocabulary |
| exec is a Jupyter kernel cell | binary contains `execute_request`, `kernel_info_reply`, `ws://shell`, `github.com/coder/websocket@v1.8.14`, `/home/runner/work/colab-cli/colab-cli/kernel.go`, and the notebook-create JSON |
| upload/download are base64-over-kernel | generated Python recovered from the binary, quoted in §2 |
| ADC is `authorized_user` | JSON parse: `type=authorized_user`, `quota_project_id=project-flash-490419` |
| GAC works for `google-auth` | `google.auth.default()` → `google.oauth2.credentials.Credentials`, refresh OK, `storage.objects.list` → HTTP 200 |
| GAC does **not** authenticate `gcloud` | empty `CLOUDSDK_CONFIG` + GAC → `ERROR: (gcloud.storage.ls) You do not currently have an active account selected.` |
| `CLOUDSDK_AUTH_ACCESS_TOKEN` does | same empty config dir + 256-char `ya29.a…` token → `rc=0`; REST `objects.list` → HTTP 200 |
| `--cred-file` rejects authorized_user | `gcloud auth login --help`: accepts workload-identity config or service-account key JSON only |
| bucket exists, region | `gcloud storage buckets describe` → `nextlat-lurestar-project-flash-490419  US-CENTRAL1` |
| T=69, vocab=106, 21.24 M params | upstream `Tokenizer` (`data/stargraph.py:9-57`) replayed on a G(5,5) line built by `data/stargraph/prepare.py:8-35`; param count from `config/stargraph/5_5/gpt_stargraph_5_5.yaml:33-38` |
| resume pointer semantics | `core_train.py:140-167`, `:931-948`, `:952-979` |
| silent scratch-init on missing pointer | `core_train.py:165-168` |
| unsafe recovery rotation | `core_train.py:970-979` (non-atomic pointer write, then unconditional delete of the previous recovery checkpoint) |
