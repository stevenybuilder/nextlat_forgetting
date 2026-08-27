# Hardened Colab profiling gate

`scripts/colab_profile_loop.py` is the supported paid-compute launcher for the profiling gate.
It is separate from the confirmatory matrix driver: profiling is nonconfirmatory, writes only
under `gs://YOUR_PRIVATE_BUCKET/lurestar/profiles/`, and cannot mark a training
job `TRAINED` or `DONE`.

Run it from the project checkout only after the T4 recovery gate passes:

```bash
.venv/bin/python scripts/colab_profile_loop.py
```

The launcher has no science-facing CLI flags. Its content-hashed sidecar freezes A100,
`bf16-mixed`, seed 1234, two 50-step HMM smoke jobs, 500-step GPT/NextLat/BST Lure-Star profiles
(100-step warmup), and 300-step GPT/NextLat HMM profiles (60-step warmup). It also freezes the
complete GCS corpus/manifest object inventory: every object name, generation, size, MD5, and
CRC32C is part of the sidecar contract and `profile_id`. The runtime lists those prefixes again
before doing work, refuses any mismatch, and downloads the exact frozen generations. The five
measured jobs are run by `scripts/profile.sh`; peak allocated/reserved VRAM comes from the
in-process `profile_entry.py` probe.

## Durability and disconnect behavior

The runtime uses the Python `google-cloud-storage` client with the uploaded mode-0600 ADC. It does
not use runtime `gcloud`. Every minute it uploads raw logs, job manifests, probes, GPU samples,
resolved configs, checkpoints, summaries, and provenance with SHA-256 metadata. It rechecks local
bytes after upload and publishes `state.json` last, so that object is the commit record for a
consistent artifact generation.

If the `colab exec` websocket returns while the owned runtime is still active, the host monitors
the GCS generation counter and leaves an advancing runtime alone. It stops the owned runtime only
after a verified terminal marker or six monitoring windows with no new durable artifact generation;
two agreeing `no_runtime` reads also establish that it has already disappeared. It never
automatically provisions a replacement. A failed run
gets `failure.json`, a final partial sync, and a failure terminal marker; that preserves useful
measurements without pretending the gate passed.

Artifact payloads use immutable content-addressed keys
`artifacts/sha256/{payload_sha256}/{relative_path}`. Changing a live log or checkpoint creates a
new object instead of overwriting bytes named by an earlier committed `state.json`; an interrupted
sync therefore cannot invalidate the last recoverable generation. State, provenance, and terminal
receipts all bind the full remote-input identity as well as the source snapshot.

On a manual rerun with the same source and remote-input hashes, verified GCS artifacts are restored. Completed HMM
smokes and complete Lure-Star/HMM profile groups are skipped. A preemption inside one group may
require rerunning that group, but every completed log, probe, config, checkpoint, and manifest
from the interrupted attempt remains durable and auditable.

## Completeness contract

A success terminal is refused unless all of the following are present and verified:

- both 50-step HMM smoke manifests, nonempty raw logs, in-process VRAM probes, resolved configs,
  and checkpoints;
- exact job manifests for Lure-Star GPT, NextLat, and BST at 500 steps and HMM GPT and NextLat at
  300 steps;
- nonempty raw logs and exactly one in-process peak-VRAM probe for every measured job;
- `profile_summary.json` with all five jobs, every required throughput/memory/checkpoint field,
  and a complete end-to-end budget projection.

The terminal marker is session-scoped at
`lurestar/profiles/{profile_id}/sessions/{session_id}/terminal.json` and binds the session, source
snapshot hash, frozen remote-input identity, and committed profile-state hash. Host teardown calls
`colab stop --session {owned_session_id}`—never an account-wide bare stop—and fails closed unless
two status reads agree on `no_runtime` and two later quota reads both report
`active_runtimes: 0` and `burn_rate_hourly: 0`.
