# Durable confirmatory HMM matrix

`scripts/run_hmm_matrix.py` is the only confirmatory HMM launch path. It fixes the matrix before
any outcome exists:

| model | regimes | seeds | updates per cell | exact job ids |
|---|---:|---:|---:|---|
| GPT | 3 | 1234–1238 | 3,000 | `gpt-seed{seed}-hmm-{regime}` |
| NextLat | 3 | 1234–1238 | 3,000 | `nextlat-seed{seed}-hmm-{regime}` |

This is 30 jobs: two models by five seeds by the three jointly frozen regimes
`persistent_moderate`, `fast_mixing_moderate`, and `persistent_high_aliasing`. It is not a sweep
whose regimes or seeds are discovered dynamically. Every job has its own
`runs/hmm_family/{regime}/{model}/seed{seed}/base` output root; its experiment directory is its exact job id. The
id contains the substring `seed`, so the pinned trainer does not append a second seed suffix.

## Before compute

The runner re-hashes the complete 31-row family corpus/posterior/manifest snapshot against
`manifests/hmm_family_inventory.sha256`, verifies the family materialization receipt, and checks
every dataset, threshold, pair-bank, and matrix artifact against the same regime HMM. All three
regimes must be present in frozen order; a favorable subset cannot be planned. It also binds the
config, matrix runner, HMM shim,
datamodule, durable-checkpoint/runtime patch sources, and the actual upstream training sources into
each job's ledger identity. The upstream checkout must still resolve to commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`; runtime hardening may make that checkout dirty, but it
may not change its commit identity.

Print the complete pre-compute plan without touching a GPU or ledger:

```bash
python scripts/run_hmm_matrix.py \
  --root /content/lurestar \
  --project-root /content/project \
  --upstream /content/project/upstream/NextLat \
  --snapshot-root /content/lurestar \
  --data-root /content/lurestar \
  --family \
  --print-plan
```

The emitted plan records all 30 exact job IDs, config paths, isolated output/checkpoint roots,
identity inputs, update target, and the 250-update recovery cadence.

## Training and recovery

Run through the hardened Colab driver after its T4 recovery gate and A100 profiles pass. The
source/config root and durable snapshot/data root are intentionally different: code is extracted
under `/content/project`, while immutable arrays, manifests, outputs, and the ledger live under
`/content/lurestar`.

```bash
python scripts/run_hmm_matrix.py \
  --root /content/lurestar \
  --project-root /content/project \
  --snapshot-root /content/lurestar \
  --data-root /content/lurestar \
  --upstream /content/project/upstream/NextLat \
  --family \
  --phase train \
  --ledger /content/lurestar/run_ledger.json \
  --driver-managed-durability
```

The command always overrides `trainer.train_batches=3000`, `compile=false`, W&B off, and
`save_recovery_checkpoint=250`. The guarded runtime patch writes a checkpoint to `.partial`,
fsyncs and atomically renames it, hashes and deserializes it, retains two verified generations,
then advances the pointer. The Colab driver uploads artifacts first and publishes durable state
last every 60 seconds. Thus an exec-stream timeout is not a stop condition: the host monitors the
owned runtime's durable step and terminal marker. A stopped runtime restarts from the newest
verified generation; if that generation is corrupt, resolution rolls back one.

An interrupted subprocess appends `INTERRUPTED` with the newest verified step and a bounded last
40-line diagnostic tail. A failure before any valid checkpoint appends `FAILED`; successful exit
at 2,999 or 3,001 is also `FAILED`. Only a verified step-3,000 checkpoint plus the upstream
materialized config and nonempty metrics log reaches `TRAINED`. Rerunning the command skips a
TRAINED or DONE job only after its recorded hashes re-verify.

`--driver-managed-durability` is a required execution contract, not an optional convenience. The
runner itself does not own a live GCS daemon; the driver does. The deprecated `--bucket` flag now
fails closed because passing a bucket to a post-process checkpointer does not protect checkpoints
written while the trainer subprocess is still running. `--dry-run` and `--print-plan` remain safe
outside the driver: both return before a `Ledger` is constructed and mutate no state.

### Driver dispatch hook

The Colab job sidecar needs a validated runner selector rather than a shell fragment. Its HMM train
hook should resolve to the command above, with the driver adding
`--driver-managed-durability` only after its 60-second artifact-first/state-last sync thread is
running. Recommended sidecar fields:

```json
{
  "runner": "hmm",
  "runner_phase": "train",
  "family": true
}
```

Allow only `runner in {lurestar,hmm}` and `runner_phase in {train,evaluate}`; pass arguments as an
argv list, never a joined shell string. The HMM hook must set source root `/content/project`,
snapshot/data root `/content/lurestar`, and the freshly cloned/patched upstream root. The driver,
not this runner, owns GCS retries, restore, terminal markers, and runtime shutdown.

## Evaluation is a separate, non-selective phase

Training never claims a scientific result. It stops at `TRAINED`. The evaluator must write
`evaluation/hmm_geometry.json` and its SHA sidecar below each job root. The schema is
`nextlat_forgetting/hmm_geometry/1`; it binds job/model/seed, the final checkpoint SHA, evaluator
source, and frozen manifests. It must attest that no metric selection was performed and report
the complete preregistered set:

- H1 centered-cosine predictive equivalence and its whitened robustness check;
- H2 Spearman, partial Spearman, and neighborhood retrieval;
- HMM Bayesian posterior and future-distribution decoding at lengths 32 and 64. Historical metric
  keys use the `h3_` prefix, but these are frozen HMM calibration diagnostics—not the permanently
  retired Lure-Star H3 adaptation/interference estimand or its mechanism probes.

Only then run:

```bash
python scripts/run_hmm_matrix.py \
  --root /content/lurestar \
  --project-root /content/project \
  --snapshot-root /content/lurestar \
  --data-root /content/lurestar \
  --upstream /content/project/upstream/NextLat \
  --ledger /content/lurestar/run_ledger.json \
  --family \
  --phase evaluate \
  --driver-managed-durability
```

This phase cannot launch training. It promotes `TRAINED` to `DONE` atomically in the append-only
ledger after re-verifying the checkpoint, training artifacts, evaluation receipt, sidecar,
evaluator, manifests, and exact metric-key set. Missing or extra metric keys fail closed, which
prevents a favorable subset from becoming the terminal scientific record.

The evaluation hook is also driver-owned: restore the TRAINED checkpoint and its ledger first,
produce the frozen receipt, invoke `--phase evaluate`, then include every ledger-recorded
evaluation artifact in the driver's final state-last durability transaction before publishing the
evaluation terminal marker. A local host invocation cannot use `/content/...` checkpoint paths
after the runtime has stopped and is intentionally refused without the driver contract.

Partial subsets may be selected with exact IDs only for recovery or operational scheduling, for
example `--only gpt-seed1234-hmm-persistent_moderate`; this does not change the frozen 30-job
confirmatory family.
