# HANDOFF — NextLat × Predictive Geometry

Written 2026-08-23 for a cold pickup by another agent or session. Read this file, then
`nextlat_v4_predictive_geometry_spec.md`, then `docs/FOUNDATIONS.md`. Everything below is
verified fact with the command that established it, not intention.

## What this project is, in one paragraph

The NextLat paper (arXiv:2511.05963v4) adds a latent transition model to next-token training and
proves the hidden state becomes a belief state — a sufficient statistic of the history for
predicting the future. It shows the representation gets compressed (effective rank 52.7 vs GPT's
160.1) but never asks what the geometry *is*. Its own §6 says so: *"On the analysis side, we do
not study the structure of the learned representations under NextLat, leaving open questions about
how the method shapes latent spaces."* This project fills exactly that gap by asking whether
NextLat separates histories whose futures differ while keeping future-equivalent histories close,
whether that geometry predicts planning behaviour, and whether it predicts which memories interfere
under later training. A 4-state HMM supplies exact Bayesian ground truth that Path-Star cannot.

## Status: BUILD COMPLETE, SWEEP NOT STARTED

No model has been trained. 0.25 of 1788 compute units spent. No Colab runtime is active.

### Verified done

| Thing | Evidence |
|---|---|
| Upstream pinned | `upstream/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`, read-only |
| Paper-scale corpus | `data/stargraph/`, 200k train sha256 `d13199b0…`, 20k test `f52fb14e…`, byte-identical to upstream's serial render, generated in 4.6s by `scripts/generate_corpus.py` |
| Corpus on GCS | `gs://nextlat-lurestar-project-flash-490419/lurestar/corpus/stargraph/`, hash-verified on an A100 runtime |
| Stimuli | 2,000 quartets in `manifests/e_lure.jsonl` + `a_pair` (1,000) + `b_near` (5,000) + `b_far` (15,000), each with `.sha256` |
| HMM frozen | `manifests/hmm_matrices.json`, chosen by a 2,592-point **model-blind** grid search, 83 candidates passed, payload sha256 recorded |
| Profiling gate | `results/profile_summary.json` — A100, GPT 0.2017 s/step, NextLat 0.2335 s/step, batch 512 fits in 40GB, no gradient accumulation, bf16-mixed available |
| Tests | 369 passed / 5 skipped / 0 failed as of the pre-BST baseline. **Re-run and re-baseline before trusting this line.** |

### In flight when this was written

Two background workflows may have completed after this file was written — check for their outputs
before redoing their work:

* `lurestar-bst-arm` → writes `docs/review/bst-arm.md`, edits `configs/`, `scripts/run_matrix.py`,
  `scripts/config_lib.py`, `src/lurestar/representations.py`.
* `standing-agents-sweep-1` → writes `docs/QA_LOG.md`, `docs/QA_STATUS.md`,
  `docs/THROUGHPUT_LOG.md`, `docs/UNSLOP_REPORT.md`, `docs/UNSLOP_LOG.md`.

## The immediate next actions, in order

1. **Apply the queued seed change.** See `.agent_state/PENDING_SEED_CHANGE.md`. The spec already
   says five confirmatory seeds; the code still says three. `scripts/run_matrix.py:72` has
   `SEEDS = (1234, 1235, 1236)` → make it `(1234, 1235, 1236, 1237, 1238)`, same in
   `scripts/config_lib.py`, re-materialize configs, and verify the seed-1234–1236 configs come back
   **byte-identical** (they were frozen earlier and must not move).
2. **Re-run the full suite** and re-baseline: `.venv/bin/python -m pytest tests/ -q`.
3. **Read `docs/QA_LOG.md`** if it exists. Do not launch the sweep with an open P0.
4. **Launch the sweep**: `.venv/bin/python scripts/colab_train_loop.py` from the project root. It
   packages the project, pushes it to GCS, starts an A100, execs itself as the DRIVER, and restarts
   on a fresh runtime whenever Colab drops it. Resume state lives in GCS, so a re-exec resumes.
5. **Run the standing agents on cadence** while it trains — QA after the first seed lands, unslop
   before anything is shared.
6. **Figures, then the writeup.** `report/blog.md` is the single source of truth; `{{live:KEY}}`
   tokens are filled from `results/live_numbers.json` by `scripts/build_report.py`, which renders
   the Word doc. An unfilled key renders as `[pending]` on purpose.
7. **Publish**: public GitHub repo, no secrets. `.env`, `.secrets/`, and the ADC path must never be
   committed — secret-scan before pushing.

## Five landmines. Every one of these fails SILENTLY.

1. **D-19 — adaptation jobs that do nothing.** Branching a 20,000-step parent restores
   `training_steps`, so the loop starts at step 20,001. An adapt config with `train_batches: 500`
   returns *immediately* and looks like a clean completion with **zero optimizer updates** — a
   fabricated H3 null. Set `20500`, or reset `training_steps` first, and keep the test that catches
   a zero-update job.
2. **D-07 — the wrong model for 20,000 steps.** `proj_factor: 0.5` appears upstream *only* inside
   the `sweep:` block. Delete the sweep and it silently defaults to `1.0`: dynamics hidden 768
   instead of 384, ~+885k params, no warning. Assert the resolved value and the parameter count at
   step 0.
3. **D-06 — a 100× tighter gradient clip.** The spec's key name `clip_gradient_norm` does not
   exist; the real key is `optimizer.grad_clip`. Under the wrong name it falls back to `1.0` while
   the paper uses `100`.
4. **D-11 — measuring a state that cannot vary.** The token generated at the `=` delimiter
   (index 62) is the *source*, which is already in the prompt and identical across every member of
   a quartet. The first real branch decision is index 63. Extract **both**; index 62 stays the
   preregistered primary for PSI, and every correct-branch margin comes from index 63.
5. **D-20 — halting on a correct run.** The paper puts GPT on G(5,5) at ~18.6%, which is 1/d
   chance. The 90% competence gate therefore applies to **NextLat and BST only**. See
   `docs/DECISION_D20_competence_gate.md`.

## Colab transport — three failures already hit, all fixed, all easy to reintroduce

1. **A child process's stdout does not reach the `colab exec` stream.** Relay it in-process with
   `Popen` + readline, or the run is blind. The first smoke test "passed" while running blind.
2. **`cmd | tail -N` returns tail's exit status.** The second smoke test reported `RC=0` while
   `train.py` was dying on `Missing key test_generalization`. Never pipe a checked command.
3. **`colab exec` forwards no argv and leaves `__file__` undefined.** Parameters arrive via an
   uploaded sidecar JSON; guard every `__file__` with a `try/except NameError`.

Also: never hand-write a training config. Derive it from the official YAML and override only
permitted keys — a reconstructed config is what caused failure 2.

## Credentials and GCS

Service-account keys are blocked by the org policy `constraints/iam.disableServiceAccountKeyCreation`.
The working pattern, verified both directions on a runtime: upload the local `authorized_user` ADC
(`~/.config/gcloud/application_default_credentials.json`) to `/content/adc.json`; the python
`google-cloud-storage` client reads it from `GOOGLE_APPLICATION_CREDENTIALS`, and the `gcloud` CLI
works from an access token minted in-process into `CLOUDSDK_AUTH_ACCESS_TOKEN`. The refresh token is
long-lived so a run can re-mint indefinitely. **This credential has full `cloud-platform` scope and
must never be committed, echoed, or persisted to Drive** (risk R12 in `docs/FOUNDATIONS.md`).

## Design decisions already made — do not silently revisit

* **Three architecture-matched arms: GPT, NextLat, BST.** BST is the competence-matched control:
  the paper puts it at ~99.9% on G(5,5) *without* a latent-transition objective, so **NextLat vs
  BST is the primary cross-model contrast**, NextLat vs GPT is secondary and competence-confounded,
  and BST vs GPT shows how much is competence alone. Cost 3.7 GPU-h / ~20 CU.
* **Five confirmatory seeds** (1234–1238), matching the paper. Decided before any training, so
  nothing was chosen on results.
* **Manhattan priced and declined**: 359 GPU-h / 1,902 CU / 106% of balance for one seed pair;
  1,076 GPU-h / 319% for three. No checkpoints were released. A 15%-scale fallback (60k steps, one
  seed pair, 54 GPU-h / 285 CU) is recorded in spec §14 if it is ever justified.
* **Total projected: 23.5 GPU-h, ~125 CU, 7% of balance.**
* **The stimulus invariant is weaker than the spec's literal text, deliberately and provably.**
  `docs/STIMULUS_DESIGN.md` proves identical absolute edit positions are impossible with a shared
  anchor, refutes the obvious repair (it would turn a 2-token perturbation into a ~10-token one),
  and states the adopted invariant plainly. Do not "fix" this without reading that proof.

## What honesty requires of the writeup

* Report every preregistered metric, including the nulls. Path-Star has one consistent algorithm,
  so H3 may legitimately come out flat. A clean null is the result, not a failure.
* Seeds are the inferential unit for cross-model contrasts; items never substitute for seeds.
* State what effect size five seeds could not have detected.
* No biological claims. "Pattern separation" only as an explicitly labelled computational abstraction.
* Do not present as novel anything the paper already owns: ordinary Path-Star accuracy, lower
  effective rank, "NextLat is more compressed", t-SNE/UMAP plots, generic forgetting.
* The audience is Pratyusha Sharma. Her forgetting work is *parameter-space* (intruder dimensions,
  spectral shift); this is *representation-space*. Name the disanalogy rather than blurring them.

## Where things live

```
nextlat_v4_predictive_geometry_spec.md   the spec (revised: scope constraints are priced decisions)
PROGRAM.md                               what an autonomous agent may and may not change
HANDOFF.md                               this file
docs/FOUNDATIONS.md                      37-row deviation ledger + 12 ranked risks
docs/UPSTREAM_REPORT.md                  1085-line repo cartography, file:line cited
docs/PAPER_NOTES.md                      grounded paper extraction incl. verbatim Limitations
docs/STIMULUS_DESIGN.md                  the impossibility proof and the adopted invariant
docs/EXTRACTION.md                       which hidden state, at which position, and why
docs/DECISION_D20_competence_gate.md     why GPT is allowed to fail, and how BST repairs it
docs/RUNLOG.md                           append-only log of what was actually established
docs/review/*.md                         adversarial reviews of each build track
src/lurestar/                            generate, validate, representations, evaluate, durable_checkpoint
src/hmm_geometry/                        generate, forward, pair_bank, evaluate
scripts/colab_train_loop.py              dual-role resumable sweep driver (LOOP on Mac, DRIVER on Colab)
scripts/run_matrix.py                    idempotent job matrix + ledger
scripts/build_report.py                  report/blog.md + results/live_numbers.json -> .docx
report/blog.md                           the writeup, single source of truth
```

Durable root: `gs://nextlat-lurestar-project-flash-490419/lurestar/`
