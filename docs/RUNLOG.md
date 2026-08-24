# Run log

Append-only. Every entry is a fact with the command or artifact that established it.
Timestamps are local (America/Los_Angeles).

---

## Session 1 — Friday setup

**Environment.** macOS host, no local GPU, no local torch. Analysis venv at `.venv`
(python 3.12, numpy 2.5.2). GPU via Colab Pro+ `colab` CLI: 1788 compute units, H100 /
A100 / L4 / T4 / G4 available. Durable store: `gs://nextlat-lurestar-project-flash-490419`
(us-central1), created for this project.

**Upstream pinned.** `JaydenTeoh/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
("Initial public release", 2026-05-25), checked out read-only at `upstream/NextLat`.

**G(5,5) configs exist as the spec assumes.** `config/stargraph/5_5/{gpt,nextlat}_stargraph_5_5.yaml`
match the spec's expected values: `train_batches: 20000`, `effective_batch_size: 512`,
`n_layer: 12`, `n_head: 6`, `n_embd: 384`, `stargraph_max_nodes: 100`, AdamW at 5e-4 with
betas (0.9, 0.95), weight decay 0.1, grad clip 100, constant schedule (`warmup_iters: 0`,
`warmdown_iters: 0`). NextLat adds `mtp_horizon: 3`, `lambda_mse: 1.0`, `lambda_kl: 1.0`,
`proj_factor: 0.5`. The shipped sweep is five seeds `[1234..1238]`; we preregister the
first three.

**Two spec-vs-repo deviations already confirmed by reading the source.**
1. The shipped configs set `compile: true`; the spec requires `false`, and the upstream
   README itself reports inconsistent Path-Star results under `torch.compile`. Override.
2. The shipped launch scripts use `--devices 2 --strategy ddp`. We run `--devices 1`.
   Effective batch size is held at 512 either way.
3. The spec's `data.train_graphs` / `data.heldout_graphs` keys do not exist upstream; the
   repo takes file paths (`stargraph_train_data_path`, `stargraph_test_data_path`) whose
   filenames encode the sample count. Dataset size is controlled at generation time.
4. The spec says node IDs are sampled from `1...100`; `data/stargraph/prepare.py` samples
   from `range(maxNodes)`, i.e. `0...99`. Cosmetic, but the manifests record the real range.

**Serialized example format** (`data/stargraph/prepare.py:68-73`), verbatim structure:

```
u1,v1|u2,v2|...|uN,vN/source,goal=p0,p1,...,pL
```

Arm 0 is always the goal arm (the path is accumulated only while `p == 0`), the edge list
is shuffled, and each sample is generated under `random.seed(seed)` with `seed` incrementing
from 0 — so the corpus is reproducible and the test split continues the same counter.
G(5,5) gives 20 edges over 21 distinct nodes and a 5-node answer path.

**Colab transport: two bugs found and fixed before any real run.**
1. A child process's stdout does **not** reach the `colab exec` stream. The first smoke
   test ran completely blind — `ls`, `head`, and the training command all produced no
   visible output. Fixed by relaying child lines in-process via `Popen` + readline.
2. `cmd 2>&1 | tail -60` returns *tail's* exit status, so a crashed training run reported
   `RC=0`. The first smoke test's apparent success was an artifact of this. All checked
   commands now run unpiped.

**Durable auth resolved.** The org policy `constraints/iam.disableServiceAccountKeyCreation`
blocks service-account keys, so the standard "upload a key to the runtime" pattern is
unavailable. The local credential is an `authorized_user` ADC (long-lived refresh token).
Two working paths, both verified on an L4 runtime: the python `google-cloud-storage` client
reads it straight from `GOOGLE_APPLICATION_CREDENTIALS`, and the `gcloud` CLI accepts an
access token minted from it in-process via `CLOUDSDK_AUTH_ACCESS_TOKEN`. Verified
`PY_CLIENT_WRITE=True`, `PY_CLIENT_READ` round-tripped, `GCS_CLI_WRITABLE=True`.

**Runtime identity (L4).** torch 2.11.0+cu128, CUDA 12.8, compute capability (8, 9),
`bf16_supported=True`. The spec's `bf16-mixed` precision is therefore available; the
`16-mixed` fallback is not needed on this GPU class.

**First throughput signal.** `data/stargraph/prepare.py` reseeds the RNG once per sample and
prints a 50-character progress bar per sample. Generating 2,000 tiny samples on the Colab
CPU was already visibly slow. Extrapolated to the required 200,000 train + 20,000 heldout
graphs this is a real bottleneck, and it is on the critical path for every run. Handing to
the throughput agent: generate the corpus once, hash it, store it on GCS, and never
regenerate it during a resume (the spec's own instruction).

**The gap this project targets, in the paper's own words** (arXiv:2511.05963v4, §6
Limitations and Future Work):

> On the analysis side, we do not study the structure of the learned representations under
> NextLat, leaving open questions about how the method shapes latent spaces.

and, for the later bottleneck-width ablation:

> We also do not study how the width of the hidden layers in the latent dynamics MLP
> affects learning, even though it effectively acts as a bottleneck that constrains
> belief-state capacity and may influence performance across tasks.

**Smoke test green on L4 (third attempt).** 20 NextLat steps at the paper architecture
(12L/6H/384, `mtp_horizon=3`, `lambda_mse=lambda_kl=1.0`, `proj_factor=0.5`) on a 2,000-graph
toy corpus. Train loss 4.38 -> 4.31, validation loss tracked it, and the run wrote
`ckpt_iter_*.pt`, a `latest_ckpt` pointer, `materialized_config.yaml`, and `version_0/metrics.csv`.
End-to-end transport is therefore proven: local script -> Colab runtime -> pinned repo ->
training -> durable artifacts -> GCS.

The third attempt is the point. Attempt 1 reported success while running blind. Attempt 2
reported success while `train.py` was in fact dying on `Missing key test_generalization`,
because a hand-written config had dropped a key the trainer requires. Only attempt 3 — with
child output relayed, exit codes unmasked, and the config *derived from the official YAML by
overriding permitted keys only* — actually trained. The spec's rule "copy the official Path-Star
configuration, do not reconstruct an approximate one from this document" earned its place.

Also observed: upstream retains only the latest and best checkpoints (`ckpt_iter_10` was deleted
once `ckpt_iter_20` existed). The spec requires two *verified* recovery checkpoints with the
oldest deleted only after the newest is loaded and hashed, so the durable layer has to own
retention rather than delegating it upstream.

**H1's extraction point needed correcting before a single confirmatory run.** The spec says to
take the state at the final prompt delimiter `=` (index 62). But the token generated from index
62 is `path[0]`, the *source* — which is already printed in the prompt's `/source,goal` field, so
predicting it is a copy, and it is identical across every member of a quartet by construction. The
first genuine *branch* decision is `path[1]`, generated from index 63. Extracting only at `=`
risked measuring a state whose immediate prediction target cannot differ between near-safe and
near-critical. Resolution, frozen before any model is trained: extract both positions, keep index
62 as the preregistered primary for PSI, and compute every correct-branch logit margin from index
63.

## Profiling gate — passed on A100, at exact paper scale

Ran spec section 11's 500-step gate for both objectives on the real 200,000-graph corpus, with the
config derived from the official YAML and only step count, out_dir, compile and corpus paths
overridden. The corpus SHA-256 was verified on the runtime against the manifest before training
(`d13199b0...`, `f52fb14e...`) - the file that trained is provably the file that was generated.

| | GPT | NextLat |
|---|---|---|
| steady s/step (steps 100-500) | 0.2017 | 0.2335 |
| wall for 500 steps | 96.7 s | 93.6 s |
| checkpoint size | 256 MB | 274 MB |

NextLat's overhead is **1.16x**, not the 1.6x the pre-run model assumed. Both fit the paper's
physical batch of 512 on a 40 GB A100, so the gradient-accumulation fallback is not needed and the
confirmatory path carries no batch-related deviation. `bf16-mixed` is available (capability 8.0), so
no precision deviation either.

Measured A100 burn rate is **5.3 CU/h**, against a pre-run prior of 11.77 - the budget was 2.2x too
pessimistic. Full workload projects to 9.4 GPU-h and **50 compute units, 2.8% of the 1788 balance**.
Credits are not the constraint; wall-clock across Colab session limits is, which is what the durable
resume layer exists for.

GPT reached train loss 0.444 by step 500 and NextLat 0.459 - both learning, neither diverging.

Two measurement bugs in the profiler itself, both fixed for later runs and both worth recording
because they are the same class of error as the `tail` exit-code bug: the driver kept only a 60-line
tail of child output, so its own steady-state estimate silently returned `null` (the numbers above
were recovered by parsing the full local log); and peak VRAM was read from `torch.cuda` in the driver
process rather than the `fabric run` child, so it reported 0.00 GB. **Peak VRAM is therefore still
unmeasured** and must be captured on the first confirmatory run.

## Durable checkpoint / resume contract — implemented and forced-interruption tested

`src/lurestar/durable_checkpoint.py`, `scripts/run_matrix.py`, `tests/test_resume.py`,
`tests/test_run_matrix.py`. 35 tests, all passing on the CPU-only host in 12 s.

**The recovery test passes bitwise -- on a CPU surrogate, not on the real trainer.** The host has
no GPU and no torch, so the trainer under test is `src/lurestar/toy_trainer.py`: a numpy MLP that
owns the state upstream's checkpoint is missing (Adam moments with a real `t`, a scheduler with a
real `last_epoch`, a shuffled data position, and the python + numpy RNG streams). 300 steps
uninterrupted; then the same 300-step job, `SIGKILL`ed at step 150 (a real child process,
`returncode == -9`, no `finally`, no flush), resumed from the newest verified checkpoint at step
125, finished at 300. Measured
`max |delta param| = 0.000e+00`, and step, both Adam moment buffers, `lr_scheduler_state`,
data epoch/cursor/permutation and every `metrics/step_*.json` are equal. The stated tolerance is
therefore exact zero, not a bound.

That only means something because the falsifier is also tested: with the python and numpy RNG
states *dropped* from the checkpoint — which is upstream's real behaviour, since its checkpoint
carries only model/optimizer/scheduler/step (UPSTREAM_REPORT §3.1) and `train.py:170` reseeds on
every launch — the same experiment diverges by `7.012e-02`. So the spec's contingency ("if the
trajectories materially diverge, add Python, NumPy, CPU and CUDA RNG states") is confirmed as the
expected path, and the RNG state is in the checkpoint from the start.

One real bug found while building it, worth recording because it is the kind that survives a
casual resume test: restoring the numpy Generator and *then* constructing the data sampler drew a
permutation from the freshly restored Generator, forking the trajectory while every counter still
matched. Step, `opt.t`, `last_epoch`, epoch and cursor all agreed; only the weights differed.
Data position is now asserted separately from RNG restoration for exactly that reason.

Four mutations were run against the finished layer to check the tests can fail: disabling the
rollback in `resolve()` (4 failures), publishing the pointer before verification (1), keeping
three checkpoints instead of two (1), and skipping RNG restore (2). Each broke the intended test
and nothing else.

Retention, atomicity and the pointer are now owned by our layer, closing UPSTREAM_REPORT §3.5
items 1-4: `.partial` + fsync + rename for both checkpoints and pointers, hash *and* read-back
before a record enters the index, two-deep retention with the oldest deleted only after the newest
verifies, and `finalize()` clearing the stale `recovery_ckpt` that would otherwise hard-fail the
next resume at `core_train.py:148-150`.

## Representation extraction and the H1/H2 evaluation math — implemented and mutation-tested

`src/lurestar/representations.py`, `src/lurestar/evaluate.py`, `tests/test_representations.py`,
`docs/EXTRACTION.md`. 37 tests, all passing on the CPU-only host
(`.venv/bin/python -m pytest tests/test_representations.py` → `37 passed in 5.46s`).

The file is split at a hard line: **Layer A** is pure numpy (distances, centering, whitening,
shrinkage, margins, index resolution) and **Layer B** is the only torch-touching code, with
`import torch` inside the functions so the module imports cleanly here. `evaluate.py` is entirely
Layer A. Layer B encodes the forward-pass asymmetry and nothing else: GPT returns `(logits, h)`
(`model_gpt.py:290-291`), NextLat early-returns `(token_embeds, text_embd)` at
`model_nextlat.py:199-200` and never applies `lm_head`, so the caller applies it
(`model_nextlat.py:121`). The state is `transformer.norm(x)` — RMS-normalized, because
`bias: false` sends `LayerNorm.forward` to `F.rms_norm` (`model_base.py:823-830`).

The extraction correction is now frozen in code as `PSI_EXTRACTION_INDEX = 62` /
`BRANCH_MARGIN_INDEX = 63`, with `resolve_extraction_indices()` re-deriving both from a real
token batch and refusing a ragged or shifted one.

**The suite was mutation-tested rather than merely run.** Fourteen deliberately wrong
implementations were built and the suite had to kill each: swapped extraction indices, centered
cosine ignoring the pool mean, a leaked cross-fit, a too-narrow bootstrap, a reversed PSI sign, a
dropped whitening transform, a full-vocab branch margin, a removed leakage guard, a removed
shrinkage floor, a removed seeds-are-not-items guard, a removed ragged-batch check.

One mutant survived the first pass: **taking the centering mean from the scored pair instead of
the declared E_lure pool**. That is exactly the silent way to manufacture a PSI effect, and the
first version of the suite could not see it — the analytic geometry was built from mirrored item
pairs whose per-pair mean coincides with the pool mean. Fixed by
`test_psi_distances_center_on_the_DECLARED_pool_and_not_on_the_scored_pair`, which pins the
identity against the declared pool *and* asserts the pair-derived pool gives a materially
different answer. Both halves are needed; the first alone would pass whenever the two pools
happen to coincide. All fourteen mutants now die.

Inferential units are separated by argument type, not by convention: item-level functions take
arrays, and `model_contrast_seed_level` takes a `Mapping {seed_id: statistic}` and raises
`TypeError` on an array — so a 20,000-item array cannot be passed where three seeds belong. With
three seeds the exact two-sided sign-flip p has a floor of 0.25, which is reported alongside the
interval and every per-seed value.

## Lure-Star stimuli — generated, solver-verified, and one impossibility documented

`src/lurestar/generate.py` + `src/lurestar/validate.py` + `tests/test_lure_generator.py`.
48 tests, all passing (`.venv/bin/python -m pytest tests/test_lure_generator.py`, 28.9 s).

**Spec §5's matching requirement is not literally satisfiable, and the reason is structural.**
The critical swap must edit the goal arm's depth-k edge; the safe swap must not. Those are
different edges, one edge ordering gives distinct edges distinct slots, so no ordering makes
the two edits land on the same absolute token positions of a shared base string. The proposed
fix — one edge permutation per condition, chosen so both edits land at indices (i,j) — was
refuted rather than adopted: relocating the safe pair displaces whatever occupied (i,j), so
base→near-safe becomes a ten-token perturbation instead of a two-token one, which destroys the
premise of a *near* lure and does not even make the two edit distances equal without further
rejection sampling. Full argument, including the proof that no other two-token
multiset-preserving edit yields a valid star, in `docs/STIMULUS_DESIGN.md`.

What is enforced instead, both per item and both asserted on the written manifests:

- **LS-1 (primary).** `near_safe` and `near_critical` are each an exact two-token edit of the
  *same* base string, their slot pairs are disjoint with **identical gap**, and a fair coin
  decides which pair each condition gets — so edit position is exchangeable between conditions
  rather than confounded with them. This keeps the spec's PSI formula literally. It is weaker
  than the spec text on exactly one dimension, absolute within-item edit position, and
  §2 of `STIMULUS_DESIGN.md` says so in those words.
- **LS-2 (declared robustness check).** A sixth condition, `near_safe_aligned`, is the same
  safe swap applied to `repeat` — serialized so the safe pair sits exactly where the critical
  pair sits in `base`. Its edit is at the *identical absolute token positions* as
  `near_critical`'s. Anchor differs; position matches. LS-1 and LS-2 disagree on which nuisance
  to absorb, both are generated, and the primary was fixed here, before any model exists.

**Measured, on the shipped 2,000-quartet pool.** Every condition is 63 prompt tokens; both near
lures change exactly 2 tokens, always edge *head* tokens; base/repeat/near-safe/near-safe-aligned
answers identical; near-critical changes the first branch and the path while the goal *token* is
byte-identical (the goal node moves to the other arm — asserted directly, not assumed);
far-critical edge overlap with base is `{0: 288, 1: 820, 2: 892}` out of 20 edges (mean 1.28,
cap 2 by rejection, 72% first-try acceptance) against 18/20 for the near lures. Pools:
`E_lure` 2,000 quartets, `A_pair` 1,000 training items, `B_near` 5,000, `B_far` 15,000 candidates
— all in `manifests/` with `.sha256` sidecars and `stimuli_provenance.json`.

**A leakage positive control caught a real bug.** `TrainingIndex.build` canonicalized a graph by
sorting edge *strings* while `canonical_graph_key` sorted edge *tuples* — "10,5" < "2,3" as text,
the reverse as numbers. The two therefore never agreed, so the "no evaluation item is in the
training corpus" check would have passed vacuously on every input, including a training line
copied verbatim. The test that failed was the one asserting the index *does* flag a known
training item. Without that positive control the leakage guarantee would have been decorative.
Both sides now use the lexicographic form and the control is permanent.

Round-trip fidelity to upstream is asserted, not assumed: `graph_from_line` → `serialize` is
byte-identical on sampled lines of the real 200,000-line corpus, and our solver reproduces
upstream's recorded answers from the edge list alone.

**Correction, appended after adversarial review (`docs/review/durable-checkpoint.md`).** The
paragraph above originally read "the mandatory recovery test passes bitwise" without naming the
surrogate. Spec §9's mandatory recovery test, and spec §10's stop condition "interrupted training
cannot resume reproducibly enough for the stated analysis", are about the 12L/6H/384 model on the
real corpus. That test has **not** been run. `docs/FOUNDATIONS.md` D-23 conditions the
`--strategy ddp --devices 1` sampler decision on exactly this check ("Verify empirically with the
300 vs 150+150 test before trusting it"); it remains outstanding and is a blocker for the first
confirmatory run, not for the durable layer.

## Adversarial review of the durable-checkpoint track — five P0s, all fixed

`docs/review/durable-checkpoint.md`. All 35 previously reported tests reproduced, and the durable
primitive's own tests survived adversarial input (truncation, constant-length bit flips, hand-written
`.partial` files, a serializer that writes garbage, and a same-length pointer-target corruption). The
recovery test is falsifiable: `test_resume.py:183`'s `0 < resumed_from` is what would fail if the
resume silently restarted from scratch, since a scratch restart would otherwise land bit-identically
on the reference.

What the tests did not cover was the seam to the trainer that will actually run. Every runner test
drives a `FakeLauncher` that writes *through* `DurableCheckpointer`; production launches upstream
`train.py`, which writes through `fabric.save` (`models/model_base.py:417`) and knows nothing about
our index. Five P0s followed, each reproduced before it was fixed:

1. `resolve()` deleted `{out_dir}/recovery_ckpt` whenever its own index was empty — i.e. on every
   real job — destroying upstream's valid pointer (`core_train.py:968-974`). Now it clears the
   pointer only when it has records that all failed, or when the pointer is dangling (the case
   `core_train.py:148-150` hard-fails on).
2. Nothing adopted upstream-written checkpoints, so every real resume planned `init_from=scratch`
   on top of valid weights and no real job could ever reach `DONE`. Added
   `DurableCheckpointer.adopt()` / `adopt_existing()` — hash, deserialize, sidecar, index, no
   pruning of files we did not write — called from `MatrixRunner.plan()` and after each launch.
3. `build_matrix`'s default configs (`{model}_lurestar_base.yaml`) never existed, and near and far
   were handed the *same* config. Now `base -> configs/{model}_lurestar.yaml`,
   `adapt -> configs/adapt_{condition}.yaml`, and a missing config is a refusal.
4. The identity guard skipped any key recorded as `None`, which was `config_sha256` for every job.
   A run could have resumed under a different `train_batches` unnoticed. `None` is now a mismatch
   like any other, a missing config raises rather than being recorded, and the dataset manifests are
   wired into the default matrix.
5. The H3 branch command was arithmetically dead: `--checkpoint_path` restores `training_steps`
   (`model_base.py:437`) → `self.step` (`core_train.py:309`) → the loop returns at
   `core_train.py:569` with `train_batches=500` off a 20,000-step parent, i.e. **zero adaptation
   updates**, a trap `UPSTREAM_REPORT` §3.4 names verbatim and `tests/test_run_matrix.py:349`
   had pinned as the contract. The command now emits `parent_steps + adapt_steps` and refuses to
   emit at all without the parent's step count. Same command also lacked `use_nextlat=false`, so
   every GPT adaptation job would have trained a NextLat model (`configs/adapt_near.yaml:20`).

After the fixes: **45 tests pass** (35 original, 10 new regression tests, one rewritten because it
encoded P0-5). Seven P1s are recorded in the review and left open, the largest being that
`scripts/run_matrix.py` and `scripts/launch_train.sh` are two contradictory launch paths for the
same jobs — different `out_dir` layout, different experiment names, different `--strategy` default,
different branching mechanism. They must converge before the first confirmatory run.
