# Run log

Append-only. Every entry is a fact with the command or artifact that established it.
Timestamps are local (America/Los_Angeles).

---

## Session 1 — Friday setup

**Environment.** macOS host, no local GPU, no local torch. Analysis venv at `.venv`
(python 3.12, numpy 2.5.2). GPU via Colab Pro+ `colab` CLI: 1788 compute units, H100 /
A100 / L4 / T4 / G4 available. Durable store: `gs://YOUR_PRIVATE_BUCKET`
(us-central1), created for this project.

**Upstream pinned.** `JaydenTeoh/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
("Initial public release", 2026-05-25), checked out read-only at `upstream/NextLat`.

**G(5,5) configs exist as the spec assumes.** `config/stargraph/5_5/{gpt,nextlat}_stargraph_5_5.yaml`
match the spec's expected values: `train_batches: 20000`, `effective_batch_size: 512`,
`n_layer: 12`, `n_head: 6`, `n_embd: 384`, `stargraph_max_nodes: 100`, AdamW at 5e-4 with
betas (0.9, 0.95), weight decay 0.1, grad clip 100, constant schedule (`warmup_iters: 0`,
`warmdown_iters: 0`). NextLat adds `mtp_horizon: 3`, `lambda_mse: 1.0`, `lambda_kl: 1.0`,
`proj_factor: 0.5`. The shipped sweep is five seeds `[1234..1238]`; all five are
preregistered confirmatory seeds.

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
pessimistic. After adding BST and extending all arms to five seeds, the full workload projects to
23.5 GPU-h and **125 compute units, 7.0% of the 1788 balance**, including the 20% interruption
margin.
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

---

## Adversarial review of the `representations` track — 2026-08-23

`docs/review/representations.md`. Reproduced the reported `37 passed`, verified every upstream
`file:line` anchor in `representations.py` and `EXTRACTION.md` against the pinned tree, and ran
**52 independently written mutants** in an isolated copy under the scratchpad. Thirty-seven died
untouched — including asymmetric centering, pairing destroyed by independent sorting, and a
seed-contrast that secretly tiles three seeds into 30,000 units. The estimator internals are
sound. Everything that survived lived at the API boundary or in Layer B.

**2 P0, both fixed.**

1. **The centering pool was named, not enforced.** `psi_distances_centered_cosine` took a bare
   `ndarray`. Passing the scored triple raised PSI 0.0783 → 0.1013 (+29%); passing seven rows of
   noise that share no row with any scored state dropped it to 0.00035 (−99.6%), reported as
   `centering_pool_n = 7`, with all 37 tests green. `EXTRACTION.md`'s "this is enforced, not
   merely documented" was false. The implementer killed this mutant for the *implementation* and
   left it open for the *caller*, which is the only place it can actually happen. Fixed with
   `representations.CenteringPool`: every one of the five spec §5 conditions must be supplied or
   explicitly `declared_missing` (recorded in the serialized metric), every scored state must be a
   row of the pool checked per argument, condition order is canonical. A bare array is a
   `TypeError`.
2. **Layer B had zero executable coverage** — the GPT/NextLat asymmetry that the whole track rests
   on. Deleting the architecture validation, or replacing the GPT branch with
   `lm_head(hidden)`, both left the suite green; the one test aimed at it passed only because
   `_torch()` raised first. Fixed by moving validation ahead of `_torch()` and adding a
   numpy-backed stub `torch` with `_FakeGPT`/`_FakeNextLat` built so the mutants are
   distinguishable (6-feature state, 5-token vocab, `token_embeds ≠ post_norm`, and a `lm_head`
   attribute deliberately different from the projection used inside `__call__`).

**2 P1 also fixed** (cheap, independent of the P0 work): the whitening leakage guard was defeated
by `np.asarray` casting a mixed id list to strings — a leaked integer id `3` vanished the moment
the batch also carried a string id — and "held out" was opt-in, so a whitener could score its own
fitting rows in silence.

**4 P1 left open**, largest first: `similarity_dependent_interference`'s shared-parent "assertion"
is algebraically tautological (replacing every `margin_before` with `1e9` changes nothing and
raises nothing) and the real invariant — one `parent_checkpoint_sha256` across both branches — is
not representable in its signature; spec §6/H3.1–5 and §10's post-adaptation drift and
erosion-predictor estimators do not exist yet. Eight P2 recorded.

Also: commit **`fb5d035`** contains a corrupted `representations.py` — the Ledoit-Wolf shrinkage
floor replaced by `pass`. A concurrent agent's `git add -A` captured an in-flight mutant of mine.
The working tree is correct; the commit is not. All later mutation work was moved to an isolated
tree copy.

Tests: 37 → **49** in `tests/test_representations.py` (twelve added, each with a discrimination
assertion; none weakened or removed). Full suite `369 passed, 5 skipped`.

## Required experiment B — HMM belief geometry, frozen before any HMM model exists

**The forward algorithm is validated against the definition, not against itself.**
`src/hmm_geometry/forward.py` implements the scaled (explicitly renormalised) forward recursion
rather than a log-space one, for a stated reason: the per-step normaliser *is*
`P(X_t | X_1:t-1)`, so the posteriors, the exact next-observation distribution and the conditional
log-likelihoods all fall out of one pass; and because the carried vector sums to 1 after every
step, underflow is structurally impossible rather than merely improbable. A log-space
implementation (`log_forward_batch`) is kept as an independent second opinion and the tests hold
the two against each other at length 64 and at length 600.
`tests/test_hmm_forward.py::test_brute_force_agreement` compares every posterior, predictive prior,
next-observation distribution and conditional log-likelihood against explicit enumeration over all
`4**L` hidden-state paths for `L = 1..6`, on a deliberately asymmetric HMM, plus every length-4
sequence under a deterministic-emission HMM. Two negative controls check the comparison could
fail (a transposed transition matrix and a shifted observation stream must both disagree).
21 tests, all passing.

**The HMM was chosen by a model-blind grid search and frozen.** 2,592 candidates, 79 passed all
eleven acceptance intervals (spec §12's four criteria plus two degeneracy guards), selection by
maximum worst-case normalised slack with a lexicographic tie-break. The same candidate wins at both
pilot sizes tried (1,500 and 4,000 sequences), which is the only robustness check the selection rule
gets. Frozen in `manifests/hmm_matrices.json`, `hmm_sha256 = 83d24e7f...`:

| diagnostic | value | acceptance interval |
|---|---|---|
| mean state dwell time | 3.401 | [1.8, 4.0] (random 4-state chain = 1.333) |
| self-transitions | 0.66 – 0.74 | [0.35, 0.80] |
| states with `P(o|s) >= 0.10` per symbol | 3 | >= 2 |
| posterior entropy p05 / p50 / p95 (bits) | 0.434 / 1.204 / 1.762 | p05 < 0.5, p95 > 1.3, spread > 0.9 |
| Bayes next-obs accuracy | 0.4134 | [0.35, 0.80] |
| best-constant-predictor chance | 0.2697 | — |
| Bayes − chance | 0.1438 | >= 0.10 |
| stationary distribution | 0.218 / 0.237 / 0.259 / 0.286 | min >= 0.15 |

Corpus generated from the frozen matrices: 100,000 × 32 train, 10,000 × 32 validation,
10,000 × 64 length-generalisation, seeded by `SeedSequence(5963).spawn(3)` so a split does not move
if another split's size changes. Exact posteriors and exact next-observation distributions are
stored for the two evaluation splits (`manifests/hmm_dataset.json` carries every SHA-256); the
training split's posteriors are recomputable exactly and are not written.

**Two threshold-design errors were found and corrected before any model existed.** Both were
caught by looking at the yield per prefix length, not by a test.
1. An *absolute* high-edit-distance cut, fitted by pooling all prefix lengths, lands at 17. No
   prefix shorter than 17 symbols can reach it, so the bank silently contained no equivalent pairs
   below length 17, and the length-64 pool cleared the same bar with far less surface change than
   the length-32 pool — confounding exactly the length-generalisation comparison that pool exists
   for.
2. A single *rate* cut fails in the opposite direction: the Levenshtein rate between independent
   strings falls with length, so a pooled 0.737 asks for a 75th-percentile pair at length 12 and a
   90th-percentile pair at length 32 (yield collapsed to 263 and 54 pairs).
   The frozen rule is now a **per-length quantile table** — the 75th percentile of Levenshtein
   distance among calibration pairs *of that same prefix length* — fitted on a calibration half of
   each band. `PREFIX_MIN` was also raised from 8 to 16: at length 8 a 4-symbol alphabet forces
   prefix collisions between any two pools, and filtering them would delete the most probable
   histories rather than fix anything. At 16 the measured collision count between the calibration
   and test pools is **0**.

**Pair bank, built by applying the frozen thresholds unchanged** (`manifests/hmm_eval_pairs.jsonl`,
8,409 pairs; thresholds `manifests/hmm_thresholds.json`, sha256 `23881606...`):

| set | n | posterior JS (bits) | edit distance | edit rate |
|---|---|---|---|---|
| predictively equivalent | 2,218 | 0.0002 mean, max 0.0006 | 27.9 mean | 0.694 |
| predictively divergent near-lure | 3,973 | 0.378 mean, min 0.300 | 1.64 mean | 0.054 |
| history-distance-matched control | 2,218 | 0.605 mean, min 0.376 | 27.9 mean (matched pair-for-pair) | 0.694 |

Retuning is prevented structurally, not by discipline: `fit_thresholds` sees only the calibration
pools and takes no yield target; `freeze_thresholds` raises `ThresholdMismatch` rather than
overwrite; `load_thresholds` re-verifies the payload hash and the quantile-rule text and is the
only function that sets `verified=True`; and `build_bank` refuses unverified thresholds, refuses a
different HMM hash, and refuses to run on a calibration pool. `tests/test_hmm_pairs.py` verifies the
shipped artifacts by recomputing every posterior from the frozen matrices and every edit distance
from the raw symbols, and includes a negative control that builds a bank on permuted posteriors and
shows the verification rejects it.

**Deviation to record:** near-lure pairs are *constructed* by substituting one or two symbols in a
held-out prefix, not found in the pool. Two independent length-16 prefixes over four symbols are
within Levenshtein distance 2 with probability about `2.5e-7`, so a 5,000-sequence pool yields a
handful at whatever divergence they happen to have. This mirrors the Lure-Star quartets, both
members are genuine in-support sequences under the frozen HMM, and both posteriors are exact.

**Integration plan for the trainer:** `docs/HMM_DATASET_PLAN.md`. The whole integration is one dict
insertion (`DATAMODULES["hmm_belief"] = ...`) from a shim outside `upstream/`, plus a new datamodule
file; no shared training code is edited. `context_length = 0` is the load-bearing config value — it
disables Path-Star's prompt masking at `models/model_gpt.py:362` and `models/model_nextlat.py:441`,
giving next-token loss on every position. The competence gate is not an accuracy threshold but the
exactly known optimum: **1.2563 nats/token**, against 1.3845 for the best constant predictor. A run
below the optimum is a bug, not a result.

## Adversarial review of the Lure-Star generator — three P0 holes in the checker, closed

`docs/review/lure-generator.md`. The 48 reported tests reproduced (12.26 s), and the whole
shipped pool regenerated from its recorded seed at `--workers 4` to four **byte-identical**
sha256s, so reproducibility and worker-independence are confirmed rather than asserted. The
LS-1 exchangeability claim was tested rather than taken on trust: 2x20 contingency test on the
critical vs safe edit-slot marginals over 2,000 quartets gives chi2=22.59, p=0.256. The
impossibility proof in `STIMULUS_DESIGN.md` §2 is correct.

**What broke: `check_quartet` certified three kinds of record it should reject.**

1. **Mismatched suffix depths.** A quartet whose safe swap is depth-1 and whose critical swap
   is depth-3 passed with zero problems. Depth is the magnitude of the manipulation (three
   nodes moved per arm vs one) but *both* are two-token prompt edits, so every token-level
   assertion in the suite passes on an unmatched pair. Closed as invariant **LS-0**: the depth
   is re-derived from the anchor's solved arms and `record["depth"]` is cross-checked against
   it, never trusted.
2. **A near lure relabelled as the far control.** `far_critical` with 18/20 edge overlap
   passed: the checker only compared recorded-vs-recomputed overlap and had no cap, because
   `far_max_edge_overlap` lived only in the generator. H3's near-minus-far contrast would have
   been measured against a near lure. Closed: the cap is a checker parameter, threaded from
   `QuartetConfig`.
3. **A falsified `graph_key`.** `check_quartet` never recomputed the stored identities, and
   both the CLI gate and the leakage test read `graph_key` off the record — so a verbatim
   training line carrying a bogus key read as clean. This is the same failure *mechanism* as
   the `TrainingIndex` bug already fixed here: that fix removed the symptom, not the pattern of
   consulting a self-reported field. Closed: `check_quartet` recomputes `graph_key`,
   `prompt_sha256` and `answer` from the line, and the gate is now
   `generate.leaked_quartet_ids()`, which keys off the line.

Six regression tests added, each mutation-tested against a checker with only its own fix
removed; all six fail without it. No fix touches the RNG stream, and a full post-fix
regeneration reproduces all four manifest hashes exactly. `54 passed`; whole repo `389 passed`.

**Left open, deliberately, with reasons in the review.** (a) Spec §6 requires the near and far
H3 branches to match on *target-path distribution*; measured, they do not — `B_near` reproduces
the parent's target token at path positions 2/3 with probability 0.343/0.670 versus 0.049/0.026
for `B_far`, which biases `erosion_near - erosion_far` against the hypothesis. Unmet and, unlike
LS-1, not declared. (b) `near_safe_aligned` — the LS-2 robustness condition — cannot be consumed:
`representations.CENTERING_POOL_CONDITIONS` omits it and `from_conditions` raises on it, so one
sixth of the manifest is currently un-analysable. (c) E_lure base edge orders are not drawn from
the corpus's uniform-shuffle distribution: the pinned edit slots are U-shaped (chi2 p=2.9e-35 vs
uniform), a consequence of drawing the slot gap uniformly. It does not bias PSI (see the
exchangeability test above) but it does shift absolute competence on E_lure relative to the
held-out corpus, which matters for spec §10's 90% gate.

## 2026-08-23 — Colab CLI rate and identity calibration

After two agreeing `no_runtime` status reads, provisioned one L4 and one A100 sequentially for
identity/rate calibration only. No trainer or confirmatory metric ran. L4 resolved to `NVIDIA L4`,
PyTorch `2.11.0+cu128`, CUDA 12.8, BF16 true, with the account reporting 1.54 CU/h. A100 resolved
to `NVIDIA A100-SXM4-40GB`, 39.49 GiB, the same PyTorch/CUDA build, BF16 true, with the account
reporting 5.3 CU/h. Each session was stopped and followed by two agreeing `no_runtime` reads plus
a settled zero burn rate. Exact sessions and balances are in `results/compute_log.csv`.

This clears rate calibration and A100 identity/BF16 only. It does **not** clear paid training:
the real-trainer atomic/two-deep checkpoint patch, real T4 forced-interruption equivalence test,
step-0 integrity assertions, BST/HMM profiles, HMM smoke, and peak-VRAM measurement remain gates.

## 2026-08-23 — Immutable HMM and manifest staging

Uploaded the five frozen HMM arrays to `gs://YOUR_PRIVATE_BUCKET/lurestar/corpus/hmm/`
and the frozen Lure-Star/HMM manifests plus `manifest_inventory.sha256` to the durable manifest
prefix. Every object was streamed back through host-side `gcloud storage cat` and matched its
local SHA-256; no Colab runtime was active. The runtime driver now downloads these objects through
the Python storage client and runs `sha256sum -c manifests/manifest_inventory.sha256` before any
trainer launch. Future adaptation-bank materialization refreshes this inventory atomically and
must be uploaded/verified before the confirmatory matrix gate can open.

## 2026-08-23 — T4 recovery-gate attempts and circuit breaker

Two paid T4 recovery rehearsals were stopped before any scientifically useful training result was
claimed. Both sessions were explicitly torn down and followed by two agreeing `no_runtime` reads;
the settled account balance remained 1786.482212634042 CU (billing may still settle later). The
append-only event records, full failure tails, source hashes, fixed non-confirmatory seed 910241,
and session identities are retained in `results/recovery_gate_receipts.jsonl`.

1. `rg-62f25753f208-1787540074039870000-c3656d0b` reached the verified T4/CUDA environment but
   failed before training because the runtime storage client did not receive an explicit GCP
   project. The driver and recovery harness now bind `project-flash-490419` explicitly.
2. `rg-3bbc405317a2-1787540260217435000-d362144e` passed authentication, source setup, data
   generation, runtime patching, and the real trainer's step-0 contract. It then failed on the
   first optimizer step because Lightning Fabric cannot gradient-clip through PyTorch fused AdamW
   while an FP16 GradScaler is active on T4. Runtime patch v2 keeps the frozen FP16 and
   `grad_clip=100` settings, selecting non-fused AdamW only when Fabric exposes a live GradScaler;
   the BF16 A100 path remains fused.

The two-strike circuit breaker is active: no third runtime may be provisioned until runtime patch
v2 compiles against the pinned source, its FP16/BF16 selection tests pass, the recovery harness
tests pass, and a read-only audit accepts the fix. A successful next rehearsal must still prove
clean-300 versus kill-at-150/resume-to-300 equivalence and durable GCS recovery; reaching step 1 is
not itself a pass.

## 2026-08-23 — First real T4 training progress; forced-kill gate remains closed

Gate `rg-322403f1b51f-1787541480267918000-3dbe0c0d` established that runtime patch v3 fixes the
T4 FP16 failure: the real trainer passed optimizer step 1 with `grad_clip=100`, completed an
independent clean 300-step reference, and durably committed its final checkpoint, metadata, and
state-last record under the gate's GCS prefix. Verified recovery checkpoints were also produced at
steps 50, 100, and 150. The run cost 0.161765 CU; post-stop balance was
1786.3204473108256 CU with two agreeing no-runtime reads and zero burn.

The gate is **not passed**. At the deliberate step-150 interruption, `SIGKILL` terminated the
Fabric launcher/TCP store but rank 0 escaped its process group and continued as an orphan. That
orphan advanced beyond step 200 and rotated away the step-100 generation while the snapshot code
was reading it, correctly preventing `state.json` from being committed last. The host then
manually released the runtime rather than spend further compute on an invalid comparison. GCS
therefore contains a useful clean reference and an explicitly incomplete resume prefix, never a
false success marker.

The live run also showed that two separately initialized T4 FP16 lineages diverge before any
interruption, so comparing their final weights would conflate ordinary CUDA nondeterminism with
resume error. The preregistered tolerances will not be loosened. Before another paid attempt, the
harness must (1) kill and verify the entire descendant tree including escaped process groups, and
(2) compare uninterrupted and recovered continuations from a common hash-identical lineage. The
transport circuit breaker is active; no further Colab runtime starts until both changes have local
tests and read-only review.

## 2026-08-24 — Full recovery path works; nondeterministic CUDA replay fails frozen tolerance

Gate `rg-4c510af4abde-1787543838297723000-7a85f244` closed the process-control and durable-restore
questions left by the preceding rehearsal. The harness killed and verified the complete trainer
descendant tree at step 150, resumed a reference continuation from that exact checkpoint, deleted
the local recovery lineage, restored the checkpoint and metadata from immutable GCS objects, and
resumed a second continuation. Both arms reached exactly step 300 and committed checkpoint,
metadata, and state-last records under `final/reference/` and `final/recovered/`. The owned T4
session was released; two status reads reported no runtime and quota reported zero active runtimes
and zero burn. Cost was 0.174685 CU.

The gate is still **not passed**. Lightning's manual-optimization CSVLogger writes constant
`step=0`, which exposed a comparison-harness assumption after both final commits. Offline analysis
of the hash-verified durable artifacts corrected the row identity to `(logger version, row ordinal)`
without changing any preregistered tolerance. That analysis then exposed genuine replay divergence:
model maximum absolute delta `0.0265127`, optimizer maximum absolute delta `0.0147252`, and metric
loss maximum absolute delta `0.0345252`, all beyond the frozen limits. Scheduler and FP16 scaler
states were exact; the restore path itself was exercised successfully.

Runtime patch v4 now adds an opt-in recovery-only deterministic CUDA contract before model work:
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch algorithms, cuDNN deterministic mode with
benchmarking disabled, TF32 disabled, and an explicit runtime receipt. Frozen science tolerances
remain unchanged. No A100 profile or confirmatory run may start until a new common-parent T4 replay
passes this gate and the deterministic patch receives read-only review.

## 2026-08-24 — Deterministic T4 replay is exact; independent audit clears A100

Gate `rg-d21e8fee468a-1787545664418856000-9899dbda` exercised the common-parent interruption,
immutable GCS delete/restore, and two step-150-to-300 continuations under deterministic CUDA.
Both final checkpoints have SHA-256
`9b7f2d2edec3d4045ce963a4deb0179ca7f6c662090eb7a35825ec1ca38e7c04`; weights, optimizer,
scheduler, RNG, FP16 GradScaler, logits, and metrics are exact with maximum absolute and relative
difference zero. Progress and durability checks pass.

The immutable gate result remains `passed:false`: the host retained only the last 300 process
lines, so normal per-step output evicted the early `Fast forwarded data to step 150` observation.
Pinned source and the restored step-150 lineage prove the continuation cannot reach step 300
without that fast-forward path. An independent read-only audit therefore issued a separate
posthoc engineering clearance for A100 profiling without changing a tolerance, seed, scientific
result, or the original gate artifact. The diagnostic buffer is now 5,000 lines with regression
coverage; a streaming marker latch remains a future robustness improvement. Cost was 0.166697 CU,
followed by two no-runtime reads and zero active burn.

## 2026-08-24 — First A100 profile attempt fails safe on wrapper import path

Profile `a100-d44f883e387f-8d316efc9c53` provisioned session
`gpu-a100-s-kkb-usc1f2-e7mlqhd70xv3` and verified an NVIDIA A100-SXM4-40GB, PyTorch
2.11.0+cu128, CUDA 12.8, and BF16 support against the frozen 24-object input inventory. Both
50-step HMM GPT and NextLat smoke runs completed and their recovery/final checkpoints were synced.
The measured Lure-Star jobs then failed before training because Fabric/torchrun executed
`scripts/profile_entry.py`, and `runpy.run_path` did not expose the upstream trainer directory for
the sibling `core_train` import. This is an engineering failure, not a scientific result.

The driver committed a generation-3 failure state, terminal receipt, and 45 immutable
SHA-addressed artifacts to GCS, released the owned A100, and two subsequent checks reported no
runtime, zero active runtimes, and zero burn. The attempt cost 0.216229 CU. The wrapper now
prepends the absolute target-trainer directory to `sys.path`; the external sibling-import
regression and the combined profile suites pass 40/40 before rerun.

## 2026-08-24 — Corrected A100 profile reaches BST step 364 before runtime loss

Profile `a100-b15e1bc9d596-8d316efc9c53` verified the corrected distributed wrapper on the real
A100 path. Both HMM 50-step smoke jobs completed. The measured 500-step Lure-Star GPT and NextLat
jobs completed with probes, checkpoints, and observed rates near 8.2 and 7.3 steps/s. The
47.3M-parameter BST arm ran near 2.2 s/step, wrote recovery checkpoints at steps 125 and 250, and
reached step 364. The Colab execution stream then ended with EOF and the runtime itself
disappeared; this was not a trainer exception. Two subsequent status/quota reads reported no
runtime, zero active runtimes, and zero burn. Cost was 1.402500 CU.

The last committed state is generation 2 with 29 smoke artifacts. Periodic sync correctly refused
to commit a mixed generation when live GPU telemetry changed during hashing, but that meant the
state-last pointer did not reference later stable checkpoints. Failed sync attempts nevertheless
uploaded immutable SHA-addressed objects, including the complete GPT/NextLat profile artifacts and
BST step-125/250 checkpoints. These objects are retained for a fail-closed salvage/resume audit;
they are not yet promoted into a successful profile state. No replacement A100 is authorized until
cross-session profile resume is behaviorally proven and binds unchanged trainer/config/input
identity across the orchestration-only source change.

The next-run synchronizer now snapshots mutable logs/CSV files privately, defers only files that
race, carries forward prior immutable versions, and still commits stable checkpoints state-last;
terminal sync refuses any deferral. The full repository suite after this hardening is 763 passed.

## 2026-08-24 — Recovered A100 profile and strengthened-design budget reconciliation

The fail-closed salvage/resume path completed nonconfirmatory engineering profile
`a100-be81d2f1e79c-8d316efc9c53` at durable state generation 12 with 108 artifacts. The local
recovery receipt restored and verified 34 artifacts, retrained no model, and used no additional
paid compute. It binds `results/profile_summary_recovered.json` at SHA-256
`ccccea44a5f4ce321b21827a2c8276ac23d906fd538c2e9893f78696b72d0d1a` and the rendered summary at
SHA-256 `4ba74d9c2772d0e7a62de2526c9edafc5c4551ee93a6199c709a2d6d5999b84b`.
This evidence is engineering-only: no confirmatory checkpoint or H1/H2/H3/HMM geometry outcome was
inspected or produced.

The recovered projection predates the 2026-08-24 strengthened design. It gives 66.0096047653
GPU-h for the already-correct 15 base jobs, 3.3215172269 GPU-h for 30 near/far adaptation
branches, and 0.1219404929 GPU-h for 10 one-regime HMM cells, including the profiler's measured
checkpoint treatment but before its 20% interruption margin. Scaling only the changed counts to
45 near/mid/far branches and exactly 30 three-regime HMM cells gives:

```text
66.0096047653 + 3.3215172269*(45/30) + 0.1219404929*(30/10)
  = 71.3577020844 GPU-h subtotal
  = 85.6292425012 GPU-h with 20% interruption margin
  = 453.834985256 CU at the measured 5.3 CU/h
```

The operational planning values are therefore **85.629 GPU-h / 453.835 CU**, correcting the stale
23.5 GPU-h / ~125 CU entry above without rewriting append-only history. They are not a wall-clock
promise. They exclude a currently unknown strengthened-evaluation delta for held-out whitening and
dual-metric extraction, H1/H2 cross-fitting, H3 gradient/actual-update/Jacobian controls, and full
three-regime evaluation/cache work. That path must be target-profiled; no endpoint, threshold,
branch, regime, or seed may be changed in response. The recovered BST base timing dominates the
projection, and no unmeasured BST optimization is budget authority.

The outcome-blind preregistration validator exists and fails closed. Its current receipt at
`.agent_state/preregistration-freeze-block.json` reports `all_eleven_gates_pass: false`: the
evidence index is absent, so all eleven gates block. Its authority hashes became stale during this
documentation reconciliation and must be regenerated against the final immutable source bundle.
The last recorded full-suite state is also `BLOCK` (827 passed, 10 runtime-bootstrap integration
failures); independent review and confirmatory GO clearance are pending. Gate 4 has an additional
explicit blocker: no authoritative pilot architecture/seed/step/checkpoint selection is frozen,
and `B_mid` plus matched acquisition candidate generators were absent at this snapshot. The
pre-outcome A100 BST step-500 seed-1234 checkpoint
`1f5f00611e33ada0ac0a778f9d45bef9e174f1bbeedfaaa3491018a9bf400176` was **not** silently chosen.

Two agreeing Colab status/quota checks during this reconciliation reported no runtime. The latest,
at 2026-08-24 03:29 EDT, returned `status: no_runtime`, `active_runtimes: 0`,
`burn_rate_hourly: 0`, and settled `paid_balance: 1783.25765136152`. No runtime is active and no
compute units are currently burning. Confirmatory launch remains locked until the all-eleven PASS
receipt, clean full-suite receipt, independent review, immutable source bundle, and GO clearance
all exist.

## 2026-08-24 — Focused preregistration verification remains fail-closed

After the documentation reconciliation, the focused command
`.venv/bin/python -m pytest tests/test_validate_preregistration.py
tests/test_create_confirmatory_clearance.py tests/test_colab_train_loop.py -q` returned 110 passed
and 2 failed. Both failures are nominal PASS-fixture tests, and both block only at gate 8 because
the validator now requires a `subject` payload key for `hmm_family_manifest` and
`hmm_materialization_receipt` while the fixture omits it. This is a live validator/test-schema
regression, not evidence that the scientific gate passed. The validator remains fail-closed; no
runtime was launched and no threshold or endpoint was changed.

## 2026-08-24 — Exact three-regime HMM family materialized locally

After the HMM-focused suite passed 102 tests, the model-blind create-only materializer generated
the exact three frozen regimes locally. It created 31 planned scientific artifacts: corpora,
posterior arrays, future-distribution-JS thresholds, pair banks, matrices, and manifests. The
family payload is `e24a883dbc547c99f543433d290b0e228e2191427af041a851a1279d77a68b27`;
the materialization inventory is
`079eeaa283a4d91eb4512726195fdc08a291e2020bb97507c8d8380ebc32e8d2`. The receipt records three
required regimes, 2,000 target pairs, no model inputs, and no inspected model outcomes.

A second create-only invocation completed without changing the frozen receipt or inventory, and
`sha256sum -c manifests/hmm_family_inventory.sha256` verified every one of the 31 rows. The local
footprint is about 307 MB of arrays plus 22 MB of family manifests/pair banks. This clears local
HMM materialization only. The combined runtime inventory must still be refreshed after H3 banks
are frozen, the exact paths must be uploaded and reverified on the runtime, and no confirmatory
training is authorized by this CPU-only step.

## 2026-08-24 — Semantic validator and recomputed launch clearance pass focused tests

Every gate 2–11 JSON artifact now requires an exact source-archive-bound attestation envelope,
canonical finite-JSON payload hash, producer/source/test bindings, and role-specific semantic
fields. Raw HMM family/materialization subjects retain their native schemas and are wrapped with
hash-checked attestations. Placeholder schema-only JSON and mutations to status, source hash,
payload hash, H3 controls, AdamW batch identity, TE certificates, test counts, and review findings
all fail closed.

The launch clearance was then tightened so a stored PASS is never trusted on shape alone. Before
any Colab status, quota, upload, or provision action, it loads the exact bound validator source,
recomputes all eleven gates from the live evidence tree, and requires exact logical equality with
the stored freeze receipt. A structurally plausible fabricated PASS has a direct regression test.
The adjacent validator/clearance/Colab suite passes 118 tests; py_compile and `git diff --check`
also pass. No real PASS or GO receipt exists yet, and no external lifecycle action occurred.

## 2026-08-24 — Frozen H3 pilot scoring completed; fixed caliper is infeasible

The sole prospectively frozen nuisance-selection pilot (BST seed 1234, exact step 500, checkpoint
SHA-256 `1f5f00611e33ada0ac0a778f9d45bef9e174f1bbeedfaaa3491018a9bf400176`) scored all
53,000 model-blind candidate rows on one A100. The scientific score job SHA-256 was
`860aa43623ec07c1e1bf97d0bcd77629b08a1ed3853e3af45640c733d13e0feb`; the immutable
source archive SHA-256 was `724f1a4f4fef5f08247495b1b6763420ad336418abf339c50ca12d08dcc995a1`
at GCS generation `1787558668940035`. Scoring completed in 126.6 seconds. Every 1,000-row chunk
was published create-only as data, scorer receipt, and a generation/SHA-bound commit record before
the next chunk was credited. The final loss table SHA-256 is
`a562057ead0852cb2a5dd5e68f3e50b34d9f299e1cabb707b6a269f37bbc7f13`; its receipt SHA-256
is `9086a322a9ff08985b56d4230eb3739f5734776e9dcf2c5d0300462cb8352908`; complete state was
published last at generation `1787558930759654`.

The selector then applied D39 unchanged and failed closed: the absolute pilot-loss caliper 0.1
left 1,115 of 5,000 near items without an eligible middle candidate. No threshold was widened, no
candidate was regenerated, and no confirmatory checkpoint or outcome was inspected. This is an
outcome-blind feasibility result. H3 remains blocked pending an explicit prospective scientific
amendment or a decision to drop the affected hypothesis; independent agents were assigned to
evaluate defensible remedies and p-hacking risk. The A100 was stopped immediately. Two agreeing
post-stop reads reported no runtime, zero burn, and a settled balance of 1783.0722580985762 CU;
the measured debit was 0.185393 CU.

## 2026-08-24 — Prospective D40 one-shot middle-support expansion frozen

Before scoring any new candidate, Decision D40 froze a single uniform model-blind repair for the
D39 overlap failure. Each of 5,000 near items retains its original three middle candidates and
adds exactly nine unique candidates in each 1/2/3-rewire stratum: 30 per near, 150,000 total, and
135,000 new-only rows. Deterministic CPU generation reproduced create-or-verify outputs with no
paid compute: expanded SHA-256
`2effd4e13d384786546c71cc61b4138dc97f082e3992bf3cdf398e6bf93264f1`, new-only SHA-256
`4a8e906bd868c1f751ab32e0af6ad9652cf706e320b366875a21eac63d45df0c`, and generation receipt
SHA-256 `6c24eee439fcd8de02cd5d487267c408734680aefe197e33ab662b1efb75d909`.

The scorer reuses the exact D39 scientific loss implementation and sole checkpoint, recomputes
serialized identities, and is wrapped in generation-bound GCS durability with a frozen 1,000-row
maximum in-flight chunk. Combination requires the scoring receipt and state-last durable receipt;
selection requires their exact 188,000-row combined lineage. The integrated finalizer produces the
D40 middle mapping plus unchanged D39 far and three acquisition selections without invoking the
known-infeasible D39 middle selector. If any middle pair remains unmatched, it writes a permanent
H3 block and prohibits further amendments. Focused verification passed 62 tests; no GPU was
launched and no confirmatory input or outcome was inspected during this freeze.

## 2026-08-24 — D40 completed once; frozen rule permanently withdrew H3

The immutable D40 job
`393c933e9e616cd24a4b7a9b408203b0c22002c39cf97f2d72b03176fe45482a` scored all 135,000 new
model-blind rows using the frozen BST seed-1234 step-500 checkpoint. The first A100 disappeared
after 124,000 committed rows; a replacement restored all 124 generation-pinned chunks and finished
the same job without rescoring them. The new-row loss SHA-256 is
`f84c73b81d7b9e8cab44e32d89cd272d320d420583bd7badf76f3c0dade7f537`, scoring-receipt SHA-256 is
`3e487999752871048c6bc71bcf181f048b3d4c897b03a853420a06770960f621`, durable-state SHA-256 is
`e1ed1d814ea190b1602c31ef82bee86bcd0937dc26dec5963a31d692f8faa0c2`, and state was committed
last at generation `1787563059069047`. Combining D39 and D40 produced 188,000 rows with loss-table
SHA-256 `814058a162e12fde36c7204dd30798b63bfbf02294fce768046070672e5afece`.

The unchanged selector left 4/5,000 near identities unmatched. The create-only terminal receipt
`manifests/h3_selected/PERMANENT_H3_BLOCK.json` has SHA-256
`82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778` and records
`no_further_amendments_permitted: true`. This executes the prospectively frozen stopping rule:
H3 is permanently absent from confirmatory training and inference; caliper changes, weighting,
unmatched restriction, further candidates, pilot substitution, and another matching amendment are
forbidden. H1, H2, and the exactly three-regime HMM remain confirmatory. The four unmatched pairs
are a nuisance-bank feasibility result, not a confirmatory model or interference outcome.

The lost and completion runtimes debited 1.569460 and 0.410562 CU respectively, for **1.980022 CU**
total D40 spend. Two agreeing post-stop reads reported no runtime and zero burn at settled balance
`1781.092236633952` CU. Every completed scoring artifact survived the disconnection.

## 2026-08-24 — Outcome-blind HMM launch-verifier schema repair

Before HMM compute, the family launch verifier was found to expect a top-level `hmm_sha256` in
`hmm_thresholds.json`, while the already-frozen schema stores that identity under
`thresholds.hmm_sha256`. The verifier was corrected to read the existing nested field. No HMM
matrix, threshold, pair bank, corpus, manifest byte, or model outcome changed.

The exact family preflight
`.venv/bin/python scripts/run_hmm_matrix.py --root . --snapshot-root . --data-root . --project-root . --upstream upstream/NextLat --family --print-plan`
completed with schema `nextlat_forgetting/hmm_matrix_plan/1` and exactly 30 jobs, from
`gpt-seed1234-hmm-persistent_moderate` through
`nextlat-seed1238-hmm-persistent_high_aliasing`. This print-plan remains mandatory before paid HMM
compute.

## 2026-08-24 — H3 naming collision resolved without changing the HMM schema

Independent review found that `src/hmm_geometry/aggregate.py` uses legacy metric keys beginning
`h3_posterior_*` and `h3_future_*`. Those keys name prespecified HMM Bayesian posterior- and
future-distribution-decoding diagnostics; they are not Lure-Star adaptation/interference H3.
The permanent D40 block therefore excludes only the Lure-Star adaptation branches, interference
estimand, and gradient/shadow-update/Jacobian mechanism probes. It does not exclude these HMM
calibration outputs. The existing metric names remain stable before outcomes; no schema key,
scientific estimator, artifact byte, or outcome was changed by this documentation clarification.

## 2026-08-24 — First HMM interruption exposed and repaired a ledger/state split

The first confirmatory HMM A100 runtime completed and durably committed all ten
`persistent_moderate` checkpoints at exactly step 3,000 under source
`a962cdb94c865e16c2c7c86d5c18b9cc2d3bd301feeea12e42075751f52c9285`, then disconnected.
The replacement runtime restored the checkpoint objects but trusted a regressed global ledger and
attempted to resume completed GPT jobs, producing invalid step-3,001 incident artifacts. The host
stopped that runtime immediately after detecting the mismatch. Two settled status/quota pairs
reported no runtime, zero burn, and balance `1779.3814353793016` CU.

All ten original step-3,000 checkpoints, sidecars, final summaries, older step-2,500/2,750 recovery
generations, and telemetry remained recoverable. The five GPT terminal states and their original
materialized configs were recovered byte-for-byte from exact GCS soft-delete generations. Current
live state once again reports ten `TRAINED` jobs at step 3,000. Before restoration, every replaced
live object was copied to a content-addressed incident archive; every restored byte sequence was
hash-checked and copied to a separate content-addressed recovery archive. The outcome-blind plans
and receipts are `.agent_state/hmm-soft-delete-recovery-plan.json`,
`.agent_state/hmm-soft-delete-recovery-receipt.json`, and
`.agent_state/hmm-exact-target-state-repair.json`. No HMM evaluation metric was opened or used.

Decision D41 freezes an operational-only successor source: exact-target checkpoints terminalize
without launching the trainer; over-target checkpoints fail closed; recovery can accept only the
one clearance-bound predecessor source; new state is stamped with the successor source; and HMM
evaluation representation-cache chunks now cross GCS state-last so an evaluation disconnect loses
at most one sync interval. Scientific data, models, seeds, steps, estimands, thresholds, endpoints,
and the exactly thirty-cell aggregation are unchanged. A new full suite, independent review,
eleven-gate freeze, and job-specific clearance are required before another paid launch.
