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

**The mandatory recovery test passes bitwise.** 300 steps uninterrupted; then the same 300-step
job, `SIGKILL`ed at step 150 (a real child process, `returncode == -9`, no `finally`, no flush),
resumed from the newest verified checkpoint at step 125, finished at 300. Measured
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
