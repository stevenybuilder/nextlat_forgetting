# D47 — NextLat seed-1235 reproducibility and execution-parity diagnostic

**Date:** 2026-08-25  
**Role:** nonconfirmatory diagnosis after the frozen competence result  
**Primary-study effect:** none; this decision adds no seed and replaces no checkpoint

## Observed result

The preregistered one-GPU NextLat seed-1235 run completed all 20,000 scratch updates but failed
the frozen held-out competence gate: 3,663/20,000 exact paths (18.315%). Training loss fell while
held-out middle-path tokens stayed near chance. NextLat seed 1234, trained under the same local
configuration and runtime, reached 19,991/20,000 (99.955%). Both checkpoints and receipts remain
immutable experimental outputs.

No new confirmatory seed is added. No diagnostic checkpoint may replace seed 1235 in H1, H2,
H1-BD-1, CFS-2, or any confirmatory aggregate.

## Design problem identified

This project correctly trains fresh weights from the official NextLat source and recipe; it does
not use an unpublished author checkpoint. Fresh initialization is required to test the learning
procedure across seeds. However, the project overstated execution parity.

The official G(5,5) NextLat and GPT launchers use two GPUs under DDP, and the official BST launcher
uses eight. The observed seed-1235 run used one RTX 3090 Ti, PyTorch 2.11.0+cu128, BF16 mixed
precision, and `compile: false`. The project preserved the global effective batch of 512 and the
20,000 optimizer updates, but that does not make one-GPU and multi-GPU execution identical. DDP
changes per-rank computation, gradient-reduction arithmetic, and kernel execution. On a task that
the upstream authors explicitly describe as numerically sensitive, those small changes can alter
which optimization trajectory a nominal seed follows.

The official source commit, architecture, objective, corpus construction, optimizer, learning
rate, global batch, and update count are matched. The paper does not publish per-seed Path-Star
traces, checkpoint weights, or an exact PyTorch/CUDA environment. The strongest current hypothesis
is therefore execution-sensitive optimization, with world size and runtime stack as concrete
unresolved variables. This is a testable hypothesis, not an established cause.

## Staged diagnostic

Run these only after costing them and without interrupting already-running production cells.

1. **Same-runtime replay.** Start seed 1235 from scratch in a new diagnostic namespace on the same
   one-GPU software stack. Compare the validation trace at updates 1,000–4,000 with the immutable
   original. This tests whether the failed early trajectory repeats or whether low-level
   nondeterminism changes the outcome.
2. **Author-launch parity test.** Start seed 1235 from scratch with two same-host GPUs and DDP,
   keeping source, corpus, model, optimizer, BF16 precision, and `compile: false` fixed so world
   size is the first isolated variable. Compare updates 1,000–4,000. Continue to 20,000 only if
   the early result is ambiguous or a final competence result is worth the separately approved
   compute.
3. **Request missing provenance.** Before claiming exact author parity, obtain or explicitly note
   the absence of the authors' Path-Star per-seed metrics, checkpoint hashes, PyTorch/CUDA versions,
   GPU allocation, and actual compile setting. The public paper and launcher do not fully specify
   this execution state.
4. **Interpret without substitution.** A repeatable one-GPU failure plus two-GPU success supports
   world-size execution sensitivity. A one-GPU replay success supports numerical nondeterminism.
   Failure in both environments leaves an unresolved author-runtime difference. None of these
   outcomes changes the original seed-1235 record.

Every diagnostic records GPU model/count, driver, PyTorch, CUDA, cuDNN, Lightning, precision,
compile flag, upstream commit, corpus hashes, materialized configuration, commands, checkpoint
hashes, and per-evaluation-step metrics. It may not inspect H1/H2 geometry or CFS-2 outcomes.

## Scientific and workshop consequence

The one-GPU five-seed cohort is an environment-defined independent reproduction, not an exact
replay of the authors' private training execution. The seed-1235 failure is a trainability and
generalization result. Unless the frozen five NextLat/BST pairs meet their competence contract,
the original five-pair confirmatory H1/H2 classification is incomplete; competent-subset geometry
is qualified analysis. HMM is logically separate. CFS-2 may not silently substitute diagnostic
weights for an incompetent frozen parent.

For a workshop submission, report this boundary directly: the measurement and causal questions
remain legitimate, while base-training portability across execution environments becomes a
limitation and potentially a reproducibility finding. The diagnostic may explain that boundary;
it cannot repair the confirmatory cohort.

## Completed diagnosis (2026-08-25)

The diagnostic no longer supports the earlier single-factor explanation that one GPU versus two-
GPU DDP caused seed 1235 to fail. The failure is an optimization bifurcation into a shortcut basin,
and several small execution-path changes can select a different basin.

### Failure phenotype

The pinned public source, recommended `compile: false`, two-GPU DDP launcher, frozen corpus, and
seed 1235 completed all 20,000 updates in the current environment. Frozen greedy evaluation was
5/20,000 exact paths (0.025%). Token accuracies were 99.99%, 6.18%, 6.535%, 7.165%, and 99.985%.
Thus the run learned the easy boundary tokens while the three path-dependent interior tokens stayed
near chance. The low training loss and chance held-out interior tokens identify memorization plus the
Path-Star shortcut, not a timeout, evaluator error, or merely slower convergence.

### Factor isolation

All early-stop cells below used seed 1235 and were evaluated at update 2,000:

| Source/runtime path | Sampler/topology | Compile | Exact-path accuracy |
|---|---|---:|---:|
| clean public source | one-device random sampler | false | 85.50% |
| clean public source | one-device distributed sampler | false | 0.03% |
| project runtime patch | one-device distributed sampler, GPU 0 | false | 96.36% |
| project runtime patch | one-device distributed sampler, GPU 1 | false | 95.99% |
| clean public source | two-GPU DDP | false | 0.06% |
| clean source plus shared distributed-sampler seed | two-GPU DDP | false | 0.04% |
| clean public source | two-GPU DDP | true | 11.82% |
| clean source plus deterministic CUDA algorithms | two-GPU DDP | false | 0.03% |

The shared-sampler cell corrected a real launcher problem: the public code calls
`seed_everything(base_seed + global_rank)`, while the installed Lightning version also uses
`PL_GLOBAL_SEED` as the `DistributedSampler` seed. The ranks therefore do not shard one common
permutation. Correcting only that behavior did not rescue seed 1235, so it is not the root cause.

The public `compile: true` path changed the trajectory materially but did not reproduce the fast
high-accuracy transition by update 2,000. This result is consistent with the upstream README's own
warning that compilation is inconsistent on Path-Star. It is evidence of sensitivity, not a valid
checkpoint substitution.

### First-step parity result

The most controlled clean-versus-patched one-device distributed-sampler trace used byte-identical
initial parameters, the identical first batch, and identical CPU and CUDA RNG states. Seventy-eight
of 79 parameter gradients were byte-identical. Only `token_embedding.weight` differed, at about
`6e-8` maximum scale. PyTorch documents CUDA `nn.Embedding` backward as normally nondeterministic
and provides a deterministic alternative under `torch.use_deterministic_algorithms(True)`.

That microscopic embedding-gradient difference is a concrete first divergence between a failing
and a succeeding trajectory. It is not, however, a sufficient single-factor explanation: forcing
deterministic CUDA algorithms in the clean two-GPU run still produced 0.03% at update 2,000.

### Root-cause statement

Seed 1235 is not intrinsically a defective seed, and no single GPU, DDP, sampler, timeout, corpus,
or evaluator defect explains the result. In the current implementation, NextLat Path-Star training
has at least two reachable solution regimes: a generalizing path-solving regime and a
memorization/shortcut regime. Seed 1235 lies close enough to their optimization boundary that data
order, sampler construction, compiler kernels, and microscopic floating-point differences can
change which regime is reached. The original frozen run entered the shortcut regime.

The exact private execution path that produced the paper's five successful seeds cannot be
reconstructed from the public artifacts because the authors did not publish their package lock,
per-seed traces, or checkpoints, and the public config's `compile: true` conflicts with the README's
Path-Star recommendation to disable compilation. Therefore the defensible conclusion is
execution-sensitive optimization and a public-reproduction gap—not a claim that DDP alone, the
dataset generator, or seed 1235 alone caused the failure.

## Sampler-factorial continuation (2026-08-25)

The follow-up used a fresh same-host pair of RTX 3090 GPUs, the untouched pinned upstream commit,
PyTorch 2.11.0+cu128, Lightning 2.6.5, BF16 mixed precision, and `compile: false`.  An external
diagnostic wrapper supplied a shared distributed-sampler seed independently of the model seed and
recorded every per-rank sampler permutation.  All cells ran 2,100 requested updates and were
evaluated at update 2,000.

| Initialization seed | Sampler seed | Recorded epoch behavior | Repeat | Exact-path accuracy |
|---:|---:|---|---|---:|
| 1235 | 1235 | Fabric-advanced epochs | control | 0.02% |
| 1235 | 1235 | Fabric-advanced epochs | A | 0.03% |
| 1234 | 1234 | Fabric-advanced epochs | — | 0.04% |
| 1234 | 1235 | Fabric-advanced epochs | — | 0.02% |
| 1235 | 1234 | Fabric-advanced epochs | — | 0.02% |
| 1235 | 1235 | Fabric-advanced epochs | B | 9.34% |

The cell named `fixed-i1235-d1235` was intended to repeat epoch zero, but its trace showed epochs
0, 1, 2, ... and the same permutation hashes as the explicit-reshuffle cells.  Lightning Fabric's
configured dataloader advances the sampler epoch even though `core_train.py` does not call
`set_epoch()` directly.  The earlier project claim that DDP repeats one fixed permutation every
epoch is therefore false in this installed runtime.  The historical cell name is retained for
provenance; it does not describe the observed behavior.

The four initialization-by-order combinations all remained near chance.  Thus neither seed 1235
alone, sampler seed 1235 alone, nor their simple interaction explains the failed public DDP path in
this runtime.

### Exact-repeat localization

The two nominally identical 1235/1235 repeats had byte-identical initial parameters, identical
per-rank first batches, and identical sampler-permutation traces, yet reached 0.03% and 9.34% at
update 2,000.  Per-parameter first-update traces localized the initial divergence to exactly one of
79 gradients: `token_embedding.weight`.  All other gradients were byte-identical on both ranks.
The embedding-gradient sums differed by approximately `4.56e-7`; their L2 norms differed by
approximately `5.62e-9`.

Repeating the two-update comparison under `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` made all 79 gradients byte-identical.  This confirms that CUDA
embedding-backward nondeterminism is the initiating source of same-condition run-to-run divergence
in this environment.  It is not a sufficient explanation for failure versus success: the earlier
deterministic 2,000-update DDP cell still achieved only 0.03%.

### Revised causal boundary

The diagnosis now separates two levels:

1. **Within-environment trajectory variation:** initiated by nondeterministic CUDA embedding
   backward and amplified by the Path-Star optimization dynamics.
2. **Public-reproduction failure:** not explained by the tested initialization seeds, sampler
   seeds, sampler epoch advance, rank-seed correction, topology alone, compilation alone, or
   deterministic CUDA alone.  The public recipe remains strongly biased toward the shortcut basin
   in the frozen environment, while the successful private execution path remains unavailable.

The sampler remains scientifically important, and the rank-dependent sampler-seed interaction is a
real defect, but neither is the single root cause.  The strongest supported statement is now:
**CUDA embedding nondeterminism explains why nominally identical runs diverge; unstable Path-Star
optimization explains why that microscopic divergence becomes behaviorally large; missing private
execution provenance prevents attribution of the paper-versus-public reproduction gap.**

Local evidence is retained under `results/d47_sampler_factorial/`,
`results/d47_first_gradient_repeat/`, and `results/d47_first_gradient_deterministic/`.
