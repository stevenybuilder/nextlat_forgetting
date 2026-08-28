# Path-Star reproducibility finding

## What happened

In the original fixed five-seed NextLat cohort, seed 1234 reached 99.955% exact-path accuracy while
seed 1235 completed 20,000 one-GPU updates at 18.315%, near the Path-Star shortcut level. A later
run through the clean public two-GPU launcher reached 5/20,000 exact paths. Both learned the easy
boundary tokens while the three path-dependent interior tokens remained near chance.

This is a completed training outcome, not a timeout or an evaluator failure.

## What the diagnostics ruled out

Changing one GPU to two GPUs did not by itself recover the paper's behavior. Neither did fixing
the rank-dependent sampler seed, changing initialization and sampler seeds factorially, enabling
compilation, or forcing deterministic CUDA algorithms. Some one-device runtime paths reached high
accuracy quickly, while nominally similar public paths remained near chance.

Two nominally identical two-GPU runs had identical initial parameters, first batches, and recorded
sampler permutations but diverged behaviorally. The first update differed in exactly one of 79
parameter gradients: `token_embedding.weight`, at sub-micro scale. Deterministic CUDA settings made
those first gradients identical, but deterministic training still did not guarantee the
generalizing solution.

## Supported conclusion

CUDA embedding-backward nondeterminism explains one source of same-condition trajectory
divergence. Unstable Path-Star optimization amplifies microscopic numerical and execution changes
into a choice between two basins: a generalizing planner and a memorizing/shortcut solution.

The broader public-versus-paper reproduction gap remains unresolved. GPU count, sampler behavior,
compiler kernels, and determinism each affect the trajectory, but none singly explains it. The
paper did not release per-seed checkpoints, full package locks, or the private execution path, so
an exact replay is not currently possible.

This does not show that seed 1235 is intrinsically bad, and it does not show that NextLat cannot
solve Path-Star. It shows that the public training procedure is not robustly reproducible across
the execution paths tested here.

## Consequences for the new studies

- Report the original five-seed cohort unconditionally. Never replace seed 1235 with a successful
  diagnostic run.
- Make solver probability and time to the generalizing basis the primary outcomes of Study 1.
- Freeze a single-GPU common runtime so topology is not a between-method confound.
- For representation results, show both the unconditional method population and the explicitly
  conditional competent-checkpoint population.
- Build forgetting parents from a separately predeclared ordered seed stream; report how many
  attempts were required and do not use that cohort to rewrite the original replication.
- Treat per-seed traces, sampler hashes, and exact interior-token accuracies as required evidence,
  not optional debugging output.

The complete legacy diagnostic remains recoverable from public Git history with:

```bash
git show 5c71e5f:docs/DECISION_D47_NEXTLAT_SEED1235_REPRODUCIBILITY_DIAGNOSTIC.md
```
