# Shared experiment design

## Three evidence tiers

The eventual objective comparison uses three explicitly separated tiers. The active phase precedes
that comparison and contains only the NextLat basin gate.

| Tier | Purpose | What is held fixed | Permitted claim |
| --- | --- | --- | --- |
| Paper-native reproduction | Verify that the implementation can recover its published behavior | The method's own data, architecture, optimizer, horizon, topology, and evaluation | Reproduction or implementation-gap claim for that method only |
| Common objective comparison | Isolate objective differences | Shared corpus, backbone where compatible, update budget, runtime, seed roster, checkpoints, and evaluator | Comparative trainability, geometry, or exact-state claim |
| Downstream conditional study | Ask representation or forgetting questions in a usable model | Competent parents selected by a rule frozen before training | Conditional claim given that the method reached the solver basis |

Do not merge tiers in one aggregate. In particular, BST's 8-million-example `N=50` setting,
FSP's 500-epoch `G(2,6)`/`G(2,8)` setting, and NextLat's fixed 200,000-example `N=100` setting are
native validations, not comparable samples.

## Seed policy and experimental unit

The training seed is the inferential unit. Thousands of prompts from one checkpoint improve
measurement precision but do not create thousands of independent model replicates.

- Freeze an ordered list of 24 NextLat basin seeds before training.
- Preserve the same ordered seeds for the conditional stabilization cohort if it is triggered.
- Report every attempted seed, including crashes, numerical failures, and shortcut solutions.
- Resume a mechanically interrupted run only from a hash-verified checkpoint; never restart until
  it succeeds and call that the same replicate.
- Never replace a completed incompetent seed in an unconditional cohort.
- A separate competent-parent stream may consume a predeclared ordered list until it obtains the
  declared number of parents. Report attempts, acceptance rate, and stopping point.
- Keep initialization seed, data-order seed, and sampler seed separately recorded even when they
  are intentionally equal.

## Power and sample size

Twenty-four seeds is the active budget-capped target. This phase estimates one method's solver
probability rather than a paired between-method effect.

For the one-sided solver-probability claim `H0: p <= 0.80`, `n=24` rejects at 23 or more successes.
The exact type-I error at `p=0.80` is 0.0331; power is 0.661 at `p=0.95` and 0.839 at `p=0.97`.
The original five seeds are therefore a reproduction audit, not an adequately powered population
claim. Even 11 trials, as used in a detailed Path-Star follow-up, was explicitly described by its
author as insufficient for significance.

If pilot variance differs materially from planning assumptions, update the prospective power
calculation without opening condition labels. Do not stop early for a favorable effect. A future
sequential design would require frozen alpha-spending and futility rules before the first outcome.

## Common statistical rules

Only the exact binomial solver-count analysis and interval-censored transition analysis are active
for the basin gate. The comparison, representation, forgetting, and exact-state rules below remain
prospective until that gate passes.

- Report estimates and 95% confidence intervals before p-values.
- Use paired seed-level contrasts whenever methods share seeds.
- Use exact binomial intervals/tests for solver counts.
- Use interval-censored survival models for transitions observed only at checkpoint boundaries.
- Use hierarchical models for prompt-level outcomes, with graph/process identity and model seed as
  random effects; cluster bootstrap at the seed level as a distribution-light robustness check.
- Treat each numbered study as a separate family. Use one primary endpoint per study and Holm
  correction for its planned secondary contrasts.
- Report unconditional method results first. Competent-only analyses are explicitly conditional
  and never presented as estimates of method reliability.
- Freeze transforms, covariance regularization, probe hyperparameters, checkpoint selection, and
  exclusion rules in code before evaluation.

Binary accuracy is summarized with a binomial or beta-binomial model as appropriate. Continuous
seed-level contrasts use paired permutation tests and a hierarchical estimate. The permutation
test is the confirmatory distribution-free test; the hierarchical model supplies effect sizes and
variance decomposition.

## Counterfactuals

| Question | Necessary counterfactual | Main threat it blocks |
| --- | --- | --- |
| Active NextLat basin gate | Same frozen procedure and evaluator across predeclared seeds; solver and shortcut outcomes both retained | Selecting favorable restarts or treating prompt count as independent training evidence |
| Later objective comparison | Same data, backbone, updates, runtime, and seed; different objective | Calling architecture or data-scale differences an objective effect |
| Future-sensitive representation | Same visible edit burden; future preserved versus changed | Mistaking lexical or structural distance for predictive distance |
| Controlled forgetting | Same overlap; future same versus conflicting, crossed with low/high overlap | Conflating input similarity with output conflict |
| Exact predictive states | Same next-token distribution but different full future, especially RRXOR | Reducing belief-state recovery to next-token decoding |

Additional negative controls are an untrained network, shuffled labels, serialization-only edits,
self patches, unrelated patch donors, and matched random patches. Positive controls must be defined
before data are opened and must test the evaluator rather than guarantee the scientific result.

## Stimulus construction and leakage control

All controlled stimuli are generated from exact solvers. Each materialization records generator
commit, parameters, RNG streams, corpus hash, split hash, and solver version.

For Path-Star pairs:

- enforce graph validity and a unique target path;
- match prompt length, output length, node multiset, degree profile, token counts, edit count, and
  edit positions;
- compute whether the next token, later path, and full future distribution changed;
- split by base graph before variant generation so siblings never cross train/test boundaries;
- publish failure and attrition counts for every matching stage.

For stochastic processes:

- sample from the published transition matrices and stationary initial distribution;
- group splits by history or predictive-state family;
- compute beliefs with an independent analytic implementation and cross-check against author
  artifacts before training;
- preserve cases with identical next-token distributions but different futures as a named RRXOR
  test set.

## Execution reproducibility

Path-Star is unusually optimization-sensitive. Each run therefore records:

- source and container digest;
- GPU model, count, topology, driver, CUDA, cuDNN, PyTorch, Lightning, and precision;
- compilation and deterministic-algorithm settings;
- all RNG seeds and per-epoch sampler hashes;
- resolved config, corpus hashes, command, checkpoint hashes, and evaluation receipts;
- step time, wall time, memory, loss, first-decision margin, and exact-path checkpoints.

The basin cohort uses one fixed verified RTX 4090 runtime, contingent on a profile within 20% of
the A100 anchor. Existing legacy reproducibility diagnostics already record the
multi-GPU/topology tests; this phase does not pay to repeat that factorial.

## Competence and missingness

A completed run can be a competent solver, a shortcut solution, a numerical failure, or a
mechanical interruption. Only the last category may resume without changing its scientific
identity. Missingness is summarized by method and reason.

Representation results are reported in two views:

1. unconditional, treating all completed seeds as the method's actual training population;
2. conditional, restricted to the predeclared competence gate and interpreted only as geometry
   given successful planner learning.

The forgetting study uses competent parents by construction and must report the cost and selection
probability required to obtain them.

## Freeze checklist

Before a confirmatory launch, the repository must contain a hash-bound manifest specifying:

- basin, paper-native, or common tier;
- source commits and licenses;
- exact data identities and split groups;
- seed list and RNG mapping;
- objective and architecture configuration;
- competence, interruption, and exclusion rules;
- checkpoint and stopping schedule;
- primary and secondary endpoints;
- statistical code and multiplicity family;
- profile-derived wall-time and cost cap;
- artifact retention and secret-handling rules.

Only then is a GPU launch scientifically authorized.
