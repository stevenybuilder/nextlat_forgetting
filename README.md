# Predictive objectives, planning, representation, and forgetting

This repository asks how training objectives that predict more than the next token change what a
transformer learns. It uses controlled planning and stochastic-process tasks because success,
future equivalence, and the relevant counterfactuals can be computed exactly.

The completed first release is a retrospective case study of two NextLat Path-Star trajectories.
One selected run crosses from 0% to 99.015% exact-path accuracy between retained updates 1,000 and
3,000; the other plateaus near 18% and becomes increasingly confident in the wrong branch even as
its training loss falls. This is a useful optimization case study, not an estimate of NextLat's
population success probability.

## The four studies

1. **Reliable planner training:** first establish and characterize a NextLat procedure that
   reliably reaches the generalizing Path-Star basin; cross-objective comparison comes later.
2. **Future-sensitive representation:** test whether internal distances track changes to the
   future rather than superficial changes to the serialized graph.
3. **Controlled forgetting:** test whether structural overlap is especially destructive when new
   training assigns a conflicting future to a familiar decision state.
4. **Exact predictive-state geometry:** test how closely learned representations recover exact
   belief states on Mess3, RRXOR, and controlled grid processes.

The literature ladder includes standard next-token prediction, multi-token prediction,
joint-token prediction, the Belief State Transformer, NextLat, future-summary prediction, and
Hierarchical Latent Prediction. None of that comparison is active yet. The previously proposed
24-seed reliability cohort is deferred on cost grounds. The retrospective evaluation performed no
new training, cost approximately $0.382, and is complete.

## Start here

- [Basin case study](docs/BASIN_CASE_STUDY.md) reports the completed two-trajectory evaluation,
  controls, results, and claim boundary.
- [Blog post](docs/BLOG_POST.md) is the concise research-facing version of the result.
- [Future-sensitive representation pilot](docs/FUTURE_SENSITIVE_GEOMETRY_PILOT.md) specifies the
  recommended low-compute next experiment, directly testing study 2 with no new model training.
- [Research plan](docs/RESEARCH_PLAN.md) defines the hypotheses and claim boundaries.
- [Experiment design](docs/EXPERIMENT_DESIGN.md) defines stimuli, counterfactuals, seed policy,
  power, and statistical analysis.
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) distinguishes reusable code from missing
  work and gives the gated execution order.
- [Compute budget](docs/COMPUTE_BUDGET.md) defines the active $5 evaluation-only cap and records
  the larger seed-cohort estimate as deferred work.
- [Literature ledger](docs/LITERATURE.md) records the paper, repository, and author-source anchors
  adopted by the design.
- [Current status](docs/CURRENT_STATUS.md) says what is implemented, historical, or not yet run.

## Repository layout

| Path | Role |
| --- | --- |
| `src/` | Existing Path-Star, predictive-geometry, HMM, and controlled-forgetting components |
| `scripts/` | Data, training, evaluation, profiling, and recovery entry points |
| `configs/` | Materialized configurations; current files belong to the legacy replication phase |
| `manifests/` | Dataset identities, construction receipts, and run provenance |
| `results/` | Compact tracked evidence, diagnostics, and the case-study results release |
| `tests/` | Scientific invariants, evaluator checks, and recovery tests |
| `docs/` | Active research design and implementation roadmap |
| `upstream/NextLat/` | Read-only NextLat checkout pinned at commit `3770be6009cea2b3c455a9ce7f2ca88b504bb955` |

Large datasets, checkpoints, credentials, local environments, and runtime state are excluded from
Git. Never commit `.env`, `.secrets/`, provider credentials, private bucket configuration, model
weights, or raw recovery bundles.

## Reproducibility boundary

The original five NextLat seeds remain an unconditional reproduction cohort. A failed seed is not
replaced after the fact. Separately, later studies may train a predeclared sequence of additional
seeds to obtain competent parent models; that creates a new conditional population and cannot
rewrite the original five-seed result. See the concise
[reproducibility finding](docs/REPRODUCIBILITY_FINDING.md).

No further GPU launch is implied by this repository. Freeze a new study manifest, pass its
implementation and profile gates, and obtain compute authorization before starting the proposed
future-sensitive representation pilot or any confirmatory matrix.
