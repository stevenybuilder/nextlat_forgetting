# Implementation plan

## What can be reused

The repository is not starting from zero. The reset changes the scientific contract more than the
low-level infrastructure.

| Existing asset | Reuse | Required change before promotion |
| --- | --- | --- |
| Pinned `upstream/NextLat` checkout | GPT, MTP, JTP, BST, NextLat models and native Path-Star configs | Add project adapters, provenance tests, common-runtime configs, and evaluator parity |
| `src/lurestar/` and Path-Star scripts | Exact solving, dataset variants, evidence extraction, competence evaluation | Replace legacy pair taxonomy with the four matched future-edit classes |
| `src/hmm_geometry/` | Belief updates, HMM training/evaluation scaffolding | Match the official Mess3/RRXOR architecture, optimizer, process parameters, and grouped analysis |
| `src/cfs2/` | Balanced intervention generation, branch evaluation, activation patching | Express the design as structural overlap by future conflict; remove legacy study identifiers from active outputs |
| Profiling and durable-run scripts | Wall time, memory, checkpointing, recovery | Generalize to all objectives and make billing-wall-time the budgeting field |
| Tests | Many generator, hash, evaluator, and recovery invariants | Add method parity, matching, grouped-split, power, and statistical golden tests |
| Compact manifests/results | Provenance and the reproducibility finding | Keep legacy namespace; never mix with new study registries |

Existing implementation is not automatically part of the new protocol. Promotion requires a test,
a frozen config, and a manifest that names the new study and evidence tier. Superseded narrative
records remain available from Git history rather than an active in-tree archive.

## Missing components

### Objective adapters (blocked until the basin gate passes)

- Later, add MTP and JTP common Path-Star configs directly derived from
  `upstream/NextLat/config/stargraph/5_5/` and verify loss masks and horizon 3.
- Add a common model/evaluator interface that exposes backbone states, objective-specific latents,
  logits, parameter counts, and resolved loss terms.
- Implement FSP-BoW from the paper: one auxiliary head predicts the bag of all future target tokens
  while the autoregressive head remains unchanged. Reproduce `G(2,6)` and `G(2,8)` before use.
- Implement HiLP's lower latent predictor, sliding-window higher latent, higher-level predictor,
  and combined next-token head from the August 2026 algorithm and hyperparameters. Because there is
  no public task code pinned here, label this a port and validate every loss path independently.
- Keep objective heads out of greedy inference when the source paper removes them at inference.

### Active planner-basin study

- Use the frozen roster of all 20 non-recovery checkpoints from the selected generalizing and
  shortcut runs; do not train or substitute runs.
- Preserve the historical exact-path evaluator byte-for-byte and use a separate case-study
  evaluator for per-position accuracy and the teacher-forced first branch decision.
- Bind every evaluation to checkpoint, corpus, config, evaluator, and upstream hashes.
- Use actual checkpoint steps and interval bounds because the two historical save schedules differ.
- Report held-out intervals only as conditional measurement precision, never as uncertainty over
  seeds or solver frequency.

### Representation study

- Build a constraint-based variant generator for serialization-only, future-preserving,
  immediate future-changing, and delayed future-changing edits.
- Add an independent exact-solver audit and machine-readable matching diagnostics.
- Split by base graph before generating variants.
- Fit whitening only on the training split and store the covariance regularizer in the manifest.
- Extract final pre-normalization backbone states for every objective; expose HiLP's higher latent as
  a diagnostic channel only.
- Implement paired and hierarchical analysis from a synthetic golden dataset before real outcomes.

### Forgetting study

- Materialize the full two-by-two intervention bank with exact token and acquisition matching.
- Add a parent-lineage manifest tying every branch to one competent checkpoint and baseline receipt.
- Use full-parameter adaptation as the primary path; place LoRA in a separate namespace.
- Evaluate parent margin and exact path before, during, and after new-task training.
- Retain activation patching at fixed layers with self, unrelated-donor, and norm-matched random
  controls; hash-bind every patch artifact to parent, adapted checkpoint, and stimulus set.

### Exact-state study

- Verify the official Mess3 and RRXOR process outputs and saved-checkpoint analyses first.
- Add the exact four-layer, width-64, one-head, ReLU TransformerLens-compatible baseline.
- Implement each objective without changing context, online sampling, optimizer, or update count.
- Add analytic Bayes loss and exact belief-vector oracles independent of the training dataloader.
- Add grouped history/state-family splits, affine probe cross-validation, shuffled-label controls,
  same-next-token/different-future subsets, and horizon-1 through horizon-8 prediction metrics.
- Add controlled grid walkers only after the two inherited processes reproduce.

## Gated execution order

### Gate 0: design freeze

Deliver the active documents, machine-readable schemas, statistical golden tests, and secret scan.
No scientific GPU training.

### Gate 1: deterministic evaluation parity

Run the scientific-contract tests, verify all frozen hashes, capture one full-host 4090 runtime,
and complete one checkpoint/evaluator round trip. No training smoke test is needed because training
is outside the active question.

### Gate 2: retrospective trajectory

Evaluate every frozen checkpoint on the common held-out corpus, repeat one final checkpoint, and
transfer the compact receipts before destroying the instance. Stop on any identity or determinism
failure rather than changing the runtime.

### Gate 3: analysis and release

Plot exact-path, per-position, first-decision, and saved-loss trajectories. Report the transition
only to the resolution of the checkpoint schedule. Produce a concise report and blog post, run
secret and reproducibility checks, and publish the repository.

### Gate 4: deferred program decision

The 24-seed reliability cohort, stabilization cohort, competent-parent selection, and all four
larger studies require a new decision and budget. They do not follow automatically from this case
study.

### Gate 5: controlled forgetting (future tranche)

No adaptation matrix launches under the $5 case-study budget. If a later reliability study passes and a later
geometry study justifies the forgetting question, select only through a newly frozen
competent-parent rule, begin with 16 paired parents, and obtain separate authorization. A failed
acquisition match stops interpretation rather than triggering post-hoc stream tuning.

### Gate 6: exact predictive states (checkpoint reproduction only)

Reproduce official saved-checkpoint figures and exact-oracle controls without new large-scale
training. The 1,000,000-update objective matrix and grid walkers require a later budget and are not
part of the active execution plan.

## Suggested active namespaces

New work should use descriptive names rather than legacy decision numbers:

```text
configs/studies/planner_training/
configs/studies/future_sensitive_geometry/
configs/studies/controlled_forgetting/
configs/studies/exact_predictive_states/
manifests/studies/<study>/<freeze_id>/
results/studies/<study>/<freeze_id>/
scripts/studies/<study>/
```

Do not rename the remaining legacy source/result paths until their imports and hashes are migrated
with tests. Their directory READMEs and descriptive new-study namespaces mark the authority boundary.

## Definition of done

An experiment is complete only when it has:

- a frozen, hash-bound design and seed registry;
- a validated implementation provenance label;
- complete terminal receipts including failures;
- immutable checkpoint and dataset hashes;
- preregistered aggregation produced without manual row edits;
- confidence intervals, counterfactual controls, and multiplicity handling;
- a result table that separates unconditional, conditional, native, and common-comparison claims;
- enough metadata for another researcher to rerun it without private runtime knowledge.

A trained checkpoint alone is not a completed experiment.
