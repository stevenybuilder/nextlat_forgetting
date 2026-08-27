# PROGRAM.md — the autonomous loop's constitution

This file governs what the autonomous agents may change while the human is away. It is the
analog of the `program.md` in Karpathy's autoresearch harness, with one deliberate inversion:
**autoresearch lets the agent mutate `train.py` freely and keeps whatever lowers val_bpb.
Here, the training configuration is frozen and hillclimbing a scientific metric is forbidden.**

The project is a preregistered confirmatory study (spec §6, §10). An agent that tuned anything
toward a better PSI, a larger interference gap, or a stronger belief-geometry correlation would
be manufacturing the result, not measuring it.

## Frozen surface — never modified by any agent, for any reason

From spec §8 and §11:

- model width, depth, heads, `n_embd`, `proj_factor`, `mtp_horizon`
- optimizer, learning rate, schedule, betas, weight decay, gradient clipping
- `lambda_mse`, `lambda_kl`
- effective batch size (512 Lure-Star / 256 HMM) and the optimizer-update count
- base dataset size (200,000 train / 20,000 heldout graphs; 100,000 HMM sequences)
- base training steps (20,000 Lure-Star / 3,000 HMM)
- the five confirmatory seeds: 1234, 1235, 1236, 1237, 1238
- graph topology `G(5,5)`, `stargraph_max_nodes: 100`
- the surviving confirmatory hypotheses H1/H2 and the HMM calibration, and the two co-primary distances: centered cosine and held-out
  whitened Mahalanobis, scored on identical frozen `E_score` items under the binding
  intersection-union rules in the 2026-08-24 amendment; nPSI is diagnostic and cannot rescue a
  failed co-primary result
- the HMM centering population and evaluator hash ratified pre-compute in
  `docs/DECISION_D38_HMM_CENTERING_AMENDMENT.md`
- legacy H1 at h62 remains unchanged; the separately numbered, prospectively
  declared H1-BD-1 analysis at h63 is governed by
  `docs/DECISION_D46_H1_BRANCH_DECISION_ANALYSIS.md` and cannot rescue or relabel H1
- exactly three model-blind HMM regimes and exactly 30 HMM cells (3 regimes × 5 seeds ×
  GPT/NextLat), including their family selection rule, `TE` gates, regime weights, corpus seeds,
  pair banks, and future-JS/edit-distance thresholds
- the 15 Lure-Star base jobs (GPT/NextLat/BST × five seeds); H3's prospectively frozen D40
  stopping rule has fired, so the confirmatory matrix contains zero adaptation branches
- every surviving threshold and membership frozen into a manifest: `E_white`/`E_score` and HMM
  future-JS/edit-distance cuts; the H3 permanent-block receipt and its no-further-amendment rule
- CFS-1 is an immutable blocked design record. Its exact 18/18/8/7 overlap table is
  construct-confounded and no CFS-1 branch may launch. CFS-2 is the separately numbered
  outcome-blind successor; only its exact 18/18/8/8 construction may enter future causal
  forgetting execution, after its own runner/evaluator clearance. CFS-2 completion requires a
  patching artifact for every branch at frozen blocks 3/7/10 with matching-parent,
  unrelated-donor, norm-matched-random, and self-patch controls

The one permitted execution fallback is gradient accumulation that preserves effective batch
size *and* update count. It must be recorded as a deviation in `docs/FOUNDATIONS.md`.

## Mutable surface — where iteration is legitimate and encouraged

- data generation and serialization throughput, provided output stays hash-identical
- dataloader workers, prefetch, pinning, and checkpoint I/O scheduling
- durable-checkpoint and resume machinery, the run ledger, and recovery logic
- GCS transfer batching and region placement
- analysis, plotting, and bootstrap code, provided the estimator and its inferential unit
  do not change
- the writeup in `report/` and the docs in `docs/`
- test coverage
- outcome-blind external-validity protocol work, provided it is a separately numbered study and
  cannot change, rescue, or relabel any core Path-Star/HMM/CFS result

## Loop invariants

1. **Append-only ledger.** Every run writes to `results/run_ledger.json` and
   `results/metrics.jsonl`. Nothing is ever rewritten in place. A wrong entry is corrected by
   appending a superseding entry with a reason.
2. **Fixed budget per iteration.** Profile before sweeping (spec §11). No job launches
   without a measured seconds-per-step and a projected compute-unit cost.
3. **Durability before speed.** No required artifact may exist only on a Colab runtime.
   A step that is not on GCS did not happen.
4. **Report all preregistered metrics.** Including the nulls. Spec §10 is explicit: report
   every preregistered metric even if only one is positive. Dropping a null is fabrication.
5. **Stop rather than shrink.** If the hardware cannot run the exact configuration, the loop
   pauses for a compute decision. It never quietly reduces the confirmatory model.
6. **Every claim carries its rung.** R0 speculation through R4 replicated-and-controlled, as
   defined in the `unslop` agent. The writeup states the rung.
7. **Fail-closed scientific clearance at source freeze.** A new scientific source bundle is not
   provisioned until its hash-bound preregistration checks, impacted tests, independent review,
   and immutable clearance pass. Once that bundle is running, do not repeat a repository-wide
   gate for every checkpoint, reconnect, evaluator, or provider restart; verify only the exact
   job identity, required artifacts, and affected component.
8. **No metric or regime shopping.** H1 and the equal-regime HMM primary pass only when both
   co-primary metrics pass. Regimes are averaged equally inside seed; seeds, not items or regimes,
   are the inferential unit. Emit every null and invalid cell.
9. **No silent pilot choice.** A pre-amendment nonconfirmatory checkpoint is not automatically the
   model-blind pilot. Pilot architecture, seed, step, checkpoint SHA, evaluator, and selection rule
   must be explicitly frozen before it can select or loss-match an H3 bank.
10. **No Lure-Star H3 resurrection.** D40 left 4/5,000 pairs unmatched under the unchanged eligibility rule.
    The create-only permanent-block receipt is terminal: no caliper, weighting, restriction,
    candidate, pilot, matching amendment, adaptation job, interference estimand, or mechanism probe
    may be added. Legacy HMM `h3_posterior_*`/`h3_future_*` keys remain required calibration
    diagnostics and must not be renamed or removed before outcomes.
11. **Exact HMM family preflight.** Before paid HMM compute, the 30-job `--family --print-plan`
    must pass against the frozen inventory. Matrix identity is read from the frozen
    `thresholds.hmm_sha256` field; no validator may rewrite an artifact to fit its expectation.
12. **Construct validity is not a unit-test result.** Every stimulus family needs a
    human-readable, model-outcome-blind audit of the balance quantities that identify its stated
    construct. A mechanically valid generator can still be scientifically confounded, as CFS-1
    demonstrates.
13. **No external-validity laundering.** Path-Star and HMM support controlled symbolic and oracle
    calibration claims, not general language-model claims. Any language extension is separately
    numbered, preregistered before its outcomes, reports nulls, and cannot rescue the core study.
14. **Language work is outside the current milestone.** TS-1 and NL-1 remain frozen future-study
    records, but no agent may download their corpora, profile them, provision a worker for them, or
    launch them without explicit authorization for a new milestone. Deferral does not authorize a
    smaller or outcome-adaptive substitute.
15. **CFS-2 patching is mandatory and outcome-blind.** After each CFS-2 branch is complete, run
    `scripts/run_cfs2_patching.py` against its exact parent and retention manifest. Retain all
    three frozen layers and named controls. Missing cells remain incomplete; do not choose layers,
    donors, controls, or exclusions from observed effects. This does not resurrect Lure-Star H3.

## Minimal operational loop

1. At worker start, verify the frozen source/input identity and the worker's exact job allowlist.
2. Resume only from a checkpoint whose path, step, payload, sidecar, and SHA-256 agree.
3. Train to the next useful recovery point; do not checkpoint or run a full test suite per batch.
4. Sync the newest resumable state and verify remote size/hash. Do not re-download historical
   checkpoints when the worker ledger and local terminal artifacts already verify.
5. Evaluate each terminal base once against the identity frozen before training, promote it to
   `DONE`, and then advance to the next job.
6. Retry incomplete training or explicit transport failures. Quarantine deterministic identity,
   schema, hash, and unknown post-training failures without an autorestart loop.
7. Run focused tests for a changed component. Run the full suite only before freezing a new
   scientific source bundle or after a genuinely cross-cutting change.

Colab and Vast are separate provider adapters. Neither inherits the other's reconnection or retry
policy merely because they call the same scientific runner.

## Milestone review agents

- `run-qa` — at a milestone boundary, sweeps the ledger, checkpoint lineage, focused tests,
  leakage hashes, provider state, and rendered figures. Ranks findings by downstream impact.
- `data-throughput` — classifies the bottleneck before touching anything; only data-path
  levers are in scope.
- `unslop` — before public sharing, audits the writeup and implementation against the actual
  arXiv:2511.05963v4 paper and the spec. `BLOCKED` means nothing is shared.

These are milestone checks, not continuously running agents and not per-cell gates.

## Stop conditions (spec §10) — halt and document, do not work around

- safe and critical lures cannot be exactly matched
- interrupted training cannot resume reproducibly enough for the stated analysis
- NextLat or BST stays below the 90% exact-path base competence gate; GPT is the preregistered
  chance-level replication arm under `docs/DECISION_D20_competence_gate.md`
- the executed D40 all-5,000 feasibility rule fails (already occurred: 4 pairs unmatched); retire
  H3 without changing the matcher and continue only the independently preregistered H1/H2/HMM scope
- any CFS construction has unequal same/different structural overlap within an overlap level;
  retain it as a design record and do not launch adaptation branches
- a geometry effect appears only under a post-hoc metric or layer
- the result depends on a single seed
- the available hardware cannot run the exact paper-scale configuration reliably

Every base parent must be `DONE` before H1/H2 evaluation, with a ledger-hashed competence receipt
binding model, seed, final-checkpoint SHA, evaluator/output SHA, held-out dataset/manifest SHA,
greedy decoding, and exact-path counts. GPT's receipt is mandatory for reporting, but its score is
threshold-exempt. No adaptation plan may be emitted: H3 is permanently out of confirmatory scope.

The recovered A100 profile is nonconfirmatory engineering evidence. Removing all 45 H3 adaptation
branches from the same count-scaled calculation gives 66.375 GPU-h before contingency and
**79.651 GPU-h / 422.148 CU** with the frozen 20% interruption margin. This still excludes the
unmeasured H1/H2 dual-metric and three-regime HMM evaluation delta. Target profiling may refine
that engineering budget, but it cannot alter an endpoint, threshold, seed, or regime.
