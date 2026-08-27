# Decision D42 — complete the frozen measurement surface before further training

**Date:** 2026-08-24  
**Outcome visibility:** no HMM scientific evaluation was opened or interpreted; no Lure-Star
confirmatory model exists. This decision was made during outcome-blind pre-launch review.

## Why this decision exists

The binding preregistration amendment already required paired Student-t inference, nuisance-
normalized PSI, exact hash-fixed H2 folds, nested H2 models, whitener provenance, intermediate
layers, BST's TextHead state, raw cosine, uncentered Euclidean, and lure-condition accuracy and
exact-path analyses. The reduced v3 evaluator schemas did not implement all of those requirements.
Passing tests therefore established consistency with an incomplete implementation, not
completeness against the amendment.

Further confirmatory training is blocked until the complete measurement surface below passes
source-bound semantic tests and independent review. Earlier successor snapshots and clearances are
historical NO_GO evidence and cannot be reused.

## Frozen operationalization

The two H1 co-primary endpoints remain unchanged: final post-RMSNorm state at index 62 under
centered cosine and held-out whitened Mahalanobis, scored on the same fixed 1,600 `E_score` items.
Only those endpoints enter H1's intersection-union decision.

Mandatory secondary geometry is now a complete, outcome-independent grid:

- every forward-stream transformer block output, blocks 0 through 11, before final RMSNorm;
- both frozen positions, 62 and 63;
- centered cosine and held-out whitened Mahalanobis for every block-position cell;
- final-state index 63, raw cosine, and uncentered Euclidean;
- BST TextHead pre-logit state under the inference-time lone-EOS backward input.

All 12 blocks are retained because choosing a subset after results would create a layer-selection
surface. BST's backward stack remains excluded because it can see the answer suffix. Secondary
geometry is descriptive, labeled `promotion_prohibited`, and cannot rescue either co-primary
metric.

For every one of the five fixed lure conditions, evaluation performs exactly five autoregressive
steps from the 63-token prefix ending in `=`. Each step uses explicit argmax (`top_k=1`,
`temperature=0`). Generated token tails, per-item exact-path indicators, first-branch indicators,
and integer correct/total counts are retained. Secondary H2 accuracy and exact-path models reuse
the exact `int(SHA256(base_id), 16) mod 2` folds. A constant training-fold binary predictor is
reported as not estimable because of ceiling, never jittered or allowed to invalidate the primary
margin analysis.

## Provenance and failure semantics

Every whitener binds ordered calibration base identities, condition-qualified fit-row identities,
dtype, shape, state bytes, shrinkage, condition number, feature count, and a canonical fit-source
SHA-256. Fit-group and score-group identities share the same base-ID domain, so leakage cannot be
hidden by condition suffixes. All 15 cells must reuse the same ordered 400 calibration base IDs.

The report schema retains invalid cells with machine-readable reasons, emits the fixed null
interpretation that unresolved effects are not evidence of equivalence, and carries an explicit
non-applicable manipulation record for the permanently retired Lure-Star H3 branch. Missing,
extra, nonfinite, blocked, or provenance-inconsistent fields fail closed.

Evaluation is atomic at the matrix level: all 15 base parents must be DONE and competence-gated
before the first extraction subprocess; all 15 evidence cells must validate before the sole final
evaluator invocation. HMM evaluation independently preflights all 30 trained cells before opening
the first checkpoint for evaluation.

## Statistical repair

H1 uses a two-sided paired Student-t 95% interval over the five training-seed differences, with
the exact two-sided sign-flip p-value and its 0.0625 floor, paired standardized effect, 80%-power
MDE, and all five leave-one-seed-out summaries. H2 reports out-of-fold M0, M1, and delta R-squared
on identical hash-fixed folds. HMM sign-flip p-values and MDEs are likewise consistently
two-sided; Holm consumes exactly the five named unadjusted secondary p-values.

No number from these analyses may be used to revise this decision or select a layer, metric,
condition, seed, or HMM regime.
