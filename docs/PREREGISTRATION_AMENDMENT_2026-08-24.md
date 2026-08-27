# Outcome-blind preregistration amendment — metric robustness, H3 mechanism controls, and HMM calibration

**Frozen:** 2026-08-24 America/New_York, before confirmatory base training and before inspection of
any confirmatory H1/H2/H3 or HMM geometry outcome. The completed 500-step engineering profiles,
recovery tests, and HMM smoke runs are nonconfirmatory and cannot enter any scientific estimator.

**Authority:** This amendment supersedes conflicting analysis language in
`nextlat_v4_predictive_geometry_spec.md`, `docs/FOUNDATIONS.md`, and `docs/EXTRACTION.md`. It does
not authorize training by itself. Confirmatory compute remains blocked until every item in
"Pre-compute freeze gate" has a hash-bound artifact and automated validator.

**Executed prospective disposition, 2026-08-24:** D40 was the one and only permitted H3
nuisance-support expansion. Its unchanged selector left 4 of 5,000 pairs unmatched, so the frozen
contingency has fired and H3 is permanently withdrawn from confirmatory training and inference.
This records execution of the pre-outcome stopping rule; it is not a new matching amendment and no
confirmatory model outcome was inspected. `manifests/h3_selected/PERMANENT_H3_BLOCK.json`
(SHA-256 `82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778`) forbids candidate
expansion, caliper changes, weighting, unmatched-item restriction, pilot substitution, and another
matching amendment. H1, H2, and the exactly three-regime HMM calibration remain confirmatory.
For avoidance of doubt, the block applies only to the Lure-Star adaptation/interference estimand
and its gradient, shadow-update, and Jacobian mechanism probes. The prespecified HMM Bayesian
posterior- and future-distribution-decoding diagnostics remain required. Their existing
`h3_posterior_*` and `h3_future_*` result keys are legacy schema names and must remain byte-stable;
they do not resurrect Lure-Star H3.

## 1. Contribution and identification

Established predictive-state theory already says that a sufficient state retains the information
needed to predict the future. It does not identify a unique coordinate system or require ordinary
distances to reflect predictive similarity. For any invertible map `g`, `g(h)` preserves
sufficiency while generally changing cosine and Euclidean distances. Exact HMM filtering,
predictive-state representations, causal-state equivalence, representation-similarity analysis,
and gradient-based accounts of interference are therefore related work, not claims of novelty.

The comparative contribution tested here is narrower:

> A solver-verified, competence-aware benchmark of whether GPT, BST, and NextLat allocate
> representational distance to matched future-relevant rather than future-irrelevant changes, and
> whether that geometry predicts planning behavior, calibrated against exact predictive-state
> relationships in three model-blind HMM regimes.

`NextLat - BST` is the primary competence-matched contrast. It is **not** an objective-only causal
contrast: BST adds a backward transformer, TextHead, more parameters, and O(T^2) prefix-suffix
supervision. `NextLat - GPT` is secondary and competence-confounded. No result may be described as
showing that "the NextLat objective alone caused" a difference.

The project can contribute a measurement framework and comparative empirical result. It cannot
claim a new theory of belief states, a unique/minimal manifold, biological pattern separation, or
that representational distance itself causes retention. Causal language about geometry is allowed
only for a separately reported, successfully completed activation-intervention experiment under
section 6.

This boundary is grounded in prior mathematical work, not caution by fiat. Predictive-state
representations encode state through predictions of future observations
([Littman, Sutton & Singh, 2001](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html));
causal states formalize predictive-equivalence classes and their minimality
([Shalizi & Crutchfield, 2001](https://arxiv.org/abs/cond-mat/9907176)); and bisimulation work
connects behavioral equivalence to state metrics
([Taylor, Precup & Panangaden, 2008](https://proceedings.neurips.cc/paper/2008/hash/6602294be910b1e3c4571bd98c4d5484-Abstract.html)).
Transformers trained only on next-token prediction have already been shown to linearly represent
nontrivial HMM belief geometry
([Shai et al., 2024](https://papers.nips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html)).
Gradient inner products are also an established interference baseline
([Bengio, Pineau & Precup, 2020](https://proceedings.mlr.press/v119/bengio20a.html)); recent
continual-learning work explicitly analyzes forgetting through gradient alignment and
representation correlation
([Bai et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/81b00efcbc755bd0b8dc6c0d15e9d0b1-Abstract-Conference.html)).
Our novelty claim must sit beyond, and be compared against, those results.

## 2. H1 metric-invariance contract

Let `b`, `s`, and `c` denote the base, near-safe, and near-critical final RMS-normalized states for
one quartet. The item statistic under metric `d` is

```text
PSI_d = d(b, c) - d(b, s).
```

Positive PSI means that an equally sized, future-changing edit is represented farther away than a
future-preserving edit. It does not mean that the representation is sufficient or minimal.

### Co-primary metrics

No finite choice of ordinary metric can make a learned representation uniquely meaningful under
all invertible reparameterizations. Full-covariance Mahalanobis is affine-invariant in the ideal
population/full-rank limit, but held-out finite-sample shrinkage weakens that exact invariance.
Accordingly, the rule below tests **operational robustness across two complementary geometries**;
it does not claim coordinate-free identification.

Both metrics below are co-primary at extraction index 62 and on the same frozen scoring items:

1. **Centered cosine:** `1 - cos(x - mu, y - mu)`, with one mean per
   `(model, seed, index)` over all five frozen `E_lure` conditions. No condition-specific
   centering is permitted.
2. **Held-out whitened Mahalanobis:** `||W(x-y)||_2`, with shrinkage covariance fit only on the
   frozen whitening-calibration split, never on a scored quartet. Shrinkage, condition number,
   calibration IDs, and fit-source hash must travel with every result.

The 2,000 `E_lure` base IDs are split by ascending SHA-256 of the canonical base serialization:
the first 400 form `E_white`; the remaining 1,600 form `E_score`. `E_white` is used only to fit the
whitener. Both co-primary PSI estimators score exactly `E_score`. Centered cosine keeps its existing
complete-pool centering contract; its scored rows are nevertheless restricted to `E_score` so the
two metrics see identical items. The split is global and reused for every arm and seed.

For each `(metric, model, seed)`, also report the dimensionless nuisance-normalized cell statistic

```text
nPSI = 2 * sum_i[d(b_i,c_i) - d(b_i,s_i)]
           / sum_i[d(b_i,c_i) + d(b_i,s_i)].
```

The denominator uses the same scored items and must be strictly positive and finite; otherwise the
cell is invalid. `nPSI` is a mandatory scale diagnostic, not a third metric that can rescue a failed
co-primary result.

### Binding H1 decision rule

The sole confirmatory H1 contrast is paired by training seed:

```text
Delta_metric(seed) = mean_PSI_NextLat(seed) - mean_PSI_BST(seed).
```

The central H1 claim receives:

- **metric-robust confirmatory support** only if, for both co-primary metrics, the paired-seed mean
  is positive and the lower bound of its two-sided 95% paired Student-t interval is above zero;
- **directionally consistent but unresolved evidence** if both paired-seed means are positive but
  either interval includes zero;
- **metric-dependent evidence** if signs disagree or only one co-primary metric is positive; and
- **no support** if both means are nonpositive.

This is an intersection-union rule: both co-primary metrics must pass, so there is no multiplicity
discount for succeeding on only one. Report the interval, every seed, the exact two-sided sign-flip
randomization p-value, its attainable floor (`0.0625` at five seeds), the paired standardized effect,
the 80%-power minimum detectable effect, and leave-one-seed-out sensitivity. Item bootstraps describe
conditional within-checkpoint uncertainty only. Index 63, BST TextHead state, raw cosine, uncentered
Euclidean, and intermediate layers are mandatory labeled secondary analyses and cannot promote H1.

## 3. H2 geometry-to-behavior contract

H2 remains predictive, not causal. Correct-branch margins are taken at index 63. Both co-primary
distances are fit and evaluated separately on identical folds over `E_score`; fold assignment is
`int(SHA256(base_id), 16) mod 2`, so it is frozen without outcomes. Standardization is learned in
the training complement only.

For each metric, compare nested out-of-fold models:

```text
M0: critical_margin ~ base_margin
M1: critical_margin ~ base_margin + base_to_critical_distance
```

Report pooled held-out R2 for both models, `Delta R2 = R2(M1)-R2(M0)`, held-out Spearman,
per-fold coefficients, and fold sign consistency. Geometry is said to add predictive value only if
`Delta R2 > 0` and the distance coefficient has the preregistered positive sign in both folds for
both co-primary metrics. Any other pattern is described as metric-dependent or inconclusive.
Accuracy and exact-path analyses are secondary because of ceiling effects.

## 4. H3: permanently withdrawn after the frozen feasibility gate

This section is retained as the immutable record of what H3 would have tested. None of its
adaptation branches, baselines, folds, endpoints, or causal follow-ups may enter the confirmatory
job matrix or inferential report. D40 scored all 135,000 newly generated rows under job SHA-256
`393c933e9e616cd24a4b7a9b408203b0c22002c39cf97f2d72b03176fe45482a`; the loss table SHA-256 is
`f84c73b81d7b9e8cab44e32d89cd272d320d420583bd7badf76f3c0dade7f537`, the durable-state SHA-256
is `e1ed1d814ea190b1602c31ef82bee86bcd0937dc26dec5963a31d692f8faa0c2`, and the combined
188,000-row loss table SHA-256 is
`814058a162e12fde36c7204dd30798b63bfbf02294fce768046070672e5afece`. Four unmatched pairs
triggered the permanent block. These are nuisance-feasibility facts, not confirmatory interference
outcomes.

Every adaptation branch starts from the same hash-identical parent and uses **the same
next-token-only full-parameter adaptation estimand for GPT, NextLat, and BST**. For NextLat this
means both auxiliary coefficients are zero. For BST it means teacher-forced next-token cross-entropy
through the generation-time forward-encoder/TextHead path with the item-independent lone-EOS
backward state; the dense prefix-suffix BST training objective is disabled. The existing production
path that leaves `use_bst=true` and calls the ordinary BST training loss does **not** satisfy this
contract and must be replaced and parity-tested before H3. Bank selection, order, optimizer state,
update count, acquisition probes, and all analysis folds are frozen before the parent is trained.

### Three structural-distance branches

H3 uses `near`, `mid`, and `far`, not a post-hoc dichotomy. `B_mid` contains 5,000 solver-verified
items selected before confirmatory training from the pre-generated candidate bank by structural
graph distance: after the exact near/far matching constraints and pilot-loss caliper are satisfied,
select candidates closest to the fixed 50th percentile of normalized edge disagreement, breaking
ties by candidate SHA-256. `B_near`, `B_mid`, and `B_far` must match target-path distribution,
example count, initial pilot-loss deciles, update count, and held-out acquisition-set size. The
model-blind pilot and selector source, checkpoint, manifests, calipers, and tie-breaks are hashed.

Decision D39's complete frozen pilot showed that the original three candidates per near item left
1,115 items unmatched. Decision D40 is the sole prospective feasibility repair: for each near item,
retain those three and add exactly nine unique candidates in each 1/2/3-rewire stratum (30 total;
150,000 globally), using a new deterministic RNG namespace and no model information. Only the
135,000 new rows are newly scored. Middle deciles are recomputed across all 150,000 while near
deciles and every eligibility/selection rule above remain unchanged. If even one of 5,000 pairs is
unmatched, H3 is permanently blocked and no further caliper, weighting, restriction, candidate,
or pilot amendment is allowed. See `docs/DECISION_D40_h3_overlap_expansion.md`.

The labels are structural. If realized hidden distances do not order `near < mid < far`, retain all
branches, report the failed manipulation check, and do not claim a distance dose-response. Never
reselect a bank using a confirmatory checkpoint.

### Frozen mechanistic baselines

For each parent, item `i`, and branch `k`, compute before adaptation:

```text
Gdot_ik = <grad_theta m_i, grad_theta L_Bk>
Gcos_ik = cosine(grad_theta m_i, grad_theta L_Bk)
```

where `m_i` is the original correct-branch margin and `L_Bk` is mean next-token loss on a
the exact first effective adaptation batch (512 examples in the frozen configuration) of branch
`k`. Gradients use all parameters updated during adaptation, float32 accumulation, eval mode, no
persistent optimizer step, and the exact parent checkpoint. Because the actual optimizer is AdamW,
also make one disposable shadow copy of the parent model/optimizer/scheduler, execute that exact
first update including clipping, moments, weight decay and schedule, and record

```text
Udot_ik = -<grad_theta m_i, Delta_theta_Bk_first_update>.
```

`Gdot` is the raw gradient-alignment baseline; `Udot` is the first-order prediction under the
actual first optimizer update. For plain gradient descent the bridge is

```text
m_i(theta - eta * grad L_Bk)
  = m_i(theta) - eta * <grad m_i, grad L_Bk> + O(eta^2).
```

Also compute the hidden-state parameter-Jacobian overlap

```text
Jov_ik = <J_i, mean_j J_Bk,j>_F
         / (||J_i||_F * ||mean_j J_Bk,j||_F),
```

for index-63 states. Estimate Frobenius products with 16 shared Rademacher output projections,
seed `20260824`, using the same first branch batch. The estimator implementation must be validated
against exact Jacobians in a tiny model before use. Missing, nonfinite, or mismatched-probe
baseline values invalidate H3; they are not silently dropped.

### Incremental predictive model

Let `e_ik = margin_before_i - margin_after_ik`. Fit separately within each
`(model, parent_seed, metric)` cell, producing one inferential result per training seed. Use
five-fold cross-fitting with folds fixed by `SHA256(A_pair_id) mod 5`; all three branches of an item
remain in one fold. Within each training fold, standardize continuous predictors and fit:

```text
M0: e ~ branch + initial_margin + lure_loss + acquisition_margin
        + Gdot + Gcos + Udot + Jov

M1: M0 + distance + (distance^2 - training_fold_mean(distance^2))
```

Fit this model separately for centered-cosine and held-out-whitened distance. The quadratic term is
always present: it prevents a nonmonotonic relationship from being forced into a favorable linear
summary. No distance bins are tested. Fixed deciles may be used only as x-coordinates for a plot of
the fitted continuous curve.

Report held-out `Delta R2`, held-out Spearman, linear and quadratic coefficients, the predicted
curve over the observed 5th–95th percentile, and whether its turning point lies inside that range.
The preregistered monotonic prediction is a negative linear distance effect (closer items erode
more). If an interior turning point is present or the quadratic term reverses the fitted slope over
the central range, report a **nonmonotonic association** and do not summarize H3 as monotonic.

Geometry has incremental predictive support only when `Delta R2 > 0`, the central-range fitted
slope is negative, and the direction agrees across both co-primary metrics. Seed-level contrasts,
intervals, sign-flip p-values, MDE, and leave-one-seed-out sensitivity remain mandatory. Model
comparisons aggregate within seed before inference. Acquisition failure or materially unmatched
pilot-loss distributions invalidates the affected branch contrast; it does not license retuning.

## 5. HMMs are calibration, not a second discovery surface

The HMM experiment checks whether the measurement framework behaves correctly when exact Bayesian
predictive distributions are known. It is not evidence that NextLat discovers a unique posterior
coordinate system, and it cannot by itself establish the Lure-Star contribution.

### Frozen model-blind family

Use three 4-state/4-observation stationary regimes chosen without model representations or model
outcomes from the existing deterministic candidate grid:

1. **persistent/moderate-aliasing** — the currently frozen matrix, retained unchanged;
2. **fast-mixing/moderate-aliasing** — the passing candidate with the lowest mean dwell time;
3. **persistent/high-aliasing** — among passing candidates with mean dwell at least the median,
   the candidate with the highest mean posterior entropy.

All selections use the same acceptance box already recorded in `manifests/hmm_matrices.json`.
Ties are broken lexicographically by the existing candidate tuple. Before training, publish one
family manifest containing all matrices, exact selection diagnostics, the complete ordered list of
passing candidate hashes, `TE = transition @ emission`, its rank and singular values, and all
corpus/pair-bank hashes. A regime with nonfinite diagnostics, nonstationarity, `rank(TE) < 4`, or
`sigma_min(TE) <= 0.05` fails closed and must be replaced by the next candidate under the same
predeclared ranking, with the rejection recorded. This algebraic gate ensures that distinct beliefs
in these calibration regimes induce meaningfully distinct one-step predictive distributions; it is
not claimed as a new theorem.

Each regime uses the same five paired seeds (`1234`–`1238`), architecture, update count, corpus
sizes, and evaluator. Regime-specific corpora and pair banks are generated from fixed seeds derived
as `1105963 + regime_index * 100000`, where regime order is the list above.

### Ground truth and primary aggregate

For belief `b_t`, score the exact future distribution

```text
q_t = (b_t @ transition) @ emission.
```

Future-distribution Jensen–Shannon divergence is the primary ground-truth distance. Belief JS is a
mandatory secondary calibration quantity because posterior coordinates can contain distinctions
that are irrelevant to one-step prediction.

Regime-specific pair banks are therefore selected from exact **future-distribution JS**, not belief
JS: predictive-equivalence pairs combine high history edit distance with low future JS, and
divergent near-lures combine low history edit distance with high future JS. Thresholds are fixed
from validation quantiles and applied unchanged to disjoint test pools; ties break by canonical pair
SHA-256. Belief-JS versions of the analyses are mandatory secondary outputs and cannot redefine the
pair bank.

For each regime, seed, model, and co-primary hidden metric, compute partial Spearman correlation
between hidden distance and exact future-distribution JS while controlling history edit distance
and prefix length. The single primary HMM aggregate for metric `d` is

```text
A_d(seed) = mean_over_3_regimes[
    atanh(rho_NextLat,d) - atanh(rho_GPT,d)
].
```

The HMM calibration claim passes only if the paired-seed 95% interval for `A_d` lies above zero for
**both** centered-cosine and held-out-whitened Mahalanobis. This is again an intersection-union rule.
Regimes are aggregated inside each seed and never counted as independent replications.

Secondary regime-aggregated endpoints are: predictive-equivalence contrast, future-distribution
probe JS at length 32, future-distribution probe JS at length 64, posterior-probe JS at length 32,
and posterior-probe JS at length 64. Apply Holm correction across these five named endpoints after
first aggregating regimes within seed. Report every unadjusted and adjusted p-value and effect; no
secondary endpoint can rescue a failed primary aggregate.

For every HMM endpoint report all regime/model/seed cells, paired-seed 95% intervals, exact
sign-flip p-values and their attainable floor, MDE at 80% power, and leave-one-seed-out results.
A null is written as "not resolved at the detectable effect size" together with the MDE, never as
evidence that the systems are equivalent. HMM results that reverse across regimes are heterogeneity,
not an invitation to select a favorite matrix.

## 6. Causal-language gate and optional Lure-Star activation intervention

This section governs only the core Lure-Star H1/H2 study. It does not govern the separately
numbered CFS-2 causal-forgetting experiment. CFS-2 freezes matching parent-to-adapted patching at
blocks 3, 7, and 10 as a required per-branch endpoint under
`DECISION_CFS2_STIMULUS_REPAIR.md`; that requirement is not relaxed by the word “optional” below.

Without an activation intervention, allowed verbs are `predicts`, `is associated with`,
`separates`, `tracks`, and `is consistent with`. Forbidden causal summaries include `geometry
causes/protects against forgetting`, `mediates`, and `the objective alone causes`.

The optional intervention patches the penultimate block-10 state at index 63 from the matched
near-critical item into the base computation, then runs block 11, final RMSNorm, and the output
head. The preregistered outcome is change in correct-branch log odds. Near-safe, random-graph, and
norm-matched patches are mandatory controls; patch donor/recipient pairs and random controls are
fixed by manifest hash. Patching the final pre-logit state is disallowed. A successful controlled
intervention permits a local causal statement about that activation at that site and checkpoint;
it does not establish that global pairwise distance is a causal mechanism or that NextLat alone
caused it.

## 7. Multiplicity, exclusions, deviations, and nulls

- H1 and the HMM primary aggregate each use intersection-union co-primary metric rules.
- H2 must report both metrics and cannot promote a one-metric result. H3 has no confirmatory
  endpoint because its prospective feasibility gate failed.
- The three preregistered Lure-Star arm contrasts retain their order: NextLat–BST primary,
  NextLat–GPT secondary/confounded, BST–GPT reference. Secondary families use Holm correction.
- Seeds, not items, are the inferential unit for model contrasts. Items quantify conditional
  precision only.
- There are no outcome-based seed, item, layer, distance, HMM-regime, or time-point exclusions.
  Integrity exclusions require a named machine-checkable rule and a retained failure receipt.
- Missing cells are not imputed. A missing seed blocks the confirmatory aggregate unless recovered
  under the exact checkpoint contract. Reduced-seed results are labeled incomplete/descriptive.
- Every preregistered endpoint is emitted in a fixed-schema receipt, including nulls, sign
  disagreements, failed manipulation checks, and nonfinite/invalid cells.
- Any scientific change after this freeze requires a new dated amendment that states which
  outcomes, if any, were visible. Outcome-aware changes are exploratory and can never alter the
  confirmatory decision rule.

## 8. Pre-compute freeze gate

No confirmatory training may start until automated validation proves all of the following:

1. this amendment and the authoritative spec are hashed into the source snapshot;
2. `E_white` and `E_score` memberships are immutable/nonoverlapping, all five condition manifests
   are complete, and the full evaluation pool is disjoint from training inputs;
3. the held-out whitener and both co-primary metric schemas are executable on synthetic fixtures;
4. the D40 job, loss, combined-loss, durable-state, and permanent-block receipts match the hashes
   above, record exactly 4/5,000 unmatched, and retain `no_further_amendments_permitted: true`;
5. confirmatory source, job, result, and clearance schemas contain no Lure-Star H3 adaptation
   branch, interference estimand, or mechanism-probe endpoint and fail closed if one is introduced,
   while continuing to require the unchanged HMM `h3_posterior_*`/`h3_future_*` calibration keys;
6. the H1 paired-seed intersection-union implementation and H2 fixed two-fold cross-fitting emit
   every metric, fold, diagnostic, and null required by sections 2–3;
7. every Lure-Star base checkpoint has a hash-bound competence/evaluation receipt before H1/H2;
8. all three HMM regimes, corpora, pair banks, `TE` certificates, and one family hash are frozen,
   and the exact 30-job family print-plan verifies each nested `thresholds.hmm_sha256` binding;
9. the cross-seed aggregate, exact sign-flip test, MDE, leave-one-seed-out, intersection-union and
   Holm procedures pass deterministic unit tests;
10. the fixed result schema refuses missing or extra preregistered metrics; and
11. the full local suite passes with no unresolved P0/P1 scientific finding.

If any surviving-scope gate is absent, training remains blocked. A validator that still requires
H3 execution, or that permits H3 jobs despite the permanent-block receipt, is stale and cannot
authorize compute. Engineering profiles may be repeated to validate runtime behavior, but their
checkpoints and metrics remain forever ineligible for confirmatory analysis.
