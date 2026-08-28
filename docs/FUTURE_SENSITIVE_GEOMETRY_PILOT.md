# Future-sensitive representation pilot

## Decision

The next low-compute experiment should test the program's second hypothesis:

> After controlling the visible graph edit, are internal representations more sensitive to
> changes in the future than to changes that preserve the future?

This is a better use of the archived checkpoints than another retrospective seed analysis. It
directly answers one of the four original questions, requires no new training, and turns the
observed solver/shortcut contrast into a mechanistic experiment. The result will remain conditional
on the two selected trajectories; it will not estimate a NextLat population effect or compare
training objectives.

## Why this appears additive

The NextLat paper reports final Path-Star performance and future-token probes on TinyStories, but
does not test matched Path-Star counterfactual geometry across training or intervene on the state
that selects an arm. Existing Path-Star work introduced structured same-graph/different-target
samples and training interventions, but not this representation test. Future-state probing and
causal tracing are established methods, so the contribution is the controlled Path-Star assay and
its solver-versus-shortcut transition, not activation patching by itself.

Primary anchors:

- [NextLat](https://arxiv.org/abs/2511.05963)
- [Original Path-Star pathology](https://arxiv.org/abs/2403.06963)
- [Structured Path-Star samples](https://arxiv.org/abs/2410.13779)
- [Path-Star supervision interventions](https://arxiv.org/abs/2503.10542)
- [Future Lens](https://arxiv.org/abs/2311.04897)
- [Do language models plan ahead?](https://arxiv.org/abs/2404.00859)
- [Causal tracing across a grokking transition](https://arxiv.org/abs/2405.15071)

## Frozen-model contrast

Use every retained checkpoint from the generalizing seed 1234 and shortcut seed 1235 trajectories.
The primary post-transition checkpoint is seed 1234 at update 3,000; update 1,000 is the
pre-transition comparator. The two final update-20,000 checkpoints compare the stable solver and
shortcut phenotypes. The behavioral transition is interval-censored to (1,000, 3,000]; its exact
update is unknown. The remaining checkpoints describe the time course and are not independent
replicates.

For a target arm `s-a-b-c-t` and distractor arm `s-d-e-f-u`, generate four
exact-solver-verified variants of every untouched base graph:

1. **Serialization-only:** exchange two disjoint edge records without changing the graph or correct
   future; select the records so the prompt-token Hamming distance matches the node-swap variants.
2. **Future-preserving:** swap same-depth internal node labels between two non-target arms across
   all their edge-list occurrences, leaving `s-a-b-c-t` unchanged.
3. **Immediate future-changing:** swap the depth-1 labels `a` and `d` across all edge-list
   occurrences, keeping query target `t` fixed; the correct first decision changes from `a` to `d`.
4. **Delayed future-changing:** swap same-depth non-endpoint labels `b` and `e` (or `c` and `f`)
   across all edge-list occurrences, keeping `a` and query target `t` fixed while changing a later
   correct token.

The primary delayed-versus-preserving pair must match the number of token substitutions, graph
depth of swapped nodes, node multiset, degree sequence, unchanged query, prompt/answer length, and
serialized-position distribution. Define base difficulty from graph and serialization properties
before model inference, never from model margins or hidden states. The serialization control is
dose-matched on prompt-token Hamming distance rather than graph edits, because it changes no edge.
Publish standardized matching diagnostics, tolerances, failures, and attrition.

An independent validator must verify one center, five directed length-5 arms, unique node IDs,
unchanged query, valid ordered edges, and the intended exact path for every sibling. Split base
graphs into disjoint development/pilot, covariance-calibration, and untouched confirmatory-test
pools before creating siblings.

## Geometry test

Extract the final pre-normalization backbone residual after the teacher-forced start node—the state
that must choose the first nontrivial branch. For each checkpoint separately, fit a shrinkage
covariance matrix and calibration mean only on that checkpoint's calibration graphs. Apply them
only to untouched test states from the same checkpoint. Measure whitened Mahalanobis distance from
each base representation to each variant; centered cosine distance using the corresponding
calibration mean is a robustness metric.

The primary per-graph contrast is:

```text
distance(delayed future-changing) - distance(future-preserving)
```

The immediate-change contrast `distance(immediate)-distance(preserving)` is a positive control.
The serialization control requires `distance(delayed)-distance(serialization)>0`; serialization
distance itself need not be zero. The primary result is whether the delayed contrast is positive at
seed 1234 update 3,000. Direct post-minus-pre and final-solver-minus-final-shortcut interactions
test whether that contrast changes across the behavioral transition and phenotype.

### Hypotheses

- **Null:** after matching, delayed future-changing edits do not move the branch-state
  representation farther than future-preserving edits.
- **Alternative:** delayed future-changing edits produce a larger representational displacement at
  the generalizing solver checkpoint.

### Inference and success rule

Graph identity is the inferential unit for effects conditional on a fixed checkpoint. The primary
analysis is the paired mean contrast with a graph-level bootstrap confidence interval. A paired
sign-flip test is a robustness analysis and requires a symmetric paired-difference distribution;
edit-class labels are constructed rather than randomized. Do not use the 20 checkpoints or the
thousands of graphs as independent training runs. Standardize a cross-time contrast by the
checkpoint-specific standard deviation of its per-graph paired differences, not a pooled state-
distance standard deviation.

Support requires all of the following on the untouched test set:

- a positive delayed-minus-preserving contrast at seed 1234 update 3,000, with a standardized
  paired effect of at least 0.2 and a 95% graph-bootstrap interval above zero;
- positive immediate-minus-preserving and delayed-minus-serialization control contrasts;
- a positive direct post-minus-pre contrast to support emergence across the retained transition
  interval, and a positive final-solver-minus-final-shortcut contrast to support phenotype
  selectivity.

Nonsignificance at the pre-transition or shortcut checkpoint is not evidence of absence. To claim
equivalence, predeclare `|standardized contrast|<0.10` and require a 90% equivalence interval inside
that margin. Otherwise report the effect as smaller or not detected. A positive pre-transition
contrast may indicate that the representation precedes behavior and does not by itself refute the
primary H2 result.

With 2,000 independent base graphs, a normal-approximation planning calculation gives about 80%
power for a paired standardized mean difference near 0.063 at two-sided alpha 0.05:
`(z_0.975 + z_0.80) / sqrt(2000) = 0.0626`. Actual bootstrap or sign-flip power depends on the
paired-difference distribution. This quantifies generalization over graphs for fixed checkpoints,
not over seeds or NextLat training runs.

## Causal check

If the geometry gate passes, run activation patching on delayed-change pairs at the seed-1234
update-3,000 checkpoint. Truncate the causal input immediately before the delayed divergent token:
for `s-a-b` versus `s-a-e`, run through the shared `s,a` prefix. At the output-start-node `s`
position only, replace the post-block-7/pre-block-8 residual (module index 6 in a zero-based
12-block implementation) with the donor value, then read the `e`-versus-`b` logits at `a`. Human
layers 3 and 10 are secondary sites. Seed 1234 update 20,000 is a robustness checkpoint;
pre-transition and shortcut checkpoints are developmental controls, not primary patching tests.

Let `q=logit(e)-logit(b)`. The primary endpoint is the graph-paired raw donor-direction shift
`mean(q_patch-q_base)`. Secondary normalized recovery is the ratio of means
`R=mean(q_patch-q_base)/mean(q_donor-q_base)`, computed only when the calibration-set donor gap is
positive and bounded away from zero. Do not average unstable per-graph ratios.

Controls are an implementation-check self patch, serialization and future-preserving donors,
unrelated donors, and norm-matched random patches. In addition to natural full patches, use
`h_base + alpha*(h_donor-h_base)` to equalize patch-delta norms across delayed, preserving,
serialization, and unrelated donors. This distinguishes future content from intervention size.

A practical positive result is a positive layer-7 raw shift with a 95% graph-bootstrap interval
above a predeclared combined/max-control shift and secondary normalized recovery of at least 25%.
Layer 7 is tested once as primary. Apply Holm-adjusted p-values to layers 3 and 10, or use
simultaneous max-T/bootstrap bounds; do not describe ordinary confidence intervals as
Holm-corrected.

A positive patch shows that the patched mid-layer, branch-position activation is locally
sufficient to shift the later donor-versus-base margin in this checkpoint. It does not establish
that the site is necessary, unique, or the cause of the training transition. A null single-site
patch is ambiguous because the representation may be distributed.

## Staged cost and stopping rule

1. **Local construction gate:** validate 200 development quartets and matching diagnostics on CPU.
2. **GPU pilot:** use only those 200 development quartets to extract all layers at the four named
   checkpoints and run the layer-7 patch. Stop if matching controls fail or the delayed contrast has
   the wrong sign at the post-transition solver.
3. **Freeze:** after the pilot, freeze construction code, matching tolerances, checkpoints, hook
   definitions, thresholds, endpoints, and analysis without opening calibration or test outcomes.
   No pilot graph enters confirmatory estimation.
4. **Full conditional study:** fit covariance only on the disjoint calibration pool, then evaluate
   2,000 untouched test quartets at every retained checkpoint and the three predeclared patching
   layers only if the pilot gate passes.

Expected full evaluation is roughly 1–6 RTX 4090 GPU-hours, or about $0.50–$2.50 at the observed
rental rate. Engineering and stimulus validation are the main costs. No new model training is
required.

## Claim boundary

The strongest permitted positive claim is:

> At the seed-1234 update-3,000 NextLat G(5,5) checkpoint, the output-start-node representation is
> more sensitive to matched delayed-future changes than to future-preserving and serialization
> edits, conditional on the frozen generator and test-graph population.

If the direct interactions pass, add that the contrast increased across the retained (1,000,
3,000] transition interval and was larger than in the final shortcut checkpoint. If equivalence
also passes, the stronger “not present” language is permitted; otherwise say smaller or not
detected. If patching succeeds, add that the patched mid-layer, branch-position activation was
locally sufficient to shift the later donor-versus-base logit margin in this checkpoint.
Replication across prospectively sampled runs is still required for a population-level NextLat
claim.
