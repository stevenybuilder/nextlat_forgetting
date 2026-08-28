# Task 16 pilot preregistration

Status: frozen before the first Task 16 policy outcome. The configuration SHA-256 is
`b6d3f37959c7075b082e5fcfc62f94bc11c8c2feec0f931cee83e25041b0f686`.

## Research question

Can an outcome-blind measure of interference among target, receptacle, and spatial-relation
directions in a VLA's action-facing hidden state prospectively predict closed-loop compositional
failures?

This is narrower than generic VLA failure detection. SAFE already learns a supervised detector
from successful and failed rollouts. Here, factor directions are estimated from initial-state
activations without using any reward, success label, executed action, or behavioral seed.

## Hypotheses

**Primary alternative.** Combinations with greater cross-fitted geometric interference have lower
closed-loop success on the official VIMA-Bench combinatorial-generalization partition.

**Null.** Geometric interference has no prospective association with combination-level success.

**Difficulty alternative.** Object or direction identity, action confidence, or layout difficulty
predicts behavior, but representation geometry adds no held-out predictive value.

**Routing alternative.** The factors are additively represented at the first decision, but failures
arise later from memory or action routing and are not predicted by first-decision geometry.

The study does not assume that orthogonality is necessary for every VLA. That claim is established
only under specific theoretical conditions for vision embeddings and is being tested here as an
extrapolated predictive hypothesis.

## Benchmark and population

The target population is the frozen official VIMA-92M checkpoint on Task 16,
`manipulate_old_neighbor`, in VIMA-Bench's official `combinatorial_generalization` partition. The
VIMA paper reports 42.0% success for this model/task/partition, avoiding the ceiling that made Task
1 unsuitable.

The 32 experimental cells are the Cartesian product of:

- target shapes: `letter A`, `block`, `heart`, `L-shaped block`;
- receptacle shapes: `bowl`, `square`;
- remembered neighbor directions: `north`, `south`, `west`, `east`.

The official Task 16 L2 generator is not narrowed or modified. All 20 target-texture values, 19
receptacle-texture values, 14 target-shape values, four receptacle-shape values, distractors, and
layouts retain their official support; seeds are merely conditioned on membership in the 32
predeclared cells. No custom image, object, task, or success criterion is introduced.

## Sampling and seed separation

Each cell has eight representation resets and twenty closed-loop behavior rollouts. Two cells have
two additional smoke seeds. All three samples are disjoint.

The exact seeds are resolved by the frozen `official_task16_factor_stratified_v1` algorithm. It
mirrors Task 16's pre-render RNG calls and selects only on target shape, receptacle shape, and
neighbor direction. It cannot access the policy, actions, rewards, success labels, or rollout
state. Before any policy outcome, two real resets in every cell (64 total) matched the mirror and
the adapter was verified byte-for-byte not to alter the official task arguments. Every executed
episode independently validates its actual factors and requested seed; mismatches are failures and
are never replaced.

Representation seeds estimate a cell's hidden state. Separate behavior seeds estimate its success
probability, preventing the same difficult layout from mechanically driving both predictor and
outcome.

## Endpoints and controls

The primary representation is the final VIMA transformer token directly consumed by the action
decoder at the first environment decision. The cell representation is the mean over its eight
independent nuisance layouts.

The primary geometric predictor is mean absolute cosine similarity among the selected centered
factor-value effects. Effects are estimated by an additive ridge model using only training cells in
five fixed combination-level folds. Higher values mean more interference.

The primary behavioral endpoint is success proportion over twenty independent closed-loop
rollouts. The inferential unit is the 32-cell combination, not a frame, activation, action, or
rollout.

The preregistered difficulty baseline includes symmetric one-hot factor-identity main effects,
mean initial action confidence, target-to-receptacle distance, and initial target and neighbor
coordinates. The logistic ridge coefficient is fixed at 1.0.

## Analysis and power

1. Estimate every cell's interference from a model that excluded that entire cell.
2. Report Spearman correlation between interference and success with a 10,000-replicate cell
   bootstrap interval and a 10,000-replicate two-sided permutation test.
3. Compare leave-one-cell-out binomial log loss for the difficulty baseline against the same model
   plus interference.
4. Report additive held-out reconstruction R-squared as a representation diagnostic, not evidence
   of behavioral relevance.

With 32 cells, a two-sided alpha-0.05 Fisher-z approximation gives about 80% power for a population
correlation near |rho| = 0.48. Twenty behavior trials per cell limit binomial measurement noise;
they do not inflate the inferential sample size. Secondary endpoints use Benjamini-Hochberg
correction if more than one is promoted beyond exploratory status.

## Decision gate

The pilot advances to causal interventions only if:

- the exact checkpoint, upstream commits, source hash, one-GPU float32 topology, and smoke gate
  validate;
- every preregistered behavior rollout produces a valid terminal outcome; frozen failures are
  retained and abort confirmatory analysis rather than being replaced;
- Spearman rho is at most -0.50, its 95% cell-bootstrap upper bound is below zero, and the
  two-sided permutation p-value is below 0.05; and
- adding interference improves leave-one-cell-out binomial log loss by at least 5% over the full
  difficulty baseline.

A smaller or non-robust association is reported as a negative pilot and does not trigger a causal
stage.

## Conditional causal stage

Only after the gate passes, remove the target-shape or direction subspace at the action-facing token
and compare the intended action-distribution change with equal-rank, norm-matched random subspaces.
Closed-loop interventions remain secondary until action-distribution selectivity is established.

## Scope boundary

This pilot concerns one older, object-centric simulated VLA and one official task. Even a positive
result is proof of concept, not a state-of-the-art VLA claim. A broader claim requires preregistered
replication on an unsaturated modern checkpoint and benchmark such as SmolVLA on LIBERO, where
current public evaluation reproducibility issues must be resolved first.
