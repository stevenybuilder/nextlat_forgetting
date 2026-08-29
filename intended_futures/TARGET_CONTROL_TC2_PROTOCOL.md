# Layout-identifiable intended-future geometry

## Question

After removing everything predictable from the exact instruction pair, does π0.5's fixed
PaliGemma layer 13 still encode where the instruction-selected pickup target is in a new layout,
and is that layout-specific information causally usable?

This is the identified follow-up to TC1. TC1's activation observer was accurate in absolute terms,
but an exact-prompt lookup was better because each prompt pair was tied to an almost fixed layout.
TC2 holds each ordered instruction pair fixed while the target geometry changes across official
LIBERO-Plus layouts.

## Why these stimuli

[LIBERO](https://arxiv.org/abs/2306.03310) supplies the manipulation environment and fixed-state
evaluation convention. Its original placement regions, and the inherited LIBERO-CF scenes used in
TC1, move the relevant objects by only about 2 cm. Drawing more reset seeds from those regions does
not fix the identification problem.

[LIBERO-Plus](https://arxiv.org/abs/2510.13626) is an established robustness benchmark with an
explicit **Objects Layout** condition covering target displacement and confounding objects. TC2 uses
only its pre-generated `level{1..5}_sample{1..4}` target-displacement tasks. It excludes the `add_*`
clutter tasks because they change distractors without necessarily changing the target, and excludes
the top-drawer family because its target is hidden while the primary label is visible workspace XY.
No task is selected using π0.5 output.

Each stimulus is an official, unmodified LIBERO-Plus BDDL file plus its official fixed initial
state. The image and robot state are paired with two distinct, verbatim LIBERO-Spatial instructions:
one truthfully selects `akita_black_bowl_1`, and one truthfully describes the initial relation of
`akita_black_bowl_2`. Thus pixels and robot state are identical within a pair; only the intended
target changes. This matched-prompt pairing is a derived analysis set, not an unmodified leaderboard
task, and will be described that way.

The official public `pi0.5-LIBERO` checkpoint and OpenPI inference settings are unchanged. Moving to
DROID would simultaneously change the checkpoint, embodiment, observation distribution, and target
annotation scheme; DROID also does not provide the identical-scene counterfactual target pairs this
test needs. It remains a later external-validity study rather than a substitute for this controlled
identification test.

## Frozen population and splits

- Source: LIBERO-Plus commit `4976dc30028e805ff8094b55501d532c48fec182`.
- Population: 158 official layout tasks across nine visible LIBERO-Spatial families.
- Observer fit: official samples 1 and 2, 81 tasks.
- Untouched observer validation: official sample 3, 38 tasks.
- Causal test: official sample 4, 39 tasks; it is never loaded unless every observer gate passes.
- Inference cluster: base spatial-task family, not individual layouts.

The pre-model BDDL audit passed before freezing. Target pairs are at least 12.1 cm apart; after
subtracting observer-fit prompt-family means, pooled layout RMS is 3.48 cm and untouched validation
RMS is 3.87 cm. All nine families span more than 11 cm. These quantities use only official BDDL
placement regions and no model output.

## Hypotheses

**Null.** Conditional on the ordered prompt family, PaliGemma-13 differences contain no predictive
information about held-out layout-specific target displacement. Any apparent compact intervention
is no better than matched random or the training-only prompt-family mean.

**Alternative.** PaliGemma-13 differences predict held-out layout-specific target displacement, and
a decoder-derived intervention using the actual layout redirects behavior beyond both matched
random and a controller given only the training prompt-family mean.

## Observer analysis

For each fit record, subtract its observer-fit prompt-family mean from both the full token-preserving
activation difference and the simulator-measured target XY difference. Ridge strength is selected by
leave-one-difficulty-level-out cross-fitting; prompt-family means are recomputed inside every fold.
The selected ridge is then fit on all 81 fit records and evaluated once on the 38 sample-3 records,
using training means unchanged.

The observer advances only if all four conditions hold:

1. held-out layout-residual R² is at least 0.10;
2. activation SSE is at least 10% lower than the training-only exact-prompt-family mean;
3. mean residual-direction cosine is positive in at least seven of nine families; and
4. the clean first-action-chunk positive control has nonnegative held-out residual R².

Failure stops the study before causal rollouts and no controller artifact is created. Raw absolute
R² may be reported descriptively but cannot substitute for the residualized gate.

## Causal test

If the observer passes, a three-coordinate decoder is refit on fit plus validation records. Six
conditions are randomized within each untouched sample-4 layout: clean task A, clean task B, the
minimum-norm actual-layout controller, the same controller given only the observer training
prompt-family mean, a norm-matched random controller, and complete donor replay. Every condition
uses the same one-GPU topology, bfloat16 checkpoint, OSMesa simulator, fixed within-condition noise,
12 replans, and five executed actions per replan.

Complete replay must first pass the manipulation check: at least a 20-point increase in task-A first
touch over clean task B, a family-cluster 95% interval above zero, positive effects in at least seven
families, and exact patch receipts.

Compact layout-specific control is supported only if every frozen criterion passes:

- minimum norm improves task-A first touch over clean task B by at least 15 points;
- minimum norm improves it over matched random by at least 10 points;
- both family-cluster 95% interval lower bounds exceed zero;
- minimum norm improves task-A progress over the prompt-family-mean controller by at least 5 mm,
  with its family-cluster interval lower bound above zero;
- at least seven families have positive primary touch effects;
- all patch receipts are exact and there are no invalid units.

This is an intersection-union rule: no favorable secondary endpoint can rescue a failed condition.

## Power and claim boundary

Nine independent task-family clusters and 38–39 held-out layouts are adequate only for large,
consistent effects. A negative pilot does not exclude small effects or effects restricted to one
relation. A positive result is an exploratory causal pilot because there is no independent
replication split.

The strongest allowed conclusion is about visible instruction-selected pickup-target XY geometry at
one fixed PaliGemma layer in `pi0.5-LIBERO`. It is not evidence for temporally extended planning, a
general world model, all π0.5 checkpoints, or real-robot generalization.

## Compute stop rule

The persistent Vast instance stays stopped until this protocol, manifest, analysis code, and tests
are committed. The new-rental cap is **$2.00** on one RTX 4090. The observer should yield the first
terminal answer in roughly 20–30 minutes of active compute. Causal rollout runs only after an
automatic gate and is expected to bring total active time to roughly 2.5–4 hours. The instance is
stopped immediately after a terminal observer failure, causal completion, error, or cap breach.
