# Compact target control at PaliGemma layer 13

Status: **frozen before model outcome collection**  
Study ID: `pi05-selective-target-control-tc1-v1`

## Question

Can the distributed donor state that redirected target choice under full PaliGemma layer-13 replay
be compressed into a small linear intervention that selectively changes which object the frozen
π0.5-LIBERO policy touches first?

This is not a test of a temporally extended future, autonomous planning, task completion, or a
general world model. The controller receives the simulator-measured donor-minus-recipient object
displacement. It therefore tests **oracle-coordinate-assisted causal controllability**, not a
deployable language-only steering method.

## Why the intervention site is fixed

Layer 13 was selected before M0 outcomes because Action Atlas reported the maximum PaliGemma goal
classification accuracy there. M0 then compared that site with the literature-selected expert
layer 9. Full PaliGemma-13 replay produced donor-like target behavior; expert-9 replay did not.

The follow-on holds PaliGemma layer 13 fixed to avoid jointly searching the layer, intervention,
strength, and endpoint. It does not claim that layer 13 is uniquely optimal. Establishing a unique
causal locus would require a separately frozen layer-localization study.

## Official data and independent splits

The model, suite, scenes, prompts, renderer, and single-GPU topology remain identical to v4 and
M0. The stimuli are constructed matched contrasts from official LIBERO-CF/Spatial scenes, initial
states, and verbatim BDDL instructions. Within a contrast, pixels, robot state, object layout,
model, and action noise are held fixed; only the selected pickup object changes.

The previous study observed state indices 0–9. TC1 uses the official state bank as follows in every
one of the 12 frozen scenes:

| Split | State indices | Units | Permitted use |
| --- | --- | ---: | --- |
| Observer fit | 10–19 | 120 | Ridge selection and initial decoder fit |
| Observer validation | 20–29 | 120 | Frozen observer gate and later final refit |
| Causal test | 30–39 | 120 | Primary causal outcomes |
| Reserve | 40–49 | 120 | Exact replication only after a fully positive causal test |

The reserve split may never rescue a negative or inconclusive primary test. States are replicates
nested in 12 scene-prompt contrasts; they do not turn this into a 120-scene population.

Before any model request, a runtime receipt must bind the frozen protocol and manifest to the
simulator-only 480-state preflight, the parent checkpoint hash, exact LIBERO-CF and OpenPI commits,
the source tree, and one visible non-distributed RTX 4090.

## Observer and pre-causal gate

The feature is the per-token donor-minus-recipient residual output at PaliGemma layer 13. Token
positions are never pooled. A zero-intercept ridge map predicts the three-dimensional
donor-minus-recipient object displacement. The zero intercept makes zero activation change map to
zero decoded target change, which is required for a meaningful minimum-norm inverse.

Ridge strength is selected from the frozen scale-relative grid in the JSON protocol using
leave-one-scene-out prediction on observer-fit states. The resulting decoder is evaluated on the
observer-validation states against global, ordered-subject-pair, and exact-prompt-pair means fit
only on observer-fit states.

Causal rollout is authorized only if validation:

1. has activation R² at least 0.05;
2. reduces squared error at least 10% relative to the exact-prompt-pair baseline; and
3. has positive mean target-direction cosine in at least 9 of 12 scenes.

After this gate passes, the selected decoder is refit on fit plus validation states. Causal-test
and reserve files are neither loaded nor inspected during fitting. A create-only clearance binds
the exact protocol, manifest, analysis, and controller hashes.

## Interventions

Every causal-test unit runs six conditions in a deterministic shuffled order:

1. clean donor;
2. clean recipient;
3. minimum-norm target controller;
4. the full donor difference projected into the decoder column space;
5. an independently sampled random direction norm-matched to the minimum-norm controller; and
6. complete donor replay at PaliGemma layer 13.

For the primary controller, the intervention is the smallest decoder-space activation change that
maps to the desired 3D target displacement under a damped inverse. Its norm is capped at the norm
of the corresponding complete donor-minus-recipient activation difference. No strength is tuned
against behavior. All patch conditions execute on every matching layer call during each replan.

Each condition runs at most 12 replans with five actions executed per replan and stops at first
contact with either candidate object. This design measures target selection, not completion of the
subsequent placement task.

## Endpoints and decision rule

The primary effect is the paired change in donor-object first touch between the minimum-norm
controller and clean recipient. Selectivity is the same comparison against the norm-matched random
direction. Complete donor replay is a required assay manipulation check on the same untouched
population.

Scene-cluster bootstrap intervals and exact scene sign-flip tests use scenes as inference clusters.
Compact target control is supported only if every frozen criterion in
`config/target_control_tc1.json` passes, including:

- full replay improves donor first touch by at least 20 percentage points with an interval above
  zero and at least 9 positive scenes;
- the minimum-norm controller improves donor first touch by at least 15 points over clean
  recipient and 10 points over matched random;
- both controller intervals are above zero, at least 9 scenes are positive, every patch receipt is
  exact, and no unit is invalid.

A positive primary test authorizes an unchanged reserve replication. A final positive claim
requires the entire rule to pass again. Failure of full replay makes the controller result
assay-inconclusive. Failure of the learned controller after successful replay is evidence only
against this controller, layer, checkpoint, and frozen scene population.

## Precision and power boundary

The design is intended to detect a large redirection effect comparable to the M0 replay result.
Only 12 scene clusters exist. A null result does not exclude a 15-percentage-point effect unless
the scene-cluster interval itself excludes that value. Using M0's across-scene standard deviation
of 0.332, a two-sided noncentral-*t* approximation gives about 30% power for a 15-point effect,
48% for 20 points, and 81% for 30 points. The practical 80%-power effect is therefore about 30
points under M0-like heterogeneity. Additional state replicates reduce within-scene noise but do
not create independent scenes. This calibration is recorded in
`results/target_control_m0/tc1_power_calibration.json`.

## Compute and stopping

The study is capped at one non-distributed RTX 4090 and USD 5 of new rental cost. The execution
order is simulator preflight → observer collection → observer gate → causal test → conditional
reserve replication. Any failed gate stops later stages. There is no model training, layer sweep,
invalid-unit retry, or outcome-conditioned stimulus replacement.
