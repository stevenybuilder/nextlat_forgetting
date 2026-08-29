# Geometry of intended futures: π0.5 engineering pilot

> Completion note: this is the frozen v2 protocol and is retained unchanged as historical
> provenance. V2's target labels were invalidated after extraction, v3 failed stimulus preflight,
> and the corrected balanced v4 study is governed by `config/pilot_v4.json` and
> `config/causal_v4.json`. Outcomes are reported in `RESULTS.md`.

Status: frozen before any π0.5 activation or behavioral outcome from this study. The exact stimulus
population is recorded in `manifests/pilot_stimuli.json`.

## Research question

When two language instructions specify different valid futures from the same physical state, does
π0.5's action expert contain a structured representation of that intended future, and is that
representation causally used to form the subsequent action chunk?

This is a planning-representation question. A positive result is not, by itself, proof that π0.5
learned a general environment dynamics model.

## Population and stimuli

The population is the official π0.5 LIBERO checkpoint evaluated on the five exact matched-state task
pairs in the official LIBERO-CF Spatial-Focused suite at commit
`8460457bfca6e0ef2e856bc104e2c60b023ef2a7`.

The pair resolver hashes every official `.pruned_init` file and admits a pair only when:

1. exactly two task files have byte-identical initial-state files;
2. their language instructions differ;
3. their task indices and intended subjects equal the values frozen in `config/pilot.json`.

The five contrasts are cookie versus ramekin, cookie versus a bowl next to the plate, ramekin versus
a bowl on the cabinet, ramekin versus a bowl in the cabinet, and one black bowl versus the other
black bowl under different spatial descriptions. Each contrast uses official initial-state indices
0 through 9, for 50 matched stimuli.

The same model noise is used within each prompt pair. There is no seed replacement after observing
an action, activation, reward, touch, or success.

## Representation and future target

Candidate action-expert layers are 5, 11, and 17. Activations are captured at the first denoising
call with all ten action-token positions retained. Layer choice is a discovery result; any later
confirmatory run must freeze one layer in advance.

For each matched state, the target is the three-dimensional difference between the two instructed
objects' simulator positions relative to the common end-effector position. The activation target is
the corresponding prompt-conditioned action-expert grid difference.

A rank-three reduced-rank ridge map with ridge coefficient 0.01 is evaluated with
leave-one-task-pair-out cross-validation. No frame, token, denoising step, or action is counted as an
independent observation.

## Hypotheses

**Representation alternative.** Prompt-conditioned activation differences encode the change in
the intended target position and generalize to a held-out task pair.

**Representation null.** The activation difference contains no held-out information about the
intended target beyond pair-specific language or motor structure; cross-validated future-delta
prediction is no better than the frozen controls.

**Causal alternative.** Replacing only the recipient coefficients in the fitted future subspace
with donor coefficients at the first denoising call increases early end-effector progress toward
the donor target.

**Causal null.** Donor-subspace patching has no larger directional effect than a seeded, same-rank
random subspace and merely introduces generic action corruption.

**Shortcut alternative.** The model exposes intended-goal information in its activations but its
closed-loop behavior follows a visually cued memorized target. This predicts a representation
effect without the selective causal effect required for advancement.

## Endpoints and controls

The primary representation endpoint is leave-one-task-pair-out R-squared for intended target-position
differences. Secondary diagnostics report per-pair R-squared, action difference, activation norm,
and layer-wise results.

The primary causal endpoint is donor-target progress over three replans relative to the unpatched
recipient policy. The intervention swaps the projected donor-minus-recipient activation at the
first denoising call while preserving all action-token positions.

Controls are:

- no intervention with the identical state, prompt, and noise;
- a seeded same-rank random-subspace intervention with matched donor, recipient, and strength;
- a full donor activation replacement as a positive but nonspecific control;
- action-norm, invalid-action, and generic trajectory-degradation checks.

Full task success and LIBERO-CF faithful grounding are secondary because this small pilot is powered
around the continuous early-trajectory endpoint.

## Gates

The study advances to the causal stage only if the best discovery layer has cross-validated future
R-squared of at least 0.05 and at least four of five task-pair effects have the predicted sign.

The pilot recommends a larger confirmation only if the future-subspace intervention has a
standardized donor-progress effect of at least 0.30, at least four of five task pairs have positive
effects, and the random-subspace effect is no more than 0.10 standard deviations above the
unpatched policy.

With only five task-pair clusters, the minimum nonzero two-sided exact sign-flip p-value is 0.0625.
Consequently these thresholds are engineering go/no-go rules, not confirmatory significance tests.

## Runtime and baseline gate

The model server must use exactly one visible RTX 4090, bfloat16 inference, no distributed process,
official normalization statistics, the official ten-action horizon, and five executed actions per
replan. The simulator uses deterministic OSMesa CPU rendering because the rented container exposes
the 4090 for compute but not the matching NVIDIA EGL graphics libraries; model inference remains on
`cuda:0`. The renderer is fixed across every task, prompt, and condition. Checkpoint and
converted-weight hashes are recorded before any pilot output.

Before the matched pilot, the converted PyTorch policy must complete a standard LIBERO smoke
baseline with valid actions and at least 80% success over 20 preregistered episodes: official
LIBERO-Spatial task indices 0 and 1, each at official initial-state indices 0 through 9. Failed or
invalid episodes are not retried or replaced. A failed baseline blocks interpretation and triggers
engineering diagnosis; it does not permit changing pilot states, prompts, layers, or thresholds.

## Claim boundary

A representation-only result supports linear accessibility, not causal use. A full-activation-only
effect supports entangled task information, not a specific future subspace. A selective projected
effect supports a causally used representation of intended targets in this checkpoint and benchmark.

No result from this pilot establishes a general VLA mechanism, causal effects of pretraining, or
real-robot validity. Those require a larger task population, a second architecture such as VLA-JEPA,
and eventually a matched training-objective ablation.
