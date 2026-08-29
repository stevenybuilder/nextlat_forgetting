# Selective target control in π0.5

Status: **frozen engineering protocol before target-control outcomes**

This follow-on does not claim to measure a predicted future or a world model. It asks a narrower
causal question: can an intervention make a frozen π0.5-LIBERO policy act more like the policy
conditioned on a different instruction-selected object, without merely adding generic activation
noise?

## Why this stage exists

The completed v4 pilot decoded target displacement from action-expert activations, but its causal
test patched only the first of ten flow-denoising calls. Full donor replacement did not move
behavior at that site, so the experiment did not validate its own causal assay. Published
activation-replay work instead replaces the source activation on every matching forward call.

Target control therefore proceeds through a strict manipulation-check gate. A learned controller
is not evaluated unless full activation replay first produces selective donor-like behavior.

## Reference implementations

- Action Atlas, commit `b8b0db331df18fc30a3fd92c45ec721d35d3ee52`, supplies the sequential
  capture/replay convention and the prior same-scene injection result.
- LIBERO-CF, commit `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`, supplies scenes, initial
  states, condition object identities, and gripper-contact target-touch evaluation.
- OpenPI is pinned to reference commit `215abfb217dbac7d5f1273282331b9b1866c0479`; the runtime
  receipt must additionally record the exact converted checkpoint and runtime source hashes.

No code or model weights from those repositories are silently vendored. Adapted measurement logic
is attributed in source comments, and exact upstream commits are recorded here.

## Stage M0: manipulation check

### Population

M0 uses the already frozen balanced LIBERO-CF-derived v4 manifest. It is an engineering discovery
population, not a new confirmatory population. Three initial states (`0`, `4`, and `9`) from each
of the 12 scenes yield 36 matched units. The same pixels, robot state, simulator state, action
noise, GPU, renderer, and checkpoint are held fixed within a condition block.

### Conditions

Every unit includes clean donor and clean recipient rollouts. At each of two literature-selected
sites, the recipient additionally receives either full donor activation replay or a seeded random
direction matched to the donor-minus-recipient activation norm:

1. PaliGemma layer 13, selected because prior work localizes stronger goal discrimination there;
2. action-expert layer 9, selected because prior work reports maximal expert goal classification
   near that layer.

Source activations are captured from the donor prompt at the recipient's current observation and
noise draw on every replan. Full replay replaces every matching layer call in the recipient
forward pass. The random control applies an independent, norm-matched perturbation at every call.
Token positions are never pooled.

Each condition runs for at most 12 replans with five executed actions per replan and stops on the
first donor/recipient object contact or environment termination. Condition order is deterministically
shuffled within unit. Invalid units are retained and are never retried or replaced.

### Endpoints and gate

The manipulation check is about assay validity, not a scientific effect estimate. A site advances
only if all of the following hold:

1. every valid full-replay forward call is patched, with no shape mismatch;
2. the first patched action chunk is more similar to the clean donor than to the clean recipient
   in at least 9 of 12 scene-level means;
3. the mean donor-versus-recipient action-similarity margin is at least `0.10` and exceeds the
   matched-random margin by at least `0.10`;
4. donor-target progress versus the clean recipient has a scene-standardized effect of at least
   `0.50` with at least 9 of 12 scene means positive; and
5. the invalid-unit rate is zero.

Official target-touch identity is reported as a secondary behavioral endpoint. It is not required
to pass M0 because 60 actions may end before contact in some scenes.

If neither site passes, target-control work stops. Probe quality cannot rescue a failed
manipulation check.

## Stage TC1: learned target control

TC1 is authorized only after M0 passes. Its configuration and held-out manifest must be frozen and
hashed before any TC1 rollout. The M0-selected site and replay schedule cannot be changed.

The primary learned intervention is the minimum-norm activation change that moves a cross-fitted
linear target observer from the recipient target toward the donor target. The existing rank-three
projection is a secondary comparator. Conditions are:

1. clean recipient;
2. minimum-norm target controller;
3. existing rank-three target projection;
4. norm-matched random direction;
5. full donor replay.

The primary endpoint is donor-object first touch relative to clean recipient. Secondary endpoints
are recipient-object first touch, target progress, environment success, invalid actions, action
norm, and generic trajectory degradation. Scenes/tasks, not states, replans, actions, tokens, or
denoising calls, are the inference units.

TC1 supports selective target control only if the learned controller:

- improves donor first touch by at least 15 percentage points over clean recipient;
- has a scene-cluster confidence interval above zero;
- exceeds the random intervention by at least 10 percentage points;
- preserves valid actions and loses no more than five percentage points of general task success;
  and
- does not merely match full-donor trajectory replay while failing target selectivity.

A failed learned controller after a passed M0 gate is evidence only about the tested checkpoint,
site, schedule, observer family, and LIBERO-CF task population. It is not evidence that π0.5 lacks
goal representations or that VLAs lack controllable goals generally.

## Compute and stop boundary

M0 is capped at one single-GPU run and USD 5 of new rental cost. No VLA or SAE training is
authorized. TC1 receives a separate USD 5 cap only after M0 passes. A cap overrun stops the run;
it does not authorize reducing controls or replacing failed units.

