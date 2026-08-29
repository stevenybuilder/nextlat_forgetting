# Intended-futures geometry in π0.5: pilot report

## Bottom line

In a frozen π0.5-LIBERO policy, action-expert activations linearly encode which object an official
language instruction selects from an otherwise identical scene. That signal generalizes across
held-out scenes and explains scene-specific target variation beyond prompt identity. However, a
preregistered rank-three activation patch at the selected layer and first denoising call did not
selectively redirect the robot toward the donor target.

This is evidence for **linearly accessible, scene-conditioned intended-future geometry**, not yet
evidence for a causally controlling future subspace or a general world model.

## Question and hypotheses

The experiment asks: when two instructions specify different valid futures from one physical
state, does π0.5's action expert represent the resulting change in intended target, and does the
policy use that representation to form its action chunk?

- **Representation alternative:** prompt-conditioned activation differences predict the
  three-dimensional change in intended target position in a held-out scene.
- **Representation null:** activation differences add no held-out information beyond task-language
  structure or action differences.
- **Causal alternative:** inserting the donor-minus-recipient component in the learned future
  subspace makes the recipient rollout progress selectively toward the donor target.
- **Causal null:** the learned intervention is no more directional than a norm-matched random
  subspace or generic activation corruption.

## Stimuli and controls

The final v4 population uses 12 official LIBERO-CF Spatial-Focused scenes and ten official initial
states per scene. Within each state, two verbatim official BDDL instructions select different
objects from cookie, ramekin, and the faithful black bowl. The paired policies receive the same
pixels, proprioception, initial simulator state, model, and random action noise.

Before model output, an exhaustive simulator preflight verified all three candidate objects in all
120 states, restricted them to the workspace, and required at least 5 cm separation. Ordered
subject direction was balanced: each object appeared four times on each side across scenes, and
all six ordered subject contrasts appeared twice. One proposed scene was excluded during preflight
because two targets could be only 7.8 mm apart; it was replaced before v4 was frozen.

The policy baseline used official LIBERO-Spatial tasks 0 and 1 at states 0–9. It succeeded in
19/20 episodes, exceeded the frozen 80% gate, produced zero invalid episodes, and used no retries
or replacements.

Candidate action-expert layers were 5, 11, and 17. We retained the full 10×1024 action-token grid
at the first of ten flow-denoising calls. A rank-three ridge map predicted the within-state target
position difference with leave-one-scene-out cross-validation; scenes, not tokens or states, were
the inference clusters.

## Representation result

| Model or control | Held-out R² | Mean cosine |
| --- | ---: | ---: |
| Global intercept | -0.189 | 0.337 |
| Ordered subject-pair mean | 0.677 | 0.871 |
| Exact prompt-pair mean | 0.764 | 0.863 |
| Action-chunk difference | 0.420 | 0.849 |
| Layer-17 activation difference | **0.850** | **0.935** |

All candidate layers passed the representation screen; layer 17 was best and was positive in
12/12 held-out scenes. Relative to the exact-prompt-pair mean, layer-17 activations reduced
residual squared error by 36.6%. The mean target-displacement norm was 36.8 cm, while the mean
within-scene RMS variation was 2.0 cm. The activation result therefore reflects both strong task
structure and additional scene-conditioned information.

The prompt and action controls were added after the primary representation screen, so the clean
claim is descriptive and exploratory rather than confirmatory. They make an important distinction:
the high R² is not wholly visual geometry, but it is also not reducible to prompt identity alone.

## Causal result

Layer 17 and its fitted rank-three subspace were frozen before causal outcomes. The causal stage
used states 0, 4, and 9 from every scene: 36 independent matched units. Each unit ran four
deterministically shuffled conditions for three replans and five executed actions per replan:

1. no patch;
2. learned future-subspace patch from task A into task B;
3. seeded rank-three random-subspace patch with matched norm; and
4. full donor-activation replacement as a nonspecific positive control.

All 36 units were valid and none was retried or replaced.

| Contrast: donor-target progress | Mean | Scene-standardized effect | Positive scenes | Exact p |
| --- | ---: | ---: | ---: | ---: |
| Learned future subspace − none | **1.20 mm** | 0.125 | 7/12 | 0.796 |
| Random subspace − none | 0.77 mm | 0.128 | 6/12 | 0.664 |
| Learned − random | 0.42 mm | 0.042 | 7/12 | 0.900 |
| Full donor − none | 2.03 mm | 0.224 | 4/12 | 0.567 |

The primary learned-patch 95% scene-cluster bootstrap interval was [-3.22, 6.93] mm. Its
standardized effect missed the frozen 0.30 gate, only 7/12 rather than 9/12 scene effects were
positive, and the random control exceeded its maximum permitted effect of 0.10. Consequently all
three causal gates failed.

The null causal result is local to this intervention: layer 17, a linear rank-three subspace, the
first denoising call, donor A into recipient B, and the first 15 executed actions. Full donor
replacement also failed to redirect behavior reliably, so this run cannot distinguish an
epiphenomenal code from a causal code that requires another layer, denoising time, nonlinear
coordinate system, or multi-site intervention.

## Invalidated and aborted studies

- **v1** is an infrastructure abort. OpenPI cached the compiled sampler before instrumentation;
  the first request never returned an action. Its single invalid record was retained, and v1 was
  never reused.
- **v2** passed the behavioral baseline and extracted 50 matched pairs, but its target-label
  resolver confused a biased benchmark goal object with the object named by language. Its negative
  representation result is invalid for the intended-future hypothesis and is reported only as an
  engineering trace.
- **v3** was rejected before model output because exhaustive preflight found an unsuitable
  near-coincident target contrast. No v3 model outcome exists.
- **v4** is the corrected balanced study reported above.

These exclusions were not seed replacement: each new version corrected a documented measurement
or stimulus-contract failure and received a new study identity before its outcomes were observed.

## Compute and provenance

The final representation collection took 652.8 seconds and the causal collection took 1,365.5
seconds. Including checkpoint setup, baseline, invalid-design diagnosis, validation, and all study
versions, the single RTX 4090 rental ran for approximately 6.01 hours at $0.3611/hour, or about
**$2.17** of GPU rental before storage/network charges. The instance was stopped after local backup;
it was not destroyed because it also contains earlier JEPA work.

The public model weights total 7.23 GB and are identified by a tracked tree hash. The final
representation manifest, runtime source, causal config, fitted subspace, and causal runtime source
all have tracked SHA-256 receipts. Every retained raw v4 pair and causal record has a published hash;
the large raw arrays themselves are intentionally excluded from Git.

## Interpretation and next experiment

The useful finding is a dissociation: **decodable future geometry is much easier to establish than
causal control through that geometry**. That matters for representation-learning work because a
high-quality linear probe—even one that generalizes across scenes and beats language baselines—does
not establish that the policy uses the decoded coordinates.

A follow-up is justified only if it targets the failed causal identification directly: freeze a
small layer × denoising-time intervention grid, require a positive full-donor control at each site,
and test a held-out confirmatory scene set. Retraining π0.5 or expanding to another architecture is
not warranted until an intervention site can first move behavior in this checkpoint.
