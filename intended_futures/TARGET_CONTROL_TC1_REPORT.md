# When high VLA probe accuracy is mostly prompt identity

## Result in one sentence

A linear observer recovered instruction-selected target displacement from π0.5's PaliGemma layer
13 with validation R² 0.953, but a training-only prompt-pair lookup reached 0.995, so the frozen
experiment correctly stopped before claiming state-conditioned geometry or running a causal patch.

## Why this test mattered

Earlier full activation replay at PaliGemma layer 13 changed which object π0.5 touched first: donor
touch rose from 11.1% under the recipient prompt to 58.3% under replay, near the clean donor's
61.1%. That established a useful broad sufficiency result, but not a compact representation. TC1
asked whether a small linear controller could isolate the target information responsible for the
effect.

The important confound was known in advance. Each LIBERO-CF instruction pair names a particular
ordered pair of objects, and the official initial-state bank may vary their relative positions only
slightly. A decoder can therefore look geometric while primarily identifying the prompt pair. TC1
required the activation model to improve on a prompt-pair predictor before any intervention.

## Frozen design

- Public π0.5-LIBERO checkpoint; one RTX 4090; bfloat16 inference; no distributed execution.
- Twelve official LIBERO-CF Spatial-Focused scenes and verbatim BDDL instructions.
- Previous outcome-bearing states 0–9 excluded.
- States 10–19: 120 observer-fit units.
- States 20–29: 120 untouched observer-validation units.
- States 30–39: unopened causal-test bank.
- States 40–49: unopened conditional-replication bank.
- Feature: the full donor-minus-recipient 968×2048 residual grid at zero-indexed PaliGemma layer
  13; no token pooling.
- Target: the simulator-measured 3D donor-minus-recipient object displacement.
- Model: zero-intercept dual ridge, with ridge selected by leave-one-scene-out fit-bank R².

Advancement required all three validation checks: R² ≥ 0.05, positive direction cosine in at least
9/12 scenes, and at least 10% residual-error reduction over a prompt-pair mean estimated only from
fit states.

## What happened

| Validation model | R² |
| --- | ---: |
| Global mean | -0.000 |
| Ordered object-pair mean | 0.915 |
| Exact prompt-pair mean | **0.995** |
| PaliGemma-13 activation observer | 0.953 |

The activation observer's mean cosine was 0.993 and every scene was positive. Those numbers would
look compelling without the negative control. Against the prompt-pair baseline, however, its
residual error was 9.80 times larger. The incremental-information criterion failed.

No controller or causal-clearance artifact was created. No causal-test or reserve state was loaded.
This was not a null behavioral intervention; the experiment stopped before such an intervention
was scientifically authorized.

## Interpretation

The result separates two claims that are easy to conflate:

1. PaliGemma layer 13 participates in instruction-conditioned target routing. Broad donor replay
   provides descriptive causal evidence for this.
2. A compact linear code at that layer represents state-specific target coordinates. TC1 does not
   support this claim on the official state bank.

The two findings are compatible. A distributed prompt-conditioned state can be causally sufficient
under full replay while a linear coordinate observer adds no information beyond prompt identity.
High absolute probe accuracy is therefore not enough; the relevant test is improvement over the
strongest non-representational baseline available from the stimulus design.

## What would make the next experiment identifiable

The next dataset must repeat the same instruction contrast across materially different object
layouts. The primary held-out split should be by layout or scene, and the activation observer must
beat a training-only prompt-pair predictor on those layouts. Only after that gate passes is it
meaningful to compare a minimum-norm coordinate intervention with a norm-matched random direction
and full replay.

Weakening the baseline, pooling more states from the same narrow layout distribution, or fitting a
controller to this failed observer would produce a cleaner-looking figure but not stronger
evidence.

## Reproducibility

The protocol, 480-row manifest, simulator preflight, runtime receipt, analysis, source, and compact
raw-archive receipt are in this repository. The 240 raw activation files total 594.4 MB and are
excluded from Git; all 243 remote artifact hashes were verified after local transfer. The session
stopped at $4.13 accumulated Vast rental, below the frozen $5 cap.
