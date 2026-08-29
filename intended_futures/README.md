# Geometry of intended futures in a modern VLA

This completed study asks whether a frozen action-chunking VLA represents which future an
instruction selects from the same physical scene, whether that representation is causally used to
construct the next action chunk, and whether a donor instruction's target can be transplanted by
replaying internal activations.

Using the public π0.5-LIBERO checkpoint and official LIBERO-CF/LIBERO stimuli, the final study
evaluated 120 matched prompt contrasts across 12 scenes. The image, robot state, object layout,
checkpoint, action noise, GPU topology, and renderer were fixed within each contrast; only the
official language instruction changed. Every scene contained all candidate objects with at least
5 cm separation, and instruction order was balanced across subjects.

## Result

The action-expert residual stream contains a strong linearly accessible signal for the intended
target displacement. At layer 17, leave-one-scene-out prediction reached R² = **0.850** with a mean
cosine of **0.935**, positive in all 12 held-out scenes. The activation model reduced residual
squared error by **36.6%** relative to an exact-prompt-pair mean baseline, showing that the result is
not only a language-template lookup.

The original causal result was negative. Replacing the recipient policy's coefficients in the learned
rank-three future subspace moved early behavior toward the donor target by only **1.20 mm** on
average (95% scene-cluster bootstrap CI **[-3.22, 6.93] mm**; standardized scene effect **0.125**;
7/12 scenes positive; exact sign-flip p = **0.796**). A matched random subspace produced a similar
effect, and even full donor-activation replacement was not reliably directional at this intervention
site and denoising call. None of the three frozen causal gates passed.

A gated follow-on corrected the intervention schedule and tested both model pathways. Replaying
PaliGemma layer 13 on its one matching call nearly reproduced the clean donor policy: donor-object
first touch was **58.3%**, versus **61.1%** for clean donor, **11.1%** for clean recipient, and
**13.9%** for a norm-matched random perturbation. Donor-target progress improved by **17.7 cm**
over clean recipient (scene-cluster 95% CI **[11.6, 24.2] cm**) and was positive in all 12 scenes.
Action-expert layer 9 instead moved behavior away from the donor.

The manipulation check nevertheless failed its frozen all-or-nothing advancement rule: PaliGemma's mean
donor-versus-recipient action-cosine margin was **0.051**, below the predeclared **0.10** minimum,
even though it passed the other six checks. The learned minimum-norm controller was therefore not
run in that study.

An independently frozen follow-up then collected 240 new PaliGemma-13 observer records. Its linear
decoder achieved validation R² **0.953** and positive direction cosine in 12/12 scenes, but a
training-only exact-prompt-pair baseline achieved **0.995**. Because the decoder had 9.80 times the
baseline residual error, it failed the required incremental-information gate. No controller was
fitted, and neither the causal-test nor reserve bank was opened. The supported claim is therefore
narrow: full PaliGemma replay can redirect target behavior, but this official state bank does not
show compact state-conditioned coordinate geometry beyond prompt identity. See the
[full results and limitations](RESULTS.md).

## Study design

- **Model:** public Physical Intelligence π0.5-LIBERO PyTorch-converted checkpoint.
- **Benchmark:** official LIBERO-CF Spatial-Focused scenes and verbatim official BDDL language.
- **Representation units:** 120 matched initial states nested in 12 held-out scene clusters.
- **Causal units:** three frozen states from every scene, for 36 units and four rollout conditions.
- **Intervention:** learned future subspace, matched random subspace, full donor replacement, and
  no intervention; 15 early closed-loop steps per condition.
- **Target-control gate:** 36 fixed units, clean donor/recipient, full replay and norm-matched random
  controls at PaliGemma layer 13 and expert layer 9, up to 60 actions, and official first-touch
  attribution.
- **Runtime:** one RTX 4090, bfloat16, eager PyTorch, no distributed execution, fixed within-unit
  noise, OSMesa simulation rendering, and no retry or replacement after an outcome.

The standard-policy baseline passed 19/20 official LIBERO-Spatial episodes with zero invalid
episodes before activations were interpreted.

## Reproducibility map

| Path | Purpose |
| --- | --- |
| `RESULTS.md` | Results, interpretation, exclusions, compute, and claim boundary |
| `PREREGISTRATION.md` | Original frozen v2 hypotheses and gates |
| `config/pilot_v4.json` | Corrected balanced representation protocol |
| `config/causal_v4.json` | Frozen causal protocol |
| `TARGET_CONTROL_PROTOCOL.md` | Frozen all-call replay and target-control advancement rules |
| `config/target_control_m0.json` | Exact manipulation-check population, sites, and thresholds |
| `TARGET_CONTROL_TC1_PROTOCOL.md` | Frozen independent observer/controller confirmation |
| `TARGET_CONTROL_TC1_REPORT.md` | Self-contained negative-result report and next-design criterion |
| `config/target_control_tc1.json` | Disjoint fit, validation, causal-test, and reserve banks |
| `TARGET_CONTROL_TC2_PROTOCOL.md` | Frozen LIBERO-Plus layout-identifiable observer and causal-control protocol |
| `config/layout_shift_tc2.json` | Exact TC2 hypotheses, gates, topology, and $2 stop rule |
| `manifests/target_control_tc1.json` | Exact 480-state TC1 population and split assignments |
| `manifests/pilot_v4_stimuli.json` | Exact 120 matched contrasts |
| `manifests/layout_shift_tc2_stimuli.json` | Exact 158 official LIBERO-Plus layout variants and splits |
| `src/intended_futures/` | Validation, geometry, intervention, and statistical primitives |
| `scripts/` | Collection, provenance, and analysis entry points |
| `tests/` | Leakage, topology, geometry, instrumentation, and inference-unit checks |
| `results/pilot_v4/` | Compact representation evidence and raw-file hashes |
| `results/causal_v4/` | Compact causal evidence and raw-file hashes |
| `results/target_control_m0/` | Frozen gate result, post-hoc touch audit, receipt, and raw hashes |
| `results/target_control_tc1/` | Preflight, runtime receipt, failed observer gate, and archive receipt |
| `results/target_control_tc2/` | Pre-model geometry audit and, only after execution, compact TC2 evidence |
| `ENGINEERING_LOG.md` | Aborted/invalidated designs and runtime deviations |

Raw activations, checkpoints, simulator caches, credentials, and provider state are excluded from
Git. Compact summaries, exact configurations, upstream commits, checkpoint hashes, source hashes,
and hashes for every raw record are tracked.
