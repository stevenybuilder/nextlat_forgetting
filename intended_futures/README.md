# Geometry of intended futures in a modern VLA

This completed pilot asks whether a frozen action-chunking VLA represents which future an
instruction selects from the same physical scene, and whether that representation is causally used
to construct the next action chunk.

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

The causal result was negative. Replacing the recipient policy's coefficients in the learned
rank-three future subspace moved early behavior toward the donor target by only **1.20 mm** on
average (95% scene-cluster bootstrap CI **[-3.22, 6.93] mm**; standardized scene effect **0.125**;
7/12 scenes positive; exact sign-flip p = **0.796**). A matched random subspace produced a similar
effect, and even full donor-activation replacement was not reliably directional at this intervention
site and denoising call. None of the three frozen causal gates passed.

The supported claim is therefore narrow but substantive: π0.5 exposes scene-conditioned geometry
about an instruction-selected future at this action-expert site, but this experiment does not show
that the fitted linear subspace at the first denoising call causally controls behavior. See the
[full results and limitations](RESULTS.md).

## Study design

- **Model:** public Physical Intelligence π0.5-LIBERO PyTorch-converted checkpoint.
- **Benchmark:** official LIBERO-CF Spatial-Focused scenes and verbatim official BDDL language.
- **Representation units:** 120 matched initial states nested in 12 held-out scene clusters.
- **Causal units:** three frozen states from every scene, for 36 units and four rollout conditions.
- **Intervention:** learned future subspace, matched random subspace, full donor replacement, and
  no intervention; 15 early closed-loop steps per condition.
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
| `manifests/pilot_v4_stimuli.json` | Exact 120 matched contrasts |
| `src/intended_futures/` | Validation, geometry, intervention, and statistical primitives |
| `scripts/` | Collection, provenance, and analysis entry points |
| `tests/` | Leakage, topology, geometry, instrumentation, and inference-unit checks |
| `results/pilot_v4/` | Compact representation evidence and raw-file hashes |
| `results/causal_v4/` | Compact causal evidence and raw-file hashes |
| `ENGINEERING_LOG.md` | Aborted/invalidated designs and runtime deviations |

Raw activations, checkpoints, simulator caches, credentials, and provider state are excluded from
Git. Compact summaries, exact configurations, upstream commits, checkpoint hashes, source hashes,
and hashes for every raw record are tracked.
