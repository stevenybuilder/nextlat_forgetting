# Geometry of intended futures in a modern VLA

This study asks whether a frozen action-chunking VLA represents the future requested by language,
rather than merely replaying the motor program most strongly cued by the current scene.

The first experiment uses the official public π0.5 LIBERO checkpoint and the official
LIBERO-CF Spatial-Focused benchmark. Five pairs of LIBERO-CF tasks have byte-identical simulator
initial-state files but different language instructions and intended target objects. Evaluating the
same ten official initial states for both members of each pair yields 50 matched contrasts in which
the scene, robot state, object layout, checkpoint, action noise, and GPU topology are fixed.

The primary representation is the action-expert residual stream at the first flow-denoising call.
Action-token positions are retained. A reduced-rank ridge map is fit to the difference between the
two prompt-conditioned activation grids and the difference between their intended target positions.
Generalization is evaluated by leaving out an entire task pair. Probe performance alone is not a
positive result: the project advances only if a donor future-subspace intervention moves the early
closed-loop trajectory toward the donor target more than matched random-subspace controls.

This pilot performs no model training and creates no new task dataset. It generates evaluation
rollouts and activations from public checkpoints and public benchmark states.

## Layout

| Path | Purpose |
| --- | --- |
| `PREREGISTRATION.md` | Frozen hypotheses, estimands, gates, and claim boundary |
| `config/pilot.json` | Machine-readable scientific and runtime contract |
| `manifests/pilot_stimuli.json` | Fifty exact matched-state contrasts resolved before outcomes |
| `src/intended_futures/` | Pair validation, future-subspace geometry, interventions, and statistics |
| `scripts/` | Manifest, runtime, rollout, and analysis entry points |
| `tests/` | Leakage, topology, geometry, and inference-unit checks |

Raw activations, videos, converted checkpoints, simulator caches, credentials, and provider state
are never tracked. Compact summaries and provenance receipts may be published after validation.

## Current execution state

The v2 stimulus manifest is frozen at SHA-256
`6db156eedbc0555a6355533b144110a7d14879d8f4131b3f5275144924c173ce`. Runtime provisioning and
checkpoint conversion must complete before the standard LIBERO baseline gate. Pilot outcomes must
not be inspected until the checkpoint/runtime receipt is written. V1 was aborted before a policy
action returned because the instrumented wrapper left OpenPI's cached compiled sampler active; its
single invalid infrastructure record is retained and is not part of v2.
