# Engineering deviations and aborted runs

## Pilot v1 — aborted before scientific output

The v1 baseline was started on 2026-08-28 with the scientific population and thresholds now used
by v2. OpenPI's `Policy` caches `model.sample_actions` during initialization. The initial
instrumentation replaced only `model.sample_actions`, leaving the cached compiled callable active.
The first request entered TorchInductor compilation and never returned an action before the server
was intentionally stopped. LIBERO recorded `task-00-state-00` as invalid because the WebSocket
closed with zero executed policy steps.

That invalid episode was not deleted or retried under the v1 study ID. V1 is aborted. V2 was frozen
with a new study ID and manifest hash after fixing and testing both sampler references, and before
its first policy request. The stimuli, states, prompts, checkpoint, thresholds, and analysis are
unchanged.

## Pilot v2 — scientific target invalidated

V2 passed its preregistered behavioral gate at 19/20 successful standard LIBERO-Spatial episodes,
with zero invalid episodes and no retries. It then completed all 50 matched activation pairs.

Post-run inspection found that the target-position resolver had treated LIBERO-CF's biased BDDL
goal object as the language-selected object in scenes containing two black bowls. In the focused
counterfactual suite, the object named by the instruction is often `akita_black_bowl_2`, while the
unchanged training-style goal expression can still refer to `akita_black_bowl_1`. Because the
dependent variable did not consistently describe the instruction-selected future, v2's negative
representation screen cannot answer the research question. The outputs and receipt remain tracked;
they were not reclassified as a model failure.

## Pilot v3 — rejected by stimulus preflight

V3 replaced the brittle pair resolver with balanced same-scene contrasts and required every
candidate object to be present in every scene. Exhaustive simulator validation found that the
proposed task-6 bowl-versus-cookie contrast included states where the two object centers were only
7.8 mm apart, below the frozen 5 cm endpoint-separation contract. V3 was rejected before any model
output. Task 7's official bowl-on-stove instruction replaced that contrast in v4.

## Pilot v4 — completed

V4 validated 120/120 official initial states across 12 scenes before model output, then completed
120/120 matched activation records. Layer 17 passed the representation gate with leave-one-scene-out
R² 0.850 and positive cosine in 12/12 scenes. The subsequent frozen causal stage completed 36/36
units with zero invalid records. Its learned future-subspace patch failed all three advancement
gates; see `RESULTS.md` for the effect estimates and claim boundary.

## Renderer

The Vast container exposes the RTX 4090 for CUDA compute but was created with
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so matching NVIDIA EGL graphics libraries were not
mounted. Installing guest EGL packages was not viable because the host driver was 595.71 while the
available package was 595.84. The attempted packages were removed and the original 595.71 CUDA and
NVML links were restored. OSMesa passed a render-only simulator smoke test and was then frozen for
deterministic CPU simulation rendering in every study version; all model inference remained on the
single 4090. The behavioral baseline passed before representations were interpreted.
