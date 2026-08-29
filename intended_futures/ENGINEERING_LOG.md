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

## Target-control M0 — completed without TC1 advancement

The follow-on froze 36 units and two literature-selected sites before outcome collection. An
excluded state verified that PaliGemma layer 13 patched its single call and expert layer 9 patched
all ten denoising calls with zero shape mismatches. The production run then completed 36/36 valid
units in 4,424.3 seconds, with no retries or replacements.

PaliGemma full replay produced donor-like target touch and progress and passed six of seven frozen
checks. It failed the absolute 0.10 action-cosine-margin check with an observed margin of 0.0509.
The all-or-nothing gate was honored: the minimum-norm learned controller was not fitted or run.
Expert layer 9 moved behavior away from the donor. Raw records were copied locally and verified
against all 36 remote SHA-256 hashes before the Vast instance was stopped.

## Target-control TC1 — stopped at the observer gate

TC1 froze four disjoint official-state banks after a simulator-only 480-state preflight and
published the protocol and single-4090 runtime receipt before any model request. Observer
collection completed 240/240 records in four resumable batches with no invalid or non-finite
record. All raw records were copied locally and verified against their remote SHA-256 hashes.

The layer-13 decoder reached validation R² 0.953 and positive cosine in 12/12 scenes, but the
training-only exact-prompt-pair baseline reached R² 0.995. The required 10% residual improvement
therefore failed. Fitting exited with the preregistered no-go status and did not create a controller
or causal-clearance file. Causal-test and reserve state banks were not loaded. The Vast instance
was stopped at approximately $4.13 accumulated rental, below the frozen $5 cap.

## Renderer

The Vast container exposes the RTX 4090 for CUDA compute but was created with
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so matching NVIDIA EGL graphics libraries were not
mounted. Installing guest EGL packages was not viable because the host driver was 595.71 while the
available package was 595.84. The attempted packages were removed and the original 595.71 CUDA and
NVML links were restored. OSMesa passed a render-only simulator smoke test and was then frozen for
deterministic CPU simulation rendering in every study version; all model inference remained on the
single 4090. The behavioral baseline passed before representations were interpreted.

## Layout TC2-v1 — aborted on an invalid geometry surrogate

TC2-v1 published its protocol and runtime receipt before model output, then completed 13 observer
records. Before the fourteenth policy request, the frozen 3 cm BDDL-to-simulator check failed on
`between_the_plate_and_the_ramekin-l5-s2`. The instance was stopped immediately. The records were
backed up but never fitted, statistically inspected, or reused.

A policy-free scan of all 158 fixed states showed that BDDL region centers are not valid labels for
saved MuJoCo states; the maximum difference was 89.2 cm. Two sample-1 states also placed a selected
bowl outside the physical workspace after settling. This was a measurement-design failure, not a
negative model result.

## Layout TC2-v2 — completed without causal advancement

V2 was frozen with exact simulator-state labels and a coordinate-only physical validity rule before
new model output. It excluded the two fallen-object states, retained 156 official states, and
published the revised protocol, manifest, preflight, and runtime receipt. All 117 observer records
completed without retry or invalidity.

The representation checks passed at R² 0.1066, 10.66% improvement over the prompt-family mean, and
positive cosine in 7/9 families. The action-chunk positive control failed at R² -0.0309. The frozen
all-or-nothing gate was honored: no controller or clearance was created and no sample-4 state was
loaded. Raw evidence was verified locally before the replacement instance was destroyed.
