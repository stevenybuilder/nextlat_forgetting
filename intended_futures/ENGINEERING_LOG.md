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

## Renderer

The Vast container exposes the RTX 4090 for CUDA compute but was created with
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, so matching NVIDIA EGL graphics libraries were not
mounted. OSMesa successfully passed a render-only simulator smoke test. V2 therefore freezes
OSMesa for deterministic CPU simulation rendering while retaining the single 4090 for all model
inference. The behavioral baseline gate is required before interpreting representations.
