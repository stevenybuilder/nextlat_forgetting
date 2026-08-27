# D-38 — Pre-compute HMM centering amendment

**Status:** ratified before any model training or outcome inspection, 2026-08-23.

The HMM near-lure evaluator now estimates one center per model from the union of the near-lure
and supplied control state arrays. The previous implementation estimated that center from the
near-lure arrays alone and then evaluated controls against a lure-defined origin. That asymmetry
could manufacture or erase a separation contrast. GPT and NextLat still receive the same
estimation rule; their center vectors are necessarily separate because their hidden-state bases
are unrelated.

This is an estimator change, so it is recorded as a preregistration amendment rather than hidden
as an implementation cleanup. It was accepted while the project had no trained models and no
scientific result table. It does not authorize any further estimator, layer, threshold, seed, or
hypothesis changes after compute begins.

Frozen implementation identity:

```text
src/hmm_geometry/evaluate.py
sha256 d285ecdd250a62117f2ec61649b37329df598e393b67642067ff2a67e7249db7
```

The evaluator also reports whether the supplied control is edit-distance and prefix-length
matched. A separation index with `control_is_edit_matched: false` is diagnostic only and must not
be presented as the confirmatory HMM-H1 effect.
