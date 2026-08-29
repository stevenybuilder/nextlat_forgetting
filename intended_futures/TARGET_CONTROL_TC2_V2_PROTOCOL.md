# Simulator-bound layout geometry

## Correction boundary

TC2-v1 stopped before its fourteenth observer request. Its BDDL-center validity check rejected
`between_the_plate_and_the_ramekin-l5-s2` because the saved MuJoCo state differed from the center
of its declared placement regions by more than 3 cm. Thirteen v1 activation records exist, but
they were not statistically inspected, fitted, or reused.

The failed check revealed a measurement error in the design: a BDDL region center describes a
placement region, not necessarily the fixed state shown to the policy. This version therefore
binds every response label to the simulator-measured position in the exact official saved state,
after the same ten no-op settling steps used during evaluation. BDDL geometry remains provenance
and candidate-enumeration information only.

Before freezing v2 and without a policy server, all 158 v1 candidate states were loaded under the
pinned LIBERO-CF simulator and OSMesa renderer. Two official sample-1 states were physically
invalid for this assay after settling:

- `from_table_center-l3-s1`: bowl 2 fell outside the workspace to `y=1.197 m, z=0.425 m`.
- `next_to_the_plate-l3-s1`: bowl 1 fell outside the workspace to `y=0.749 m, z=0.582 m`.

The exclusion rule was applied to physical state coordinates, not activations or policy behavior:
both instruction-selected subjects must have absolute XY at most 0.5 m and Z in `[0.85, 2.0]`.
The resulting frozen population has 156 states: 79 observer-fit, 38 untouched observer-validation,
and 39 causal-test states across the same nine task families. Noise seeds remain attached to their
original stimulus IDs.

## Pre-model geometry gate

The simulator-bound population passed every frozen check before v2 model output:

- minimum target separation: 10.4 cm;
- pooled prompt-family-residual RMS: 3.78 cm;
- untouched sample-3 residual RMS: 4.19 cm;
- all nine task families span more than 11.4 cm; and
- every included fixed state satisfies the physical workspace contract.

At collection time, the simulator-measured target difference must reproduce its manifest value to
within `1e-6 m`; otherwise the version stops without retry.

## Hypotheses and observer gate

The scientific question and thresholds are unchanged. The null says that after subtracting the
observer-fit exact-prompt-family mean, PaliGemma layer-13 activation differences do not predict
held-out layout-specific target displacement. The alternative says that they do and that the
decoded information is causally usable.

Ridge strength is selected by leave-one-level-out cross-fitting on the 79 fit states, recomputing
prompt-family means within each fold. The 38 sample-3 states are evaluated once. Causal collection
is permitted only if all four conditions hold:

1. layout-residual R² is at least 0.10;
2. SSE is at least 10% lower than the fit-only exact-prompt-family mean;
3. mean residual-direction cosine is positive in at least seven of nine families; and
4. the first clean action-chunk positive control has nonnegative residual R².

Failure is terminal and creates no controller. The sample-4 causal states are not loaded.

## Causal gate

If the observer passes, v2 retains the preregistered six conditions: clean task A, clean task B,
the actual-layout minimum-norm controller, the prompt-family-mean controller, matched random, and
complete donor replay. Full replay must first pass its manipulation check. Compact control is
supported only if it beats clean task B and matched random on target-A touch, beats the prompt-mean
controller by at least 5 mm of target-A progress, has all required family-cluster confidence
intervals above zero, is positive in at least seven families, and has exact patch receipts with no
invalid unit.

## Claim and compute boundary

This remains an exploratory test of visible instruction-selected pickup-target XY at one fixed
PaliGemma layer in the public `pi0.5-LIBERO` checkpoint. It is not evidence of temporally extended
planning, a general world model, all VLA checkpoints, or real-robot generalization. The cumulative
Vast cap for TC2 remains $2.00 on one RTX 4090, with immediate stop after observer failure, causal
completion, error, or cap breach.
