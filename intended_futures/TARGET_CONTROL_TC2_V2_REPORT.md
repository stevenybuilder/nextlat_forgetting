# Layout-specific intended-future geometry in pi0.5-LIBERO

## Result

PaliGemma layer 13 carried a modest amount of held-out layout-specific target-position information,
but the preregistered observer gate failed because the first action chunk did not carry the required
positive-control signal. The causal intervention was therefore not run.

This is a partial result, not support for the full alternative hypothesis. It supports the narrow
descriptive statement that a linear decoder at this fixed site predicted some target displacement
left over after subtracting exact-prompt-family means. It does not show that π0.5 used that
information to choose actions, and it provides no causal evidence.

## Design

- Model: public PyTorch-converted `pi0.5-LIBERO` checkpoint.
- Site: PaliGemma layer 13, fixed before TC2 from prior localization and replay evidence.
- Stimuli: official LIBERO-Plus `Objects Layout` target-displacement BDDLs and fixed states.
- Pair: identical image and robot state with two truthful official Spatial instructions selecting
  different black bowls.
- Target: exact simulator-measured bowl-1 minus bowl-2 XY in the settled state shown to the model.
- Population: 156 physically valid states across nine task families.
- Split: official samples 1–2 for fit (79), sample 3 for one-time validation (38), and sample 4
  reserved for causal testing (39).
- Analysis: prompt-family residualization, leave-one-level-out ridge selection on fit states, then
  one evaluation on sample 3.

Two sample-1 states were excluded before v2 model output by a coordinate-only physical validity
rule because an instruction-selected bowl fell outside the workspace during settling. The excluded
IDs, exact positions, hashes, and rule are in the frozen manifest. No state was selected using π0.5
behavior.

## Observer outcomes

| Frozen check | Threshold | Observed | Result |
|---|---:|---:|---|
| Layout-residual R² | ≥ 0.10 | 0.1066 | Pass |
| SSE improvement over prompt-family mean | ≥ 10% | 10.66% | Pass |
| Families with positive mean cosine | ≥ 7/9 | 7/9 | Pass |
| First-action-chunk residual R² | ≥ 0 | −0.0309 | **Fail** |

The ridge fraction selected by fit-only leave-one-level-out cross-validation was `1.0`. Its
cross-validated fit R² was 0.0431; the untouched sample-3 R² was 0.1066. Family mean cosines were
positive for seven families and negative for `from_table_center` and `on_the_cookie_box`.

Because the action positive control failed, the script exited with the frozen no-go status. It did
not create a controller or causal-clearance artifact, and no sample-4 state was loaded.

## What the result means

The result weakens a simple claim that intended-target geometry at layer 13 is already reflected in
the first decoded action chunk. It does not erase the representational signal: the layer decoder
beat the training-only prompt-family baseline on new layouts by the preregistered margin. The two
facts can coexist if the signal is weak, not linearly expressed in the first action chunk, used only
later in the trajectory, or partly incidental to computation.

The honest conclusion is therefore:

> In this fixed π0.5 checkpoint and LIBERO-Plus population, PaliGemma layer 13 modestly encoded
> layout-specific instruction-selected target displacement beyond prompt identity, but this pilot
> did not establish behavioral use or causal control.

No frequentist significance claim is made from the R² threshold alone. There are only nine
independent task-family clusters, and the result is close to the frozen boundary.

## Design correction from v1

TC2-v1 used BDDL region centers as a validity surrogate and stopped before record 14. A subsequent
policy-free scan of all 158 fixed states showed why: BDDL centers can differ dramatically from the
saved state, with a maximum discrepancy of 89.2 cm when an object fell out of the workspace. V2
bound labels directly to simulator states and excluded exactly two physically invalid states before
new model output. The 13 v1 records were archived but never inspected statistically, fitted, or
reused.

## Integrity and compute

All 117 v2 observer records completed without retry, replacement, non-finite activation, or
manifest mismatch. The local backup contains 281.6 MB of raw NPZ data and reproduces the frozen
simulator coordinates exactly. The replacement Vast instance used about 3,701 active seconds,
approximately $0.38 at its $0.372/hour rate, and was destroyed after backup. Unrelated instances
were not touched.

## Next scientific step

Do not relax the failed gate or run the planned causal test on these data. A justified follow-up
would be a separately preregistered temporal analysis asking whether target geometry becomes more
action-aligned at later replans or action-expert layers. That is a new hypothesis and must use a new
held-out population or benchmark; it cannot retroactively rescue TC2-v2.
