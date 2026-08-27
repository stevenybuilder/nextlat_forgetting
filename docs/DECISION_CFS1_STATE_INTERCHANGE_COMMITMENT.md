# Decision: execute the CFS-1 state-interchange study

**Recorded:** 2026-08-24T18:09:55-04:00  
**Status:** outcome-blind operational commitment

No CFS-1 adaptation-branch result, state-interchange result, or CFS-1 scientific
metric had been run or inspected when this decision was recorded. The decision is
therefore not a response to an observed effect.

The state-interchange analysis already specified in
`PREREGISTRATION_CFS1.md` is now required for full project completion. It will be
implemented and run for all 64 CFS-1 branches: eight parents, two episodes, and all
four overlap/future-relation cells. It remains a secondary mechanism analysis. Its
result cannot change branch inclusion, the primary endpoint, the exact sign-flip
test, multiplicity handling, stopping, or the interpretation of the randomized
CFS-1 difference-in-differences.

## Frozen intervention surface

- Recipient: the adapted branch processing its fixed retention probe.
- Donor: the matching unadapted parent processing the same fixed probe.
- Site: the penultimate-layer state, before the final transformer block and output
  readout.
- Target effect: change from the unpatched adapted computation in the correct-first-
  branch margin at that local readout.
- Named controls: an unrelated-anchor state swap and a norm-matched random-subspace
  intervention, evaluated on the same fixed probes.
- Coverage: all 2,000 retention probes in every one of the 64 branches. No
  outcome-based subset, layer, donor, probe, branch, or checkpoint selection is
  permitted.

The minimum committed model-side surface is therefore 384,000 patched
probe-condition evaluations (`64 * 2,000 * 3`), in addition to unpatched baselines
and activation-cache construction. Implementations should cache the parent and
adapted penultimate states and run only the downstream block/readout for each
intervention when equivalence checks show that this is exact.

## Completion and interpretation

Each branch must emit `COMPLETE_WITH_NAMED_CONTROLS` and all three named effect
fields. A missing or failed patching cell is retained and reported as an incomplete
committed secondary analysis; it is never replaced, hidden, or used to tune the
intervention. This completeness rule is required for calling the whole project
finished, but it does not retroactively invalidate or alter the separately locked
CFS-1 primary confirmatory inference.

Even a positive controlled result supports only a local causal claim about changing
that activation at that site and checkpoint. It cannot establish that global
representation geometry mediates forgetting, and a null cannot rescue or overturn
the primary CFS-1 result.

The first completed branch will be used only to record runtime, memory, and artifact
size before the remaining sweep is launched. Scientific arrays and aggregate effects
remain unopened during that resource-profile check.
