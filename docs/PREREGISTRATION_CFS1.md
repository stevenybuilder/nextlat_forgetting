# CFS-1 preregistration: causal forgetting from later-learning structure

## Scope and status

This document governs **CFS-1 only**. It is a new experiment; retired H3 remains
closed. At preregistration time no CFS-1 parent, adaptation branch, CFS-1 evaluation,
or CFS-1 scientific metric has been run or inspected.

## Fixed inputs and branches

The immutable materialization receipt binds `src/cfs1/generate.py`,
`src/cfs1/validate.py`, the retention and global-control manifests, all four update
JSONL provenance manifests, eight episode/arm raw trainer streams, hash codebook,
and the outer update manifest. The update schema is
`nextlat_forgetting/cfs1_update_manifest/1`. Each branch is identified by exactly
`parent_id`, `episode` (0 or 1), `overlap` (`high`/`low`), and `future_relation`
(`same`/`different`), and uses 500 CE-only updates from the committed raw Path-Star
bank. The JSONL manifest is an evidence/provenance artifact, never a trainer input.

Primary population: eight independent NextLat parents: planned seeds 1234–1238 and
CFS-only seeds 2234–2236. Each is forked into eight branches (two episodes × four
cells), yielding 64 adaptation branches. Parent checkpoint, optimizer state, RNG,
batch budget, episode codebook order, and global controls are identical across its
four cells within an episode.

## Primary endpoint and test

For each parent `s` and episode `e`, let `M` be mean correct-first-branch retention
margin erosion over the fixed 2,000 untouched probes. Define

\[
D_{s,e}=(M_{high,different}-M_{high,same})-
        (M_{low,different}-M_{low,same}).
\]

The primary parent-level quantity is `D_s = mean_e(D_{s,e})`. Test the two-sided null
that the median parent effect is zero with the exact two-sided sign-flip/permutation
test over all 2^8 signs. Report the observed effect, exact p-value, sign pattern,
95% bootstrap interval across parents, individual parent effects, episode effects,
and a minimum detectable effect calculated before opening outcomes. A two-sided
familywise decision over this single primary contrast uses alpha = .05.

If a parent has a terminally invalid branch, it is reported and excluded only under
the predeclared all-cells-required rule: no primary confirmatory decision is issued
until every required branch has a terminal durable artifact. No replacement parent,
episode, bank, or branch is allowed.

## Secondary and mechanism analyses

Secondary outcomes, reported without displacing the primary endpoint, are exact-path
retention, retention CE, adaptation acquisition, untouched global-control change,
and representation drift. Pre-adaptation predictive geometry is a prespecified
moderator in a parent-level interaction/reporting model. It is not a mediator and
cannot support a causal-mechanism claim.

The state-interchange test uses a parent-to-adapted penultimate-layer swap,
an unrelated-anchor swap, and norm-matched/random-subspace controls, all specified
before results. It can establish a local readout intervention only; it does not prove
that geometry globally mediates forgetting.

### Outcome-blind execution disposition (2026-08-24)

Before any CFS-1 adaptation-branch outcome, state-interchange result, or CFS-1
scientific metric was run or inspected, the already specified state-interchange test
was made mandatory for full project completion. It will cover all 64 branches and
all 2,000 fixed retention probes per branch, with the parent-state intervention,
unrelated-anchor control, and norm-matched random-subspace control. The binding
operational details and completion semantics are recorded in
`DECISION_CFS1_STATE_INTERCHANGE_COMMITMENT.md` and its machine-readable receipt.
This disposition adds no model, branch, condition, selected layer, or primary
estimand, and the secondary result cannot alter the primary confirmatory decision.

## Prohibitions and amendment rule

No use of model loss, acquisition, retention, geometry, checkpoint, pilot, or
evaluation outcome is permitted in CFS-1 item generation, matching, selection,
counterbalancing, stopping, or analysis-plan changes. Any material change creates a
new numbered CFS experiment with a new generator, manifest, receipt, and
preregistration; it cannot amend this one after outcomes are opened.
