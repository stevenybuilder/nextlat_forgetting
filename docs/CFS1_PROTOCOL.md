# CFS-1: construction-level causal forgetting protocol

> **Outcome-blind construct-validity disposition (2026-08-24): blocked for
> confirmatory causal launch.** The frozen CFS-1 materials remain immutable for
> provenance and design diagnostics, but no CFS-1 adaptation branch may be trained
> or interpreted as confirmatory evidence. The low-overlap cells differ by one
> exact shared edge (`same=8`, `different=7`), which can mechanically bias the
> primary difference-in-differences in the hypothesized direction. The production
> runner enforces `manifests/cfs1/STIMULUS_VALIDITY_BLOCK.json`. A fresh, separately
> numbered CFS-2 construction must resolve this before causal branch execution.

## Question and estimand

CFS-1 asks whether later learning with a conflicting future causes more retention
loss when it has high rather than low structural overlap with an untouched probe.
For parent seed `s` and episode `e`, the primary endpoint is correct-first-branch
margin erosion on the fixed retention set, before versus after the 500-update branch.
The primary contrast is

\[
[(\mathrm{high,different}-\mathrm{high,same})-
 (\mathrm{low,different}-\mathrm{low,same})]_{s,e}.
\]

A positive value is extra causal forgetting from conflicting, high-overlap learning,
beyond generic adaptation and benign transfer. Exact-path retention, retention CE,
adaptation acquisition, untouched global controls, and representation drift are
secondary outcomes. Pre-update predictive geometry may moderate this effect; it is
not tested or reported as causal mediation.

## Fresh construction, not matching

`src/cfs1/generate.py` makes 2,000 new retention probes, 2,000 disjoint untouched
global controls, and four 5,000-example update banks. Every update unit has all four
cells:

| overlap | future relation | exact edge overlap with its probe |
| --- | --- | ---: |
| high | same | 18 |
| high | different | 18 |
| low | same | 8 |
| low | different | 7 |

Those low-overlap values are the minimum compatible with a G(5,5) source fan-out
and, for the same/different future relation, the answer path that must be held fixed
across high/low. The full answer is exactly equal for high/low `same`, and exactly
equal (but different from `same`) for high/low `different`. Every cell in a unit
also has the same source, goal, node-token multiset, prompt token length, answer
length, G(5,5) degree structure, and deterministic stream position.

A SHA-256 codebook assigns every probe two or three update units (exactly 1,000
of 2,000 receive the third), fixes the 5,000 unit order, and fixes two independently
hash-ordered episodes. All four banks share that exact codebook. It is not an
outcome-dependent randomization.

Before any write, the independent solver checks every line and the generator rejects
any prompt/graph/identifier collision with the training corpus, H1/H2, legacy H3,
or HMM manifests. CFS-1 has no loss selection, pilot, learned-distance threshold,
caliper, middle matching, model import, checkpoint access, or result access.

`scripts/materialize_cfs1_banks.py` writes the sole opaque runner/evaluator envelope,
`manifests/cfs1/cfs1_update_manifest.json`. It is marked `FROZEN`, binds an
outcome-blind construction receipt, generator receipt and manifest, retention probes,
global-control manifest, exactly six paired evaluation inputs, and each arm for each
numeric episode. The actual training artifacts are immutable raw Path-Star
`graph_5_5_cfs1_*.txt` streams; JSONL files are provenance/evidence manifests and
must never be passed as `data.stargraph_train_data_path`.

## Branches and execution

The primary analysis uses eight independent NextLat parents: five planned base
parents (seeds 1234–1238) and three CFS-only bases (2234–2236). For each parent,
episodes 0/1, and all four cells, execution creates a branch keyed by:

`parent_id`, `episode`, `overlap`, and `future_relation`.

Each branch restores the exact immutable parent state, including optimizer and RNG,
then receives 500 CE-only updates from its hash-bound bank/order. Retention probes
and global controls are never used for update selection, early stopping, checkpoint
selection, or branch retry. A failure may resume only from an exact durable
checkpoint; it cannot replace an episode, alter a bank, or change the parent fork.

## Analysis discipline

The parent seed is the primary independent unit; episodes are planned robustness
replicates, not extra parent seeds. Report all branch outcomes, parent-level effects,
two-sided exact/sign-flip inference as specified in `PREREGISTRATION_CFS1.md`, effect
sizes and intervals, nulls, heterogeneity, failed branches, and all durable restart
events. Do not select a layer, time point, geometry metric, endpoint, or subset based
on outcomes.

The previously specified state-interchange analysis is now a committed secondary
deliverable for every branch, as recorded before any CFS-1 branch outcome was opened
in `DECISION_CFS1_STATE_INTERCHANGE_COMMITMENT.md`. It uses the matching parent
penultimate state plus unrelated-anchor and norm-matched random-subspace controls on
all 2,000 fixed retention probes. Completion is required to call the full project
finished, while the CFS-1 primary inference remains locked and independently valid.
