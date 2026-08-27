# CFS-2 stimulus repair: outcome-blind exact-overlap construction

**Decision date:** 2026-08-24  
**Status:** implemented and solver-validated before CFS adaptation outcomes  
**Scope:** CFS-2 stimulus construction, adaptation estimand, and the mandatory
activation-state intervention; CFS-1 artifacts and the primary base-model training snapshot are
unchanged

## Why CFS-2 exists

CFS-1 assigned exact retention/update edge overlaps of 18, 18, 8, and 7 to
high/same, high/different, low/same, and low/different. The one-edge imbalance
inside the low-overlap level could induce a positive future-relation-by-overlap
interaction even if future conflict had no special interaction with structural
overlap. Because no CFS adaptation or scientific outcome was used to find or repair
this construction defect, the appropriate response is a fresh, outcome-blind
successor rather than a post-hoc covariate adjustment.

CFS-2 keeps the scientific question and G(5,5) task but replaces the stimulus
construction. CFS-1 remains a documented blocked predecessor and must not be
silently relabeled as CFS-2.

## Feasibility result

Exact total-edge balance **is feasible** without leaving G(5,5).

Represent each graph as five arms with four edges per arm. A column-wise
permutation of the 20 non-source nodes preserves the node-token multiset, degrees,
source, goal, prompt length, answer length, and serialization order. Every such
construction retains the five source fan-out edges.

- A low/same graph must retain the unchanged answer. Its three post-source answer
  continuations plus the five fan-out edges force an overlap of eight.
- The selected low/different answer contains two post-source continuations already
  present in the probe. Together with fan-out, this forces seven edges.
- CFS-2 deterministically preserves exactly one additional **distractor**
  continuation in low/different. Total overlap is therefore eight, while the
  different answer remains exactly the same as high/different.

The resulting per-unit contract is:

| Cell | total shared edges | update-answer shared edges | other shared edges |
| --- | ---: | ---: | ---: |
| high / same | 18 | 4 | 14 |
| high / different | 18 | 3 | 15 |
| low / same | 8 | 4 | 4 |
| low / different | 8 | 3 | 5 |

The one-edge answer/non-answer decomposition difference is inherent to the intended
future intervention: a different valid path to the same goal must introduce at
least one new path edge. It is not differentially coupled to the overlap factor:

- answer-edge overlap is identical across high/low within each future relation;
- the non-answer high-minus-low contrast is exactly 10 in both future relations;
- total overlap is exactly equal across future relation within each overlap level.

Thus the construction removes CFS-1's differential low-cell nuisance without
pretending that a future-changing answer can have the same path identity as a
future-preserving answer.

## Frozen design contract

CFS-2 constructs, without model access:

- 2,000 untouched retention probes;
- 2,000 disjoint untouched global controls;
- four 5,000-example update banks;
- exactly two independently SHA-ordered episodes;
- the same two-or-three update-unit assignment per retention probe;
- a solver-certified unique answer for every serialized Path-Star line.

The downstream experiment retains eight independent parents (base seeds
1234--1238 and CFS-only base seeds 2234--2236), two episodes per parent, and all
four factorial cells: 64 adaptation branches total, each restored from its exact
immutable parent and trained for 500 CE-only updates. The parent seed remains the
independent unit; episodes remain planned robustness replicates.

The primary endpoint and contrast are unchanged from the predecessor design:
correct-first-branch margin erosion on the fixed retention set, with

\[
[(\mathrm{high,different}-\mathrm{high,same})-
 (\mathrm{low,different}-\mathrm{low,same})]_{s,e}.
\]

Only the confounded stimulus construction changes. CFS-1 branch streams must not
be mixed with or substituted for CFS-2 streams.

Within every four-cell update unit, the validator requires:

- exact 18/18 and 8/8 total edge overlap;
- the exact decomposition shown above;
- equal high/low answer sequence within future relation;
- different same/different answer sequence;
- identical source and goal;
- identical prompt node-token multiset;
- identical prompt and answer length;
- identical probe assignment and deterministic stream position;
- no prompt, graph, or identifier collision within CFS-2, the base corpus, CFS-1,
  or legacy Path-Star stimulus manifests.

The construction imports no model, checkpoint, activation, loss, pilot outcome,
learned distance, caliper, matching result, or scientific metric. Candidate retries
are permitted only for exact identity collisions.

## Implementation and evidence

- Generator: `src/cfs2/generate.py`
- Independent validator: `src/cfs2/validate.py`
- Immutable materializer: `scripts/materialize_cfs2_banks.py`
- Focused tests: `tests/test_cfs2_generate.py`
- Materialized artifacts and SHA-256 sidecars: `manifests/cfs2/`
- Activation-patching primitives: `src/cfs2/patching.py`
- Required per-branch patching runner: `scripts/run_cfs2_patching.py`
- Focused patching tests: `tests/test_cfs2_patching.py`

The validator recomputes graph solutions, hashes, overlap totals, overlap
decomposition, answer relations, balance, and disjointness from serialized lines;
it does not trust generator bookkeeping.

## Mandatory activation-patching endpoint

Activation patching is a required CFS-2 analysis, not an optional stretch goal and not part of the
permanently withdrawn Lure-Star H3 study. Every one of the 64 completed adaptation branches must
produce a `nextlat_forgetting/cfs2_activation_patching/1` artifact before CFS-2 is called complete.
A missing or failed branch artifact is reported as incomplete; it is never silently dropped.

For every fixed retention probe, run the adapted checkpoint once without intervention and then
replace its index-63 block output with the corresponding state from the exact immutable parent
checkpoint. Apply this intervention at the three outcome-blind sites frozen before CFS-2 outcomes:
blocks 3, 7, and 10 of the 12-block transformer. Continue through all unchanged downstream blocks,
final normalization, and the output head. The per-probe estimand is the patched minus unpatched
correct-first-branch margin.

Each layer must retain all of the following named comparisons:

- the matching parent state for the same retention probe;
- an unrelated parent-probe donor selected by the fixed seeded derangement;
- an isotropic random-direction replacement whose displacement from the adapted state exactly
  matches the real parent-minus-adapted displacement norm; and
- an adapted-state self patch that must pass numerical no-op parity before effects are written.

Donor assignments, layers, position, analysis seed, probe order, checkpoint hashes, and retention
manifest hash are part of the output. Layers or controls may not be selected after viewing effects.
The runner is inference-only and does not change a branch checkpoint. A successful matching-parent
effect permits a local causal statement about replacing that activation at that checkpoint and
site; it does not establish global mediation by representational distance.

## Interpretation limit

CFS-2 supports a causal claim about full-parameter adaptation on a controlled
symbolic planning task. It does not by itself establish that the same mechanism
holds in natural-language corpora. The required state-interchange analysis on CFS-2 branches
remains a local readout intervention and not proof of global causal mediation.
