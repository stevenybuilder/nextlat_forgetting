# D40 — One-shot prospective H3 overlap expansion

**Status:** Accepted before expanded-candidate generation or scoring on 2026-08-24. This is a
single prospective nuisance-feasibility repair. It inspects only the frozen nonconfirmatory BST
pilot loss table and model-blind graph structure. No confirmatory checkpoint, outcome, or result
was available or inspected.

## Trigger and diagnosis

The D39 selector applied its frozen same-loss-decile and absolute-loss-difference `<= 0.1` rule to
three model-blind middle candidates per near item. It matched 3,885/5,000 items and failed closed
for 1,115. Of those failures, 1,014 had no same-decile candidate and only 101 failed solely at the
raw-loss caliper. Complete-case restriction would therefore select a pilot-matchable subpopulation;
relaxing or rescaling the caliper would alter the rule without restoring full support; weighting
cannot create missing support. D39's failed attempt and all 53,000 loss rows remain immutable audit
evidence.

## One-shot rule frozen now

For every one of the 5,000 near items, the expanded middle pool contains exactly 30 candidates:

- retain the three D39 candidates byte-for-byte, one from each 1-, 2-, and 3-rewire class;
- add exactly nine new unique candidates to each class;
- total 10 candidates per class, 30 per near item, and exactly 150,000 candidates overall;
- score only the 135,000 new rows; never rescore or replace the frozen D39 loss rows;
- use a new deterministic RNG namespace and master seed, recorded by the generator;
- independently solve every graph and require unchanged source/goal and target-path length;
- require prompt and canonical graph identities to be unique and disjoint from training,
  `E_lure`, `B_near`, `B_far`, and acquisition candidate pools;
- structural-distance duplication is allowed. Candidate count and rewire class, not observed loss
  or distance diversity, determine acceptance.

The sole pilot remains the D39 BST seed-1234 step-500 checkpoint. The frozen scientific loss
function remains `scripts/score_h3_pilot.py` at SHA-256
`f907f00eda179c23d261a78e05efc43be33084240bb3ae691a4e29bf3a2b0954`. Expanded scoring must call
its tokenizer/model/CE primitives and may add only separately hash-bound transport/orchestration.

The final logical loss table contains exactly 188,000 unique rows: all 53,000 frozen D39 rows plus
135,000 new middle rows. Its receipt binds both source tables, ordered concatenation, identity
coverage, checkpoint/config/tokenizer/scorer hashes, and the 150,000-row expanded manifest.

Selection keeps the D39 scientific rule unchanged:

1. near loss deciles remain exactly those computed from the original 5,000 near rows;
2. middle loss deciles are recomputed over all 150,000 expanded middle candidates;
3. eligibility requires same decile, absolute pilot-loss difference `<= 0.1`, path match, and an
   independent solver pass;
4. the target is the global median structural distance over every eligible candidate;
5. each near item selects its eligible candidate closest to that median, with ascending prompt
   SHA-256 as the sole tie-break.

Selection must cover all 5,000 near items without candidate reuse. If even one remains unmatched,
the selector publishes a permanent H3 `BLOCK` receipt. No further candidate expansion, caliper
change, weighting, restriction, model substitution, or matching amendment is permitted. H1/H2/HMM
may continue, but H3 cannot enter confirmatory training.

## Operational boundary

Generation and freeze are CPU-only and create-only. Expanded scoring uses generation-bound GCS
chunks with content, receipt, commit, read-back, restore, and final state publication last. The ADC
is runtime-only and is never persisted. The controller must run plan/preflight before provisioning;
this decision does not itself authorize GPU launch.
