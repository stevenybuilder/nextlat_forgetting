# Lure-Star stimulus design: what is matched, and the one thing that cannot be

This document exists because spec §5 states a matching requirement that is, read
literally, impossible to satisfy. Rather than satisfy it approximately and stay quiet, this
file states the requirement, proves the impossibility, records the construction that was
rejected, and writes down the exact invariant the generator does enforce. Every invariant
below is asserted by `tests/test_lure_generator.py`, on the same records that get written
to `manifests/`.

Code: `src/lurestar/generate.py` (construction), `src/lurestar/validate.py` (independent
solver and the invariant checker). Upstream format anchors:
`upstream/NextLat/data/stargraph/prepare.py:8-36` (graph construction) and `:70-73`
(serialization), `upstream/NextLat/data/stargraph.py:9-57` (tokenizer), at pinned commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`.

## 1. The requirement

Spec §5: near-safe and near-critical must match exactly on

> two edited endpoint tokens; **serialized edge positions**; token edit distance; node
> multiset and frequency; degree sequence; prompt and answer length; source and goal;
> graph validity.

Every item on that list except the bolded one is satisfiable and is satisfied. The bolded
one is not.

## 2. Why "identical serialized edge positions" is impossible under one edge ordering

Write an arm as a node sequence from the common source, `A = [s, a1, a2, a3, a4]`. A suffix
swap at depth `k` between arms `A` and `B` rewrites

```
(a_{k-1} -> a_k), (b_{k-1} -> b_k)   ==>   (a_{k-1} -> b_k), (b_{k-1} -> a_k)
```

The tails are untouched, so in the serialized string exactly two tokens change: the *head*
tokens of the two edited edges. Slot `t` of the edge list contributes tokens `3t` (tail) and
`3t+1` (head) — verified against the tokenization in `docs/UPSTREAM_REPORT.md` §1.6, where
slot 19's head lands at token index 58 and `=` at 62. So the edit positions are
`{3p+1, 3q+1}` where `p, q` are the serialized slots of the two edited edges.

Now the two conditions:

- **near-critical** must edit the depth-`k` edge of the goal arm and the depth-`k` edge of
  some distractor arm. It has to touch the goal arm — that is what makes the future change.
- **near-safe** must not edit any goal-arm edge. If it did, the correct path would move and
  it would not be safe.

The two conditions therefore edit **disjoint sets of edges**. A single edge ordering assigns
each edge exactly one slot, and distinct edges get distinct slots. Hence
`{p_crit, q_crit} ∩ {p_safe, q_safe} = ∅` under any ordering whatsoever, and the two edit
position pairs can never be equal.

The only escape is to relax one of the two things the statement silently assumes: that both
lures are two-token edits of the *same* string, or that the edit is two tokens at all.

A second, weaker escape was checked and closed: is there any *other* two-token edit of a
G(5,5) line that preserves the node multiset and yields a valid star? A two-token edit that
preserves the node multiset must be a transposition of two token values. Transposing the two
heads is the suffix swap. Transposing the two tails gives the same edge set. Transposing a
head with a tail — say slots `p=(u,x)`, `q=(v,y)` become `(u,v)` and `(x,y)` — leaves `v`
with in-degree 2 (it still has its own parent), so the result is not a valid star. There is
no third option.

## 3. The construction that was proposed, and why it is refuted

The proposal on the table was: give each condition its own edge-list permutation, chosen so
that in both near-safe and near-critical the two edited edges land at the same pair of
indices `(i, j)`.

That does place the edits at the same absolute indices. It fails on two counts, both fatal
to the purpose of the design:

1. **It destroys the two-token edit.** Moving the safe pair of edges to slots `(i, j)`
   displaces whatever was there. Base and near-safe then differ in four edge slots — up to
   eight node-token positions — plus the head swap. A ten-token perturbation is not a *near*
   lure. The entire premise of Lure-Star is that the safe and critical stimuli sit a minimal,
   identical distance from the base string, so that any representational separation is
   attributable to the future and not to surface form.
2. **It breaks the very acceptance test it is meant to serve.** "Token edit distance from
   base is equal for near-safe and near-critical" is not automatic under that construction:
   the number of displaced slots depends on which edges happened to occupy `(i, j)`, and
   equalising it requires per-item rejection sampling that further conditions the stimulus
   distribution.

Trading a 2-token perturbation for a 10-token one, in order to match a bookkeeping index,
inverts the cost-benefit. The proposal is refuted.

## 4. What the generator actually enforces

Two invariants, both per-item, both asserted in the acceptance tests. The pool is generated
once and both are available for every quartet, so the analysis does not choose between them
after seeing model states — the primary is fixed here, before any model is trained.

### LS-1 — shared anchor, exchangeable position (primary)

> `near_safe` and `near_critical` are each an exact **two-token edit of the same `base`
> string**. Their edited slot pairs are disjoint and have **identical gap** `q - p`. Which of
> the two candidate slot pairs is assigned to which condition is decided by a fair coin drawn
> from the item's own RNG stream.

What this buys: the spec's PSI formula holds literally,
`PSI = d(h_base, h_near_critical) - d(h_base, h_near_safe)`, both distances share the same
anchor state, both perturbations are exactly two tokens, and the *relative* geometry of the
two edits (their separation in the sequence) is identical within every item. Edit position is
not identical within an item, but it is **exchangeable between conditions by construction**:
the marginal distribution of `(p, q)` is the same for safe and critical, because the
assignment is a coin flip on two pairs that were sampled before either condition was named.

**This is weaker than the spec's literal text on exactly one dimension: absolute serialized
position of the edit within an item.** Section 2 shows that dimension is unattainable with a
shared anchor. Stated plainly so that no reader has to infer it: *the generator does not, and
cannot, place the safe and critical edits at the same absolute token indices of a common base
string. It places them at the same separation, and randomises which pair each condition gets.*

### LS-2 — matched anchor, identical absolute position (declared robustness check)

> `near_safe_aligned` is an exact **two-token edit of `repeat`** at the **same absolute token
> positions** `{3p+1, 3q+1}` that `near_critical` edits in `base`.

`repeat` is already in the quartet: spec §5 defines it as the same graph under a reshuffled
edge order, whose purpose is to quantify how far edge order alone moves the representation.
Serializing `repeat` so that the safe pair sits exactly where the critical pair sits in `base`
costs nothing extra — the reshuffle was going to happen anyway — and yields a second contrast

```
PSI_aligned = d(h_base, h_near_critical) - d(h_repeat, h_near_safe_aligned)
```

in which both terms are literally "how far does a two-token head swap at slots `(p, q)` move
the state", differing only in whether the future changes. The price is that the two distances
are measured from different anchors, and `d(h_base, h_repeat)` is exactly the nuisance
`repeat` was put in the design to measure.

**Neither invariant dominates.** LS-1 shares the anchor and randomises position; LS-2 fixes
position and varies the anchor. LS-1 is the preregistered primary because it is the spec's own
PSI formula; LS-2 is a preregistered robustness check, in the same relationship that whitened
Euclidean distance has to centered cosine in spec §6. If they disagree, both get reported, and
the disagreement is the finding.

## 5. What *is* matched exactly

For every quartet, re-derived from the written line by `validate.check_quartet` — never from
generator bookkeeping:

| property | base | repeat | near_safe | near_critical | near_safe_aligned | far_critical |
|---|---|---|---|---|---|---|
| correct path | — | same | same | **changed** | same | **changed** |
| correct first branch | — | same | same | **changed** | same | changed |
| source token | — | same | same | same | same | same |
| goal token | — | same | same | **same** | same | same |
| node set | — | same | same | same | same | same |
| per-node frequency | — | same | same | same | same | multiset only |
| degree sequence | — | same | same | same | same | same |
| prompt token length (63) | — | same | same | same | same | same |
| answer length (5) | — | same | same | same | same | same |
| solver-verified valid | yes | yes | yes | yes | yes | yes |
| prompt tokens changed vs its anchor | — | 40 reordered | **2** | **2** | **2** | — |

Three entries deserve a note.

**The goal token is unchanged under the critical swap.** After swapping the depth-`k`
suffixes of the goal arm and distractor arm `i`, the goal *node* terminates arm `i` instead of
arm 0. The `/source,goal` field is byte-identical, the graph is still valid with the goal at
depth 4 of exactly one arm, and what changed is which first branch leads there. The test
asserts all three facts separately, including that the goal arm index actually moved.

**Token edit distance is measured on the prompt**, i.e. tokens 0..62 up to and including `=`.
The answer of near-critical necessarily differs — that is the manipulation, not a confound —
and H1's state is extracted at the prompt delimiter, so the prompt is the object that has to
be matched. Full-line distance is reported alongside but is not the matched quantity.

**Per-node frequency for far-critical is matched as a multiset, not per node.** far-critical
repartitions the same 21 nodes, so a node that terminated an arm in base (frequency 1) can sit
mid-arm in far (frequency 2). The node *set*, the sorted frequency multiset, the degree
sequence, the source and goal tokens, and both lengths are all exactly preserved. Spec §5 asks
far-critical only for "the same node multiset", so this is the requirement as written.

## 6. far-critical, and the conditioning that rejection sampling introduces

far-critical holds the source and goal tokens fixed, permutes the other 19 nodes freely, and
places the goal at the terminal position of arm 0. It is then accepted only if
`|E_far ∩ E_base| <= 2` (10% of 20 edges) and its path differs from base's. Near lures share
18 of 20 edges with base, so the near/far contrast in edge overlap is roughly 90% versus under
10%.

The rejection step conditions the far distribution and that is deliberate — far-critical is a
distance control, and low overlap is its job. The manifest records both the accepted overlap
and the number of tries per item, so the acceptance rate (and therefore the unconditioned
overlap distribution near the cap) is recoverable from the artifact rather than taken on
trust. Measured on 1,000 quartets, the accepted overlap is concentrated on 0-2 with mean
around 1.2 edges, and the mean try count is around 1.4, implying roughly a 70% first-try
acceptance rate.

## 7. Pools (spec §5 "Data pools")

Two disjoint pools, both selected model-blind — no function in `generate.py` takes a model, a
checkpoint, or an accuracy as an argument, and the tests assert the pools are a pure function
of the recorded master seed.

- **`E_lure`** (`manifests/e_lure.jsonl`): held-out quartets for H1/H2. Base graphs are sampled
  fresh from the generator's own RNG, never drawn from the corpus. Absence from training is
  *checked*, not assumed: every condition's prompt SHA-256 and order-invariant graph key is
  looked up in an index built from all 200,000 training lines.
- **`A_pair`** (`manifests/a_pair.jsonl`): base-training items, drawn by seeded index from the
  training file itself — they must be items the model was trained on, since H3 measures their
  erosion. `B_near` (`b_near.jsonl`) holds near-critical lures of each `A_pair` item;
  `B_far` (`b_far.jsonl`) is a deliberately oversized far-critical candidate bank, because
  spec §6 requires the far items actually used to be picked later by a loss-quantile match on
  a non-confirmatory pilot checkpoint and then frozen.

`E_lure` graph keys are asserted disjoint from every `A_pair`/`B_near`/`B_far` key. Each
manifest ships with a `.sha256` sidecar and a shared `stimuli_provenance.json` recording the
master seed, the config, the counts, and the far-overlap histogram.

## 8. Determinism

Every stimulus is a pure function of `(master_seed, index)` through
`numpy.random.default_rng([master_seed, index, ...])`. Nothing derives from wall clock, PID,
iteration order of a set, or worker count. `build_e_lure(..., workers=1)` and
`build_e_lure(..., workers=4)` are asserted byte-identical after JSON serialization, and the
multiprocessing start method is pinned to `spawn` so the two cannot diverge through an
inherited fork context.
