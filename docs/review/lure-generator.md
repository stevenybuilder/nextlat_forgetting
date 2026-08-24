# Adversarial review — Lure-Star stimulus generator track

Reviewer: independent agent. Date: 2026-08-23.
Scope: `src/lurestar/generate.py`, `src/lurestar/validate.py`, `tests/test_lure_generator.py`,
`docs/STIMULUS_DESIGN.md`, `manifests/{e_lure,a_pair,b_near,b_far}.jsonl` +
`stimuli_provenance.json`, and the `docs/RUNLOG.md` entry "Lure-Star stimuli".

Method: read every delivered file; reran the suite; regenerated the full shipped pool from its
recorded seed and compared sha256; then fed the checker eleven deliberately broken records to
find which ones it accepts.

---

## 0. What reproduced (stated before the findings, because it is most of the delivery)

Every file the track claimed exists, is non-trivial, and does what its docstring says.

```
$ .venv/bin/python -m pytest tests/test_lure_generator.py -q
................................................                         [100%]
48 passed in 12.26s
```

Whole repo: `3 failed, 359 passed` — the three are `tests/test_hmm_pairs.py`
(`test_no_calibration_sequence_leaks_into_the_test_bank`, `test_shipped_controls_are_matched`,
`test_shipped_bank_does_not_leak_the_calibration_pool`), a different track. Zero Lure-Star
failures.

Independent re-derivations that held:

| claim | how checked | result |
|---|---|---|
| shipped manifests are reproducible from the recorded seed | full CLI rerun, `--workers 4`, into a scratch dir | all four sha256 **byte-identical** to `stimuli_provenance.json` |
| worker-count independence | shipped pool was built with `workers = cpu_count-1`; I rebuilt with `workers 4` | identical |
| tokenizer fidelity | `upstream/NextLat/data/stargraph.py:9-57` read directly; ids 0..99 one token per node, `,` dropped | matches `validate.token_ids` |
| serialization fidelity | `upstream/NextLat/data/stargraph/prepare.py:8-36,70-73` | matches; arm `p==0` carries the goal, as assumed |
| leakage index is falsifiable | injected training lines 0 / 1234 / 199999, and a *reshuffled* training graph | prompt hash and order-invariant key both fire; reshuffle caught by key only, as designed |
| shipped `graph_key` / `prompt_sha256` fields | recomputed all 24,000 | 0 mismatches |
| E_lure ∩ training, E_lure ∩ H3, B_near/B_far ∩ training | recomputed from the corpus | 0, 0, 0 |
| duplicates | base graph keys, B_near lines, B_far lines | 0, 0, 0 |
| LS-1 exchangeability (the load-bearing statistical claim) | 2×20 contingency test on the critical vs safe edit-slot marginals over 2,000 quartets | χ²=22.59, **p=0.256** — genuinely exchangeable |
| far vs near edge overlap | recomputed | far `{0:288, 1:820, 2:892}`, near exactly 18/20 |

The impossibility proof in `docs/STIMULUS_DESIGN.md` §2 is correct. Critical must edit a
goal-arm edge, safe must not, distinct edges get distinct slots under any single ordering, so
the spec's "identical serialized edge positions" is unreachable with a shared anchor. The
refutation of the one-permutation-per-condition proposal is also correct. That part of the
delivery is sound and honestly documented.

**What follows is what broke.**

---

## P0-1 — `check_quartet` accepts an unmatched near-lure pair (different suffix depths)

**Severity: P0.** This is exactly the failure mode the review brief names: a checker that
passes on an unmatched lure pair.

**Where:** `src/lurestar/validate.py:453-465` (the LS-1/LS-2 block) and the whole of
`check_quartet` (`src/lurestar/validate.py:366`). The record carries `record["depth"]`
(`src/lurestar/generate.py:377`) and the checker never reads it, never re-derives it, and never
requires the safe swap and the critical swap to be at the same depth.

**Why it matters:** the suffix depth *is* the magnitude of the manipulation. A depth-1 swap
moves three nodes of each arm; a depth-3 swap moves one. Both are two-token prompt edits, so
every token-level assertion in the suite passes either way, while the underlying graph
perturbation is three times larger in one condition than the other. A quartet whose safe lure
is depth-1 and whose critical lure is depth-3 is precisely an unmatched pair, and it is
certified clean.

**Reproduction** (`scratchpad/adv1.py`, quartet index 4 of the shipped seed):

```python
rec  = make_quartet(20260823, 4, QuartetConfig())      # critical swap at depth 3
base = graph_from_line(rec["conditions"]["base"]["line"])
s0, s1 = rec["safe_arms"]
rec["conditions"]["near_safe"] = _condition_record(suffix_swap(base, s0, s1, 1))   # depth 1
rec["edit_slots"]["near_safe"] = list(swap_edit_slots(base, s0, s1, 1))            # honest slots
check_quartet(rec)
# -> []      # zero problems.  critical depth 3, safe depth 1.
```

**Fix:** re-derive the swap depth of every edited edge from the *anchor's own solved arms*
(never from `record["depth"]`), require all four near-lure edits to sit at one common depth in
`1..ARM_LEN-1`, and cross-check `record["depth"]` against the re-derived value.

---

## P0-2 — `check_quartet` accepts a `far_critical` that is a near lure (18/20 edge overlap)

**Severity: P0.** Same class: the far control's one defining property is unenforced.

**Where:** `src/lurestar/validate.py:467-473`. The block only asserts
`recorded == recomputed` overlap. There is no cap. `QuartetConfig.far_max_edge_overlap = 2`
(`src/lurestar/generate.py:285`) lives only in the *generator*, so a manifest is certified by a
checker that has never heard of it.

**Why it matters:** `far_critical` is the spec §5 "distance control for adaptation" and the
`B_far` bank is the H3 far branch. If the far item is not actually far, the near-minus-far
interference contrast — H3's primary outcome — is measured against a control that is a near
lure, and every acceptance test still passes.

**Reproduction** (`scratchpad/adv1.py`):

```python
rec = make_quartet(20260823, 0, QuartetConfig())
fake_far = suffix_swap(graph_from_line(rec["conditions"]["base"]["line"]),
                       rec["critical_arm"], rec["depth"])          # a NEAR lure
rec["conditions"]["far_critical"] = _condition_record(fake_far)
rec["far_edge_overlap"] = 18                                        # honestly recorded
check_quartet(rec)
# -> []      # zero problems, with an 18/20-edge-overlap "far" control.
```

**Fix:** give `check_quartet` a `far_max_edge_overlap` parameter (default 2, the shipped
value), assert the recomputed overlap is within it, and have `make_quartet` pass
`cfg.far_max_edge_overlap` so the checker and the generator cannot drift apart.

---

## P0-3 — a record with falsified `graph_key` / `prompt_sha256` passes, and evades the leakage gate

**Severity: P0.** The review brief's "a checkpoint with a wrong hash" example, instantiated.

**Where:**
- `src/lurestar/validate.py:366-473` — `check_quartet` never recomputes
  `conditions[*]["graph_key"]`, `conditions[*]["prompt_sha256"]` or `conditions[*]["answer"]`
  from `conditions[*]["line"]`.
- `src/lurestar/generate.py:591-601` — the CLI's no-leakage gate tests
  `c["graph_key"] in index.graph_keys`, i.e. the record's **self-reported** key.
- `tests/test_lure_generator.py:520` — the no-leakage assertion also reads
  `entry["graph_key"]`. (Line 515 *does* recompute `prompt_sha256`; the graph key, which is the
  order-invariant half and therefore the stronger one, is taken on trust.)

**Why it matters:** the leakage guarantee is only as good as the key it hashes. The
implementer already hit this exact class once — `TrainingIndex.build` sorting edge strings while
`canonical_graph_key` sorted tuples, which made the check vacuous — and fixed the *symptom*
(the two now agree) without removing the *mechanism* (the check consults a stored field rather
than the bytes). Any future divergence between `_condition_record` and `canonical_graph_key`
silently reopens it. The shipped manifest is clean (I recomputed all 24,000 fields: 0
mismatches), so this is a live hole in the guarantee, not a live defect in the data.

**Reproduction** (`scratchpad/adv4.py`):

```python
rec = shipped_e_lure[0]
rec["conditions"]["base"]["graph_key"]     = "0"*64
rec["conditions"]["base"]["prompt_sha256"] = "1"*64
check_quartet(rec)                     # -> []

# and the leakage gate, verbatim from generate.py:595, on a real training line:
entry = {"line": train[42], "graph_key": "deadbeef"*8, "prompt_sha256": "deadbeef"*8}
entry["graph_key"] in index.graph_keys          # -> False   (gate says clean)
canonical_key_from_line(entry["line"]) in index.graph_keys   # -> True (it IS a training item)
```

**Fix:** recompute `graph_key`, `prompt_sha256` and `answer` from `line` inside
`check_quartet`; recompute the key in the CLI gate and in the leakage test rather than reading
the field.

---

## P1-4 — spec §6's "target-path distribution" match between `B_near` and `B_far` is unmet and undeclared

**Where:** `src/lurestar/generate.py:452-533` (`build_a_pair_pools`). Nothing matches, measures
or declares the target distribution; `docs/STIMULUS_DESIGN.md` §7 does not mention it.

Spec §6 lists, as a hard requirement on H3: *"Match near/far branches for: adaptation examples
and updates; initial loss quantiles; **target-path distribution**; paired item order; optimizer
and scheduler state; learning rate and batch size."* Measured on the shipped pools — positionwise
probability that the lure's target token equals its `A_pair` parent's target token:

| position | 0 (source) | 1 (first branch) | 2 | 3 | 4 (goal) |
|---|---|---|---|---|---|
| `B_near` | 1.000 | 0.000 | **0.343** | **0.670** | 1.000 |
| `B_far`  | 1.000 | 0.040 | 0.049 | 0.026 | 1.000 |

Mean target-node-set overlap with the parent: near 3.01, far 2.45.

This is partly intrinsic — a two-token lure of a training item necessarily keeps the
parent's path suffix from the swap depth onward — but it is a listed matching requirement, it
is unmet, and it has a *direction*: the near branch rehearses 34–67% of the parent's
mid-path tokens while the far branch does not, which biases `erosion_near` downward and
therefore biases H3's primary outcome `erosion_near - erosion_far` **against** the hypothesis.
The track declared one impossibility (LS-1) very loudly and left this one silent. Compare
PROGRAM.md: "Stop rather than shrink" / declare, do not quietly weaken.

**Fix (not applied — needs a design decision, not a patch):** either (a) record the measured
table above in `stimuli_provenance.json` and add a §8 to `docs/STIMULUS_DESIGN.md` declaring the
dimension unmet with its direction of bias, and add it as an item-level covariate to the H3
model; or (b) construct `B_far` so its targets have the same positionwise agreement profile with
the parent as `B_near` — which would make `B_far` no longer a repartition and needs the spec
owner's sign-off.

---

## P1-5 — LS-2, the declared robustness check, cannot be computed by the delivered evaluation stack

**Where:** `src/lurestar/representations.py:147-153` (`CENTERING_POOL_CONDITIONS`),
`:349-372` (`from_conditions` raises on unknown conditions), `:393-403` (`require_contains`).
The lure track adds a sixth condition `near_safe_aligned`
(`src/lurestar/validate.py:302-311`, `src/lurestar/generate.py:352`) and preregisters
`PSI_aligned = d(h_base, h_near_critical) - d(h_repeat, h_near_safe_aligned)` in
`docs/STIMULUS_DESIGN.md` §4 and in the RUNLOG. The evaluation layer does not know the
condition exists.

**Reproduction** (`scratchpad/adv6.py`):

```
CenteringPool.from_conditions(..., near_safe_aligned=H)
 -> ValueError: unknown centering-pool condition(s) ['near_safe_aligned'];
    the declared pool is ['base','repeat','near_safe','near_critical','far_critical']
pool.require_contains("h_near_safe_aligned", H)
 -> ValueError: 4 distinct row(s) ... are not in the centering pool
```

So one sixth of the manifest is currently un-analysable, and the "both invariants are generated,
the primary was fixed before any model exists" claim is only half true: LS-1 is runnable, LS-2 is
not.

**Fix (not applied — cross-track):** add `"near_safe_aligned"` to `CENTERING_POOL_CONDITIONS`
and extract its state alongside the other five. `representations.py` is owned by a concurrent
track and its tests pin the five-condition pool; patching it here risks that agent's work, which
the brief forbids. Hand this to the representations owner as a one-line change plus a test.

---

## P1-6 — E_lure base graphs are not drawn from the training corpus's edge-order distribution

**Where:** `src/lurestar/generate.py:214-232` (`_gap_matched_slot_pairs`) and `:340-347`
(`base_pins`). Gap is drawn uniformly from 1..18, then a start uniformly from `0..19-gap`. For
large gaps only near-terminal starts exist, so the marginal distribution of the four pinned edit
slots is strongly U-shaped.

Measured over the 2,000 shipped quartets, critical edit-slot marginal (20 slots):

```
[278 273 241 220 177 185 172 169 147 145 162 168 157 163 173 165 225 228 258 294]
chi-square vs uniform: p = 2.9e-35     (safe pair: p = 1.6e-42, same shape)
```

Upstream shuffles the edge list uniformly (`prepare.py:30`, `random.shuffle(edgeList)`), so
training graphs have a flat slot marginal, and the E_lure base graphs do not: the goal-arm and
distractor depth-*k* edges sit near the ends of the edge list about twice as often as in the
middle. Bleed-through is visible in the goal-arm depth-1 profile (χ² p=0.003 for E_lure, p=0.26
for the corpus).

**This does not bias PSI** — the critical and safe marginals are statistically identical
(χ²=22.59, p=0.256), which is the claim LS-1 actually makes and it holds. What it does bias is
*absolute* competence on E_lure relative to the held-out corpus, which matters because spec §10's
90% exact-path base-competence gate and H2's base-margin covariate are absolute quantities.

**Fix (not applied — would change every manifest hash):** sample the slot pair uniformly over all
C(20,2) pairs (equivalently, draw the gap with probability ∝ 20−gap) instead of drawing the gap
uniformly; that makes the first pair's slot marginal exactly flat. Doing so regenerates
`e_lure.jsonl` and invalidates the recorded sha256s, so it must be a deliberate, logged
regeneration — not a fix bundled with the P0 patch. Minimum action if it is not regenerated:
record the measured marginal in `stimuli_provenance.json` and state the shift in
`docs/STIMULUS_DESIGN.md`, and evaluate the competence gate on the upstream held-out file rather
than on E_lure bases.

---

## P1-7 — provenance cannot prove which corpus was used or that the leakage gate ran

**Where:** `src/lurestar/generate.py:606-627`. `stimuli_provenance.json` records `train_file` as
a *path* and nothing else about it, and does not record the value of `--skip-leakage-check`.

Keys present: `config, counts, far_edge_overlap_histogram, generator, invariants, master_seed,
sha256, train_file`. So the artifact cannot distinguish "the leakage gate ran and passed" from
"the gate was skipped", and cannot prove the corpus it was checked against was the one whose
sha256 is `d13199b0…` (which `manifests/corpus.sha256` records separately, unlinked).

**Fix:** hash the training file into the provenance, record `leakage_check: "ran" | "skipped"`,
and record the numpy/python versions the determinism claim is conditional on.

---

## P2 findings (listed, not fixed)

8. **`item_id` namespace collision between `B_near` and `B_far`.** `generate.py:498` and `:522`
   both mint `f"{item_id}:{k}"`; 5,000 ids collide across the two pools. Only the `pool` field
   separates them. H3 requires "paired item order" matched across branches, so a join on
   `item_id` is a likely downstream move and would silently pair `near 3:4` with `far 3:4` and
   drop `far 3:5..3:14`. Prefix the ids (`near:3:4` / `far:3:4`).
9. **`docs/STIMULUS_DESIGN.md` §5 table overstates the `repeat` perturbation** as "40
   reordered". Measured over 500 quartets: mean 37.2 differing prompt token positions, range
   30–40.
10. **`tests/test_lure_generator.py:665`** rebuilds with `QuartetConfig()` rather than
    `QuartetConfig(**prov["config"])`, and compares only the first 200 of 2,000 quartets. The
    full-file check I ran by hand passes; the test should do it (or at least honour the recorded
    config).
11. **Determinism is version-conditional and the version is not recorded.** `np.random.default_rng([seed, i])`
    is stable across current numpy, but nothing in the artifact pins it. See P1-7.
12. **`docs/UPSTREAM_REPORT.md:100` explicitly requires "the manifest must record `0..99`"**;
    `stimuli_provenance.json` records `max_nodes: 100` only.
13. **`near_critical`'s answer changes by 1, 2 or 3 tokens depending on `depth`** (measured mean
    2.0 full-line token distance beyond the 2 prompt tokens). `depth` is in the manifest, so this
    is controllable, but no analysis plan says PSI/H2 will be reported by depth. Say so before
    seeing model output.
14. **`check_quartet` raises rather than reports** when prompt token lengths differ
    (`validate.py:445` → `token_diff_positions` → `GraphError`). Currently unreachable because
    `validate_line` returns early on any non-G(5,5) line, but it makes the function's contract
    ("returns a list of problems") conditionally false.

---

## Things I tried to break and could not

Recorded so the absence of a finding is legible rather than inferred:

- substituting a condition from a neighbouring quartet — caught (200/200);
- swapping the `near_safe` and `near_critical` labels — caught (200/200);
- anchoring `near_safe_aligned` on `base` instead of `repeat` — caught;
- a genuinely gap-mismatched quartet built through the real constructor — caught, and the
  self-check refuses to emit it;
- tampering with an answer field — caught;
- tampering with `far_edge_overlap` — caught;
- `far_critical` set to a reshuffle of `base` — caught ("path unchanged");
- truncating an edge out of a condition — caught;
- a reshuffled training graph presented as novel — caught by the order-invariant key;
- worker count 1 vs 4, and a full regeneration of the shipped pool — bit-identical;
- LS-1's exchangeability claim, tested rather than asserted — holds (p=0.256).

---

## Fixes applied in this review

P0-1, P0-2 and P0-3 only.

**Code**

- `src/lurestar/validate.py` — new `FAR_MAX_EDGE_OVERLAP` and `_depth_of_head_node`;
  `check_quartet` gains a `far_max_edge_overlap` parameter and three blocks: **LS-0** (both
  near lures must be swaps at one re-derived common depth in `1..3`, and `record["depth"]`
  must agree), the far-overlap cap, and recomputation of every stored `graph_key`,
  `prompt_sha256` and `answer` from the condition's own `line`.
- `src/lurestar/generate.py` — `make_quartet` self-checks with
  `check_quartet(record, far_max_edge_overlap=cfg.far_max_edge_overlap)` so checker and
  generator cannot drift; the CLI leakage gate is extracted into `leaked_quartet_ids()`,
  which keys off `c["line"]` (canonical key **and** prompt hash) instead of the record's
  stored `graph_key`.
- `docs/STIMULUS_DESIGN.md` — new §4 "LS-0" documenting the added invariant and the two
  other closed holes.

**Tests** — six added, all mutation-tested (each was run against a checker with only its
own fix removed and had to fail):

| test | mutant it kills |
|---|---|
| `test_checker_rejects_near_lures_at_MISMATCHED_SUFFIX_DEPTHS` | LS-0 block removed → FAILED |
| `test_checker_rejects_a_near_lure_relabelled_as_the_FAR_CONTROL` | cap removed → FAILED |
| `test_checker_recomputes_stored_identities_and_rejects_a_wrong_one[graph_key]` | identity block removed → FAILED |
| `test_checker_recomputes_stored_identities_and_rejects_a_wrong_one[prompt_sha256]` | identity block removed → FAILED |
| `test_checker_recomputes_stored_identities_and_rejects_a_wrong_one[answer]` | identity block removed → FAILED |
| `test_leakage_gate_recomputes_the_key_instead_of_trusting_the_record` | gate reverted to `c["graph_key"]` → FAILED (`assert [] == [2]`) |

The depth-mismatch test additionally asserts that LS-1 still *passes* on its rogue, so the
new failure is attributable to depth alone; the far-control test asserts the cap is a real
parameter by loosening and tightening it.

**Manifests are byte-unchanged.** None of the fixes touches the generator's RNG stream.
Full CLI regeneration after the patch (`--n-quartets 2000 --n-a-pair 1000 --near-per-item 5
--far-per-item 15 --workers 4`):

```
e70fb087b6b1dd6fa7129303bbc4bcc30843c327fcab168937976295cbf2dd10  a_pair.jsonl
364978600eb73a6e9044e812dd974fe6a2df509b7f256079dc3c7d2ec8ab99e3  b_far.jsonl
7e4a414fc51c693e850fb5a0e01a651e3e78cb01304ddf1704cf11aad5314528  b_near.jsonl
f67765e6ea2afd4156c9d03ad0271afe224f1a54ddf1afc82118fcc3e4541495  e_lure.jsonl
```

— identical to `manifests/stimuli_provenance.json` and to the shipped files.

### After-output

```
$ .venv/bin/python -m pytest tests/test_lure_generator.py -q
......................................................                   [100%]
54 passed in 13.37s

$ .venv/bin/python -m pytest -q
389 passed in 70.83s (0:01:10)
```

(The three `tests/test_hmm_pairs.py` failures seen at the start of this review were fixed by
that track's own agent while this review ran; the repo is now green.)

### Left for the track owner

P1-4 through P1-7 and every P2. P1-5 requires editing `representations.py`, owned by a
concurrent track whose tests pin the five-condition centering pool; P1-6 requires
regenerating every manifest and invalidating four recorded sha256s; P1-4 requires a spec
decision about whether `B_far` should be reconstructed or the mismatch declared. Bundling
any of them with the P0 patch would have put the P0 fixes at risk.

**Verdict: FAIL at review time, PASS after the applied fixes** — conditional on P1-4 and
P1-5 being closed before H1/H3 are run, because as delivered the LS-2 robustness check
cannot be computed and an H3 matching requirement is unmet and undeclared.
