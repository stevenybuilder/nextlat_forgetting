# Adversarial review — the `representations` track

Reviewer: independent agent, 2026-08-23. Method: read the delivered files against the spec and
against the pinned upstream checkout `upstream/NextLat @ 3770be6`, reproduced the reported test
run, ran six hand-written probes against the public API, and ran **52 independently written
mutants** in an isolated copy of the tree (not the implementer's 14).

Verdict up front: the estimator *internals* are unusually well tested — 37 of my mutants died
without me touching a line. Everything that survived lived at the **API boundary and in Layer B**,
which is exactly where this track's science can be lost silently. Two of those were P0.

Counts: **2 P0 (both fixed), 6 P1 (2 fixed, 4 open), 8 P2 (all open).**

---

## Reproduction of the implementer's report

```
$ .venv/bin/python -m pytest tests/test_representations.py -q
.....................................                                    [100%]
37 passed in 5.51s
```

Reproduced exactly. All four files exist and are substantial
(`representations.py` 620 lines, `evaluate.py` 716 lines, `test_representations.py` 814 lines,
`docs/EXTRACTION.md` 189 lines). Nothing was described-but-not-written.

The full-suite claim was *understated*, not overstated: at review time it was `4 failed, 299
passed` — the two `test_configs.py` failures the implementer named, plus two in
`tests/test_profile_tooling.py`. None are in this track. (Both other tracks landed their fixes
during this review; the suite is green now.)

### Upstream citations: all verified

Every `file:line` anchor in `representations.py` and `EXTRACTION.md` checks out against the pinned
tree. I re-derived each rather than trusting them:

| claim | verified |
|---|---|
| `model_gpt.py:276` `x = self.transformer.norm(x)` | ✅ |
| `model_gpt.py:279-280` `targets is None → output = self.lm_head(x)` — full sequence, not last position | ✅ |
| `model_gpt.py:290-291` `return output, x` | ✅ |
| `model_nextlat.py:197` `text_embd = self.transformer.norm(x)` | ✅ |
| `model_nextlat.py:199-200` early-returns `(token_embeds, text_embd)` **before** `lm_head` | ✅ |
| `model_nextlat.py:121` `self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)` | ✅ |
| `model_base.py:823-830` `bias is None → F.rms_norm` | ✅, and `bias: false` confirmed in both `config/stargraph/5_5/*.yaml` |
| `data/stargraph.py:9-30` tokenizer → `'='=101`, EOS`=104`, `vocab_size = 100+5+1 = 106` | ✅ |
| `data/stargraph.py:245,251` `graph_description_len = prefix_len - 2 = 62 → context_length` | ✅ |

I also checked two things the report did not claim, because both could have silently invalidated
the extraction:

* `cross_entropy_loss` (`model_base.py:647-675`) applies the **same** `lm_head` with no extra
  transform, so `lm_head(h)` is faithful to the training-time logits, and NextLat's own
  `targets is None` path (`model_nextlat.py:204`) computes literally that expression.
* The loss mask really does make index 62 predict the source. `pos_ids[i] = i+1`
  (`model_base.py:598-627`), the mask is `(pos_ids <= context_length + 1) & (pos_ids != 0)`
  (`model_gpt.py:367-369`), so targets `j ≤ 61` are `-100` and the first scored prediction is
  `j = 62 → batch[63] = 49 = source`. **The implementer's index correction is correct**, and
  h[63] is causally clean: it attends to tokens 0..63, which is prompt + source only — the first
  branch node does not appear until index 64.

---

## P0-1 — the centering pool was named, not enforced; the caller was the surviving mutant

**Severity: P0. Status: FIXED.**
`src/lurestar/evaluate.py:77-100` (before), `src/lurestar/representations.py:130-138`,
`docs/EXTRACTION.md:117-131`.

The implementer's own mutation report says the one mutant that survived their first suite was
*"centering mean taken from the scored pair instead of the declared E_lure pool"*, and that they
fixed it. They fixed it **for a mutation of the implementation**. They did not fix it for the
caller — which is the only way it can actually happen in this pipeline, because
`psi_distances_centered_cosine` took a bare `np.ndarray` and asked no questions about it.

`EXTRACTION.md:124-127` states: *"This is enforced, not merely documented."* That sentence was
false. The only enforcement was that the argument had a name.

Reproduction (before the fix):

```python
declared = np.vstack([base, repeat, safe, crit, far])           # the declared E_lure pool
good = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=declared)
pair = E.psi_distances_centered_cosine(base, crit, safe,
                                       centering_pool=np.vstack([base, crit, safe]))
junk = E.psi_distances_centered_cosine(base, crit, safe,
                                       centering_pool=rng.standard_normal((7, dim)) * 50)
```

```
PSI declared pool : 0.07834418130003899
PSI pair-only pool: 0.10130624642760040     (+29%,  the mutant they thought they had killed)
PSI JUNK pool     : 0.00035056367088443     (-99.6%, seven rows of noise sharing no row with
                                             any scored state — accepted, no warning, and
                                             cheerfully reported as centering_pool_n = 7)
```

Every one of the 37 tests still passed. A 200× erasure of the primary preregistered statistic was
one keyword argument away and nothing in the codebase would have noticed.

**Fix applied.** The pool is now a constructed, checked object rather than an array
(`representations.CenteringPool`), and `psi_distances_centered_cosine` refuses anything else —
the same enforcement pattern the module already used to stop items being passed where seeds
belong:

1. `CenteringPool.from_conditions(base=…, repeat=…, near_safe=…, near_critical=…, far_critical=…)`
   raises unless **every** condition in `CENTERING_POOL_CONDITIONS` is either supplied or named
   in `declared_missing`. Dropping `far_critical` is now a recorded statement carried into the
   serialized metric (`report()["declared_missing"]`, `["complete"]`), never an omission.
2. `require_contains()` checks by exact row bytes that every scored state is actually a row of
   the pool, per scored argument separately. A pool from another `(model, seed, index)` cell, or
   noise, is rejected outright.
3. The condition order is canonicalized to `CENTERING_POOL_CONDITIONS`, so the pooled mean cannot
   depend on the order the caller wrote the keyword arguments in.

This does not — and cannot — stop an analyst who *deliberately* declares a false pool. It makes
that impossible to do by accident and impossible to do invisibly, which is the honest maximum.

New tests: `test_psi_refuses_a_bare_array_as_the_centering_pool`,
`test_centering_pool_refuses_to_drop_a_condition_silently`,
`test_centering_pool_rejects_a_pool_from_a_different_cell`,
`test_centering_pool_rejects_unknown_and_double_declared_conditions`,
`test_pool_mean_equals_the_stacked_mean_and_ordering_is_canonical`. Each carries a
discrimination assertion: the "declared missing" test asserts the partial pool gives a
*materially different* PSI, otherwise it could not detect the substitution it exists to detect.

## P0-2 — Layer B, the crux of the track, had zero executable coverage

**Severity: P0. Status: FIXED.**
`src/lurestar/representations.py:544-572` (`forward_states_and_logits`), `:575-619`
(`extract_positions`), `tests/test_representations.py:799-801` (before).

The entire scientific value of this track rests on one asymmetry: GPT returns `(logits, h)`,
NextLat returns `(token_embeds, text_embd)` and never applies `lm_head`. Get it backwards and you
get plausible numbers — a full `(N, P, D)` array of the wrong thing — and every H1/H2/H3 result is
garbage with no symptom.

That code was never executed by any test. The one test pointed at it,
`test_forward_states_and_logits_validates_the_architecture_name`, accepted
`(ValueError, RuntimeError)` and on this host always got `RuntimeError` from `_torch()`, because
`_torch()` was called *before* the architecture check. It passed for the wrong reason.

Reproduction (before the fix) — three mutants, suite green on all three:

```
SURVIVED  architecture validation deleted entirely            :: 37 passed
SURVIVED  GPT branch replaced by `logits = lm_head(hidden)`   :: 37 passed
SURVIVED  (implied) NextLat returning token_embeds as h       :: nothing could see it
```

**Fix applied.**

* The architecture check moved **before** `_torch()`, so it is reachable and testable on a
  CPU-only host and a bad name never reaches a forward pass.
* A numpy-backed stub `torch` is injected with `monkeypatch.setitem(sys.modules, …)` (auto-torn
  down, so `test_layer_b_is_lazy_and_layer_a_needs_no_torch` still holds), together with
  `_FakeGPT` and `_FakeNextLat` that reproduce upstream's two return shapes line-for-line. The
  fakes are built so the mutants are *distinguishable*: the state has 6 features, the head
  projects to a 5-token vocab, `token_embeds ≠ post_norm`, and `_FakeGPT.lm_head` is deliberately
  a different projection from the one used inside its `__call__`, so "use the value the model
  returned" is tested rather than merely intended.

New tests: `test_layer_b_gpt_uses_the_returned_logits_and_does_not_reapply_the_head`,
`test_layer_b_nextlat_applies_lm_head_and_never_returns_token_embeds`,
`test_layer_b_architecture_name_is_validated_before_torch_is_touched`,
`test_extract_positions_returns_both_frozen_indices_for_both_architectures`,
`test_extract_positions_defaults_to_the_frozen_pair_and_rejects_bad_positions`.

Mutants now killed: architecture validation removed; return values swapped; NextLat head applied
to `token_embeds`; GPT branch recomputing through a reachable head attribute; extraction positions
reversed.

## P1-1 — the whitening leakage guard was defeated by ordinary numpy coercion

**Severity: P1. Status: FIXED.**
`src/lurestar/representations.py:364` and `:383` (before).

```python
ids = frozenset(np.asarray(list(item_ids)).tolist())
```

`np.asarray([3, "x"])` is an **array of strings**. A genuinely leaked integer id `3` is stored (or
looked up) as `"3"`, stops matching its own entry, and the guard reports clean. Reproduced:

```
fit item_ids = [0..9]
  score item_ids=[3, 4]                 -> LeakageError (correct)
  score item_ids=[3, "unrelated"]       -> NO ERROR      <-- item 3 WAS in the fitting pool
```

The same coercion at fit time is equally fatal in the other direction. Fixed by dropping the
numpy round-trip: `frozenset(item_ids)` on both sides. New test
`test_leakage_guard_is_not_defeated_by_mixed_id_types` covers both sites; restoring the coercion
at either one now kills the suite.

## P1-2 — "held out" was still opt-in at the call site

**Severity: P1. Status: FIXED.** `src/lurestar/representations.py:380-389`, `evaluate.py:103-118`.

`Whitener.distance(a, b)` with `item_ids` omitted short-circuited the guard entirely, and
`psi_distances_whitened(item_ids=None)` was the default. I scored a whitener's **own fitting
rows** with no error:

```python
w2 = R.Whitener.fit(pool, item_ids=[f"q{i}" for i in range(10)])
w2.distance(pool[:3], pool[3:6])     # -> [4.236 1.513 3.654], no complaint
```

A whitener fit *without* ids could never raise at all. Fixed: a whitener fit with ids now raises
`LeakageError` if a scored batch supplies none, and `psi_distances_whitened` — a *reported*
metric — refuses both an unchecked whitener and a missing id list. New test
`test_reported_whitened_metric_demands_a_checkable_heldout_claim`.

## P1-3 — the H3 "shared-parent" assertion is algebraically tautological

**Severity: P1. Status: OPEN.** `src/lurestar/evaluate.py:701-706`, test at
`tests/test_representations.py:768-781`.

```python
if not np.allclose(diff, margin_after_far - margin_after_near):
    raise RuntimeError("erosion difference does not reduce to the shared-parent form")
```

`(b − a_near) − (b − a_far) ≡ a_far − a_near` for **any** `b`. The guard cannot fire, and
`test_similarity_dependent_interference_reduces_to_the_shared_parent_form` asserts the same
identity — a test that cannot fail on wrong data. Reproduced: replacing every `margin_before`
with `1e9` leaves `similarity_dependent_interference` at `0.2` and raises nothing, while
`erosion_near_mean` is reported as `999999996.3`.

The real invariant — that both branches descend from one `parent_checkpoint_sha256`
(spec §9: *"Near and far branches must store the same parent_checkpoint_sha256"*) — is not
representable in this signature at all. Concrete fix: take `parent_sha256_near` and
`parent_sha256_far` (or one `parent_sha256` plus the two branch records) and raise on mismatch;
replace the tautology with a check that the two branches were evaluated on the *same A_pair item
ids in the same order*. Not applied here because it changes a signature the H3 track will own and
would risk the P0 fixes.

## P1-4 — spec-required H3 and drift estimators are absent

**Severity: P1. Status: OPEN.** `src/lurestar/evaluate.py`.

`evaluate.py` is a spec §15 deliverable. Spec §6/H3 "also report" items 1–5 and spec §10's
geometry list require, and this file does not provide:

* `A_pair` cross-entropy increase and exact-path retention (§6/H3.1);
* acquisition on adaptation examples and independent near/far validation sets (§6/H3.2);
* change on an untouched base control set (§6/H3.3);
* **cosine drift of original `A_pair` states** (§6/H3.4, §10 "post-adaptation state drift");
* **pre-adaptation distance as a predictor of item-level erosion, controlling for initial margin
  and lure loss** (§6/H3.5, §10). `fit_h2` is the right shape for this but is hard-wired to the
  H2 outcome and predictor names.

`EXTRACTION.md` does not over-claim these — it lists only `similarity_dependent_interference` for
H3 — so this is incompleteness, not misrepresentation. But PROGRAM.md invariant 4 ("report all
preregistered metrics, including the nulls") means they have to exist before Sunday. Recorded so
the omission is not discovered at analysis time.

## P1-5 — `locate_prompt_delimiter`'s documented last-vs-first behaviour is untested

**Severity: P1. Status: OPEN.** `src/lurestar/representations.py:160-175`.

The docstring says: *"Uses the LAST occurrence, because a node id can never equal 101 for
maxNodes=100 but we do not want the function to silently depend on that."* Replacing the reverse
argmax with `np.argmax(hit, axis=1)` leaves the suite green (mutant M09). The stated safety
property is therefore not a property, it is a comment. One-line fix: a test with two `=` tokens in
a row asserting the later index is returned.

## P1-6 — HEAD carries a corrupted `representations.py` (my fault, working tree is correct)

**Severity: P1. Status: working tree FIXED; the commit is not.**

While my first mutation sweep was running in-place, a concurrent agent ran `git add -A` and
committed. Commit **`fb5d035`** therefore contains `src/lurestar/representations.py` with the
Ledoit-Wolf shrinkage floor replaced by `pass`:

```
-        pass
+        alpha = max(alpha, float(min_shrinkage))
```

I restored it immediately and the working tree is correct (and now further improved), but
**anyone checking out `fb5d035` gets a `Whitener` whose shrinkage floor is gone.** All later
mutation work was done in an isolated tree copy under the scratchpad. Whoever next commits should
make sure this line is in the commit; the review does not commit on its own authority.

---

## P2 — recorded, not fixed

1. **`crossfit_linear` per-fold R² uses the full-sample mean as its baseline**
   (`evaluate.py:539`, `ss_tot_k` uses `ya.mean()`). Minor, but the same file argues at
   `:470-472` that "rescaling by a full-sample mean and sd is itself a leak" and then does it.
   Use the training complement's mean per fold, or rename the field `r2_fold_vs_full_mean`.
2. **`fit_h2`'s `marginal_spearman` is in-sample and unlabelled** (`evaluate.py:606-617`). The
   held-out Spearman is `spearman_rho_pred_vs_actual`; a reader could easily quote the marginal
   one as held-out. Rename to `marginal_spearman_in_sample`.
3. **The centering mean is estimated on the same items H2 then cross-fits over.** Fold 0's
   distances were computed with a mean that saw fold 1. It is a pooled first moment over ~10^5
   states so the leak is numerically negligible, and the per-cell pooled mean is preregistered —
   which means it must **not** be quietly changed to a fold-wise mean. Record it as a known,
   accepted, declared imperfection rather than fixing it.
4. **`bootstrap_psi_items(metric=…, extraction_index=…)` are free text never checked against the
   data.** I passed whitened distances and had them serialized as
   `"metric": "centered_cosine", "extraction_index": 62`. Carry the label from the distance dict
   instead of accepting it from the caller.
5. **`min_shrinkage=1e-3` and the `shrinkage=` override are retunable knobs on the declared
   robustness metric** (`representations.py:338`) with no manifest record. Per PROGRAM.md the
   primary/robustness metric pair is frozen; the shrinkage rule should be frozen with it.
6. **Which pool the `Whitener` is fit on is undefined** — not in the spec, not in
   `EXTRACTION.md`. It must be frozen before any model exists, or it becomes a post-hoc choice.
7. **Defensive guards with no tests** (all survive deletion, none can fire in valid code):
   `len(seeds) < 2` (`evaluate.py:359`), the fit id-count check
   (`representations.py:365`), `psi_items`' 1-D check (`evaluate.py:131`), the degenerate-fold
   and train/test-intersection guards (`evaluate.py:518`, `:521`), and
   `next_token_targets`' range check (`representations.py:224`).
8. **`extract_positions` accepts but ignores an attention `mask`, and `pos_t` is built on the
   token device** — if the model is on CUDA and `device=None`, the call fails. Harmless today
   (nothing calls it yet); will bite on the first GPU extraction.

---

## Things I tried to break and could not

Reported because a review that only lists faults is not a measurement.

* **PSI sign, pairing, and centering internals.** 37 of my own mutants died on the original suite,
  including per-condition centering, centering dropped entirely, mean→median, PSI sign flip,
  pairing destroyed by sorting the two conditions independently, the whitener not being an inverse
  square root, Ledoit-Wolf replaced by a constant, cross-fitting trained on all data, standardizing
  on the full sample, the CI halved, the rng ignored, `EQ_TOKEN_ID` shifted, the branch margin
  widened to the full vocab, `BRANCH_MARGIN_INDEX` moved back to 62, `next_token_targets` off by
  one, and — the decisive one — **asymmetric centering**, where the critical distance uses the
  declared pool and the safe distance uses its own pair. That last is the subtlest way to
  manufacture PSI and the suite kills it.
* **A seed-level contrast secretly inflating n** (tiling the three per-seed differences 10,000×)
  is killed by `test_item_level_interval_is_much_narrower_than_the_seed_level_one`.
* **Determinism.** `model_contrast_seed_level` sorts its seed keys; `_bootstrap_means`' chunking
  depends only on `n` and `n_boot`; `crossfit_linear` demands an explicit `Generator` or an
  explicit fold array. I found no dict-ordering or worker-count dependence anywhere in either
  module. (`CenteringPool` now canonicalizes its condition order for the same reason.)
* **The index-62/63 correction** is not just asserted, it is derived — and I re-derived it
  independently from the loss mask and position indices above. It is right.

---

## After the fixes

```
$ .venv/bin/python -m pytest tests/test_representations.py -q
.................................................                        [100%]
49 passed in 4.18s

$ .venv/bin/python -m pytest -q
369 passed, 5 skipped in 58.65s
```

37 → 49 tests. Twelve added, all with discrimination assertions; no test was weakened or removed.
Post-fix mutation re-run: every mutant listed under P0-1, P0-2, P1-1 and P1-2 is now killed. One
equivalent mutant remains and is not worth chasing —
removing `require_conditions()` from `psi_distances_centered_cosine` survives because
`require_contains()` already subsumes it (the method itself is covered directly).
