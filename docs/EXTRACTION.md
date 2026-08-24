# Extraction contract — where the Lure-Star hidden state and logits come from

Frozen 2026-08-23, **before any confirmatory model exists**. Every claim is cited
`file:line` against the pinned upstream checkout
`upstream/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`.

Implementation: `src/lurestar/representations.py`. Tests: `tests/test_representations.py`.
Adversarial review and the two P0 corrections it forced: `docs/review/representations.md`.

---

## 1. The state

The spec (§7) says "the final post-normalization hidden state returned by the official
transformer". For two of the three arms upstream produces it in exactly one place:

| arm | line | expression |
|---|---|---|
| GPT | `models/model_gpt.py:276` | `x = self.transformer.norm(x)` |
| NextLat | `models/model_nextlat.py:197` | `text_embd = self.transformer.norm(x)` |
| BST | `models/model_bst.py:287` | `fwd = self.transformer_f.norm(fwd)` — **a choice among three candidates, argued in §3** |

Shape `(B, T, 384)` for G(5,5) in all three.

**It is RMS-normalized, not LayerNorm-normalized.** `transformer.norm` is
`LayerNorm(n_embd, bias=config.bias)`, the shipped `config/stargraph/5_5/*.yaml` set
`bias: false`, and `LayerNorm.forward` (`models/model_base.py:823-830`) dispatches to
`F.rms_norm` when `bias is None`. So the vector's direction is preserved and its scale is
normalized, but the feature-axis mean is **not** removed. A large shared mean component
therefore survives into every extracted state. This is precisely why the primary distance
re-centers explicitly over an item pool (§5 below) instead of using raw cosine.

## 2. The forward-pass asymmetry

The two forwards do **not** return the same tuple under `return_hidden_states=True`:

```python
# GPT — model_gpt.py:279-291. targets is None, so `first` IS the logits.
logits, h = gpt_inner(tokens, return_hidden_states=True)

# NextLat — model_nextlat.py:199-200 EARLY-RETURNS (token_embeds, text_embd),
# before lm_head is ever applied. The head must be applied by the caller.
token_embeds, h = nextlat_inner(tokens, return_hidden_states=True)
logits = nextlat_inner.lm_head(h)            # model_nextlat.py:121
```

`h, logits = f(model, x)` cannot be written once for both — and BST makes it a three-way
split, with a different argument type as well (§3). The asymmetry is absorbed by
`representations.forward_all_states(..., architecture=...)`, and by
`forward_states_and_logits`, which is the two-tuple wrapper over it. Nowhere else.

**Preferred capture: no hook at all.** `return_hidden_states` is a public keyword on both
forwards, so calling the inner transformer directly touches no training code. A forward
hook on `model.model.transformer.norm` (same path for both models) is provided as
`representations.hidden_state_hook` for the case where the state must be captured inside an
unmodified `compute_loss` call. It requires `trainer.compile: false` — which the spec and
`upstream/NextLat/README.md:117-122` already require — or the submodule path gains an
`_orig_mod` level.

## 3. BST — three candidate states, and why the forward encoder's is the analogue

BST is the third arm (spec §8; `docs/DECISION_D20_competence_gate.md`, "Superseded in
part"). It exists because GPT sits at 1/d chance on `G(5,5)` while BST solves it at ~99.9%
without a latent-transition objective, which is what makes the cross-model contrast
identifiable. Its config differs from the GPT config by two keys — `use_bst: true` and
`bst_pair_minimum_gap: 2`, both at
`config/stargraph/5_5/bst_stargraph_5_5.yaml:1,42`. Note that
`bst_pair_minimum_gap` lives under `model:`, not under `data:`.

**Stated plainly, up front: BST has no clean single analogue of the other two arms' final
post-normalization state.** In GPT and NextLat, one tensor is simultaneously (a) the output
of the last block's final norm and (b) the immediate input to `lm_head`. In BST those are
different tensors, and there are three candidates rather than two.

### The three candidates

| candidate | line | shape | what it is |
|---|---|---|---|
| forward encoder post-norm | `models/model_bst.py:287` | `(B, T, 384)` | `fwd = self.transformer_f.norm(fwd)` |
| backward encoder post-norm | `models/model_bst.py:313` | `(B, T, 384)` | `bwd = self.transformer_b.norm(bwd)` |
| TextHead pre-logit `x_next` | `models/model_bst.py:92-100` | `(B, T, 384)` | chunk of a third norm over `cat(fwd, bwd)` |

All three are `(B, T, 384)`. **Shape cannot distinguish them**, which is why the choice has
to be argued rather than asserted, and why `tests/test_representations.py` §13 asserts the
returned state differs numerically from both of the others.

The readout is not `lm_head(state)`. `TextHead.forward` concatenates the two encoder
outputs to `2*n_embd`, adds a SwiGLU MLP residual, applies its own `LayerNorm(768,
bias=False)`, splits the result in half and only then projects the next-token half
(`model_bst.py:92-105`). `lm_head` is `nn.Linear(n_embd, vocab_size)`
(`model_bst.py:70`), so `lm_head(fwd)` type-checks — 384 in, 106 out — and is
semantically wrong: that head was never trained on `fwd`.

### The backward encoder is excluded, and the reason is leakage

`create_attention_masks` builds the backward mask as `torch.triu(...)`
(`model_bst.py:215-217`), so `bwd[t]` attends tokens `t..T-1`. At the preregistered
extraction index that means **`bwd[62]` has read tokens 62 through 68 — the entire answer
path**. A PSI computed on the backward state, or on any state derived from it with the
full sequence as input, would be measuring the target rather than the history. Spec §7
defines the state as the encoding of the prompt "before the first answer token"; the
backward state is definitionally not that.

This is not a hypothetical. `tests/test_representations.py::
test_bst_backward_state_sees_the_answer_which_is_why_it_is_excluded` builds two items with
an identical prefix and different suffixes and shows the forward state at a shared index is
bit-identical between them while the backward state at the same index is not.

### The choice, frozen

**Primary: the forward encoder's final post-norm state, `model_bst.py:287`.** Five reasons,
in descending order of weight:

1. **It is architecturally the same object — not merely a similar one.** Compare
   `model_bst.py:141-155` with `model_gpt.py:187-201`: the same `token_embedding`, the
   same `RotaryPositionEmbedding(max_seq_len=block_size+1, head_dim=n_embd//n_head)`, the
   same `Block` class (BST imports it from `models/model_gpt.py` at `model_bst.py:28`),
   and the same `norm=LayerNorm(n_embd, bias=config.bias)`. `transformer_f` is GPT's
   `transformer` ModuleDict, key for key. So the final norm is the same `F.rms_norm`
   dispatch (`model_base.py:823-830`), the arms are architecture-matched by construction,
   distances are computed in commensurable 384-dimensional spaces, and PSI magnitudes are
   comparable across arms rather than merely correlated.
2. **It is causal over the prompt alone.** `mask_f = torch.tril(...)`
   (`model_bst.py:204-206`), so `fwd[62]` depends on tokens 0..62 and nothing later —
   exactly like the GPT and NextLat states. Feeding the full serialized sequence to the
   extractor leaks nothing into indices 62 and 63.
3. **It shares the index convention.** Training pairs `(t, t+k)` take
   `next_tokens = seq[pair_idx[:, 0] + 1]` (`model_bst.py:581`), i.e. `fwd[t]` predicts
   `seq[t+1]` — the same convention that puts the source at index 62 and the first branch
   decision at index 63 (§4). Pairs additionally start at `start_index=context_length`
   (`model_bst.py:410-416`), which is BST's version of the prompt-loss masking the other two
   arms apply.
4. **It is the only item-varying input to BST's own inference path.** `BST.generate`
   computes its backward embedding from a *lone EOS token*
   (`model_bst.py:803-809`) and holds it fixed for the whole generation. Every item-to-item
   difference in BST's next-token distribution is therefore a difference in `fwd`.
5. **It is what the objective trains.** `compute_loss` detaches the encoder outputs,
   accumulates TextHead gradients against them and backpropagates into the encoder through
   `fwd_emb`/`bwd_emb` (`model_bst.py:555-560, 635-646`).

**Declared secondary: the TextHead pre-logit state `x_next`, `model_bst.py:92-100`,
computed with the inference-time backward embedding.** It is reported alongside, always,
under the key `hidden_texthead`, for the one reason the primary does not cover: `x_next` is
the state `lm_head` actually consumes, so it is the *functional* analogue of GPT's `h` even
though it is not the architectural one. It is secondary rather than primary because with
the backward input held at its inference-time constant, `x_next` is a fixed deterministic
function of `fwd` — a reparameterization through an MLP residual, a norm and a chunk. It
can move distances; it cannot add information. If PSI at `hidden` and PSI at
`hidden_texthead` disagree, that is a result to report, exactly as with indices 62 and 63.

**Excluded: the backward encoder state**, for the leakage reason above.

### The backward input, and why the lone-EOS choice is not a hack

BST's logits need *some* backward embedding. We use the encoding of a single EOS token,
which is what `BST.generate` does (`model_bst.py:803-809`) and therefore what produced the
paper's Figure 6 accuracy. Two properties make it the defensible choice rather than a
convenient one:

* it is **item-independent** by construction, so it introduces no cross-item variance and
  cannot smuggle any part of the answer into the comparison;
* it is **on-distribution**. Under the reverse-causal document mask the terminal EOS of a
  serialized `G(5,5)` sequence attends only itself (`model_bst.py:215-217`), and EOS always
  takes document-relative position 0 (`model_base.py:572-583`). So the lone-EOS embedding
  equals `bwd[68]` of a real training sequence — and pairs `(t, 68)` are valid training
  pairs. The model was trained against this exact backward state, as the empty-suffix case.

The stargraph tokenizer has no semantic EOS; id 104 exists "for compatibility with the bst
training code" (`data/stargraph.py:27-30`) and is appended to every line by
`Tokenizer.tokenize` (`data/stargraph.py:51-57`), so index 68 of a `G(5,5)` sequence is an
EOS.

### The API asymmetry this forces

For GPT and NextLat, the extractor takes the *inner* transformer, `wrapper.model`. **BST has
no `.model`.** Its logit path needs both `.encoder` (`model_bst.py:339`) and `.text_head`
(`model_bst.py:340`), so for `architecture="bst"` the extractor takes the `BST` wrapper
itself. That is checked, with a named error rather than an `AttributeError` forty lines
deep. Two further differences: BST builds its own document masks inside the encoder
(`model_bst.py:186-219`), so an external `mask=` argument is refused; and
`BST.compile()` rebinds `raw_f_enc` itself (`model_bst.py:317-324`), which makes a compiled
BST a harder hook target than a compiled GPT — out of scope, since the spec requires
`compile: false` anyway.

In code:

```python
# GPT / NextLat — the inner transformer.
out = forward_all_states(wrapper.model, tokens, architecture="gpt")
# -> {"hidden": transformer.norm(x), "logits": ...}

# BST — the wrapper, because the logit path spans two submodules.
out = forward_all_states(bst_wrapper, tokens, architecture="bst")
# -> {"hidden": transformer_f.norm(fwd),      # model_bst.py:287, PRIMARY
#     "hidden_texthead": x_next,              # model_bst.py:100,  declared secondary
#     "logits": text_head(fwd, bwd_eos)[:,0]} # model_bst.py:109
```

`bst_texthead_prelogit` duplicates `model_bst.py:92-100` because upstream's head returns
only stacked logits when targets are `None` and we want the state. The duplication is made
self-checking: Layer B asserts that `lm_head` applied to the re-derived `x_next` reproduces
the head's *own* returned next-token logits, and raises if it does not. A future upstream
change to `TextHead` fails loudly instead of silently forking.

`representations.STATE_SOURCE`, `HIDDEN_STATE_MODULE_PATH`, `BST_STATE_ROLES` and
`BST_STATE_POLICY` carry all of the above into the code and into every serialized metric,
so the choice travels with the number.

## 4. The correction to the preregistered extraction index

### What the spec says

Spec §6/H1: *"Extract the state at the final prompt delimiter `=`, before the first answer
token."* For G(5,5) that is sequence index **62**, verified two independent ways:

* by arithmetic on the serialized format (`data/stargraph/prepare.py:68-73`) —
  20 edges × 2 node tokens + 19 `|` + 1 `/` + source + goal = 62 prompt tokens, then the
  `=`, 5 path tokens and EOS, for `total_len = 69`;
* by upstream itself, which writes that same 62 into `config.model.context_length` at
  `data/stargraph.py:251` and uses it for prompt-loss masking
  (`model_gpt.py:362-370`, `model_nextlat.py:441-453`).

### Why index 62 alone cannot test H2 or H3

The token *generated from* index 62 is `path[0]`, which for a Path-Star answer is **the
source node**. The source is already printed verbatim in the prompt's `/source,goal` field,
so predicting it is a copy — and by the spec §5 matching rules, source and goal are held
**fixed across every member of a quartet**.

Therefore a correct-branch logit margin measured at index 62 is scoring *the same token* in
the base, repeat, near-safe and near-critical conditions. It cannot differ between them, and
no amount of data would make it differ. The first genuine branch decision is `path[1]` — the
first node of the goal arm — generated from the hidden state at index **63** (the source
token, now inside the answer region).

Worked example, tokenized with upstream's own `Tokenizer` (`data/stargraph.py:9-57`):

```
prompt tail   … |53,5 / 49,33 =        answer   49,97,53,5,33
index          …  57 58  60 61  62               63 64 65 66 67   68=EOS
h[62] scores token 49  (the source — a prompt copy, constant within a quartet)
h[63] scores token 97  (the first branch node — the decision the lure is designed to move)
```

### The resolution, frozen

1. **Index 62 remains the preregistered primary extraction point for H1 PSI.** A
   preregistered primary is not moved after the fact, and the quartet's *states* at 62 can
   legitimately differ even when their next-token target does not.
2. **Index 63 is extracted alongside it, always.** Every correct-branch logit margin used
   by H2 and H3 is computed from the logits at index 63.
3. **Both indices are reported for every metric**, so the choice is auditable rather than
   load-bearing. If PSI at 62 and PSI at 63 disagree, that is a result to report, not a
   selection to make.

In code:

```python
PSI_EXTRACTION_INDEX = 62   # H1 PSI, preregistered primary
BRANCH_MARGIN_INDEX  = 63   # H2/H3 correct-branch logit margin
```

`representations.resolve_extraction_indices()` re-derives both from a real token batch and
raises if the `=` is not at 62, if the batch is ragged with respect to the delimiter, or if
63 falls outside the sequence. It is the guard that would catch a `G(d,l)` change or a
tokenizer change before any geometry is computed.

## 5. Distances

**Primary: centered cosine.** `1 - cos(a - m, b - m)`.

The centering mean `m` is computed over the **full E_lure evaluation pool for one
(model, seed, extraction_index) cell** — all base, repeat, near-safe, near-critical and
far-critical states pooled together, one mean vector per cell. It is never computed per
condition, and never over only the pair being scored.

Centering per condition would subtract exactly the between-condition shift that PSI is
trying to measure, which is a silent way to manufacture or erase an effect. Naming the
argument is not enough to stop that — an adversarial review (`docs/review/representations.md`
P0-1) showed that passing the scored triple inflated PSI by 29% and passing seven rows of
noise erased it by 99.6%, both without a single test failing. So the pool is now a
**constructed, checked object**, and `psi_distances_centered_cosine` refuses a bare array
the same way `model_contrast_seed_level` refuses one:

```python
pool = CenteringPool.from_conditions(
    base=..., repeat=..., near_safe=..., near_critical=..., far_critical=...
)
```

* `from_conditions` raises unless **every** condition in `CENTERING_POOL_CONDITIONS` is
  supplied or named in `declared_missing`, so a dropped condition is a recorded statement
  carried into the serialized metric (`report()["declared_missing"]`, `["complete"]`),
  never an omission;
* `require_contains()` verifies by exact row bytes, per scored argument, that the states
  being scored are rows of that pool — a pool from another `(model, seed, index)` cell is
  rejected outright;
* the pooled conditions are canonicalized to `CENTERING_POOL_CONDITIONS`, so the mean
  cannot depend on keyword-argument order.

`tests/test_representations.py` checks both halves of the resulting invariance — the
distance is invariant to a translation of the whole pool, and is **not** invariant to
translating one condition only — plus the five pool-construction tests above.

**Declared robustness check: whitened Euclidean.** `||W(a - b)||` with `W @ W = Sigma^-1`,
equal to the Mahalanobis distance under `Sigma` (asserted in the tests against an
independent `np.linalg.solve` reference).

Three properties are enforced by `representations.Whitener`:

* the covariance is estimated on a **held-out pool** passed to `Whitener.fit`;
* **shrinkage is always applied and always reported** — Ledoit-Wolf (2004) intensity toward
  `(tr(S)/p) I` by default, with a `1e-3` floor so an intensity of exactly zero can never be
  reported, and `Whitener.report()` carries the intensity, the pool size, the feature count
  and the condition number into every serialized metric;
* `Whitener.distance(..., item_ids=...)` raises `LeakageError` if any scored item was in the
  fitting pool, **and** if a whitener fit with ids is handed a batch with none — otherwise
  "held out" is opt-in at the call site, which the review found it was. Ids are compared as
  raw Python objects; the earlier `frozenset(np.asarray(list(ids)).tolist())` silently cast a
  mixed id list to strings and let a genuinely leaked integer id through.
  `psi_distances_whitened`, being a *reported* metric, refuses both an unchecked whitener and
  a missing id list.

## 6. Inferential units

| quantity | unit resampled | function |
|---|---|---|
| PSI within one (arm, seed) cell | **item** (quartet) | `evaluate.bootstrap_psi_items`, `evaluate.psi_per_arm` |
| safe-lure invariance | **item** (quartet) | `evaluate.safe_lure_invariance` |
| H3 near-minus-far erosion | **item** (A_pair, within parent checkpoint) | `evaluate.similarity_dependent_interference` |
| any cross-model PSI contrast | **training seed** | `evaluate.model_contrast_seed_level`, `evaluate.three_arm_contrasts` |

The model contrast takes a `Mapping {seed_id: statistic}`, not an array — so an item-level
array of 20,000 quartets cannot be passed where three seeds belong; it raises `TypeError`.
Seeds are paired (the same seed trains all three arms, spec §8), the estimator is the mean
of the per-seed differences, and the report carries every per-seed value, the interval, the
exact two-sided sign-flip randomization p over all `2^n` assignments, and the smallest p
that `n` seeds could attain (`2^{1-n}` = 0.25 for three seeds). That floor is the honest
statement of what three seeds can show, and the function warns when `n < 5`.

### The three preregistered contrasts

`evaluate.PREREGISTERED_CONTRASTS` fixes the order before any number exists, and
`three_arm_contrasts` returns them in that order and never re-sorts by effect size:

| # | contrast | role | how to read it |
|---|---|---|---|
| 1 | NextLat − BST | **primary**, competence-matched | both arms solve `G(5,5)` (~99.8% / ~99.9%) and differ only in objective, so a gap is attributable to the objective |
| 2 | NextLat − GPT | secondary, competence-**confounded** | GPT is at 1/d chance (~18.6%); report the number and the confound in the same breath |
| 3 | BST − GPT | reference, competence only | the size of the effect that solving the task alone buys; the yardstick contrast 1 is read against |

Each is reported by `evaluate.contrast_with_mde`, which never returns an effect without the
two things that bound it: the seed-level interval, and the **minimum detectable effect** —
the smallest `|Δ|` a paired `n`-seed test could have found at the observed seed-to-seed SD
(`evaluate.minimum_detectable_effect`, two-sided paired *t*, noncentral-*t* power solved
exactly, monotone decreasing in `n`).

For the confirmatory design the numbers are blunt. **Three seeds resolve nothing below
≈3.26 seed-level standard deviations** at α=0.05, power 0.80. And the exact sign-flip test
cannot reach p ≤ 0.05 at *any* effect size, because its floor is `2^-2` = 0.25; five seeds
do not fix that either (floor 0.0625), six is the first `n` that can. `MDEResult` carries
`randomization_test_can_reject` so this cannot be quietly omitted. Every null reported from
this design has to be stated as "not detectable at this resolution", not as "no effect".

## 7. H2's model

```
critical_correct_branch_margin ~ base_critical_distance + base_correct_branch_margin
```

Two-fold cross-fitting (`evaluate.fit_h2` → `evaluate.crossfit_linear`): each fold is
predicted only by a model fit on its complement, and the standardization constants are fit
on the training complement too, because rescaling by a full-sample mean and sd is itself a
leak. Reported: held-out `R²` (pooled out-of-fold, full-sample-mean baseline, so it can go
negative), per-fold `R²`, Spearman correlation of held-out prediction against outcome, the
marginal Spearman of each predictor, and the standardized coefficient sign in every fold
with a `sign_consistent` flag. Margin is primary because accuracy may be at ceiling
(spec §6/H2). Every margin is taken from index **63**.

## 8. Layering

`src/lurestar/representations.py` is split at a hard line:

* **Layer A** — pure numpy. Distances, centering, whitening, shrinkage, margins, index
  resolution. Runs on the CPU-only host and is fully unit-tested there.
* **Layer B** — the only torch-touching code. `import torch` happens *inside* the functions,
  so the module imports cleanly with no torch installed. Layer B does no arithmetic beyond
  `lm_head` and an index-select, and hands numpy back to Layer A immediately.

`src/lurestar/evaluate.py` is entirely Layer A and imports no torch at all.

Layer B's three-way dispatch is exercised without a GPU: `tests/test_representations.py`
§12 and §13 inject a numpy-backed stub `torch` and run the GPT, NextLat and BST branches
for real, including the assertions that BST's `hidden` is neither the backward state nor
the TextHead chunk (all three are `(B, T, 384)`, so shape alone can never catch a swap).
