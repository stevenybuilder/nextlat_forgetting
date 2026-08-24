# Extraction contract — where the Lure-Star hidden state and logits come from

Frozen 2026-08-23, **before any confirmatory model exists**. Every claim is cited
`file:line` against the pinned upstream checkout
`upstream/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`.

Implementation: `src/lurestar/representations.py`. Tests: `tests/test_representations.py`.
Adversarial review and the two P0 corrections it forced: `docs/review/representations.md`.

---

## 1. The state

The spec (§7) says "the final post-normalization hidden state returned by the official
transformer". Upstream produces it in exactly one place per model:

| model | line | expression |
|---|---|---|
| GPT | `models/model_gpt.py:276` | `x = self.transformer.norm(x)` |
| NextLat | `models/model_nextlat.py:197` | `text_embd = self.transformer.norm(x)` |

Shape `(B, T, 384)` for G(5,5).

**It is RMS-normalized, not LayerNorm-normalized.** `transformer.norm` is
`LayerNorm(n_embd, bias=config.bias)`, the shipped `config/stargraph/5_5/*.yaml` set
`bias: false`, and `LayerNorm.forward` (`models/model_base.py:823-830`) dispatches to
`F.rms_norm` when `bias is None`. So the vector's direction is preserved and its scale is
normalized, but the feature-axis mean is **not** removed. A large shared mean component
therefore survives into every extracted state. This is precisely why the primary distance
re-centers explicitly over an item pool (§4 below) instead of using raw cosine.

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

`h, logits = f(model, x)` cannot be written once for both. The asymmetry is absorbed by
`representations.forward_states_and_logits(..., architecture=...)` and nowhere else.

**Preferred capture: no hook at all.** `return_hidden_states` is a public keyword on both
forwards, so calling the inner transformer directly touches no training code. A forward
hook on `model.model.transformer.norm` (same path for both models) is provided as
`representations.hidden_state_hook` for the case where the state must be captured inside an
unmodified `compute_loss` call. It requires `trainer.compile: false` — which the spec and
`upstream/NextLat/README.md:117-122` already require — or the submodule path gains an
`_orig_mod` level.

## 3. The correction to the preregistered extraction index

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

## 4. Distances

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

## 5. Inferential units

| quantity | unit resampled | function |
|---|---|---|
| PSI within one (model, seed) cell | **item** (quartet) | `evaluate.bootstrap_psi_items` |
| safe-lure invariance | **item** (quartet) | `evaluate.safe_lure_invariance` |
| H3 near-minus-far erosion | **item** (A_pair, within parent checkpoint) | `evaluate.similarity_dependent_interference` |
| GPT vs NextLat PSI contrast | **training seed** | `evaluate.model_contrast_seed_level` |

The model contrast takes a `Mapping {seed_id: statistic}`, not an array — so an item-level
array of 20,000 quartets cannot be passed where three seeds belong; it raises `TypeError`.
Seeds are paired (the same seed trains both models, spec §8), the estimator is the mean of
the per-seed differences, and the report carries every per-seed value, the interval, the
exact two-sided sign-flip randomization p over all `2^n` assignments, and the smallest p
that `n` seeds could attain (`2^{1-n}` = 0.25 for three seeds). That floor is the honest
statement of what three seeds can show, and the function warns when `n < 5`.

## 6. H2's model

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

## 7. Layering

`src/lurestar/representations.py` is split at a hard line:

* **Layer A** — pure numpy. Distances, centering, whitening, shrinkage, margins, index
  resolution. Runs on the CPU-only host and is fully unit-tested there.
* **Layer B** — the only torch-touching code. `import torch` happens *inside* the functions,
  so the module imports cleanly with no torch installed. Layer B does no arithmetic beyond
  `lm_head` and an index-select, and hands numpy back to Layer A immediately.

`src/lurestar/evaluate.py` is entirely Layer A and imports no torch at all.
