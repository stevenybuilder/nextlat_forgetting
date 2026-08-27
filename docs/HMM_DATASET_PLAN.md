# `hmm_belief`: how the HMM corpus plugs into the pinned trainer

Required experiment B needs GPT and NextLat trained on the frozen 4-state HMM. This document
specifies the smallest integration that achieves it, and — equally important — states what it must
not touch. Upstream is pinned read-only at `3770be6009cea2b3c455a9ce7f2ca88b504bb955`; nothing under
`upstream/` is edited, and no shared training code is modified, on the runtime or anywhere else.

Everything below is derived from the pinned source with file:line anchors. It has not been executed
yet: this host has no torch, so the datamodule itself is written and smoke-tested on the Colab
runtime, the same way the Path-Star smoke test was (RUNLOG, "Smoke test green on L4").

---

## 1. What the trainer actually requires of a datamodule

`train.py:34-41` is a plain dict:

```python
DATAMODULES = {
    "tinystories": TinyStoriesDataModule,
    "stargraph": StarGraphDataModule,
    ...
}
```

`train.py:176-181` looks the config's `data.dataset` up in it, constructs
`DataModuleClass(fabric, config)`, calls `datamodule.update_config(config)` and
`datamodule.get_tokenizer()`. `core_train.py:337-356` then calls `train_dataloader()`,
`val_dataloader()`, `generalization_dataloader()` (only when `data.test_generalization` is true),
`get_tokenizer()`, and uses `prepare_batch` if the object happens to define one.

That is the entire contract. Concretely, a new datamodule must provide:

| member | why | anchor |
|---|---|---|
| `__init__(fabric, config)` | constructed positionally | `train.py:180` |
| `update_config(config)` | must set `model.vocab_size`, `model.context_length`, `model.block_size` | `data/stargraph.py:249-252`, called at `train.py:181` before `initialize_model` at `train.py:226` |
| `get_tokenizer()` | must return an object with `.eos_token_id` | `core_train.py:71` |
| `train_dataloader()` / `val_dataloader()` | batches of shape `(device_batch_size, seq_len)`, `torch.long` | `core_train.py:342-343` |
| `generalization_dataloader()` | only if `data.test_generalization` is true | `core_train.py:348-349` |

A batch is a single tensor, not a tuple: `compute_loss` slices it as `batch[:, :-1]` / `batch[:, 1:]`
(`models/model_gpt.py:352-354`).

**No `evaluate_*` hook is needed and none can be added without editing shared code.** The accuracy
dispatch at `core_train.py:672-735` is an if/elif chain on `config.data.dataset` covering
`stargraph`, `countdown`, `manhattan` and `fineweb*`. `hmm_belief` matches none of them, so
validation reports losses only. Section 5 explains why that is the right trade here — the HMM has a
better competence gate than accuracy anyway.

---

## 2. The additive pieces

Two new files, both outside `upstream/`:

```
src/hmm_geometry/datamodule.py   # HMMTokenizer + HMMBeliefDataModule (the only torch-dependent file)
scripts/train_hmm.py             # registration shim; imports upstream train.py and calls do_train
```

The shim is what keeps the integration additive. `train.py` guards its argument parsing with
`if __name__ == "__main__"` (`train.py:254`), so importing the module executes nothing except the
imports, and `do_train` is importable:

```python
# scripts/train_hmm.py  (runs from inside upstream/NextLat, which is on sys.path)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import train as upstream_train                     # no side effects at import time
from omegaconf import OmegaConf
from hmm_geometry.datamodule import HMMBeliefDataModule

upstream_train.DATAMODULES["hmm_belief"] = HMMBeliefDataModule   # the whole integration

config = OmegaConf.merge(
    OmegaConf.load("defaults.yaml"),
    OmegaConf.load(args.config),
    OmegaConf.from_dotlist(overrides),
)
upstream_train.do_train(config, hide_progress_bar=args.no_pbar, checkpoint_path=args.checkpoint_path)
```

Mutating `DATAMODULES` from outside is a one-line dict insertion into an in-memory object; the file
on disk is untouched and `git status` in `upstream/NextLat` stays clean. The shim re-implements only
the config merge from `train.py:295-333`, and it deliberately does *not* re-implement the `sweep:`
branch — UPSTREAM_REPORT §5.4 flags that mechanism as a reproducibility hazard, and the three
preregistered seeds are launched as three explicit runs.

---

## 3. On-disk form, and why it is not text

`data/hmm/hmm4x4_{split}_len{L}_{N}.npy` — a `uint8`/`int8` array of shape `(N, L)` holding raw
observation symbols in `0..3`. Hashes and provenance live in `manifests/hmm_dataset.json`.

Upstream's Path-Star corpus is a text file because its examples are variable-content strings that
have to be parsed back into token ids (`data/stargraph/prepare.py:70-73`, `data/stargraph.py:32-49`).
The HMM corpus has neither property: every example is a fixed-width array of small integers that is
already in token space. A `.npy` array is therefore the exact, hashable, zero-parse form — 3.2 MB for
the 100,000-sequence training split, loadable with `np.load(..., mmap_mode="r")`, and byte-identical
across machines. Re-serialising it as text would add a parser whose only job would be to undo the
serialisation.

The tokenizer is correspondingly trivial, and exists only to satisfy the `.eos_token_id` contract:

```python
class HMMTokenizer:
    def __init__(self, n_obs: int = 4):
        self.n_obs = n_obs
        self.eos_token_id = n_obs          # 4
        self.vocab_size = n_obs + 1        # 5
    def tokenize(self, symbols):
        return np.append(np.asarray(symbols, dtype=np.int64), self.eos_token_id)
```

One EOS is appended per sequence, mirroring `data/stargraph.py:51-57`. It is load-bearing, not
decorative: `create_attention_mask` (`models/model_gpt.py:234-249`, `models/model_nextlat.py:154-169`)
segments documents on EOS, and with exactly one EOS at the end each row is one document.

So a length-32 training example becomes 33 tokens: `[x_1 ... x_32, EOS]`.

---

## 4. `update_config`, and the one value that matters

```python
def update_config(self, config):
    config.model.vocab_size = self.tokenizer.vocab_size   # 5
    config.model.context_length = 0                       # no prompt: train on every position
    config.model.block_size = self.total_len              # 33 (or 65 for the length-64 split)
```

`context_length = 0` is the important one. Both loss functions gate prompt masking on it:

* GPT — `models/model_gpt.py:362-370`: `if self.config.context_length > 0:` … `masked_fill(..., -100)`
* NextLat — `models/model_nextlat.py:441-453`: the same guard, and it additionally intersects the
  mask with NextLat's own `nextlat_token_pred_mask`

With `context_length = 0` neither branch runs, so the loss is ordinary next-token prediction over
every position — which is what the HMM task is. Path-Star sets it to 62 because the shuffled edge
list is a prompt, not a target (UPSTREAM_REPORT §1.7); the HMM has no prompt.

Two consequences worth stating before anyone reads a loss curve:

1. **There is no BOS token, so `x_1` is never predicted.** Inputs are `batch[:, :-1]`, targets are
   `batch[:, 1:]` (`models/model_gpt.py:352-354`). The 32 supervised positions are `x_2 … x_32` and
   the final EOS.
2. **NextLat's EOS handling is already correct for this data.** `models/model_nextlat.py:437-439`
   builds `curr_token_is_eos` and excludes those positions from the latent-transition loss, because
   `(h_t, x_{t+1}) -> h_{t+1} -> x_{t+2}` would otherwise cross a document boundary. With one EOS per
   row that removes exactly the last position. No change required.

`block_size` must be `L + 1`; the model asserts `seq_len <= block_size`
(`models/model_gpt.py:256-259`). The length-64 generalisation split has 65 tokens, so when
`test_generalization` is true the datamodule reports `block_size = 65` — the same widening
`StarGraphDataModule` does at `data/stargraph.py:225-227`.

---

## 5. The competence gate: a known optimum, not an accuracy number

Spec §12 says to "confirm the small models learn next-observation prediction before running geometry
analysis". Because the generative process is frozen and its posteriors are exact, the *optimal*
value of the training objective is known in closed form, which is a far sharper gate than an
accuracy threshold:

```
optimal token cross-entropy = ( sum_{t=2..32} H(X_t | X_1:t-1) + H(EOS | x_1..x_32) ) / 32
                            = ( 40.202 + 0 ) / 32
                            = 1.2563 nats/token
```

computed from the frozen matrices over the 10,000-sequence validation split (the EOS term is zero
because EOS is deterministic given position). Reference points on the same split:

| quantity | value |
|---|---|
| optimal `val/loss` as the trainer computes it | **1.2563 nats/token** |
| mean conditional NLL over all 32 positions | 1.2995 nats |
| unigram (best constant) predictor | 1.3845 nats |
| uniform predictor | 1.3863 nats |
| Bayes-optimal next-observation accuracy | 0.4157 |
| best constant predictor's accuracy | 0.2697 |

The gate is therefore: **both models must reach a validation loss close to 1.2563 nats/token, and
neither may go below it** — a run reporting less than the Bayes optimum is a bug (leakage, an
off-by-one in the target shift, or a mis-set `context_length`), not a discovery. The margin between
the optimum and the unigram baseline is only `0.128` nats, so the loss axis must be read at three
decimals; that narrowness is a property of the process, and it is why accuracy is reported alongside
rather than instead.

Next-observation accuracy and the per-position breakdown are computed offline from the frozen
checkpoints by `src/hmm_geometry/evaluate.py`, using the same extraction hook as the geometry
analysis. That keeps the accuracy metric out of shared training code entirely.

---

## 6. Config

Two files, `configs/gpt_hmm.yaml` and `configs/nextlat_hmm.yaml`, derived from the shipped
Path-Star YAMLs by overriding permitted keys only — the rule that earned its place in the RUNLOG
("Missing key `test_generalization`"). Spec §12's block, plus the keys the trainer requires:

```yaml
data:
  dataset: hmm_belief
  hmm_train_data_path: data/hmm/hmm4x4_train_len32_100000.npy
  hmm_val_data_path: data/hmm/hmm4x4_val_len32_10000.npy
  hmm_generalization_data_path: [data/hmm/hmm4x4_lengen_len64_10000.npy]
  hmm_n_obs: 4
  effective_batch_size: 256      # -> device_batch_size 256, micro_batch_size 256 on one GPU
  gradient_accum_steps: 1
  test_generalization: true
model:
  n_layer: 4
  n_head: 4
  n_embd: 128
  mtp_horizon: 1                 # NextLat only
  lambda_mse: 1.0                # NextLat only
  lambda_kl: 0.0                 # NextLat only
  lambda_ce: 0.0                 # NextLat only
  proj_factor: 0.5               # NextLat only
trainer:
  train_batches: 3000
  val_interval: 300
  compile: false
seed: 1234                       # five runs: 1234, 1235, 1236, 1237, 1238
```

`device_batch_size` and `micro_batch_size` are computed by the trainer from
`effective_batch_size` (`train.py:139-153`), so only the effective size is set. `compile: false`
follows the same override as Path-Star (RUNLOG, deviation 1).

100,000 sequences at batch 256 is 390 batches per epoch, so 3,000 updates is ~7.7 epochs — the same
shape of run as Path-Star's 51 epochs, at a twelfth of the model size.

---

## 7. Hidden-state extraction, and the index that must not slip

The geometry analysis needs `h_t`, the state that summarises the prefix `x_1..x_t`, aligned with the
exact posterior `b_t` stored in `data/hmm/hmm4x4_val_posteriors.npz`.

With inputs `[x_1 ... x_L, EOS]` and no BOS, the hidden state at sequence index `t-1` is the one that
has attended to exactly `x_1..x_t` and whose head predicts `x_{t+1}`. So:

```
h_t              = hidden_states[:, t-1, :]
b_t              = beliefs[:, t-1, :]        # forward.py's storage convention, same offset
P(X_{t+1}|X_1:t) = next_obs[:, t, :]         # note: next_obs is offset by one, by construction
```

This mirrors the off-by-one that already had to be corrected once on the Path-Star side (RUNLOG,
"H1's extraction point needed correcting"). The alignment is checked directly rather than assumed:
a model with a *scrambled* index mapping cannot achieve above-chance posterior decoding, so
`tests/test_hmm_evaluate.py::test_length_generalisation_uses_the_probe_fitted_at_length_32` and the
`h3_posterior_decodability` baseline in `evaluate.py` will both collapse if the offset is wrong.

Which layer's states to take is settled by spec §7 and UPSTREAM_REPORT §2.3, not here: the final
post-normalisation hidden state, with the GPT/NextLat asymmetry handled by the hook described there.

---

## 8. What this plan explicitly does not do

* It does not edit `train.py`, `core_train.py`, `models/*`, or anything else under `upstream/`.
* It does not add an `evaluate_hmm` branch to `core_train.py`'s accuracy dispatch. Accuracy is
  computed offline.
* It does not reuse the `stargraph` dataset name to piggy-back on that dispatch. Doing so would
  route HMM batches into `evaluate_stargraph` (`core_train.py:674-681`), which slices a prompt at
  `num_target_tokens` and would fail or, worse, report a meaningless number.
* It does not change the frozen surface: model shape, optimizer, batch size, step count and seeds are
  spec §12's, and `PROGRAM.md` puts them out of reach of any autonomous edit.
