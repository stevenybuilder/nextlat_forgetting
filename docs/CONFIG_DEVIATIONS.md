# Configuration deviations

Every training configuration in `configs/` is a **copy** of an official Path-Star `G(5,5)`
YAML from the pinned repository with an explicit override set applied on top. Nothing here is
reconstructed from the spec document.

- Pinned upstream: `JaydenTeoh/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
- Sources: `config/stargraph/5_5/gpt_stargraph_5_5.yaml`,
  `config/stargraph/5_5/nextlat_stargraph_5_5.yaml`,
  `config/stargraph/5_5/bst_stargraph_5_5.yaml`, merged over `defaults.yaml`
- Generator: `scripts/materialize_configs.py` (machine-readable record: `configs/overrides.json`)
- Verification: `tests/test_configs.py` (150 checks), `tests/test_profile_tooling.py`

Spec section 8 permits exactly five classes of change — the three preregistered seeds, output
paths and experiment names, additional checkpoint/recovery frequency, model-output hooks for
saving hidden states, and paths for the immutable lure and adaptation manifests — plus the two
corrections established in `docs/RUNLOG.md`. Two further classes are *scientific* moves the
spec itself makes: the H3 adaptation objective (section 6) and the HMM architecture
(section 12). Every override below carries one of those authorities.

## Why the generator, and what it refuses to emit

`docs/RUNLOG.md`, session 1, attempt 2: a hand-written smoke config dropped
`data.test_generalization`, `train.py` died with an omegaconf `ConfigAttributeError` at the
first validation, and the driver reported success anyway. The rule "copy the official
configuration" earned its place there. `scripts/materialize_configs.py` enforces it
mechanically and refuses to write a file if any of five invariants fails:

| | invariant | what it prevents |
|---|---|---|
| I1 | no upstream key may vanish unless it is in an explicit `drops` list with a reason | the `test_generalization` crash |
| I2 | every key whose merged value differs from the merged upstream value must be a declared override — and a declared override that changes nothing is also an error | silent drift in either direction |
| I3 | a key hoisted into an explicit block must carry the value it already resolved to | a "restatement" that is really a change |
| I4 | a frozen key may move only where a config family carries a written spec authority | changing the science by changing a config |
| I5 | every stargraph data path must name the pool its config family is allowed to read | a swapped or leaked item bank |

**I5 was added after an adversarial review** (`docs/review/configs-and-launch.md`, findings 1
and 2) which showed that the near and far adaptation banks could be swapped, and that an
adaptation branch could be pointed at `E_lure`, with all 150 config tests still passing.
`data/stargraph.py:187-190` parses only `split("_")[1]` and `[2]` out of a data path — the two
`5`s — so upstream reads B_near, B_far and the 200,000-graph base corpus as the same kind of
file, and no other check in this repository could see the difference. A swap negates spec
section 6's primary outcome `erosion_near - erosion_far` exactly; a leak violates spec section
5's "No `E_lure` graph or lure may enter base or adaptation training". I5 binds three things
together: the branch name, the tag in the bank's basename (`bnear` / `bfar`), and the immutable
directory a bank may live in (`{MANIFESTS}/adapt`); it also pins each base run to the frozen
corpus and forbids any reserved pool name (`elure`, `a_pair`, `stimuli`) in a training or
validation path. The same bindings are asserted directly on the emitted YAML in
`tests/test_configs.py`, so they hold even if the generator is wrong.

Run `.venv/bin/python scripts/materialize_configs.py --check` to confirm the on-disk configs
are exactly what the generator produces from the pinned YAMLs; `tests/test_configs.py`
does this on every test run.

## The two established corrections

### 1. `trainer.compile: true` -> `false`

All five shipped `config/stargraph/5_5/*.yaml` set `compile: true`
(`gpt_stargraph_5_5.yaml:17`, `nextlat_stargraph_5_5.yaml:16`), contradicting the repository's
own `README.md:117-122`: *"We observe that `torch.compile()` produces inconsistent results on
numerically sensitive benchmarks like Path-Star and A5 ... we recommend setting
`trainer: compile: false`."* Spec section 8 requires `false`. Applied to all seven configs.

Second-order benefit: with `compile: true` the trunk becomes an `OptimizedModule` and every
submodule path gains an `_orig_mod` level, which would break the read-only forward hook on
`model.model.transformer.norm` used for hidden-state extraction.

### 2. `--devices 2 --strategy ddp` -> `--devices 1`

`scripts/stargraph/5_5/train_gpt_star_5_5.sh:3` and `train_nextlat_star_5_5.sh:3` launch two
GPUs with DDP. Spec section 8 gives the single-GPU form verbatim, and that is what
`scripts/launch_train.sh` runs:

```bash
fabric run --devices 1 --precision bf16-mixed train.py --config <config.yaml> \
  seed=<seed> trainer.out_dir=<per-job root> trainer.experiment_name=<job id>
```

`data.effective_batch_size` stays 512. `train.py:143-145` computes
`device_batch_size = effective_batch_size // world_size`, so one device carries 512 per step
where the shipped two-GPU script carried 256 per device. The optimizer sees the same effective
batch and the same update count either way; only per-step activation memory changes, which is
exactly what the profiling gate measures.

## Per-config override tables

Categories: **SEED** (preregistered seeds), **OUTPUT** (output paths / experiment names),
**CKPT** (checkpoint and recovery frequency), **MANIFEST** (immutable corpus and adaptation
manifest paths), **CORRECTION** (the two above), **H3** (spec section 6), **HMM** (spec
section 12), **EXEC** (execution-environment necessity, no scientific surface).

### `configs/gpt_lurestar.yaml` — from `config/stargraph/5_5/gpt_stargraph_5_5.yaml`

| key | value | category | authority |
|---|---|---|---|
| `seed` | `1234` | SEED | spec 8 |
| `trainer.compile` | `false` | CORRECTION | spec 8 + RUNLOG |
| `trainer.out_dir` | `/content/lurestar/runs/gpt/seed1234/base` | OUTPUT | spec 9 |
| `trainer.experiment_name` | `gpt-seed1234-base` | OUTPUT | spec 9 |
| `trainer.save_recovery_checkpoint` | `250` | CKPT | spec 9 |
| `trainer.log_to_wandb` | `false` | EXEC | — |
| `data.stargraph_train_data_path` | `/content/lurestar/data/stargraph/graph_5_5_sample_200000.txt` | MANIFEST | spec 8 |
| `data.stargraph_test_data_path` | `/content/lurestar/data/stargraph/graph_5_5_test_20000.txt` | MANIFEST | spec 8 |

Hoisted, value-preserving: `trainer.save_last_checkpoint`, `trainer.save_best_checkpoint`,
`trainer.log_to_file`, `trainer.init_from`, `model.compute_hidden_state_rank`.

### `configs/nextlat_lurestar.yaml` — from `config/stargraph/5_5/nextlat_stargraph_5_5.yaml`

Same override table as `gpt_lurestar.yaml` with `gpt` -> `nextlat` in `trainer.out_dir` and
`trainer.experiment_name`. Additional hoists: **`model.proj_factor`** and `model.lambda_ce`.

### `configs/bst_lurestar.yaml` — from `config/stargraph/5_5/bst_stargraph_5_5.yaml`

The third architecture-matched arm of spec section 8, added on 2026-08-23 as the
**competence-matched control** (`docs/DECISION_D20_competence_gate.md`, "Superseded in
part"). The paper's Figure 6 puts BST at ~99.9% exact-path accuracy on `G(5,5)` against
GPT at ~18.6%, which is `1/d`, chance. BST therefore solves Path-Star *without* a
latent-transition objective, so a NextLat-versus-BST PSI gap is attributable to the
objective rather than to task success. Primary contrast: NextLat vs BST. Secondary and
competence-confounded: NextLat vs GPT. Competence alone: BST vs GPT.

Same override table as `gpt_lurestar.yaml` with `gpt` -> `bst` in `trainer.out_dir` and
`trainer.experiment_name`:

| key | value | category | authority |
|---|---|---|---|
| `seed` | `1234` | SEED | spec 8 |
| `trainer.compile` | `false` | CORRECTION | spec 8 + RUNLOG |
| `trainer.out_dir` | `/content/lurestar/runs/bst/seed1234/base` | OUTPUT | spec 9 |
| `trainer.experiment_name` | `bst-seed1234-base` | OUTPUT | spec 9 |
| `trainer.save_recovery_checkpoint` | `250` | CKPT | spec 9 |
| `trainer.log_to_wandb` | `false` | EXEC | — |
| `data.stargraph_train_data_path` | `/content/lurestar/data/stargraph/graph_5_5_sample_200000.txt` | MANIFEST | spec 8 |
| `data.stargraph_test_data_path` | `/content/lurestar/data/stargraph/graph_5_5_test_20000.txt` | MANIFEST | spec 8 |

Hoisted, value-preserving: `trainer.save_last_checkpoint`, `trainer.save_best_checkpoint`,
`trainer.log_to_file`, `trainer.init_from`, `model.compute_hidden_state_rank`,
`use_nextlat`, **`model.bst_pair_maximum_gap`**, **`model.bst_pair_subsample_rate`**,
**`model.bst_single_gap_prediction_mode`**, **`data.pair_batch_size`**.

Nothing is dropped. No frozen key moves, so the BST arm claims no frozen-surface exemption.

#### How the official BST YAML differs from the official GPT YAML

Ignoring `trainer.experiment_name` (a label) and the `sweep:` block, the two pinned files
differ in exactly three flattened keys, and only two of them are settings:

| key | GPT YAML | BST YAML | effect |
|---|---|---|---|
| `use_bst` | `false` | **`true`** | selects `BST`/`BSTConfig` at `core_train.py:38-40` |
| `model.bst_pair_minimum_gap` | *(absent -> `defaults.yaml:98` = 1)* | **`2`** | minimum `(t, t+k)` gap BST trains on |
| `use_nextlat` | `false` | *(absent -> `defaults.yaml:3` = false)* | none; resolves false either way |

`tests/test_configs.py::test_bst_config_differs_from_the_official_gpt_config_by_two_keys_only`
asserts that set exactly against the pinned YAMLs, so the claim cannot rot.

Note that `bst_pair_minimum_gap` lives under **`model:`**, not under `data:`. It is read at
`core_train.py:80` and passed into `BSTConfig`, and `models/model_bst.py:409-414` uses it as
the `min_gap` of `_create_pair_indices`.

#### The BST keys that are written in no stargraph YAML — the D-07 hazard, again

`docs/FOUNDATIONS.md` D-07 records the `proj_factor` failure: a scientifically relevant
value that upstream only ever reaches through a fallback. BST has four more of them. Three
are consumed by `core_train.py:75-84` when it builds `model_args`, one is asserted at
`core_train.py:41-44` before the model exists, and one arrives through the trainer loop.
**None of them appears in any `config/stargraph/**/*.yaml`.** All are hoisted into
`configs/bst_lurestar.yaml` at their resolved values and added to `FROZEN_KEYS`:

| key | resolves from | value | what it controls |
|---|---|---|---|
| `model.bst_pair_maximum_gap` | `defaults.yaml:99` | `-1` | `model_bst.py:377-378` turns `-1` into `max_gap = document_len`, i.e. train on every gap |
| `model.bst_pair_subsample_rate` | `defaults.yaml:102` | `1.0` | `model_bst.py:478-483`: at `1.0` the subsample branch is skipped and every valid pair is kept; below `1.0` the training set becomes stochastic per step |
| `model.bst_single_gap_prediction_mode` | `defaults.yaml:104` | `eos` | `model_bst.py:584-588` rewrites the targets of gap-1 pairs. **Inert at `bst_pair_minimum_gap: 2`**, which produces no gap-1 pairs — but `core_train.py:41-44` asserts membership in `["next_token", "eos"]` before the model is built, so a missing value aborts the run at step 0, and lowering the minimum gap would make it live |
| `data.pair_batch_size` | `defaults.yaml:67` | `32768` | text-head chunk size — **and a loss-scaling key**, see below |
| `model.context_length` | the BST YAML (`0`) | `62` at runtime | `data/stargraph.py:243-251` overwrites it with `graph_description_len`; `model_bst.py:411` uses it as the first index a pair may start at, so BST never pairs inside the graph description |

`data.pair_batch_size` deserves the emphasis. `core_train.py:495` and `:635` pass it to
`compute_loss` for **every** arm, but GPT and NextLat absorb it into
`**kwargs  # Extra arguments ignored for compatibility with BST` (`model_gpt.py:342`,
`model_nextlat.py:418`). Only BST reads it, and it is not a pure memory knob:

```python
pair_accum_steps  = math.ceil(n_pairs / pair_batch_size)      # model_bst.py:600
texthead_loss_div = loss_div * batch_size * pair_accum_steps  # model_bst.py:601
```

With more than one chunk the text-head loss becomes a mean of chunk means rather than the
mean over pairs, and since the final chunk is short the gradient reweights. At `G(5,5)`
scale it never bites — one tokenized sequence is 69 tokens with `context_length` 62, so
`_create_pair_indices` yields `5+4+3+2+1 = 15` pairs at `min_gap: 2` against a chunk of
32,768, giving exactly one chunk — but the value is pinned and the arithmetic is asserted
rather than assumed (`test_bst_pair_batch_size_does_not_reweight_the_text_head_loss`).

At the fallback `bst_pair_minimum_gap: 1` the same sequence yields **21** pairs instead of
15, which is why losing that one key silently trains a different objective.

**`data.pair_accum_steps: 1` is a dead key.** It is written in all fifteen shipped
`config/**/*.yaml` files and read by no Python in the pinned tree — `pair_accum_steps` at
`model_bst.py:600` is a local variable derived from `pair_batch_size`. It is copied through
because it is an upstream key (I1), not because anything consumes it.

#### "Architecture-matched" — what is true, and what must be qualified

`models/model_bst.py:28` imports the very `Block` class `models/model_gpt.py:154` defines,
and builds it at the same `n_layer: 12`, `n_head: 6`, `n_embd: 384`, `bias: false`,
`dropout: 0.0`. **A BST layer and a GPT layer are the same object at the same width**, and
the optimizer, schedule, effective batch size, corpus and step count are identical across
the three arms. That is the sense in which the arms are architecture-matched, and
`test_bst_is_architecture_matched_to_gpt_and_nextlat` asserts all of it key by key.

BST is **not parameter-matched**, and the writeup must say so. `models/model_bst.py:148-161`
builds **two** independent stacks — `transformer_f` and `transformer_b` — over one shared
`token_embedding`, and `models/model_bst.py:53-77` adds a `TextHead` on top of the
concatenated forward/backward embedding:

| component | shape | params |
|---|---|---|
| `token_embedding` (shared by both stacks) | 106 x 384 | 40,704 |
| `transformer_f` — 12 Blocks + final LayerNorm | | 21,243,264 |
| `transformer_b` — 12 Blocks + final LayerNorm | | 21,243,264 |
| **`TransformerEncoder` total** | | **42,527,232** |
| `TextHead` `SwiGLU(768 -> 2048)` = `Linear(768 -> 4096)` | | 3,145,728 |
| `TextHead` `Linear(2048 -> 768)` | | 1,572,864 |
| `TextHead` `norm` `LayerNorm(768)`, weight only | 768 | 768 |
| `TextHead` `lm_head` `Linear(384 -> 106)` | | 40,704 |
| **`TextHead` total** | | **4,760,064** |
| **BST total** (`get_num_params(non_embedding=False)`, `model_bst.py:350-360`) | | **47,287,296** |

Set against the other two arms at the same `vocab_size` 106 and `block_size` 69:

| arm | parameters (incl. embedding) | vs GPT |
|---|---|---|
| GPT | 21,324,672 | — |
| NextLat, `proj_factor: 0.5` | 21,915,264 | +590,592 (latent-dynamics MLP) |
| BST | **47,287,296** | +25,962,624 (a second 12-layer stack, +4,760,064 `TextHead`, −40,704 for the embedding it shares) |

BST is 2.217x GPT. The honest phrasing for the writeup is: *the three arms are matched on
layer definition, width, depth, optimizer, schedule, batch size, corpus and step budget;
BST additionally carries a second encoder stack and a text head, as the BST objective
requires, so it is architecture-matched but not parameter-matched.* The alternative — a
6-layer BST to match parameters — would break the depth match instead and is not what the
paper's Figure 6 result was measured at, so it is not on the table.

`TextHead`'s hidden width is `128 * round((8 * 768 / 3) / 128) = 2048`, the same
`SwiGLU` sizing rule the GPT MLP uses (`model_gpt.py:139-140`), applied to `2 * n_embd`.

One consequence worth recording before extraction starts: BST's post-normalization hidden
states are at **`model.encoder.transformer_f.norm`** and `...transformer_b.norm`, not at
`model.model.transformer.norm` where the GPT and NextLat hook attaches, and there are two
of them per token. What "the representation" means for BST is an extraction decision, not a
config one, and it is not settled by this document.

### `configs/adapt_near.yaml` and `configs/adapt_far.yaml` — from the NextLat G(5,5) YAML

| key | value | category | authority |
|---|---|---|---|
| `seed` | `1234` | SEED | spec 8 |
| `trainer.compile` | `false` | CORRECTION | spec 8 + RUNLOG |
| `trainer.out_dir` | `/content/lurestar/runs/nextlat/seed1234/adapt-{near,far}` | OUTPUT | spec 9 |
| `trainer.experiment_name` | `nextlat-seed1234-adapt-{near,far}` | OUTPUT | spec 9 |
| `trainer.init_from` | `resume` | CKPT | spec 9 |
| `trainer.train_batches` | `500` | H3 | spec 6 + spec 8 |
| `trainer.val_interval` | `100` | CKPT | spec 6 |
| `trainer.test_interval` | `100` | CKPT | spec 6 |
| `trainer.save_recovery_checkpoint` | `100` | CKPT | spec 9 |
| `trainer.log_to_wandb` | `false` | EXEC | — |
| `model.lambda_mse` | `0.0` | H3 | spec 6 |
| `model.lambda_kl` | `0.0` | H3 | spec 6 |
| `data.stargraph_train_data_path` | `.../adapt/graph_5_5_b{near,far}_5000.txt` | MANIFEST | spec 6 |
| `data.stargraph_test_data_path` | `.../adapt/graph_5_5_b{near,far}val_2000.txt` | MANIFEST | spec 6 |

Hoisted: `use_bst`, `trainer.save_last_checkpoint`, `trainer.save_best_checkpoint`,
`trainer.log_to_file`, `model.proj_factor`, `model.lambda_ce`.

The two files differ in **exactly five keys**: `trainer.out_dir`,
`trainer.experiment_name`, `data.stargraph_train_data_path`, `data.stargraph_test_data_path`,
and the free-text `provenance.note`. `tests/test_configs.py::
test_near_and_far_differ_only_in_the_item_bank_and_output_root` asserts that set exactly, so
any future edit that makes the branches differ in learning rate, batch size, step count or
objective fails the suite.

### `configs/gpt_hmm.yaml` and `configs/nextlat_hmm.yaml` — from the G(5,5) YAMLs

| key | value | category | authority |
|---|---|---|---|
| `seed` | `1234` | SEED | spec 12 |
| `trainer.compile` | `false` | HMM | spec 12 |
| `trainer.out_dir` | `/content/lurestar/runs/hmm/{gpt,nextlat}/seed1234/base` | OUTPUT | spec 9 |
| `trainer.experiment_name` | `{gpt,nextlat}-seed1234-hmm` | OUTPUT | spec 9 |
| `trainer.train_batches` | `3000` | HMM | spec 12 |
| `trainer.val_interval` | `300` | HMM | spec 12 |
| `trainer.test_interval` | `300` | HMM | spec 12 |
| `trainer.save_recovery_checkpoint` | `250` | CKPT | spec 9 |
| `trainer.log_to_wandb` | `false` | EXEC | — |
| `trainer.wandb_project` | `hmm_belief` | OUTPUT | spec 9 |
| `trainer.wandb_tags` | `[hmm, 4state4obs]` | OUTPUT | spec 9 |
| `data.dataset` | `hmm_belief` | HMM | spec 12 |
| `data.effective_batch_size` | `256` | HMM | spec 12 |
| `data.hmm` | 100,000 train / 10,000 val length-32 sequences, 10,000 length-64 generalization sequences, frozen matrix manifest | HMM | spec 12 |
| `model.n_layer` | `4` | HMM | spec 12 |
| `model.n_head` | `4` | HMM | spec 12 |
| `model.n_embd` | `128` | HMM | spec 12 |
| `model.mtp_horizon` | `1` (NextLat only) | HMM | spec 12 |
| `model.lambda_kl` | `0.0` (NextLat only) | HMM | spec 12 |

Hoisted: `trainer.save_last_checkpoint`, `trainer.save_best_checkpoint`,
`trainer.log_to_file`, `trainer.init_from`, `model.compute_hidden_state_rank`, `use_bst`
(GPT), `model.proj_factor`, `model.lambda_mse`, `model.lambda_ce` (NextLat).

Spec section 12 overrides no optimizer key, so the Path-Star optimizer block survives
verbatim: AdamW at `5e-4`, `weight_decay 0.1`, betas `(0.9, 0.95)`, `grad_clip 100`, constant
schedule with `warmup_iters: 0` and `warmdown_iters: 0`.

**Dropped keys (HMM only), each read only under `dataset == "stargraph"`:**
`data.stargraph_max_nodes` (`data/stargraph.py:175`; `defaults.yaml:81` still resolves it to
50 after the merge, so nothing can raise), `data.stargraph_train_data_path`
(`data/stargraph.py:179,187-190`), `data.stargraph_test_data_path` (`data/stargraph.py:183`),
`data.stargraph_generalization_data_path` (`data/stargraph.py:203`, itself gated on
`data.test_generalization`, which stays `false`).

## Resolved paper-scale values, verified before launch

Spec section 8 requires the materialized YAML to be checked against the paper scale. These are
the values after `OmegaConf.merge(defaults.yaml, config)` (`train.py:348-351`), all asserted by
`tests/test_configs.py`:

```
trainer.train_batches            20000        data.dataset                   stargraph
trainer.val_interval              1000        data.effective_batch_size            512
trainer.test_interval             1000        data.gradient_accum_steps              1
trainer.val_batches                200        data.stargraph_max_nodes             100
trainer.test_batches               200        data.test_generalization           False
trainer.save_last_checkpoint      True        model.n_layer                         12
trainer.save_best_checkpoint      True        model.n_head                           6
trainer.save_recovery_checkpoint   250        model.n_embd                         384
trainer.compile                  False        model.dropout                        0.0
trainer.init_from              scratch        model.bias                         False
optimizer.optimizer_type          adam        model.gpt_mode                next_token
optimizer.learning_rate         5.0e-4        model.mtp_horizon        3  (NextLat only)
optimizer.weight_decay             0.1        model.lambda_mse       1.0  (NextLat only)
optimizer.beta1 / beta2     0.9 / 0.95        model.lambda_kl        1.0  (NextLat only)
optimizer.grad_clip                100        model.lambda_ce        0.0  (NextLat only)
lr_scheduler.schedule         constant        model.proj_factor      0.5  (NextLat only)
lr_scheduler.warmup_iters            0        seed                                1234
lr_scheduler.warmdown_iters          0        preregistered seeds   1234, 1235, 1236

model.bst_pair_minimum_gap             2  (BST only)   data.pair_batch_size   32768  (BST only)
model.bst_pair_maximum_gap            -1  (BST only)   data.pair_accum_steps      1  (dead key)
model.bst_pair_subsample_rate        1.0  (BST only)
model.bst_single_gap_prediction_mode eos  (BST only)
```

One parsing note that matters for this table: PyYAML follows YAML 1.1, whose implicit float
pattern requires a decimal point before an exponent, so plain `yaml.safe_load` reads the
shipped `learning_rate: 5e-4` as the **string** `'5e-4'`. OmegaConf installs its own float
resolver and reads it as `0.0005`. The generator uses plain `SafeLoader`/`SafeDumper` so the
emitted file is a literal copy of the official text; every value assertion goes through
`config_lib.load_yaml_as_trainer_sees_it`, which reproduces OmegaConf's resolver. Both facts
are pinned by `tests/test_configs.py::test_negative_control_yaml_float_resolver_matters`.

### The three-layer latent-dynamics MLP with hidden dimension 384

Spec section 8: *"The paper reports a three-layer latent-dynamics MLP with hidden dimension
384 for Path-Star. Verify that the official NextLat YAML resolves to those values."*
Traced through the pinned code, `models/model_nextlat.py:50-52`:

```python
input_dim  = config.n_embd * 2                 # 384 * 2      = 768
hidden_dim = config.proj_factor * input_dim    # 0.5 * 768    = 384
hidden_dim = 128 * round(hidden_dim / 128)     # 128 * 3      = 384
```

and `models/model_nextlat.py:60-66` builds exactly three `nn.Linear` layers with `bias=False`:

| layer | shape | params |
|---|---|---|
| `mlp.0` `Linear(input_dim -> hidden_dim)` | 768 -> 384 | 294,912 |
| `mlp.2` `Linear(hidden_dim -> hidden_dim)` | 384 -> 384 | 147,456 |
| `mlp.4` `Linear(hidden_dim -> n_embd)` | 384 -> 384 | 147,456 |
| `norm_x` `LayerNorm(input_dim)`, weight only | 768 | 768 |
| **dynamics model total** | | **590,592** |

Resolved model sizes, with `vocab_size` 106 and `block_size` 69 injected at runtime by
`data/stargraph.py:249-252`:

| | parameters (incl. embedding) |
|---|---|
| GPT | 21,324,672 |
| NextLat, `proj_factor: 0.5` (dynamics hidden **384**) | 21,915,264 |
| NextLat, `proj_factor: 1.0` — the silent fallback, **not** what we run | 22,800,000 |

**The `proj_factor` trap.** `proj_factor: 0.5` exists upstream **only inside the `sweep:`
block** (`nextlat_stargraph_5_5.yaml:61`), never under `model:`. Deleting the sweep block to
run a single seed — the obvious way to turn a sweep config into a job config, and what
`scripts/colab_smoke.py` did — silently falls back to `defaults.yaml:118` `proj_factor: 1.0`,
which builds a dynamics MLP of hidden width **768**, adds 884,736 parameters, and is not the
architecture the paper reports. The generator hoists `proj_factor` into `model:` and asserts
the hoisted value equals the value the sweep resolved to;
`tests/test_configs.py::test_negative_control_dropping_the_sweep_reverts_proj_factor`
reproduces the failure and then asserts our config does not have it.

For the HMM configs the same arithmetic gives `input_dim = 256`, `0.5 * 256 = 128`,
`128 * round(128/128) = 128` — a three-layer dynamics MLP of hidden width 128.

### Optimizer-update count

`core_train.py:564-571` increments `self.step` after each optimizer update and returns when
`self.step > train_batches`, an inclusive bound. `train_batches: 20000` therefore performs
**20,001** optimizer updates. This is upstream behaviour and the step count is on the frozen
surface, so it is recorded, not corrected. The adaptation configs express "500 updates" the
same way (`train_batches: 500` from a step-0 parent -> 501 updates), identically for near,
far, GPT and NextLat, so the near-minus-far contrast is unaffected.

## Gradient-accumulation fallback rule

`data.gradient_accum_steps` is `1` in every config and **must stay 1 unless the paper's
physical batch does not fit on the assigned GPU.**

Raising it is permitted **only** as an execution fallback, and only under all four conditions:

1. The profiling gate has measured an actual out-of-memory failure or an unsafe VRAM headroom
   at the paper's physical batch (512 for Lure-Star on one device, 256 for HMM). Spec
   section 11: *"Profile the paper's physical batch first. If it does not fit, test gradient
   accumulation only as an execution fallback."*
2. **Effective batch size stays 512** (Lure-Star) / 256 (HMM). `train.py:143-153` derives
   `device_batch_size = effective_batch_size // world_size` and
   `micro_batch_size = device_batch_size // gradient_accum_steps`, so accumulation splits the
   already-loaded batch (`core_train.py:486-499`) and never changes `effective_batch_size`.
3. **The optimizer-update count is preserved.** The accumulation loop runs inside one
   dataloader iteration and `compute_loss` divides by `loss_div = gradient_accum_steps`, so
   there is still exactly one optimizer step per dataloader batch and the total remains 20,001
   (base) / 501 (adaptation) / 3,001 (HMM).
4. It is applied identically to GPT and NextLat, to every seed, and to both the near and the
   far adaptation branch — otherwise it becomes a confound rather than an execution detail.

If used, it **must be documented as a deviation**: an appended entry in `docs/RUNLOG.md` and
`docs/FOUNDATIONS.md` (PROGRAM.md, "Frozen surface"), naming the GPU, the measured peak VRAM
that forced it, the chosen `gradient_accum_steps`, and the resulting `micro_batch_size`.
Nothing else may be changed to make the model fit: spec section 11 forbids changing width,
depth, sequence construction, precision, loss or training examples for throughput, and
PROGRAM.md stop condition 5 is "stop rather than shrink".

As of the profiling gate recorded in `docs/RUNLOG.md`, both models fit the paper's physical
batch of 512 on a 40 GB A100, so **no gradient-accumulation deviation is currently in effect.**

## Launch path

`scripts/launch_train.sh <config> <seed> [dotlist overrides...]` is the single-GPU launch path.
It resolves the per-job output root and experiment name, checks the preconditions upstream
fails silently on, prints the exact command, and execs it.

```bash
# Lure-Star base runs, three preregistered seeds, three architecture-matched arms
for s in 1234 1235 1236; do
  scripts/launch_train.sh gpt_lurestar.yaml     "$s"
  scripts/launch_train.sh nextlat_lurestar.yaml "$s"
  scripts/launch_train.sh bst_lurestar.yaml     "$s"   # competence-matched control
done

# H3 adaptation: the same two files drive both models
LURESTAR_MODEL=nextlat LURESTAR_PARENT_CKPT=/content/lurestar/parents/nextlat-seed1234.pt \
  scripts/launch_train.sh adapt_near.yaml 1234
LURESTAR_MODEL=gpt     LURESTAR_PARENT_CKPT=/content/lurestar/parents/gpt-seed1234.pt \
  scripts/launch_train.sh adapt_far.yaml 1234

# HMM
scripts/launch_train.sh gpt_hmm.yaml     1234
scripts/launch_train.sh nextlat_hmm.yaml 1234
```

`seed`, `trainer.out_dir` and `trainer.experiment_name` are passed as OmegaConf dotlist
overrides (`train.py:265,349`) rather than being baked per seed, so one file serves all three
seeds and the file on disk stays byte-identical across the matrix.

Precision is `bf16-mixed`. `docs/RUNLOG.md` records `bf16_supported=True` on both the L4 and
the A100 used so far; `LURESTAR_PRECISION=16-mixed` is the fallback spec section 8 allows for a
GPU without stable BF16 (a T4, for instance), and using it would be a recorded deviation.

### Why the `sweep:` block is deleted rather than used

`train.py:273-339` expands the sweep in one process and builds the experiment directory name
by iterating a Python **set** (`train.py:280,322`), so the directory name can change between
invocations of the same config unless `PYTHONHASHSEED` is pinned. All sweep entries also share
one `trainer.out_dir` and therefore one `latest_ckpt`/`recovery_ckpt` pair. Spec section 9
requires a separate output root per job. One file, one job, explicit seed.

### Adaptation branch launch protocol (read before the first H3 launch)

`core_train.py:139-168`: with `init_from: resume`, upstream prefers `{out_dir}/recovery_ckpt`,
falls back to `{out_dir}/latest_ckpt`, and **if neither exists it prints two "Could not find"
lines and builds a scratch model.** For an adaptation branch that means silently training a
fresh random network on 5,000 items. The branch is therefore started by pre-seeding
`{out_dir}/latest_ckpt` with the frozen parent, which `scripts/launch_train.sh` does when
`LURESTAR_PARENT_CKPT` is set, and refuses to launch without.

`DRY_RUN=1` performs every check but **writes nothing** — it prints a `# would seed` line
instead. It used to seed the pointer before the early exit, so a dry run silently decided the
branch's parent (`docs/review/configs-and-launch.md`, finding 5).

**Still unenforced, and the thing to close before the first H3 launch:** `launch_train.sh`
does not verify that `LURESTAR_PARENT_CKPT` belongs to this branch's model and seed, or that
it is step-rebased. A parent that is not rebased restores `self.step` to 20,000,
`core_train.py:569` returns immediately, and the branch performs **zero** adaptation updates
while still writing a checkpoint and a metrics.csv — `erosion_near - erosion_far` is then
`0 - 0 = 0`, a manufactured clean null. `scripts/run_matrix.py` avoids this with the
step-offset form; a manual pointer-seeded launch does not.
(`docs/review/configs-and-launch.md`, finding 4.)

There are two ways to attach a branch to its parent, and the pinned code supports both.
`--checkpoint_path` takes precedence over `init_from` unconditionally
(`train.py:262-264`, `core_train.py:130`), so `trainer.init_from: resume` in the YAML is inert
on a launch that passes it and correct on every relaunch that does not. The configs work
unchanged under either.

**(a) Step-offset attachment — what `scripts/run_matrix.py` implements.** Pass
`--checkpoint_path <parent>` on the first launch. It restores weights, optimizer *and*
`training_steps` (`model_base.py:437`), the trainer seeds `self.step` from it
(`core_train.py:309`), and the loop returns as soon as `self.step > trainer.train_batches`
(`core_train.py:569`) — so with a parent at 20,000 steps and `train_batches: 500` the branch
would perform **zero** updates. The runner therefore offsets `train_batches` by the parent's
step count at launch and keeps the YAML value as the number of *adaptation* updates, which is
the quantity PROGRAM.md freezes. Cost: `core_train.py:432-452` fast-forwards by materialising
and discarding `self.step` batches, so every launch and every resume replays ~20,000 batches
of the 5,000-item adaptation loader at `num_workers: 0`. Price that replay in the profiling
gate before the first H3 launch.

**(b) Pointer-seeded attachment — what `scripts/launch_train.sh` offers for a manual launch.**
Write the parent's path into `{out_dir}/latest_ckpt` before the first launch (the script does
this when `LURESTAR_PARENT_CKPT` is set, and refuses to launch without a pointer). Combined
with a **step-rebased** parent — the same `model`, `optimizer` and `lr_scheduler_state` payload
with `training_steps` rewritten to 0 — this makes all step arithmetic branch-local, removes the
fast-forward entirely, and starts and restarts the branch through one code path. Rebasing the
counter is checkpoint machinery, which PROGRAM.md places on the mutable surface; the model,
optimizer and scheduler state are copied byte for byte, so spec section 6's requirement that
near and far match on "optimizer and scheduler state" holds by construction — both branches
load the identical file.

Whichever is used, it must be the same for near and far, for GPT and NextLat, and for every
seed, and both branches must record the same `parent_checkpoint_sha256`.

### HMM datamodule registration

`data.dataset: hmm_belief` is not a registered datamodule at the pinned commit:
`train.py:34-42` lists `tinystories, stargraph, fineweb10B, fineweb100B, finewebedu, countdown,
manhattan`, and `train.py:176-178` asserts membership. The runtime working copy therefore
carries a one-line registration of the project's own `HMMBeliefDataModule`, applied to
`$NEXTLAT_REPO` and persisted as the uncommitted diff spec section 9 requires. **The pinned
tree at `upstream/NextLat` is never modified.** `scripts/launch_train.sh` and
`scripts/profile.sh` both refuse to start an HMM job if the registration is absent.

`data.hmm` is a namespaced sub-block rather than flat `data.*` keys so that no key we invent
can ever collide with an upstream one.

### W&B

`train.py:15,17,24` import `wandb` unconditionally, so the package must be installed even for
a fully offline run, and `defaults.yaml:34` sets `log_to_wandb: true` while none of the shipped
5_5 YAMLs override it — an unmodified run attempts `wandb.init` on the first `log_dict`. All
seven configs set `trainer.log_to_wandb: false` and keep `trainer.log_to_file: true`, so the
CSVLogger output at `{out_dir}/{experiment_name}/version_N/metrics.csv` is the metric record
that `scripts/profile_summarize.py` and the run ledger parse. This is an execution-environment
override with no scientific surface.

## Deliberate non-changes

- **`model.compute_hidden_state_rank` stays `false`.** Spec section 2 rules out effective rank
  as a standalone contribution and the in-training SVD costs throughput. Spec section 8's
  "model-output hooks needed to save hidden states" allowance is satisfied with **zero config
  surface**: the final post-normalization state is `x = self.transformer.norm(x)`
  (`model_gpt.py:276`, `model_nextlat.py:197`) and is reachable either by calling the inner
  transformer with the public `return_hidden_states=True` keyword or by a **read-only** forward
  hook on `model.model.transformer.norm`. NextLat's backward is a manual two-stage graph split
  (`model_nextlat.py:503-525`); a hook that modified the tensor in place would corrupt it, so
  extraction is offline and read-only.
- **`trainer.top_k` is not set.** `data/stargraph.py:122-127` calls `model.generate` with
  `temperature=1.0, top_k=None`, so the logged `val_(5, 5)/test_accuracy` — and therefore the
  90% base-competence gate — is measured under **ancestral sampling**, not greedy decoding.
  Adding `trainer.top_k: 1` would make it greedy and would change the reported metric relative
  to the paper. Greedy accuracy is computed offline by the project's own evaluator and reported
  alongside.
- **`data.num_workers` stays `0`.** PROGRAM.md puts dataloader workers on the mutable surface
  and raising it would speed up the resume fast-forward, but it is an unmeasured change until
  the profiling gate prices it. Determinism does not depend on it: the `RandomSampler`
  permutation is drawn in the main process.
- **`trainer.deterministic` is not set.** `defaults.yaml:38` defines it and a repo-wide grep
  finds no reader. It is a dead key.
- **`model.mtp_horizon` stays 3 in the adaptation configs.** It is on the frozen surface. The
  consequence is that the NextLat adaptation branch still computes the full 3-step rollout and
  then multiplies its losses by zero, which costs throughput but changes no gradient that
  reaches the trunk.
- **`model.lambda_ce` stays 0.0 everywhere.** `model_nextlat.py:319-324` computes the CE term
  regardless for logging (`README.md:128` says so), which costs throughput but contributes
  nothing to the loss.

## Open decision, to be closed by the section 9 recovery test

**Sampler determinism under resume.** `fabric run --devices 1` with no strategy leaves the
DataLoader's `shuffle=True` `RandomSampler`, whose per-epoch permutation is drawn from the
global torch RNG — which is **not** in the checkpoint (`model_gpt.py:426-434`,
`model_nextlat.py:577-585`). Adding `--strategy ddp` even at one device makes Fabric substitute
a `DistributedSampler` whose permutation is a fixed function of its seed, and `set_epoch` is
never called anywhere in `core_train.py`, so the order becomes identical across epochs and
across resumes — and matches the sampler regime the paper's own two-GPU script produced.

The spec-literal launch (`--devices 1`, no strategy) is the default in
`scripts/launch_train.sh`. `LURESTAR_STRATEGY=ddp` selects the alternative. Which one the
confirmatory matrix uses is decided by the spec section 9 forced-interruption test (300 steps
uninterrupted versus 150 + resume + 150), not by preference, and the outcome is recorded in
`docs/RUNLOG.md`. If neither passes the chosen deterministic tolerance, the fix is to add
Python/NumPy/CPU/CUDA RNG state to the checkpoint, as spec section 9 anticipates — not to
change the launch.

## The profiling gate

`scripts/profile.sh` runs the spec section 11 gate: 500 steps for Lure-Star GPT and NextLat
(first 100 discarded as warmup, final 400 summarized) and 300 steps for HMM GPT and NextLat
(first 60 discarded, the same 20% proportion). It uses the real confirmatory configs and
overrides only `trainer.train_batches`, `trainer.val_interval`, `trainer.test_interval` and
`trainer.save_recovery_checkpoint` — the last three so that a short run still contains a
validation and several checkpoint writes to measure — plus a profile-only output root that
cannot collide with a confirmatory run.

It records median and p95 seconds per step, examples and tokens per second, peak allocated and
reserved VRAM, GPU utilization and host-input wait, checkpoint-write duration and bytes, the
GPT-vs-NextLat throughput and memory overhead, validation accuracy, and the projected
end-to-end runtime with checkpoint overhead and a 20% interruption margin.

Peak VRAM, host-input wait and checkpoint write timing are only observable **inside the
training process**, so `fabric run` launches `scripts/profile_entry.py`, which installs three
read-only probes and then executes the pinned `train.py` unchanged via `runpy`. `docs/RUNLOG.md`
records why: the first profiling attempt read `torch.cuda.max_memory_allocated()` in the driver
process that shelled out to `fabric run`, and silently reported 0.00 GB. A missing probe is
reported as missing rather than as zero, **and it fails the gate**: `REQUIRED_MEASUREMENTS` in
`scripts/profile_summarize.py` is spec section 11's "Record:" list, every job's
`missing_required` field names the quantities it could not produce, and `main()` exits non-zero
listing them. Before the adversarial review (`docs/review/configs-and-launch.md`, finding 3) a
complete four-job gate whose probes never landed printed `-` in every VRAM row and exited 0 —
which is the RUNLOG bug with a different presentation. GPU utilization is deliberately not in
that list: `nvidia-smi` may legitimately be absent on a runtime, and `profile.sh` says so on
stderr. A half-run gate (a skipped task) also exits non-zero.

## Regenerating and re-verifying

```bash
.venv/bin/python scripts/materialize_configs.py          # rewrite configs/ + overrides.json
.venv/bin/python scripts/materialize_configs.py --check  # fail if on-disk differs
.venv/bin/python -m pytest tests/test_configs.py tests/test_profile_tooling.py -q
```
