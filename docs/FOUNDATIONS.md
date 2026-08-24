# FOUNDATIONS — the single brief every later agent reads

**Project:** NextLat × Predictive Geometry (Lure-Star + HMM)
**Sole specification:** `/Users/stevenyang/Documents/nextlat_forgetting/nextlat_v4_predictive_geometry_spec.md`
**Pinned upstream:** `/Users/stevenyang/Documents/nextlat_forgetting/upstream/NextLat` @ `3770be6009cea2b3c455a9ce7f2ca88b504bb955` ("Initial public release", Mon 25 May 2026 21:50:04 -0700)
**Paper:** arXiv:2511.05963**v4** [cs.LG], last revised 15 Jun 2026. **The pinned repo is v3-era code, three weeks older than v4.** Never assume the repo contains anything v4 added.
**Scout reports this brief supersedes for day-to-day use (but not for detail):** `docs/UPSTREAM_REPORT.md`, `docs/PAPER_NOTES.md`, `docs/STYLE_GUIDE.md`, `docs/COLAB_TRANSPORT.md`.
**Written:** 2026-08-23.

Rules of engagement for every agent downstream:

- Cite `file:line` against the pinned tree for any claim about upstream. Never invent an API.
- The spec is authoritative on *science*. The repo is authoritative on *code*. Where they disagree, §2 below records the decision, and §2 is the thing you follow.
- Do not push code, rent compute, or exceed the budget envelope in §4 without explicit approval.
- Three rows in §2 are `BLOCKED-NEEDS-DECISION`. Two of them must be resolved **before the first confirmatory GPU-hour is spent**, not after.

---

## 1. Ground truth — what the pinned repo actually provides

Direct answers to spec §16's five questions. Everything here was re-verified against the checkout, not copied on faith.

### 1.1 The Path-Star generator and the G(5,5) config

**Generator:** a single file, `data/stargraph/prepare.py`. Argparse block at `prepare.py:115-144`.

```bash
python data/stargraph/prepare.py --num_samples 200000 --num_test_samples 20000 \
  --num_paths 5 --path_length 5 --max_nodes 100 \
  --data_dir data/stargraph --generate_test_data
```

- `--max_nodes` defaults to **50** (`prepare.py:133`); the repo's own documented command uses 100 (`data/README.md:51-55`) and the shipped configs assert `stargraph_max_nodes: 100` (`config/stargraph/5_5/gpt_stargraph_5_5.yaml:28`). **The spec is right: the override is mandatory.**
- Node ids are `range(maxNodes)` = **`0…99`**, not `1…100` (`prepare.py:11`). Record `0..99` in the manifest.
- **There is no `--seed` flag.** Determinism is hard-wired: `seed = 0` at `prepare.py:52`, then `random.seed(seed); seed += 1` per sample (`prepare.py:61-62`, `prepare.py:90-91`). The test set's seeds *continue the same counter*, so the held-out 20k set is only reproducible if `--num_samples` is also identical. Train/test disjointness is by seed, never checked by content.
- Graph shape (`prepare.py:8-36`): the goal arm is always the arm generated first (`p == 0`, goal set at `prepare.py:28`), but `random.shuffle(edgeList)` at `prepare.py:31` destroys arm order. G(5,5) = 1 source + 5 arms × 4 nodes = **21 distinct nodes, 20 edges, answer path length 5**. Nodes are drawn without replacement, so **a suffix swap preserves the node multiset by construction** — which is exactly what the spec's matching requirements need.

**Serialized format** (`prepare.py:70-73`, identical at `:99-102`):

```
<e1>|<e2>|…|<e20>/<source>,<goal>=<p0>,<p1>,<p2>,<p3>,<p4>\n     where <e> ::= <int>,<int>
```

Verbatim example at `--max_nodes 100`:

```
49,97|65,62|36,85|51,38|61,45|49,12|64,17|5,33|12,79|49,64|62,51|45,74|49,61|74,27|17,36|32,68|97,53|79,32|49,65|53,5/49,33=49,97,53,5,33
```

**Tokenizer** (`data/stargraph.py:9-57`), hand-written, not HuggingFace. With `maxNodes=100`: nodes `0…99`, `|`=100, `=`=101, `/`=102, `$`=103, EOS=104, `vocab_size = 106` (`data/stargraph.py:233`) with **id 105 an unused slack row**. `encode()` drops every `,` (`data/stargraph.py:35-37`). One EOS is appended per line (`data/stargraph.py:51-57`).

**Resolved sequence geometry for G(5,5):** `T = block_size = 69`; `=` sits at **index 62**; `graph_description_len = context_length = 62`; `num_target_tokens = 5`.

**How the config consumes it.** `StarGraphDataModule.__init__` (`data/stargraph.py:165-235`) **parses the filename**: `num_arms = int(path.split("_")[1])` and asserts `num_target_tokens == int(path.split("_")[2])` (`data/stargraph.py:187-190`). **Every lure/eval file we write must keep the `graph_5_5_…` naming convention or this assert fires.** `_measure_index` (`data/stargraph.py:237-247`) reads **only `data[0]`** and assumes every line is the same length — a ragged file silently mis-slices every batch. `update_config` (`data/stargraph.py:249-252`) overwrites `model.vocab_size=106`, `model.context_length=62`, `model.block_size=69` at runtime, before `initialize_model` (`train.py:181` then `train.py:226`); the resolved values land in `materialized_config.yaml` (`train.py:192-194`), which is the file to archive.

**Prompt-loss masking** (`model_gpt.py:362-370`, `model_nextlat.py:441-453`): loss is computed on 6 positions per example — the source copy, four path continuations, and EOS.

**Accuracy metric** (`data/stargraph.py:77-162`): `prefix = batch[:, :-6]` → 63 tokens ending exactly on `=`; the logged metric is `val_(5, 5)/test_accuracy` (exact-path) plus `val_(5, 5)/token_{1..5}`. `token_1` is the trivial source copy and will sit at ~100%; **`token_2` is the first-branch accuracy the spec's H2 cares about.** Generation uses `torch.multinomial` (`model_gpt.py:555-557`, `model_nextlat.py:734-736`) with `temperature=1.0, top_k=None` from `getattr` defaults (`data/stargraph.py:126-127`) — **the 90% competence gate is a sampled metric, not greedy.**

### 1.2 The final post-normalization hidden state

| | GPT | NextLat |
|---|---|---|
| produced at | `model_gpt.py:276` `x = self.transformer.norm(x)` | `model_nextlat.py:197` `text_embd = self.transformer.norm(x)` |
| `return_hidden_states=True` returns | `(output, x)` — `model_gpt.py:290-291` | `(token_embeds, text_embd)` — `model_nextlat.py:199-200` |
| hidden state tuple index | **1** | **1** |
| tuple index 0 | logits (targets=None) or loss | **token embeddings — no logits at all** |
| early-returns before `lm_head`? | no | **yes** |

So `logits, h = f(...)` cannot be written once for both. For NextLat, apply the head by hand: `logits = model.model.lm_head(h)` (`model_nextlat.py:121`).

`transformer.norm` is `LayerNorm(n_embd, bias=config.bias)` with `bias: false`, and `LayerNorm.forward` dispatches to **`F.rms_norm`** when bias is None (`model_base.py:823-830`). **The "final post-normalization state" is RMS-normalized, not mean-centered.** Fine for the spec's centered-cosine primary (centering happens across the item pool), but say "RMSNorm" in the writeup.

**Preferred extraction: no hook at all.** Call the inner transformer with `return_hidden_states=True`. If a hook is genuinely needed, `model.model.transformer.norm.register_forward_hook(...)` works for both — but **read-only**: NextLat's backward is a manual two-stage graph split (`model_nextlat.py:503-525`) that injects gradients via `fabric.backward(combined_emb, gradient=combined_grad)`, and any in-place mutation of the hidden state during training silently corrupts it. For the §7 causal stretch goal, the penultimate state is the output of `transformer.blocks[10]` (`n_layer=12`).

**⚠️ The `=` position predicts the source, not the branch.** Index 62's target is `path[0]` = the source node, already visible in the prompt's `/source,goal` field — a trivial copy that is *identical across every condition in the quartet*. The first branch decision is at index **63**. See §2 row D-11 for the decision.

### 1.3 Checkpoint / resume path and pointer locations

**Payload is exhaustively:**

```python
{"model": ..., "optimizer": ..., "training_steps": int, "lr_scheduler_state": ...}
```

`model_base.py:404-417`; `_get_checkpoint_state` identical for both models (`model_gpt.py:426-434`, `model_nextlat.py:577-585`); `lr_scheduler_state` injected by the trainer at `core_train.py:936-939`, `:962-965`.

**Absent: Python/NumPy/CPU/CUDA RNG state, dataloader position, epoch, config, `best_val_loss`.** The spec's §9 contingency ("if trajectories diverge, add RNG state") should be planned as the **expected** path, not a fallback.

**Layout:**

```
{trainer.out_dir}/
├── latest_ckpt          ← plain-text pointer, ONE line, core_train.py:945
├── recovery_ckpt        ← plain-text pointer, ONE line, core_train.py:971
└── {resolved experiment_name}/
    ├── ckpt_iter_{step}_{val_loss:.4f}.pt      core_train.py:774-777
    ├── recovery_ckpt_iter_{step}.pt            core_train.py:961
    ├── materialized_config.yaml                train.py:192-194
    └── version_N/metrics.csv                   train.py:103-109
```

**The pointers live one directory ABOVE the experiment directory.** The shipped configs share `out_dir: output/stargraph` across all five algorithms and all five sweep seeds, so the shipped sweep leaves one `latest_ckpt` pointing at whichever run wrote last. **One `out_dir` per job is mandatory** — this is exactly spec §9's "must never cross branches", and `core_train.py:141-142` is why. The pointer stores the path *as written*, so a relative `out_dir` only resumes from the same CWD: **use absolute paths on Colab.**

**`init_from: resume` resolution** (`core_train.py:139-172`): `recovery_ckpt` wins over `latest_ckpt` **unconditionally, with no step comparison**. If neither pointer exists, upstream **silently initializes from scratch** (`core_train.py:165-168`) — a botched restore looks like a fresh run, not an error. This is the single most dangerous failure mode in the project. Conversely, `_save_recovery_checkpoint` deletes the previous recovery file (`core_train.py:976-979`) but never clears the pointer at end of training, so **a completed run leaves a stale `recovery_ckpt` aimed at a deleted file and the next resume hard-fails at `core_train.py:148-150`.** Our runner must rewrite or delete `recovery_ckpt` when a job reaches DONE.

**Durability gaps, all confirmed:** non-atomic `.pt` write (`model_base.py:417`) and non-atomic pointer write (`core_train.py:944-948`, `:970-974`); only one recovery checkpoint kept and the old one deleted before the new one is verified (`core_train.py:976-982`); `self.recovery_checkpoint_path` is in-memory only (`core_train.py:334`) so post-resume the pre-crash file leaks; `os.remove` unguarded (`core_train.py:979`); `_read_scheduler_state_from_checkpoint` does a second full `torch.load` and swallows every exception (`model_base.py:440-456`), so a truncated checkpoint resumes **silently with no scheduler state**.

**Data position is replayed, not restored** (`core_train.py:432-452`): exactly `self.step` batches are materialized, collated, and discarded. A resume at step 19,000 re-tokenizes ~9.7 M lines with `num_workers: 0`. Budget minutes and measure it at the profiling gate. Whether the replayed order *matches* depends on the sampler: plain `--devices 1` keeps a `RandomSampler` driven by the global torch RNG (which diverges on resume, because `init_module(empty_init=True)` at `core_train.py:171` skips the weight-init draws), while `--strategy ddp --devices 1` substitutes a `DistributedSampler` whose permutation is a fixed function of its seed — and `set_epoch` is **never called anywhere in `core_train.py`**, so the order is identical every epoch and across resumes.

**Off-by-one:** `if self.step > self.config.trainer.train_batches: return` (`core_train.py:569`) → **20,001** optimizer steps for `train_batches: 20000`. Harmless; record it.

### 1.4 The single-GPU training command

Shipped script is 2-GPU (`scripts/stargraph/5_5/train_gpt_star_5_5.sh:3`). `README.md:94-104` documents both supported launch forms. Ours:

```bash
fabric run --strategy ddp --devices 1 --precision bf16-mixed train.py \
  --config configs/nextlat_lurestar.yaml \
  trainer.log_to_wandb=false trainer.compile=false
```

`--strategy ddp` is a deliberate addition to the spec's `--devices 1` form: it buys the deterministic `DistributedSampler` order described above, and matches the paper's own sampler regime. CLI overrides are parsed as an OmegaConf dotlist (`train.py:265,349`).

Use `--precision 16-mixed` only on hardware without stable BF16 (T4/Turing has none). **That is a precision deviation and must not be on the confirmatory path.**

**`wandb` is imported unconditionally** at `train.py:15,17,24` and `core_train.py:9` — the package must be pip-installed even for a fully offline run. `log_to_wandb` defaults `true` (`defaults.yaml:34`) and none of the 5_5 configs override it, so the naive shipped config *will* try to `wandb.init`. CSV logging survives (`log_to_file: true`, `defaults.yaml:33`).

`README.md:40`: **PyTorch ≥ 2.6 required.** `train.py:28-32` imports every datamodule at module level, so `datasets`, `transformers`, `networkx` etc. must be importable even for a stargraph-only run.

### 1.5 Resolved model sizes (arithmetic re-verified)

`n_embd=384`, `n_layer=12`, `n_head=6`, MLP hidden `128·round((8·384/3)/128) = 1024`, SwiGLU gate+up `384→2048`, `vocab_size=106`, `block_size=69`, RMSNorm, `bias=false`.

| | total params | non-embedding |
|---|---|---|
| GPT | **21,324,672** | 21,283,968 |
| NextLat, `proj_factor=0.5` (dynamics hidden **384**) | **21,915,264** | 21,874,560 |
| NextLat, `proj_factor=1.0` (the silent fallback, hidden 768) | 22,800,000 | 22,759,296 |

Dynamics MLP at 0.5 is `Linear(768→384) → GELU → Linear(384→384) → GELU → Linear(384→384)` plus `LayerNorm(768)` = **590,592 params — three Linear layers, hidden width 384**, exactly the paper's Table 5 Path-Star row. At `proj_factor=1.0` it is 1,475,328 params, **+884,736**. `hidden_dim = 128 * round(proj_factor * 768 / 128)` at `model_nextlat.py:52-53`; default `proj_factor: 1.0` at `defaults.yaml:118` and `model_nextlat.py:44`.

**Param count at step 0 is the cheapest possible integrity check in the whole project. Assert it.**

---

## 2. Spec-vs-repo deviation ledger

Verdicts: **ADOPT REPO** = the spec is wrong or imprecise, follow the code. **OVERRIDE PER SPEC** = the repo is wrong or unsafe, follow the spec and patch/override. **BLOCKED-NEEDS-DECISION** = a human must sign off before the affected work starts.

| # | Spec says | Repo reality (`file:line`) | Decision | One-line justification |
|---|---|---|---|---|
| D-01 | `data.train_graphs: 200000` | **Key does not exist.** Zero hits repo-wide. Real: `data.stargraph_train_data_path` (`config/stargraph/5_5/gpt_stargraph_5_5.yaml:26`); the count lives in the **filename** and comes from `prepare.py --num_samples`. | ADOPT REPO | Spec invention. Writing it into a YAML is a silent no-op; the training-set size would be whatever the file contains. |
| D-02 | `data.heldout_graphs: 20000` | **Key does not exist.** Real: `data.stargraph_test_data_path` (`gpt_stargraph_5_5.yaml:27`), from `--num_test_samples`. | ADOPT REPO | Same. Also note `data/stargraph.py:187-190` parses `d`,`l` out of the filename and asserts — **lure files must keep `graph_5_5_…` naming**. |
| D-03 | `optimizer.name: AdamW` | `optimizer.optimizer_type: adam` (`gpt_stargraph_5_5.yaml:44`), which dispatches to `torch.optim.AdamW` (`model_base.py:70-77`, `:233`). | ADOPT REPO | Spec's intent is met; the key name is wrong and would be silently ignored by OmegaConf's non-struct merge. |
| D-04 | `optimizer.betas: [0.9, 0.95]` | Two scalars `beta1: 0.9`, `beta2: 0.95` (`gpt_stargraph_5_5.yaml:47-48`), combined at `core_train.py:227`. | ADOPT REPO | Same class of error. A `betas` list is accepted by the merge and read by nobody. |
| D-05 | `optimizer.schedule: constant` | Wrong section: `lr_scheduler.schedule` (`defaults.yaml:132`), read at `core_train.py:985`. Default already `constant` → `lambda _: 1.0`. | ADOPT REPO | Right value, wrong address. Under the correct key nothing changes. |
| D-06 | `optimizer.clip_gradient_norm: 100` | `optimizer.grad_clip: 100` (`gpt_stargraph_5_5.yaml:49`), used at `core_train.py:505-506`. | ADOPT REPO | Under the spec's name, grad clipping silently falls back to `defaults.yaml:128` `grad_clip: 1.0` — a **100× tighter clip than the paper**, which would change training. Highest-consequence of the four optimizer typos. |
| D-07 | `model.proj_factor: 0.5` under `model:` | Present **only inside the `sweep:` block** (`config/stargraph/5_5/nextlat_stargraph_5_5.yaml:61`). `model:` has no `proj_factor`; `defaults.yaml:118` = `1.0`. | OVERRIDE PER SPEC | **Highest-risk footgun in the repo.** Deleting the sweep to run one seed silently gives dynamics hidden 768 and +884,736 params — no longer the paper's config. Hoist `proj_factor: 0.5` into `model:` and assert total params == 21,915,264. |
| D-08 | `trainer.compile: false` | `compile: true` in all five shipped 5_5 YAMLs (`gpt_…:17`, `nextlat_…:16`), contradicting the repo's own `README.md:117-122` ("inconsistent results on numerically sensitive benchmarks like Path-Star and A₅ … we recommend `compile: false`"). | OVERRIDE PER SPEC | Spec is right, the shipped config is the bug. `compile: true` also inserts an `_orig_mod` level into module paths and breaks hidden-state hooks. |
| D-09 | `trainer.save_recovery_checkpoint: 250` | Key exists (`defaults.yaml:24`) but defaults to **`-1` (disabled)**; the 5_5 YAMLs never set it. Gate at `core_train.py:571-578`. | OVERRIDE PER SPEC | Spec is right that the key exists; it is simply off. Non-negotiable for any Colab run. |
| D-10 | `trainer.save_last_checkpoint` / `save_best_checkpoint` | Both exist and default `true` (`defaults.yaml:18,20`). | ADOPT REPO | Spec is correct; nothing to change. |
| D-11 | H1: "extract the state at the final prompt delimiter `=`" | Index 62. **Its prediction target is `path[0]` = the source node**, which is already in the prompt at `/source,goal` — a trivial copy identical across the whole quartet. The first branch decision is index **63** (`data/stargraph.py:118-119` confirms the eval prefix ends on `=`). | OVERRIDE PER SPEC **+ augment** | Keep `h[62]` as the preregistered primary exactly as written. **Additionally preregister `h[63]` now**, before any model exists, with its own decision rule, and compute all H2/H3 correct-branch margins from the logits at index 63. Reporting only `h[62]` risks measuring a state whose immediate target does not vary by condition. |
| D-12 | "node IDs sampled from `1...100`" | `nodes = list(range(maxNodes))` → **`0…99`** (`prepare.py:11`). | ADOPT REPO | Cosmetic, but the immutable manifest must record `0..99` or the generator acceptance tests will disagree with the data. |
| D-13 | `--max_nodes` must be overridden to 100 | Confirmed: default 50 (`prepare.py:133`); `data/README.md:51-55` and `gpt_stargraph_5_5.yaml:28` both use 100. | OVERRIDE PER SPEC | Spec is right and this is the one generator flag whose omission silently produces a different task. |
| D-14 | `sweep: - seed: [1234, 1235, 1236]` | Mechanism exists (`nextlat_stargraph_5_5.yaml:55-61`, expanded at `train.py:273-339`) but `all_sweep_param_names` is a **`set`** (`train.py:280`) iterated to build the directory name (`train.py:322`), so the experiment directory name can differ between processes unless `PYTHONHASHSEED` is pinned; and all sweep entries share one `out_dir`, hence one pointer pair. | OVERRIDE PER SPEC | **Do not use `sweep:` for confirmatory runs.** Emit one materialized YAML per (model, seed, phase, branch) with explicit `seed`, `experiment_name`, `out_dir`. This is what spec §9 already demands. |
| D-15 | `data.dataset: hmm_belief` (§12) | **No such datamodule.** `DATAMODULES` (`train.py:34-42`) = tinystories, stargraph, fineweb10B, fineweb100B, finewebedu, countdown, manhattan; membership asserted at `train.py:176-178`. | OVERRIDE PER SPEC | Experiment B requires writing a new datamodule and registering it. Required interface, read off `core_train.py:336-358` and `train.py:181-182`: `update_config(config)`, `get_tokenizer()`, `train_dataloader()`, `val_dataloader()`, optional `prepare_batch`. `generalization_dataloader()` is only called for stargraph/countdown/manhattan, so the length-64 eval must be driven by our own evaluator, not the trainer. |
| D-16 | `data.train_sequences` / `sequence_length` (§12) | Do not exist. | OVERRIDE PER SPEC | New keys owned by the new datamodule; harmless because OmegaConf merge is non-struct — but only *our* code will read them. |
| D-17 | §12 HMM YAML block as written | **`trainer.test_interval` is read unconditionally at `core_train.py:671` on every validation and is NOT in `defaults.yaml`.** Same for `trainer.test_batches` and `trainer.experiment_name` (`train.py:95`). | OVERRIDE PER SPEC | The spec's HMM YAML omits `test_interval`. As written, **both HMM runs crash at the first validation (step 300)** with a missing-key error. Every config we emit — Lure-Star and HMM — must carry `experiment_name`, `test_interval`, `test_batches`, `val_batches`. |
| D-18 | §9: deterministic job ids like `nextlat-s1234-base`, and a runner that predicts checkpoint paths | `train.py:98-99`: `if "seed" not in experiment_name: experiment_name += f"-seed{config.seed}"`, then `config.trainer.experiment_name` is **overwritten** at `train.py:125` and used to build every checkpoint path (`core_train.py:933,959`). | ADOPT REPO | `nextlat-s1234-base` does not contain the substring `seed`, so the on-disk directory is `nextlat-s1234-base-seed1234`. **The runner must predict `{out_dir}/{experiment_name}-seed{seed}/`**, or embed `seed` in the job id, or it will hash the wrong path and mark a good job FAILED. |
| D-19 | §6/§8: H3 adaptation = 500 updates from a frozen parent | `--checkpoint_path` (`train.py:262-264`, precedence at `core_train.py:130`) restores `training_steps`, and `core_train.py:309` sets `self.step = self.model.training_steps`. `core_train.py:569` returns when `step > train_batches`. | OVERRIDE PER SPEC | Branching a 20,000-step parent starts the adaptation run at step **20,001**. With `train_batches: 500` the loop **returns immediately and the job looks like a clean completion with zero updates.** Set `train_batches: 20500`, or reset `training_steps` before the trainer is constructed. Record which. |
| D-20 | §8: "If either model fails the base competence gate — initially 90% exact-path accuracy — debug data, configuration, numerical precision, and repository parity" | The paper's own Fig. 6 puts **GPT on G(5,5) at ≈18.6%**, i.e. 1/d chance, versus NextLat ≈99.8% (digitized, ±0.5 pp, `docs/PAPER_NOTES.md`). GPT failing Path-Star **is the paper's headline result**, not a bug. | **BLOCKED-NEEDS-DECISION** | Read literally, spec §10's stop condition ("either model remains below the base competence gate after a reasonable step increase") halts the whole project on a *correct* GPT run. It also hollows out H2/H3: a model at chance has a degenerate correct-branch margin, so "margin erosion" is measured on a capability it never had. **Resolve before launch.** Recommended resolution in §6/R1. |
| D-21 | §8: `effective_batch_size: 512` | `512` in the shipped YAML (`gpt_stargraph_5_5.yaml:22`), but `device_batch_size = effective // world_size` (`train.py:143-145`) → **512/GPU on one device, 2× the shipped 2-GPU script's 256**. Accumulation slices an already-loaded batch (`core_train.py:486-499`) and preserves the optimizer-update count. | ADOPT REPO **+ profile** | Do not change 512. This is precisely what the §11 profiling gate exists to test. If it does not fit, `gradient_accum_steps: 2` is a legitimate execution fallback that preserves effective batch and update count — document it as a deviation. |
| D-22 | §10: 90% exact-path competence gate | `evaluate_stargraph` generates with `torch.multinomial` (`model_gpt.py:555-557`, `model_nextlat.py:734-736`), `temperature=1.0`, `top_k=None` from `getattr` defaults (`data/stargraph.py:126-127`). | ADOPT REPO **+ report both** | The gate is a *sampled* metric and will read lower than greedy. Add `trainer.top_k: 1` to make the in-training metric greedy without touching code, and report sampled and greedy side by side. State which one the gate is evaluated on **before** the first run. |
| D-23 | §9: "verify … data position" in the forced-interruption test | Data position is not checkpointed; it is replayed (`core_train.py:432-452`), and with plain `--devices 1` the `RandomSampler` order diverges post-resume. `set_epoch` is never called, so `--strategy ddp --devices 1` gives a fixed `DistributedSampler` order. | OVERRIDE PER SPEC | Run confirmatory jobs as `fabric run --strategy ddp --devices 1`. This is an addition to the spec's launch line, adopted because it makes the mandatory recovery test passable without patching the trainer. **Verify empirically with the 300 vs 150+150 test before trusting it**; if it fails, add RNG state to the checkpoint as spec §9 already anticipates. |
| D-24 | §9: atomic checkpoints, two verified recovery files | Upstream writes the `.pt` and both pointers non-atomically (`model_base.py:417`, `core_train.py:944-948`, `:970-974`) and deletes the previous recovery checkpoint unverified (`core_train.py:976-979`). | OVERRIDE PER SPEC | Spec §9.2 items 2–4 are real fixes for real bugs, not belt-and-braces. Implement as a **minimal patch on the pinned tree** and archive the diff in `source_snapshot/`. |
| D-25 | §9: `init_from: resume` is the recovery path | If neither pointer exists, `core_train.py:165-168` **silently initializes from scratch**. A completed run leaves a stale `recovery_ckpt` → next resume hard-fails at `core_train.py:148-150`. | OVERRIDE PER SPEC | The runner must (a) verify the pointer and its target hash **before** passing `init_from=resume`, and (b) clear `recovery_ckpt` when a job reaches DONE. Never treat "training started" as "resume succeeded". |
| D-26 | §8/§11: bit-comparable reruns, "chosen deterministic tolerance" | `train.py:171-172` force-enables TF32 unconditionally; `trainer.deterministic` (`defaults.yaml:38`) is **never read anywhere** — dead key. `fabric.seed_everything(seed + global_rank)` is called once, at `train.py:170`, and nothing re-seeds on resume. | ADOPT REPO | Bit-exactness is not available out of the box. The spec's "deterministic tolerance" framing is correct; pick and record the tolerance rather than expecting exactness. Do not set `trainer.deterministic` and believe it did something. |
| D-27 | §8: "copy the official Path-Star G(5,5) YAML; permissible changes are limited to seeds, paths, checkpoint frequency, hooks" | `data.stargraph_data_path` (`defaults.yaml:80`) is dead; `epochs` and `pair_accum_steps` in the 5_5 YAMLs are never read for GPT/NextLat; `test_interval`/`test_batches`/`val_printsamples` are in the 5_5 YAMLs but absent from `defaults.yaml`. | ADOPT REPO | Copy the 5_5 YAML wholesale (including the dead keys) rather than reconstructing it, exactly as the spec says — the dead keys are inert and the live ones you would forget are not. |
| D-28 | §12: NextLat HMM with `lambda_kl: 0.0` | `lambda_ce` defaults to `0.0` (`defaults.yaml:116`) but the CE term is **computed regardless, for logging** (`model_nextlat.py:488-493`), costing throughput; `README.md:128` says so. `lambda_mse`/`lambda_kl` feed `NextLatConfig` at `core_train.py:91-92` and multiply the loss at `model_nextlat.py:490-494`, so `0.0` cleanly disables a term. | ADOPT REPO | Setting `lambda_mse=0, lambda_kl=0` for the H3 adaptation branch is a valid, clean reduction of NextLat to next-token-only, as spec §6 requires. Just know the KL/CE forward work is still paid. |
| D-29 | §2 / §12: `lambda_mse` is an MSE | The repo's `lambda_mse` is the paper's `λ_next-h`, and the loss is **Smooth L1** (paper Eq. 3; the repo detaches the target at `model_nextlat.py:303`). The metric logged as `mse_loss` is a Huber loss. | ADOPT REPO | Naming only, but the writeup must say Smooth L1. Appendix E of the paper lists "replacing Smooth L1 with MSE" as an *unsuccessful* fix — do not silently swap it. |
| D-30 | §5: "200,000 fixed training graphs and 20,000 held-out tests" reproducible | No `--seed` flag; seeds are `0…num_samples-1` for train and **continue from `--num_samples`** for test (`prepare.py:52,61-62,90-91`). | ADOPT REPO | The held-out set is only reproducible if `--num_samples` is byte-identical. Pin the full command in the manifest, and hash both files. Disjointness is by seed, never content-checked — add our own content-level disjointness assertion for `E_lure`. |
| D-31 | §11: "Prefer an A100 40/80 GB, L40S, H100, A6000" | `colab` CLI v0.2.0 accepts `--gpu t4\|l4\|a100` only; H100 appears in `eligible_gpus` but is not a selectable value (`docs/COLAB_TRANSPORT.md` §2). | ADOPT REPO | A100 it is. See §4. One budgeted H100 probe is allowed; do not plan around it. |
| D-32 | §9: "MyDrive/lurestar/" durable layout | Transport reality: `colab upload`/`download` are base64-over-Jupyter-kernel, not file transfers (`docs/COLAB_TRANSPORT.md` §2). A ~256 MB checkpoint cannot go through them. | OVERRIDE PER SPEC | Keep the spec's *directory schema* but root it at `gs://nextlat-lurestar-project-flash-490419/lurestar/`, with all runtime GCS I/O through the Python client (`GOOGLE_APPLICATION_CREDENTIALS=/content/adc.json`); **`gcloud storage` does not authenticate inside Colab** and is banned in the runtime. |
| D-33 | §5: "five seeds" (paper config) vs §8: "three preregistered confirmatory seeds" | Repo sweep ships `[1234,1235,1236,1237,1238]` (`nextlat_stargraph_5_5.yaml:56`). | ADOPT REPO **for the values**, spec for the count | Confirmatory = `1234, 1235, 1236` (the first three). The reduction from five to three is a documented deviation from the paper and must be named as such in the writeup, together with what three seeds could not have detected. |
| D-34 | §7: causal patch of the penultimate-layer state | Penultimate = output of `transformer.blocks[10]` for `n_layer=12`. Patching `transformer.norm`'s output is what the spec explicitly forbids, because `lm_head` consumes it directly (`model_gpt.py:280`). | ADOPT REPO | The spec's warning maps onto a concrete module index. Stretch goal only; drop it before dropping H3 or the HMM. |
| D-35 | §9: `E_lure` and `A_pair` must never enter training | Nothing upstream enforces this; `_measure_index` reads only line 0 and the datamodule has no dedup (`data/stargraph.py:237-247`). | OVERRIDE PER SPEC | Our own hashed-manifest disjointness check is the only guard. It belongs in `src/lurestar/validate.py` and in the acceptance tests, and must run **before** every training launch, not once. |
| D-36 | §9 security: credentials on the runtime | `/content/adc.json` would be a live `authorized_user` refresh token for `<redacted-account>` with full `cloud-platform` scope. | **BLOCKED-NEEDS-DECISION** | Recommend a dedicated service account with `roles/storage.objectAdmin` on `nextlat-lurestar-project-flash-490419` only. Blast radius one bucket, independently revocable, and it works with the identical `GOOGLE_APPLICATION_CREDENTIALS` pattern. The user-ADC route is the working fallback, not the plan. Needs a yes/no before the first upload. |
| D-37 | §13: bottleneck-width ablation "directly addresses the paper's stated uncertainty" | Paper §6 already states the *direction*: "Empirically, we observe that using smaller latent dimensions is beneficial on tasks such as Path-Star graph and Countdown." | ADOPT REPO (paper) | Accuracy-vs-width is **not novel**. Only the effect of width on *geometry* (PSI, predictive-equivalence collapse, posterior decodability) is. Frame it that way or drop it. |

Three rows carry a `BLOCKED-NEEDS-DECISION`-class urgency: **D-20** (GPT competence gate — resolve before launch), **D-36** (credential scope — resolve before first upload), and, at lower cost, **D-11** (which hidden-state index is primary — resolve before the first plot, and it is free to resolve now by preregistering both).

---

## 3. Novelty guardrail — what the paper already owns

The paper is arXiv:2511.05963v4. Every quote below is verbatim from `docs/paper_source/v4.txt` (sha256 of the archived HTML: `572bb951092ef764db4838c4af880a3bf9b6bdf80ec395ed0dc9f6678bddcae2`). **If a sentence in our writeup could be mistaken for one of these, it is not a finding.**

**Owned claim 1 — belief-state theory.** Abstract: *"Theoretically, we show that these latents provably converge towards belief states, compressed information about the history necessary to predict the future."* Theorem 3.2 plus the Appendix B backward-induction proof. **Any** claim that the latent is a sufficient statistic is theirs. We may cite it; we may not re-derive it as our own.

**Owned claim 2 — compression and effective rank.** Table 1: sequence compression **0.71 (NextLat) vs 0.65 (GPT)**; effective latent rank **52.7 vs 160.1**. Prose: *"NextLat has the lowest effective latent rank of 52.7—over 3x smaller than GPT's."* **Rank-only or compression-only results are established.** Note that their "Sequence Compression" is a *behavioral output-identity* metric ("Percentage of cases where the model produces identical continuations when prompted with two different traversals arriving at the same state"), measured on generated continuations, not on hidden-state geometry — which is why our HMM-H1 is not the same measurement, and we must say so explicitly rather than let a reader assume it.

**Owned claim 3 — Path-Star solving.** Fig. 6 (digitized, ±0.5 pp): G(5,5) NextLat ≈99.8, BST ≈99.9, JTP ≈47.3, MTP ≈21.2, **GPT ≈18.6**; G(7,7) NextLat 94.3 vs BST 9.7. Prose: *"NextLat maintains close to 100% solve rate for all topologies of the Path-Star graphs."* **Reproducing Path-Star accuracy is replication, not contribution** — and it is also our competence gate, so say "we replicate" in the same breath as reporting it.

**Owned claim 4 — Manhattan world-model coherence.** Valid trajectories 98.7% vs 97.0%; detour robustness 95.0% vs 85.0%.

**Owned claim 5 — Countdown planning.** NextLat 54.8/57.6/58.7 at d=1/4/8 vs GPT 33.1, BST 42.3; *"substantially surpasses MTP and JTP trained with the same horizon (>35.7% improvement)."*

**Owned claim 6 — future-token decodability.** *"NextLat matches GPT's next-token performance across both $d\in\{1,8\}$ and exhibits the strongest long-range predictive capability (up to 20 tokens ahead)."* **"NextLat's states contain more information about the future" is already shown by linear probing.** Our HMM-H3 posterior probe must be framed as *decoding an exact known Bayesian posterior*, which they could not do, not as "probing shows more future information".

**Owned claim 7 — A5 length generalization / state tracking.** The co-trained latent dynamics RNN generalizes 12→36 tokens at >95% while the transformer does not.

**Owned claim 8 — data efficiency from dense latent supervision.** §3.1 "Better Data Efficiency".

**Owned claim 9 — self-speculative decoding, up to 3.3× speedup.** Explicitly out of scope for us.

**Owned claim 10 — efficiency vs BST** (O(T) vs O(T²) gradients, >3× faster) **and horizon-independence vs JTP's `d ≥ k`.**

**Owned claim 11 — preservation of next-token perplexity relative to MTP/JTP** (Table 2), with the authors' own hedge: *"NextLat ($d=2$) does show a modest gain in average accuracy over GPT (59.21 vs. 58.82), but these improvements are not consistent across tasks."*

**Owned claim 12 — the objective ablations.** Smooth L1 alone strongest at d=1; KL+SmoothL1 best at d=8; stop-gradients on both components best. Our HMM's "Smooth L1 alone at d=1" choice is *supported by* their Appendix D, not discovered by us.

**Owned claim 13 — smaller latent dimensions help Path-Star and Countdown.** §6, verbatim: *"Empirically, we observe that using smaller latent dimensions is beneficial on tasks such as Path-Star graph and Countdown."* See D-37.

**Owned claim 14 — the optimization quirks**, including Appendix E's *"the rise in smooth L1 loss may reflect changes in the scale or geometry of the latent states"* — the only place the paper uses "geometry" of latents at all, and it is an unmeasured conjecture about scale.

**Additionally already claimed as future work by the authors, therefore not ours to propose:** post-hoc finetuning objective for pretrained transformers, RL post-training, hierarchical / higher-dimensional latents, adaptive-length speculative decoding, comparison against newer MTP variants. The spec already excludes all five; that exclusion is now grounded in the authors' declared agenda.

**The one sentence our novelty case rests on**, §6, verbatim:

> "On the analysis side, we do not study the structure of the learned representations under NextLat, leaving open questions about how the method shapes latent spaces."

**Quote that sentence. Do not paraphrase it.**

**What is therefore genuinely open**, and is the only ground we may claim: *which* distinctions the geometry encodes under **matched** future-relevant vs future-irrelevant perturbation; whether that geometry predicts behavior **item by item**; whether it predicts **later interference**; and whether the states respect **exact Bayesian predictive equivalence** when ground truth is available. The paper supplies no pairwise-distance, predictive-equivalence, belief-divergence, posterior-decoding, or forgetting result of any kind.

**Two more guardrails, from the spec and from honesty:**

- **Non-uniqueness.** Definition 3.1 defines sufficiency only up to invertible re-encoding. Nothing in the paper claims the belief state is minimal, unique, or coordinate-aligned to a posterior simplex, and neither may we. Prioritize predictive equivalence, relative divergence, decodability, and future-distribution prediction over literal alignment to the simplex.
- **No neuroscience overreach.** "Pattern separation" is used as a computational abstraction — *similar histories + different futures → greater separation; similar histories + same future → relative invariance*. Do not claim hippocampus, dentate gyrus, place cells, manifold homology, or brain-like circuits. Population-level analysis is a methodological analogy, not homology.

**Related-work links the writeup must make explicitly** (per `docs/STYLE_GUIDE.md`, for Pratyusha Sharma as a reader): intruder dimensions / spectral forgetting (arXiv 2410.21228, 2607.23711) and OP-Mix (arXiv 2605.15220) — while flagging that NextLat's constraint is **representational, not parameter-space**, which is exactly why the interference question is not a re-run of the LoRA-vs-full-FT result.

---

## 4. Compute plan

### 4.1 Hardware decision

**Confirmatory runs: A100** (`colab exec --gpu a100`). It is the only CLI-reachable option that gives native BF16 (spec §8's `bf16-mixed`), removes every memory question at 40 GB, matches the class of hardware the paper used (RTX A5000 / H100 NVL / B200), and fits the weekend on wall-clock.

**T4: smoke tests and the mandatory forced-interruption test only.** Turing has no BF16, so a confirmatory T4 run would force `16-mixed` — a precision deviation on the confirmatory path, which spec §11 forbids ("Never silently reduce model scale to fit a runtime"; the same logic applies to precision). The recovery test is a test of the *transport layer*, not the GPU, so cheap hardware is correct there.

**L4: overflow lane only**, if A100 assignment fails repeatedly. ~2× A100 wall-clock for ~40% of the CU — a bad trade when CU is not the binding constraint.

**H100 is not selectable** at `colab` CLI v0.2.0 (`--gpu` accepts `t4|l4|a100`). One budgeted probe is permitted; do not plan around it.

### 4.2 Rates — one measured, two are placeholders

| GPU | CU/h | Status |
|---|---|---|
| T4 | **1.54** | **MEASURED** (`colab quota --json` with a live T4 assignment) |
| L4 | ~4.82 | **PLACEHOLDER — prior, unverified.** Replace at gate 1. |
| A100 | ~11.77 | **PLACEHOLDER — prior, unverified.** Replace at gate 1. |

Short-session billing floor ≈ **0.17 CU** (measured: a sub-minute accidental T4 cost 0.16875 CU). `quota` and `status` are eventually consistent and have returned a phantom record — **drop detection must require two agreeing polls.**

`colab start --help` **is not parsed as help and provisions a real runtime.** Only `colab -h` is safe. Never append `--help` to a subcommand.

### 4.3 Workload and modelled budget

```
base       = 6 runs  × 20,000 steps = 120,000 optimizer steps   (3 seeds × {GPT, NextLat})
adaptation = 12 branches × 500 steps =   6,000 optimizer steps   (2 models × 3 seeds × {near, far})
HMM        = 6 runs  ×  3,000 steps =  18,000 optimizer steps   (3 seeds × {GPT, NextLat}, small)
                                       ---------
                                        144,000 optimizer steps
```

Workload constants, derived from the pinned configs: T=69, 512×69 = **35,328 tokens/step**, **4.51 × 10¹² FLOP/step** (6N, fwd+bwd, N=21.24 M non-embedding); HMM model 0.79 M non-embedding, 3.87 × 10¹⁰ FLOP/step; checkpoint ≈ **256 MB**.

| GPU | GPT s/step | NextLat s/step | Base h | Adapt h | HMM h | Ckpt h | Total h | **+20% margin** | CU |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0.577 | 0.924 | 25.02 | 1.25 | 0.20 | 0.64 | 27.10 | **32.5** | **50** (measured rate) |
| L4 | 0.310 | 0.496 | 13.44 | 0.67 | 0.20 | 0.64 | 14.95 | **17.9** | ~86 *(placeholder rate)* |
| **A100** | **0.180** | **0.289** | 7.82 | 0.39 | 0.20 | 0.64 | 9.04 | **10.9** | **~128** *(placeholder rate)* |

Every s/step figure above is **modelled, not measured** — MFU assumptions 12%/12%/8%, a 30 ms per-step floor, and a 1.6× NextLat multiplier (plausible band 1.4–1.9×). **The §11 profiling gate exists to replace all of them.** Treat the table as a planning prior with an explicit expiry date, not a result.

### 4.4 Budget envelope and stop-line

Balance at last check: **1788.61 CU**. Free tier is **0** — there is no free fallback; every runtime-second is paid.

- **Expected:** ~130 CU (7.1% of balance).
- **Planning envelope:** **400 CU.** Survives reality being ~3× worse than modelled.
- **HARD STOP-LINE: 600 CU cumulative (~33% of balance).** On crossing it, halt every runtime, write the ledger, and escalate for an explicit compute decision. Do not continue "just to finish this one seed".
- **Per-job stop:** if a single base run exceeds **3× its profiled projection**, kill it and investigate before relaunching.
- **Circuit breaker (transport):** progress is defined **only** by the durable step counter in `gs://…/runs/{run_id}/state.json`, never by `colab exec`'s exit status. Two consecutive fast returns (<120 s) with no durable step advance ⇒ **ABORT, surface the last 200 log lines, do not start another runtime.** Two strikes, not three.

**CU is not the binding constraint. Wall-clock and interruption exposure are.** Optimize for those.

### 4.5 Gates, in order — do not skip one

1. **Calibrate L4 and A100 rates** (~0.5 CU). Replace the two placeholders in §4.2 with measured `burn_rate_hourly` before any real spend.
2. **Probe A100 and confirm BF16:** `colab exec --gpu a100 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"`.
3. **Forced-interruption test on T4** (spec §9): 300 steps clean, then 150 + kill + resume + finish at 300; verify step, optimizer/scheduler state, data position, metrics, and final weights/logits within a recorded tolerance.
4. **Profile on A100** (spec §11): 500 Lure-Star steps (first 100 warmup, summarize the last 400) and 300 HMM steps. Record median and p95 s/step, examples and tokens/s, peak allocated and reserved VRAM, GPU utilization and host-input wait, checkpoint write duration and bytes, GPT-vs-NextLat overhead, and the projected end-to-end runtime. Substitute measured s/step into §4.3 and recompute. **If measured step time exceeds the modelled A100 figure by more than ~2.5×, stop and investigate the dataloader first** (`data.num_workers: 0` in both stargraph configs is the obvious suspect for a 35k-token step).
5. **Only then launch the sweep**, one seed per job, one `out_dir` per job, recovery loop and circuit breaker armed.

Record into `run_ledger.json` for every job: session id, GPU name, zone, `paid_balance` before and after, wall-clock, measured s/step, peak VRAM, checkpoint count and bytes, resume count, and `parent_checkpoint_sha256`.

### 4.6 Transport rules (non-negotiable, from `docs/COLAB_TRANSPORT.md`)

- **`colab exec` does not forward argv** — it runs a Jupyter cell. `train.py:259` makes `--config` required, so `train.py` cannot be what `exec` runs. Pass parameters via an uploaded `/content/job.json` read from a hard-coded absolute path, and launch upstream as a `fabric run` **subprocess** so it gets a real argv. Do not `import train` and poke `sys.argv`.
- **`__file__` is undefined in a cell.** A single `RUNTIME_ROOT = "/content/lurestar"` constant, and a `grep -n '__file__' <script>` gate that must return nothing before every upload.
- **Never capture the child's stdout.** Stream line-by-line (`Popen(..., bufsize=1, text=True)`) with a 30 s heartbeat thread, or the websocket dies during a trailing checkpoint write — exactly when results exist and are not yet durable.
- **`upload`/`download` are base64-over-kernel, not transfers.** Sidecars only. A 256 MB checkpoint goes runtime↔GCS directly.
- **All runtime GCS I/O through the Python client** with `GOOGLE_APPLICATION_CREDENTIALS`. **`gcloud storage` does not authenticate inside Colab** and is banned in the runtime.
- Durable root: `gs://nextlat-lurestar-project-flash-490419/lurestar/` (US-CENTRAL1), with spec §9's `source_snapshot/ manifests/ runs/{model}/{seed}/{phase}/{condition}/ results/ run_ledger.json` schema underneath.

---

## 5. Build order

Dependency-ordered. File names are spec §15's deliverables verbatim. **Each phase has an exit test; do not start the next phase until it passes.**

### P0 — Pin and record (DONE)
`docs/UPSTREAM_REPORT.md`, `docs/PAPER_NOTES.md`, `docs/STYLE_GUIDE.md`, `docs/COLAB_TRANSPORT.md`, this file.
Add `.gitignore` entries for `application_default_credentials.json`, `adc.json`, `*.tgz`, `.colab_session` **before the first commit**.
**Exit:** D-20 and D-36 have written decisions.

### P1 — Lure-Star stimuli (blocks everything else on task A) — *parallel with P2*
`src/lurestar/generate.py`, `src/lurestar/validate.py`, `tests/test_lure_generator.py`, `manifests/stimuli.jsonl`, `manifests/stimuli.sha256`.
Base graphs from the upstream generator command in §1.1 (`--max_nodes 100`), then quartets (base / repeat / near-safe / near-critical) plus the far-critical distance control, all solver-verified.
Constraints that are easy to get wrong: keep `graph_5_5_…` filenames (D-02); node ids `0..99` (D-12); every line the same length or `_measure_index` mis-slices (§1.1); `E_lure` and `A_pair/B_adapt` disjoint from training by **content hash**, not by seed (D-30, D-35).
**Exit:** ≥1,000 quartets pass every spec §5 acceptance test, deterministically, under a recorded seed. **If exact matching fails, stop and fix the generator — do not regress it away later** (spec §10).

### P2 — HMM ground truth — *parallel with P1*
`src/hmm_geometry/generate.py`, `src/hmm_geometry/forward.py`, `tests/test_hmm_forward.py`, `manifests/hmm_matrices.json`.
Preregistered 4-state / 4-observation stationary HMM chosen by a **model-blind** generator test; matrices frozen into the manifest before any training.
**Exit:** the normalized forward algorithm matches brute-force enumeration on short sequences, to machine precision.

### P3 — Configs
`configs/gpt_lurestar.yaml`, `configs/nextlat_lurestar.yaml`.
Copy the shipped 5_5 YAMLs wholesale (D-27), then apply: `compile: false` (D-08); **hoist `proj_factor: 0.5` into `model:`** (D-07); absolute unique `out_dir` per job (§1.3); explicit `experiment_name` and `seed` (D-18); `save_recovery_checkpoint: 250` (D-09); `log_to_wandb: false`; delete the `sweep:` block (D-14); absolute data paths keeping `graph_5_5_…` naming; keep `test_interval`/`test_batches`/`val_batches` present (D-17); `top_k: 1` if the gate is to be greedy (D-22).
**Exit:** a materialized-config diff against the shipped YAML that a reviewer can read in 30 seconds, plus a **step-0 assertion that total params == 21,324,672 (GPT) / 21,915,264 (NextLat)** (§1.5).

### P4 — Durability and the runner (blocks all GPU spend)
`src/lurestar/durable_checkpoint.py`, `scripts/run_matrix.py`, `results/run_ledger.json`, `tests/test_resume.py`.
Minimal patch on the pinned tree for `.partial` + atomic rename, atomic pointer writes, two verified recovery checkpoints, guarded `os.remove` (D-24); runner-side pointer verification before `init_from=resume` and `recovery_ckpt` cleanup on DONE (D-25); path prediction that accounts for the `-seed{seed}` suffix (D-18); the two-strike circuit breaker keyed on GCS `state.json` (§4.4).
**Exit:** the spec §9 forced-interruption test passes on T4 under `--strategy ddp --devices 1` (D-23), with the divergence tolerance recorded. If it fails, add RNG state to the checkpoint and re-run — do not proceed on a "close enough" resume.

### P5 — Profiling gate
`scripts/profile.sh`, `results/compute_log.csv`.
500 Lure-Star steps (100 warmup) and 300 HMM steps on A100, per §4.5 gate 4. Profile the paper's physical batch **first** (D-21).
**Exit:** measured s/step substituted into §4.3, recomputed budget under the 400 CU envelope, and a written go/no-go.

### P6 — HMM datamodule and configs
New `hmm_belief` datamodule registered in `train.py:34-42` (D-15) implementing `update_config` / `get_tokenizer` / `train_dataloader` / `val_dataloader`; `configs/gpt_hmm.yaml`, `configs/nextlat_hmm.yaml` **including `test_interval` or they crash at step 300** (D-17).
**Exit:** 50 steps run clean end-to-end including one validation pass.

### P7 — Base training (6 runs)
3 seeds × {GPT, NextLat}, `1234/1235/1236`, one `out_dir` per job, 20,000 steps.
**Exit:** competence gate evaluated per the D-20 decision; `materialized_config.yaml` archived and hashed for every run; checkpoint lineage recorded.

### P8 — Representations and H1/H2
`src/lurestar/representations.py`, `src/lurestar/evaluate.py`, `results/metrics/`, `results/metrics.jsonl`.
Extract **both** `h[62]` and `h[63]` with the asymmetric API of §1.2 (D-11). Centered cosine primary, whitened Euclidean robustness. Two-fold cross-fitting for H2.
**Exit:** PSI with paired 95% bootstrap intervals, per-seed values plus across-seed spread, and an explicit statement of what magnitude three seeds could not have detected.

### P9 — H3 interference
`configs/adapt_near.yaml`, `configs/adapt_far.yaml`.
Both branches from the identical parent checkpoint, same `parent_checkpoint_sha256`, separate `out_dir` per branch. `lambda_mse=0, lambda_kl=0` for the primary NextLat branch (D-28). **`train_batches: 20500`, not 500** (D-19). `B_far` chosen on a model-blind pilot checkpoint and frozen.
**Exit:** both branches acquire; near/far initial-loss imbalance reported and controlled item-level; near-minus-far computed within each parent, bootstrapped over paired items, every seed reported. **A clean null is an acceptable outcome** — do not force one with an extreme LR or contradictory labels.

### P10 — HMM pair bank and evaluation
`src/hmm_geometry/pair_bank.py`, `src/hmm_geometry/evaluate.py`, `manifests/hmm_eval_pairs.jsonl`, `tests/test_hmm_pairs.py`.
Thresholds frozen from the validation pool and applied unchanged to the test pool, **before any model state is inspected**.
**Exit:** HMM-H1/H2/H3 metrics computed, including length-64 generalization.

### P11 — Figures, tables, README
`results/figures/` (the six required figures), one results table per task, `README.md`.
The README must distinguish paper results from ours, neuroscience inspiration from evidence, and positive findings from nulls (spec §15).
Writeup follows `docs/STYLE_GUIDE.md`: 2,000–3,000 words, hero figure first with a caption that already discloses the caveat, failures in the introduction rather than a limitations section, one visual per ~200 words, spaced hyphens not em dashes, and a specific ask at the end.

**Only after all of the above:** the §7 causal-patching stretch goal (D-34), then the §13 bottleneck-width ablation framed as geometry-not-accuracy (D-37). **Drop causal patching before dropping H3 or the HMM.**

---

## 6. Open risks, ranked

Each risk names the **specific early signal** that should trigger the corresponding spec §10 stop condition. "Stop" means halt, write it down, and escalate — not quietly re-parameterize.

### R1 — GPT is *supposed* to fail the competence gate, and the spec says that stops the project. **[BLOCKED — resolve before launch]**
The paper puts GPT on G(5,5) at ≈18.6%, which is 1/d chance. Spec §8 tells us to debug, and spec §10 tells us to stop, if "either model remains below the base competence gate". Worse than the process contradiction is the scientific one: a model at chance has a degenerate correct-branch margin, so H2's "geometry predicts planning" and H3's "margin erosion" are being measured on a capability GPT never had, and the GPT-vs-NextLat difference-in-differences loses its meaning.
**Early signal:** GPT `val_(5, 5)/test_accuracy` still ≈0.20 and `token_2` ≈0.20 at step 5,000.
**Recommended resolution (needs sign-off):** apply the 90% gate to **NextLat only**, preregister GPT's chance-level accuracy as a replication of the paper's Fig. 6 rather than a failure, and preregister *before any training* how GPT's H2/H3 quantities will be interpreted at chance — most defensibly as a floor, with margin analyses reported descriptively and the DiD contrast bounded out loud. Do **not** resolve it by switching topology to G(2,10) (broadens the benchmark, spec §1 forbids) or by conditioning `A_pair` on model correctness (spec §5 forbids).
**Stop condition if unresolved:** spec §10, "either model remains below the base competence gate".

### R2 — `proj_factor` silently reverts to 1.0 and we train the wrong model for 20,000 steps.
Deleting the `sweep:` block — which P3 requires — drops the only place `proj_factor: 0.5` appears (D-07). The fallback is `1.0`: dynamics hidden 768, +884,736 params, no longer the paper's configuration. Nothing warns.
**Early signal:** `materialized_config.yaml` shows `model.proj_factor: 1.0`, or the step-0 param assertion reports 22,800,000 instead of 21,915,264.
**Stop condition:** spec §10, "the available hardware cannot run the exact paper-scale configuration" — read here as "we are not running the paper-scale configuration". Kill the run immediately; a 20k-step run on the wrong width is unusable and unrecoverable.

### R3 — Resume is not reproducible enough for H3's within-parent design.
Checkpoints carry no RNG state and no data position (§1.3); data order is *replayed*, and with plain `--devices 1` it diverges. TF32 is force-enabled and `trainer.deterministic` is dead, so bit-exactness is unavailable at any setting (D-26).
**Early signal:** the mandatory 300 vs 150+150 test shows final weights or logits diverging beyond the recorded tolerance; or `--strategy ddp --devices 1` fails to give an identical sampler order.
**Stop condition:** spec §10, "interrupted training cannot resume reproducibly enough for the stated analysis". Mitigation before stopping: add Python/NumPy/CPU/CUDA RNG state to the checkpoint (spec §9 already anticipates this).

### R4 — The exact matched-lure design does not survive contact with the graph.
Near-safe and near-critical must match on two edited endpoint tokens, serialized positions, token edit distance, node multiset, degree sequence, prompt and answer length, source, goal, and validity. Node sampling without replacement (§1.1) makes the multiset free, but degree-sequence and validity preservation under an equal-depth suffix swap between the *goal* arm and a distractor is the part that can fail.
**Early signal:** fewer than 1,000/1,000 quartets pass the §5 acceptance tests, or any single acceptance criterion fails at a nonzero rate.
**Stop condition:** spec §10, "safe and critical lures cannot be exactly matched". **Fix the generator; do not regress the mismatch away in analysis afterward** — the spec explicitly forbids that.

### R5 — H3 near/far branches are not comparably difficult, so the interference contrast is confounded.
`B_far` must be matched to `B_near` on initial loss quantiles using a **separate, non-confirmatory pilot checkpoint**, then frozen across every model and seed.
**Early signal:** in the model-blind pilot, near/far initial-loss quantiles fail to overlap; or, at adaptation time, acquisition on the two branches' own examples differs materially.
**Stop condition:** spec §10, "near/far branches cannot achieve comparable acquisition or initial difficulty". Report any residual imbalance and control for it item-level rather than hiding it.

### R6 — A botched restore looks exactly like a fresh run.
`core_train.py:165-168` silently initializes from scratch when the resume pointer is missing, and `core_train.py:976-979` deletes the previous recovery checkpoint after a non-atomic pointer write with no verification of the new one. Both failure modes produce a running job and a plausible loss curve.
**Early signal:** `state.json` step resets to 0, or `training_steps` in a loaded checkpoint disagrees with the ledger, or the driver logs "Could not find checkpoint file".
**Stop condition:** transport-level, not spec §10 — the two-strike circuit breaker (§4.4) plus a hard runner assertion that the pointer and its target hash match `state.json` **before** passing `init_from=resume`.

### R7 — Measured throughput blows the wall-clock budget.
Every s/step in §4.3 is modelled. `data.num_workers: 0` with a 35k-token step and `num_workers: 0` fast-forward replay on resume are both plausible ways reality comes in slow.
**Early signal:** at the §4.5 gate-4 profile, measured A100 s/step exceeds the modelled figure by more than ~2.5×; or cumulative spend crosses **400 CU** (envelope) with base runs incomplete.
**Stop condition:** spec §10, "the available hardware cannot run the exact paper-scale configuration reliably. In that case, pause for a compute decision rather than shrinking the confirmatory model." Hard stop at **600 CU** (§4.4). Reduce optional analyses before reducing scientific fidelity.

### R8 — The geometry effect exists only under one metric or one layer.
The spec preregisters centered cosine as primary and whitened Euclidean as the robustness check, and designates the final post-norm state as primary with intermediate layers descriptive only. D-11 adds a second, equally preregistered position.
**Early signal:** PSI is positive under centered cosine and null under whitened Euclidean; or positive at `h[63]` and null at `h[62]`, or vice versa, without a stated mechanism.
**Stop condition:** spec §10, "a geometry effect exists only under a post-hoc distance metric or layer". Report the disagreement as the result. Do not go metric-shopping.

### R9 — The result depends on one seed.
Three confirmatory seeds is already a documented reduction from the paper's five (D-33).
**Early signal:** the across-seed spread of PSI or of `erosion_near − erosion_far` includes zero once any single seed is dropped.
**Stop condition:** spec §10, "the result depends on one seed". Run the remaining two paper seeds if time permits; otherwise bound the claim out loud and state what magnitude three seeds could not have detected.

### R10 — The new `hmm_belief` datamodule breaks the trainer in a way that costs a runtime.
`trainer.test_interval` is read unconditionally at `core_train.py:671` and is absent from `defaults.yaml`, and the spec's §12 YAML omits it (D-17). `experiment_name` is likewise required by `train.py:95`.
**Early signal:** the first validation (step 300) raises a missing-key error — which, under `colab exec`, returns "cleanly" in ~40 s and invites the orchestrator to burn another runtime.
**Stop condition:** transport circuit breaker. Mitigate cheaply by running P6's 50-step exit test on **T4** before touching A100.

### R11 — The competence gate is measured under multinomial sampling, not greedy.
`evaluate_stargraph` samples (D-22). The number we quote as "exact-path accuracy" is therefore not the number a reader will assume.
**Early signal:** sampled accuracy sits a few points below greedy on the same checkpoint at the first validation.
**Stop condition:** none — but the gate's sampling regime must be fixed **in writing before the first run**, and both numbers reported. Silently switching to greedy after seeing a gate failure is exactly the post-hoc move the spec forbids.

### R12 — Credential blast radius on a machine we do not control. **[BLOCKED — resolve before first upload]**
`/content/adc.json` as a user ADC is a live refresh token for `<redacted-account>` with full `cloud-platform` scope, uploaded to a Google-managed VM with an uncontrolled lifetime (D-36).
**Early signal:** none — this risk is silent by construction, which is the argument for fixing it up front.
**Stop condition:** none in spec §10; this is a project-hygiene gate. Recommended: a dedicated service account scoped to `roles/storage.objectAdmin` on `nextlat-lurestar-project-flash-490419` only. If the user ADC is used anyway: `chmod 0600` as the driver's first action, uploaded per session, never persisted to Drive, never in the source snapshot, never in git, never echoed, and never inside `job.json`.
