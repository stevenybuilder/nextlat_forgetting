# Upstream Repository Cartography — NextLat @ `3770be6`

**Pinned checkout:** `/Users/stevenyang/Documents/nextlat_forgetting/upstream/NextLat`
**Commit:** `3770be6009cea2b3c455a9ce7f2ca88b504bb955` — "Initial public release", Mon May 25 21:50:04 2026 -0700
**Spec answered:** `nextlat_v4_predictive_geometry_spec.md` §16 (questions 1–5) plus the four extra reporting items.

Every claim below is cited `file:line` against the pinned tree. Nothing here is inferred from the paper or
from the GitHub web UI. Verbatim excerpts are copied from the checkout.

---

## 0. TL;DR — the decision-relevant facts

| # | Finding | Consequence |
|---|---|---|
| 1 | `data/stargraph/prepare.py:133` defaults `--max_nodes 50`; `data/README.md:54` uses `--max_nodes 100` | Must pass `--max_nodes 100` explicitly. Spec is right. |
| 2 | `proj_factor: 0.5` lives **only in the `sweep:` block** of `nextlat_stargraph_5_5.yaml:61`, not in `model:` | Deleting the sweep block (to run one seed) silently reverts to `defaults.yaml:118` `proj_factor: 1.0` → dynamics MLP hidden 768 not 384, +885k params. **Highest-risk footgun in the repo.** |
| 3 | `compile: true` in all five shipped `config/stargraph/5_5/*.yaml` (e.g. `gpt_stargraph_5_5.yaml:17`) | Directly contradicts the repo's own `README.md:117-122`. Spec's `compile:false` is correct; the shipped YAML is wrong. |
| 4 | Resume pointers `latest_ckpt`/`recovery_ckpt` live at `trainer.out_dir`, **not** at `out_dir/experiment_name` (`core_train.py:945,971`) | All sweep seeds under one `out_dir` overwrite one pointer. One `out_dir` per job is mandatory, exactly as the spec says. |
| 5 | Checkpoint dict = `{model, optimizer, training_steps, lr_scheduler_state}` only (`model_gpt.py:430-434`, `model_nextlat.py:581-585`, `model_base.py:411-417`) | **No RNG state, no dataloader position, no epoch, no config.** Spec §9 recovery test will need the RNG extension. |
| 6 | `Transformer.forward(..., return_hidden_states=True)` returns `(output, x)` for GPT (`model_gpt.py:290-291`) but `(token_embeds, text_embd)` for NextLat (`model_nextlat.py:199-200`) — and NextLat's early-returns **before** computing logits | Extraction code cannot be shared naively. NextLat needs `lm_head(text_embd)` applied by hand. |
| 7 | `train_graphs` / `heldout_graphs` **do not exist** anywhere in the repo | Spec inventions. Real keys are `stargraph_train_data_path` / `stargraph_test_data_path`. |
| 8 | `save_last_checkpoint`, `save_best_checkpoint`, `save_recovery_checkpoint` **do exist** (`defaults.yaml:18,20,24`) | Spec is right on these three. |
| 9 | `optimizer.name` / `optimizer.betas` / `optimizer.schedule` / `optimizer.clip_gradient_norm` do not exist | Real: `optimizer.optimizer_type`, `beta1`/`beta2`, `lr_scheduler.schedule`, `optimizer.grad_clip`. |
| 10 | StarGraph accuracy uses **multinomial sampling**, not greedy (`model_gpt.py:555-557`, `model_nextlat.py:734-736`; no `temperature`/`top_k` in the 5_5 YAML → `getattr` defaults `1.0`/`None` at `data/stargraph.py:126-127`) | The 90% competence gate is measured under ancestral sampling. |
| 11 | The state at the `=` delimiter predicts the **source node** (a trivial copy), not the branch | The first *branch* decision is one position later. See §2.4 — this materially affects H1/H2. |
| 12 | `wandb` is imported unconditionally at `train.py:15,17,24`; `log_to_wandb: true` by default (`defaults.yaml:34`) and the 5_5 configs never override it | Offline Colab needs `trainer.log_to_wandb=false` (CLI override) or `WANDB_MODE=offline`. `wandb` must still be pip-installed. |
| 13 | `trainer.deterministic` (`defaults.yaml:38`) is **never read anywhere in the codebase** | Dead key. Do not rely on it. |
| 14 | Sweep experiment names are built by iterating a Python `set` (`train.py:280,322`) | Directory name can change across process invocations (str hash randomization). Do not use the sweep block for confirmatory runs. |

---

## 1. Question 1 — the Path-Star generator, its CLI, defaults, serialized format, and how G(5,5) consumes it

### 1.1 The generator CLI

Single file: **`data/stargraph/prepare.py`**. The `argparse` block is `prepare.py:115-154`:

```python
115	if __name__ == "__main__":
116	    parser = argparse.ArgumentParser(description="Generate star or sink graph data")
117	    parser.add_argument(
118	        "--num_samples", type=int, default=200000, help="Number of samples to generate"
119	    )
120	    parser.add_argument(
121	        "--num_test_samples",
122	        type=int,
123	        default=20000,
124	        help="Number of test samples to generate",
125	    )
126	    parser.add_argument(
127	        "--num_paths", type=int, default=5, help="Number of paths from the source"
128	    )
129	    parser.add_argument(
130	        "--path_length", type=int, default=5, help="Length of each path"
131	    )
132	    parser.add_argument(
133	        "--max_nodes", type=int, default=50, help="Maximum number of nodes in the graph"
134	    )
135	    parser.add_argument(
136	        "--data_dir",
137	        type=str,
138	        default="data/stargraph",
139	        help="Directory to save the generated data",
140	    )
141	    parser.add_argument(
142	        "--generate_test_data", action="store_true", help="Generate test data"
143	    )
144	    args = parser.parse_args()
```

**Defaults:** `--num_samples 200000`, `--num_test_samples 20000`, `--num_paths 5`, `--path_length 5`,
**`--max_nodes 50`**, `--data_dir data/stargraph`, `--generate_test_data` off.

**There is no `--seed` flag.** Determinism is hard-wired: `prepare.py:52` sets `seed = 0` and
`prepare.py:61-62` / `prepare.py:90-91` do `random.seed(seed); seed += 1` per sample. The test set
continues the same counter, so its seeds are `num_samples … num_samples+num_test_samples-1`.
**Consequence:** the 20,000-graph held-out set is only reproducible if `--num_samples` is also
identical. Train/test disjointness is by seed, not by content check — collisions are astronomically
unlikely with 100 nodes but are not asserted anywhere.

### 1.2 The `max_nodes` 50-vs-100 discrepancy — confirmed, spec is correct

- Code default: `--max_nodes 50` (`prepare.py:133`).
- The repo's own documented command uses 100 (`data/README.md:51-55`):

```bash
python data/stargraph/prepare.py --num_samples 200000 --num_paths 5 --path_length 5 --max_nodes 100 --generate_test_data
```

- The shipped G(5,5) configs assert 100: `stargraph_max_nodes: 100`
  (`config/stargraph/5_5/gpt_stargraph_5_5.yaml:28`, `nextlat_stargraph_5_5.yaml:27`).
- `defaults.yaml:81` also says `stargraph_max_nodes: 50`.

**Verified empirically** by running the generator from the local venv (with a stub for the unused
`import torch` at `prepare.py:1`) — `--max_nodes 50` produces node ids ≤ 49; `--max_nodes 100`
produces ids ≤ 99.

**Node id range is `0…max_nodes-1`, i.e. `0…99` — not `1…100`** as spec §8 says
(`prepare.py:11`: `nodes = list(range(maxNodes))`). Cosmetic, but the manifest must record `0..99`.

**The exact command to run:**
```bash
python data/stargraph/prepare.py --num_samples 200000 --num_test_samples 20000 \
  --num_paths 5 --path_length 5 --max_nodes 100 --data_dir data/stargraph --generate_test_data
```
This writes `data/stargraph/graph_5_5_sample_200000.txt` and `data/stargraph/graph_5_5_test_20000.txt`
(filename templates at `prepare.py:56` and `prepare.py:85`), which are exactly the two paths hard-coded
in the shipped 5_5 configs (`gpt_stargraph_5_5.yaml:26-27`).

### 1.3 Graph construction

`prepare.py:8-36`:

```python
 8	def star_or_sink_graph_maker(
 9	    numOfPathsFromSource, lenOfEachPath, maxNodes, shuffle_edge_lists, reverse=False
10	):
11	    nodes = list(range(maxNodes))
12	    random.shuffle(nodes)
13	
14	    source = nodes.pop()
15	
16	    edgeList = []
17	    path = [source]
18	
19	    for p in range(numOfPathsFromSource):
20	        oldNode = source
21	        for i in range(lenOfEachPath - 1):
22	            newNode = nodes.pop()
23	            edgeList.append((oldNode, newNode))
24	            oldNode = newNode
25	            if p == 0:
26	                path.append(oldNode)
27	        if p == 0:
28	            goal = oldNode
29	
30	    if shuffle_edge_lists:
31	        random.shuffle(edgeList)
```

Key facts for the Lure-Star generator:
- **The goal arm is always arm `p == 0`, the first one generated**, but `random.shuffle(edgeList)`
  (`prepare.py:31`) destroys positional information, so the model cannot exploit arm order.
- For G(5,5): 1 source + 5 arms × 4 nodes = **21 distinct nodes**, **20 edges**, path length 5.
- Nodes are sampled without replacement from a shuffled `range(100)` — no node repeats within a graph.
  A suffix-swap lure therefore preserves the node multiset by construction, which is what the spec's
  matching requirements need.

### 1.4 Exact serialized text format

`prepare.py:70-73` (identical at `prepare.py:99-102` for the test file):

```python
70	            file.write(
71	                "|".join([",".join([str(i) for i in edge]) for edge in edgeList])
72	                + f"/{source},{goal}={','.join([str(i) for i in path])}\n"
73	            )
```

Grammar:

```
<edge>       ::= <int> "," <int>
<edge_list>  ::= <edge> ( "|" <edge> )*
<line>       ::= <edge_list> "/" <source> "," <goal> "=" <path_node> ("," <path_node>)* "\n"
```

Delimiters: `|` between edges, `,` inside an edge and between path nodes, `/` before the
source/goal query, `=` before the answer. One example per line, `\n`-terminated.

**Verbatim G(5,5) example** produced by this generator at `--max_nodes 100` (first line of a
3-sample run):

```
49,97|65,62|36,85|51,38|61,45|49,12|64,17|5,33|12,79|49,64|62,51|45,74|49,61|74,27|17,36|32,68|97,53|79,32|49,65|53,5/49,33=49,97,53,5,33
```

Here source=`49`, goal=`33`, answer path=`49,97,53,5,33`.

### 1.5 Tokenizer and vocabulary

`data/stargraph.py:9-57`:

```python
 9	class Tokenizer:
10	    def __init__(self, maxNodes):
11	        self.maxNodes = maxNodes
12	        self.encoder = {str(i): i for i in range(maxNodes)}
13	        self.encoder["|"] = maxNodes
14	        self.encoder["="] = maxNodes + 1
15	        self.encoder["/"] = maxNodes + 2
16	        self.encoder["$"] = maxNodes + 3  # Padding token
...
23	        self.decoder[maxNodes + 4] = ""
...
30	        self.eos_token_id = maxNodes + 4
```

With `maxNodes = 100`:

| token | id |
|---|---|
| node `0`…`99` | `0`…`99` |
| `\|` | 100 |
| `=` | 101 |
| `/` | 102 |
| `$` (pad, unused for stargraph) | 103 |
| EOS (decodes to `""`) | 104 |

`encode()` (`data/stargraph.py:32-49`) **drops every `,`** (line 35-37) and greedily consumes digit
runs, so `,` is purely a text separator with no token. `tokenize()` (`data/stargraph.py:51-57`)
appends **one EOS at the end** of every line.

**`vocab_size = maxNodes + 5 + 1 = 106`** (`data/stargraph.py:233`). Note ids `0..104` are the only
ones ever emitted — id **105 is an unused slack row** in the embedding/lm_head. Harmless, but the
extraction code must not assume `vocab_size-1` is meaningful.

The comment at `data/stargraph.py:26-30` is worth quoting: *"stargraph has no eos token, padded
sequence masking doesn't need to happen, but this property is needed for compatibility with the bst
training code."* In practice the appended id-104 **is** treated as EOS by
`create_attention_mask` (`model_gpt.py:235`, `model_nextlat.py:155`) and by
`create_position_indices` (`model_base.py:593`), so each line is exactly one "document".

### 1.6 Verified tokenization of the example above

Reproduced locally with the repo's own `Tokenizer` logic:

```
token ids: [49, 97, 100, 65, 62, 100, 36, 85, 100, 51, 38, 100, 61, 45, 100, 49, 12, 100,
            64, 17, 100, 5, 33, 100, 12, 79, 100, 49, 64, 100, 62, 51, 100, 45, 74, 100,
            49, 61, 100, 74, 27, 100, 17, 36, 100, 32, 68, 100, 97, 53, 100, 79, 32, 100,
            49, 65, 100, 53, 5, 102, 49, 33, 101, 49, 97, 53, 5, 33, 104]
seq_len (total_len)                 = 69
index of the '=' token (id 101)     = 62
graph_description_len (=context_length) = 62
num_target_tokens                   = 5
vocab_size                          = 106
```

Arithmetic: 20 edges × 2 node tokens = 40; 19 `|`; 1 `/`; source + goal = 2; 1 `=`; 5 path tokens;
1 EOS → **69**.

### 1.7 How the G(5,5) config consumes it

`StarGraphDataModule.__init__` (`data/stargraph.py:165-235`):

```python
172	        self.batch_size = config.data.device_batch_size
...
175	        maxNodes = config.data.stargraph_max_nodes
176	        self.tokenizer = Tokenizer(maxNodes)
178	        # Load data
179	        with open(config.data.stargraph_train_data_path, "r") as f:
180	            train_data = f.readlines()
182	        with open(config.data.stargraph_test_data_path, "r") as f:
183	            val_data = f.readlines()
185	        graph_description_len, total_len = self._measure_index(train_data)
186	        num_target_tokens = total_len - graph_description_len - 2
187	        num_arms = int(config.data.stargraph_train_data_path.split("_")[1])
188	        assert num_target_tokens == int(
189	            config.data.stargraph_train_data_path.split("_")[2]
190	        ), f"num_target_tokens {num_target_tokens} does not match name in file: ..."
```

**The filename is load-bearing.** `num_arms` and the expected answer length are parsed out of
`stargraph_train_data_path.split("_")[1]` and `[2]`. For
`data/stargraph/graph_5_5_sample_200000.txt` the split is
`['data/stargraph/graph', '5', '5', 'sample', '200000.txt']` → `num_arms=5`, expected target len `5`.
**Any lure/eval file we generate must keep the `graph_<d>_<l>_...` naming convention or this assert
fires.**

`_measure_index` (`data/stargraph.py:237-247`) only reads `data[0]` — **the first line** — and assumes
every line has identical length. True for a fixed `G(d,l)`, and true for our lures by construction,
but it means a ragged file silently mis-slices every batch.

`update_config` (`data/stargraph.py:249-252`) **overwrites three model keys at runtime**:

```python
249	    def update_config(self, config):
250	        config.model.vocab_size = self.vocab_size          # 106
251	        config.model.context_length = self.graph_description_len   # 62
252	        config.model.block_size = self.total_len           # 69
```

So the YAML's `vocab_size: 0` and `block_size: 1024` (`gpt_stargraph_5_5.yaml:38-39`) are placeholders.
This runs at `train.py:181`, **before** `initialize_model` at `train.py:226`, so the model is built at
`block_size=69`, `vocab_size=106`, `context_length=62`. The `materialized_config.yaml` dumped at
`train.py:192-194` records the resolved values — that is the file to archive per the spec.

**`context_length: 62` drives prompt-loss masking** (`model_gpt.py:362-370`, `model_nextlat.py:441-453`):
targets for positions inside the prompt are set to `-100`. Traced for G(5,5): position ids are
`[1..68, 0]`; `context_length_mask = (pos_ids <= 63) & (pos_ids != 0)` masks token indices `0..62`;
after the `[:, 1:]` shift, targets `0..61` are ignored and target index 62 (predicting the answer's
first token, the source) **is** trained. So loss is computed on 6 positions per example: the source
copy, the 4 path continuations, and EOS.

**Dead key:** `defaults.yaml:80` defines `stargraph_data_path: ""`, which no code path reads. The real
keys `stargraph_train_data_path` / `stargraph_test_data_path` are supplied only by the task YAMLs and
exist in no defaults file.

Batching: `train_dataloader` (`data/stargraph.py:257-266`) uses `shuffle=True, drop_last=True`,
`num_workers` from config (0 in the 5_5 YAMLs), `collate_fn` = plain `torch.stack`
(`data/stargraph.py:294-297`). 200,000 / 512 = **390 batches per epoch**, so a 20,000-step run is
~51.3 epochs.

Accuracy eval: `evaluate_stargraph` (`data/stargraph.py:77-162`) splits
`prefix = batch[:, :-(num_target_tokens+1)]` (line 118) → 63 tokens, **ending exactly on `=`** — and
`target = batch[:, -(num_target_tokens+1):-1]` (line 119) → the 5 answer tokens. The logged metric is
`val_(5, 5)/test_accuracy` (line 155, prefix built at lines 95-99) — exact-path accuracy — plus
`val_(5, 5)/token_{1..5}`. `token_2` is the first-branch accuracy the spec cares about; `token_1` is
the trivial source copy and will sit at ~100%.

---

## 2. Question 2 — where the FINAL POST-NORMALIZATION hidden state is produced, and the minimal hook

### 2.1 GPT

`models/model_gpt.py:251-293`:

```python
251	    def forward(
252	        self,
253	        batch: torch.Tensor,
254	        mask: Optional[torch.Tensor] = None,
255	        targets: Optional[torch.Tensor] = None,
256	        return_hidden_states: bool = False,  # Whether to return hidden states for CVAE
257	    ) -> torch.Tensor:
...
272	        x = self.token_embedding(batch)
273	
274	        for block in self.transformer.blocks:
275	            x = block(x, mask=mask, rope=rope)
276	        x = self.transformer.norm(x)
277	
278	        # If no targets given, return logits
279	        if targets is None:
280	            output = self.lm_head(x)
281	
282	        # If targets are given, compute loss
283	        else:
284	            output = self.cross_entropy_loss(
285	                input=x,
286	                last_layer=self.lm_head,
287	                targets=targets,
288	            )
289	
290	        if return_hidden_states:
291	            return output, x
292	
293	        return output
```

**`x = self.transformer.norm(x)` at `model_gpt.py:276` is the final post-normalization hidden state.**
Shape `(B, T, 384)`. It is returned as the **second** element when `return_hidden_states=True`.

### 2.2 NextLat

`models/model_nextlat.py:171-217`:

```python
191	        # token embeddings of shape (b, t, n_embd)
192	        x = self.token_embedding(batch)
193	        token_embeds = x
194	
195	        for block in self.transformer.blocks:
196	            x = block(x, mask=mask, rope=rope)
197	        text_embd = self.transformer.norm(x)
198	
199	        if return_hidden_states:
200	            return token_embeds, text_embd
201	
202	        # If no targets given, return logits
203	        if targets is None:
204	            output = self.lm_head(text_embd)
```

**`text_embd = self.transformer.norm(x)` at `model_nextlat.py:197` is the same quantity.**

### 2.3 The asymmetry — read this before writing `representations.py`

| | GPT | NextLat |
|---|---|---|
| module producing it | `transformer.norm` (`model_gpt.py:276`) | `transformer.norm` (`model_nextlat.py:197`) |
| `return_hidden_states=True` returns | `(output, x)` — output is logits (targets=None) or loss | `(token_embeds, text_embd)` — **no logits at all** |
| tuple position of the hidden state | index **1** | index **1** |
| tuple position 0 | logits/loss | token embeddings `(B,T,384)` |
| does it early-return? | no (falls through lines 279-291) | **yes** (line 199-200, before `lm_head`) |

So `h, logits = f(model, x)` cannot be written once for both. For NextLat you must apply the head
yourself:

```python
# NextLat
token_embeds, h = nextlat.model(prompt, return_hidden_states=True)   # model_nextlat.py:199-200
logits = nextlat.model.lm_head(h)                                    # model_nextlat.py:121
# GPT
logits, h = gpt.model(prompt, return_hidden_states=True)             # model_gpt.py:290-291 (targets=None)
```

Two more facts:
- `self.transformer.norm` is `LayerNorm(n_embd, bias=config.bias)` with `bias: false` in the 5_5 configs,
  and `LayerNorm.forward` (`model_base.py:823-830`) dispatches to **`F.rms_norm`** when `bias is None`.
  The "final post-normalization state" is therefore an **RMS-normalized** vector — direction is
  preserved, magnitude is not mean-centered. Since the spec's primary metric is centered cosine
  distance (centering across the item pool), this is fine, but it should be documented as RMSNorm.
- The two `compute_loss` paths feed **different sequence lengths** to the trunk:
  GPT uses `inputs = batch[:, :-1]` (`model_gpt.py:353`, T=68), NextLat uses the full `batch`
  (`model_nextlat.py:430`, T=69). Irrelevant for offline extraction where we drive the forward pass
  ourselves, but it does mean a training-time hook would see different `T` per model.

### 2.4 ⚠️ The `=` position predicts the source, not the branch

Spec §6/H1 says "extract the state at the final prompt delimiter `=`, before the first answer token."
That is index **62** (verified §1.6). But the token generated from index 62 is `path[0]` = **the
source**, which is already visible in the prompt at the `/source,goal` field — a trivial copy. The
first *branch* choice is `path[1]`, generated from the hidden state at index **63** (the source token
inside the answer region).

Concretely, for the verified example: prompt `…53,5/49,33=` → answer `49,97,53,5,33`. `h[62]` scores
`49`; `h[63]` scores `97`.

Recommendation: extract **both** `h[62]` (the spec's preregistered primary, keep it) **and** `h[63]`,
and compute the "correct-branch logit margin" of H2/H3 from the logits at index 63. Freeze this choice
before looking at any model. Reporting only `h[62]` risks measuring a state whose immediate prediction
target is identical across every condition in the quartet.

### 2.5 Minimal hook, no training-code change

**Preferred — no hook at all.** Call the inner `Transformer`/`NextLatTransformer` directly with
`return_hidden_states=True`. This is a public keyword on both forwards and touches nothing.

**If a hook is genuinely needed** (e.g. capturing during an unmodified `compute_loss` call), register a
forward hook on the final norm module. The module path is **identical for both models**:

```python
store = {}
def _cap(mod, inp, out):
    store["h"] = out.detach()            # (B, T, 384) post-norm
handle = model.model.transformer.norm.register_forward_hook(_cap)
```

Caveats, all verifiable in the tree:
- `model.model` is wrapped by `fabric.setup_module` (`model_gpt.py:329`, `model_nextlat.py:249`).
  Lightning's `_FabricModule` forwards attribute access, so `model.model.transformer.norm` resolves;
  if it does not on your Lightning version, use `model.model._forward_module.transformer.norm`.
- With `trainer.compile: true` (`core_train.py:209-211`) the trunk is an `OptimizedModule` and
  submodule paths gain a `_orig_mod` level. **Run with `compile: false`** — which the spec and
  `README.md:117-122` already require — and this problem disappears.
- For the §7 causal-patching stretch goal, the penultimate-layer state is the **output of
  `transformer.blocks[10]`** (`n_layer=12`, so index 10 feeds the last block); hook
  `model.model.transformer.blocks[10]`. Do **not** patch `transformer.norm`'s output and call it a
  circuit — `lm_head` consumes that directly, exactly as the spec warns.

---

## 3. Question 3 — checkpoint save/load, contents, pointer files, and `init_from: resume`

### 3.1 What is written

`models/model_base.py:404-417`:

```python
404	    def save_checkpoint(self, file_path: str):
...
409	        self._assert_fabric_is_setup()
410	        self.fabric.print(f"Saving checkpoint to {file_path}")
411	        state = self._get_checkpoint_state()
412	        if "training_steps" not in state:
413	            # Make sure to save training steps
414	            state["training_steps"] = self.training_steps
415	        if self.lr_scheduler_state is not None:
416	            state["lr_scheduler_state"] = self.lr_scheduler_state
417	        self.fabric.save(file_path, state)
```

`_get_checkpoint_state` is identical for GPT (`model_gpt.py:426-434`) and NextLat
(`model_nextlat.py:577-585`):

```python
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "training_steps": self.training_steps,
        }
```

`lr_scheduler_state` is stuffed in by the trainer immediately before each save
(`core_train.py:936-939` and `core_train.py:962-965`).

**Checkpoint payload (exhaustive):**

| key | contents | present? |
|---|---|---|
| `model` | full model `state_dict` (trunk + `lm_head` + `dynamics_model` for NextLat) | ✅ |
| `optimizer` | AdamW `state_dict` (exp_avg, exp_avg_sq, step, param groups) | ✅ |
| `training_steps` | `int` | ✅ |
| `lr_scheduler_state` | `LambdaLR.state_dict()` (or a list if multiple optimizers) | ✅ |
| Python / NumPy / torch-CPU / CUDA RNG state | — | ❌ **absent** |
| dataloader / sampler position | — | ❌ **absent** |
| `epoch` | — | ❌ **absent** (resets to 0 on resume, `core_train.py:308`) |
| resolved config | — | ❌ absent from the `.pt` (dumped separately to `materialized_config.yaml`, `train.py:192-194`) |
| `best_val_loss` | — | ❌ absent (resets to `None`, `core_train.py:311`) |

This is the direct answer to spec §9: **"The official checkpoint includes model, optimizer, scheduler,
and training step"** — correct, and nothing else. The spec's contingency ("If the trajectories
materially diverge, add Python, NumPy, CPU, and CUDA RNG states") should be treated as the expected
path, not a fallback (see §3.5).

### 3.2 Loading

`models/model_base.py:419-456`:

```python
419	    def load_checkpoint(self, file_path: str, strict: bool = True):
...
426	        state = self._get_checkpoint_state()
427	        if self.optimizer is None:
...
433	            state.pop("optimizer", None)
434	        # fabric.load() will in-place modify all objects in the state
435	        self.fabric.load(file_path, state, strict=strict, weights_only=False)
436	        # Update training steps manually because it is an int
437	        self.training_steps = state["training_steps"]
438	        self.lr_scheduler_state = self._read_scheduler_state_from_checkpoint(file_path)
```

Note `weights_only=False` — the `.pt` is unpickled with full trust. Fine for our own files; do not
load third-party checkpoints. `_read_scheduler_state_from_checkpoint` (`model_base.py:440-456`) does a
**second full `torch.load` of the whole file** just to pull one dict, and swallows every exception
(`except Exception: return None`). A corrupt or truncated checkpoint therefore resumes **silently with
no scheduler state** rather than failing. With `schedule: constant` this is harmless (the lambda is
`lambda _: 1.0`, `core_train.py:993-994`); with any other schedule it would be a silent wrong-LR bug.

The scheduler is then re-aligned by stepping it forward (`core_train.py:326-329`):

```python
326	        # Align optimizer group lrs to current training step without deprecated epoch arg.
327	        for scheduler in self.lr_schedulers:
328	            while scheduler.last_epoch < self.step - 1:
329	                scheduler.step()
```

### 3.3 Where files land

`_save_checkpoint` (`core_train.py:925-950`) and `_save_recovery_checkpoint` (`core_train.py:952-982`):

```
{trainer.out_dir}/
├── latest_ckpt                     ← pointer file, plain text, ONE line  (core_train.py:945)
├── recovery_ckpt                   ← pointer file, plain text, ONE line  (core_train.py:971)
└── {trainer.experiment_name}/
    ├── ckpt_iter_{step}_{val_loss:.4f}.pt      (core_train.py:774-777)
    ├── recovery_ckpt_iter_{step}.pt            (core_train.py:961)
    ├── materialized_config.yaml                (train.py:192-194)
    └── version_N/metrics.csv                   (CSVLogger, train.py:103-109)
```

**The two pointer files sit at `out_dir`, one directory ABOVE the experiment directory.** Verbatim:

```python
942	        # Save the file path to the latest checkpoint
943	        if self.fabric.global_rank == 0:
944	            with open(
945	                os.path.join(self.config.trainer.out_dir, "latest_ckpt"),
946	                "w",
947	            ) as f:
948	                f.write(ckpt_path)
```

```python
968	        # Save the most recent file path to the recovery checkpoint pointer file
969	        if self.fabric.global_rank == 0:
970	            with open(
971	                os.path.join(self.config.trainer.out_dir, "recovery_ckpt"),
972	                "w",
973	            ) as f:
974	                f.write(ckpt_path)
```

The pointer holds an **absolute-or-relative path string as written**, i.e. whatever
`os.path.join(out_dir, experiment_name, filename)` produced. If `out_dir` is relative
(`output/stargraph` in the shipped YAML, `gpt_stargraph_5_5.yaml:14`), the pointer contains a
**relative path** and resume only works from the same CWD. Use an absolute `out_dir` on Colab.

**This confirms the spec's §9 rule verbatim:** *"the official resume pointer lives at the output-root
level and must never cross branches."* Since the shipped configs share `out_dir: output/stargraph`
across all five algorithms and all five sweep seeds, running the shipped sweep leaves one
`latest_ckpt` that points at whichever run wrote last. **Every job in our matrix needs its own
`trainer.out_dir`.**

### 3.4 How `init_from: resume` resolves

`core_train.py:139-172`:

```python
139	    # Resume from a previous training run
140	    elif config.trainer.init_from == "resume":
141	        recovery_ckpt_pointer = os.path.join(config.trainer.out_dir, "recovery_ckpt")
142	        latest_ckpt_pointer = os.path.join(config.trainer.out_dir, "latest_ckpt")
143	
144	        # Use the recovery checkpoint if it exists
145	        if os.path.isfile(recovery_ckpt_pointer):
146	            with open(recovery_ckpt_pointer, "r") as f:
147	                checkpoint_path = f.read().strip()
148	            assert os.path.isfile(
149	                checkpoint_path
150	            ), f"Checkpoint file {checkpoint_path} does not exist"
151	            fabric.print(f"Resuming from recovery checkpoint {checkpoint_path}")
152	
153	        # Otherwise, use the latest validation checkpoint if it exists
154	        elif os.path.isfile(latest_ckpt_pointer):
...
164	        # If no checkpoint file is found, initialize a new model
165	        else:
...
168	            checkpoint_path = None
169	
170	        # Empty init if we have found a checkpoint
171	        with fabric.init_module(empty_init=(checkpoint_path is not None)):
172	            model = ModelClass(model_config)
```

Precedence: **`recovery_ckpt` strictly wins over `latest_ckpt`**, and it wins even if the recovery
checkpoint is *older* than the latest validation checkpoint — there is no step comparison. Since
`_save_recovery_checkpoint` deletes the previous recovery file (`core_train.py:976-979`) but the
pointer is never cleared at end of training, a completed run leaves a stale `recovery_ckpt` pointing
at a deleted file → the `assert os.path.isfile` at line 148-150 **hard-fails on the next resume**.
Our runner must delete or rewrite `recovery_ckpt` when a job finishes.

`--checkpoint_path` (`train.py:262-264`) takes precedence over `init_from` entirely
(`core_train.py:130`) and loads **weights + optimizer + step**, so it is a valid way to branch H3's
near/far adaptation runs from a frozen parent — but note it also restores `training_steps`, so the
adaptation run's step counter starts at 20001 and `trainer.train_batches` must be set accordingly
(or `training_steps` reset before the trainer is constructed).

### 3.5 Durability gaps the spec must close (all confirmed, not hypothetical)

1. **Non-atomic writes.** `save_checkpoint` writes the `.pt` directly (`model_base.py:417`) and the
   pointer is a plain `open(..., "w")` (`core_train.py:944-948`, `970-974`). No `.partial` + `rename`.
   A kill mid-write leaves a truncated `.pt` *and/or* a pointer to it. Spec §9.2 items 2–3 are
   mandatory, not optional.
2. **Only one recovery checkpoint is kept, and the old one is deleted before the new one is
   verified** (`core_train.py:976-982`). Spec §9.2 item 4 (keep two, delete oldest only after
   verifying newest) is a real fix.
3. **`self.recovery_checkpoint_path` is in-memory only** (`core_train.py:334`). After a resume it is
   `None`, so the pre-crash recovery file is never garbage-collected → disk leak across many resumes.
4. **`os.remove` is unguarded** (`core_train.py:979`) — a missing file raises and kills training.
5. **RNG / data position are not checkpointed.** `_train_loop` "resumes" data by *replaying* the
   dataloader (`core_train.py:432-452`):

   ```python
   432	        # If resuming from a previous checkpoint, fast-forward data
   433	        if self.step > 0:
   434	            do_fast_forward = True
   435	            fast_forward_steps = 0
   ...
   442	                # Fast forward data
   443	                if do_fast_forward:
   444	                    fast_forward_steps += 1
   445	                    if fast_forward_steps >= self.step:
   ...
   450	                        do_fast_forward = False
   451	                    # Skip this batch
   452	                    continue
   ```

   The count is correct (exactly `self.step` batches are skipped), but **the batches are fully
   materialized and collated before being discarded**, so a resume at step 19,000 re-tokenizes
   19,000 × 512 ≈ 9.7 M lines with `num_workers: 0`. Budget minutes, and measure it in the profiling
   gate.

   More importantly, **whether the replayed order matches the original run depends on the sampler**:
   - Plain `fabric run --devices 1` → single-device strategy → Fabric does **not** substitute a
     `DistributedSampler`, so the DataLoader keeps its `shuffle=True` `RandomSampler`, whose per-epoch
     permutation is drawn from the **global torch RNG**. On resume the global RNG is re-seeded by
     `fabric.seed_everything` (`train.py:170`) but the intervening consumption differs — notably
     `init_module(empty_init=True)` (`core_train.py:171`) skips the weight-init draws that the scratch
     run performed. **Expect a different data order after resume.**
   - `fabric run --strategy ddp --devices 1` → distributed strategy → Fabric substitutes a
     `DistributedSampler`, whose permutation depends on `(seed, epoch)`. **`set_epoch` is never called
     anywhere in `core_train.py`** (grep confirms the only `set_epoch` in the repo is
     `data/fineweb.py:49,132,147,167`, for its own iterable dataset). So the order is a fixed function
     of the sampler's default seed and is **identical every epoch and across resumes** — deterministic,
     and also identical to what the paper's shipped 2-GPU `--strategy ddp` script produced.

   **Recommendation:** run the confirmatory jobs as `fabric run --strategy ddp --devices 1 …`. It
   matches the paper's sampler regime and makes the spec's forced-interruption test reproducible
   without patching the trainer. Verify this empirically with the spec's 300-step vs 150+150-step test
   before trusting it, and add RNG state to the checkpoint if it fails.

6. **Off-by-one:** `if self.step > self.config.trainer.train_batches: return` (`core_train.py:569`)
   → the loop performs **20,001** optimizer steps for `train_batches: 20000`. Harmless but record it.

---

## 4. Question 4 — the single-GPU launch command and every config key that must change

### 4.1 The shipped 2-GPU script

`scripts/stargraph/5_5/train_gpt_star_5_5.sh` (whole file):

```bash
1	#!/bin/bash
2	
3	fabric run --precision bf16-mixed --devices 2 --strategy ddp train.py --config config/stargraph/5_5/gpt_stargraph_5_5.yaml
```

`scripts/stargraph/5_5/train_nextlat_star_5_5.sh:3` is identical except for the config path.
(`train_bst_star_5_5.sh:3` uses `--devices 8` and no precision flag — not our path.)

`README.md:94-104` gives the two supported forms:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config <training_config>
fabric run --strategy ddp --devices <num_gpus> --precision bf16-mixed train.py --config <training_config>
```

### 4.2 The command to run

Matching the spec §8 exactly:

```bash
fabric run --devices 1 --precision bf16-mixed train.py --config configs/nextlat_lurestar.yaml
```

Recommended variant (see §3.5 item 5 — deterministic sampler, matches the paper's regime):

```bash
fabric run --strategy ddp --devices 1 --precision bf16-mixed train.py \
  --config configs/nextlat_lurestar.yaml
```

Offline / no-W&B variant (CLI overrides are parsed by `OmegaConf.from_dotlist`, `train.py:265,349`):

```bash
fabric run --strategy ddp --devices 1 --precision bf16-mixed train.py \
  --config configs/nextlat_lurestar.yaml \
  trainer.log_to_wandb=false trainer.compile=false
```

Use `--precision 16-mixed` only if the assigned GPU lacks stable BF16 (spec §8). T4 has no BF16.

### 4.3 Every config key that must change from the shipped 2-GPU config

Starting from `config/stargraph/5_5/{gpt,nextlat}_stargraph_5_5.yaml`:

| key | shipped | required | why |
|---|---|---|---|
| `trainer.compile` | `true` (`gpt_…:17`, `nextlat_…:16`) | **`false`** | Spec §8; `README.md:117-122` says compile is non-reproducible on Path-Star, especially Hopper. The shipped YAML contradicts the repo's own README. |
| `trainer.out_dir` | `output/stargraph` (`gpt:14` / `nextlat:13`) | **absolute, unique per job** e.g. `/content/drive/MyDrive/lurestar/runs/nextlat/1234/base` | Pointer files live here (`core_train.py:945,971`); sharing it across jobs corrupts resume. Relative paths break resume from a different CWD. |
| `trainer.experiment_name` | `GPT` / `NextLat` (`gpt:18` / `nextlat:17`) | explicit, e.g. `nextlat-s1234-base` | `train.py:98-99` appends `-seed{seed}`; with a sweep the name is built from **set iteration order** (`train.py:280,322`) and can differ between processes. |
| `sweep:` block | 5 seeds + NextLat hyperparams (`nextlat_…:55-61`) | **delete it — but see the `proj_factor` trap below** | One job per process; the runner owns the matrix. |
| `model.proj_factor` | **absent from `model:`**; only `sweep.model.proj_factor: [0.5]` (`nextlat_…:61`) | **add `proj_factor: 0.5` explicitly under `model:`** | Without the sweep it falls back to `defaults.yaml:118` `proj_factor: 1.0` → dynamics hidden 768 instead of 384 (+884,736 params). **This is the single most dangerous edit in the whole migration.** |
| `seed` | `defaults.yaml:7` = 1234; sweep supplies 1234-1238 | explicit top-level `seed: 1234` / `1235` / `1236` | Three preregistered confirmatory seeds. |
| `trainer.log_to_wandb` | not set → `defaults.yaml:34` = `true` | `false` (or `WANDB_MODE=offline`) | Offline Colab. |
| `trainer.save_recovery_checkpoint` | not set → `defaults.yaml:24` = `-1` (off) | **`250`** | Spec §9; required for any Colab run. |
| `trainer.wandb_project` / `wandb_tags` | `"stargraph"` / `["5_5"]` | irrelevant once W&B is off | — |
| `data.stargraph_train_data_path` / `_test_data_path` | `data/stargraph/graph_5_5_sample_200000.txt` / `..._test_20000.txt` | absolute paths, **keep the `graph_5_5_…` naming** | `data/stargraph.py:187-190` parses `d` and `l` out of the filename and asserts. |
| `data.effective_batch_size` | `512` | **`512` — do not change** | 1 GPU → `device_batch_size = 512` (was 256/GPU on 2 GPUs). Doubles per-step activation memory vs the paper's script. This is the profiling question. |
| `data.gradient_accum_steps` | `1` | `1`; raise to 2 **only** if 512 does not fit | `train.py:147-153` + `core_train.py:486-499`. Preserves effective batch and optimizer-update count. Document as a deviation per spec §11. |

**Keys that must NOT change** (verified present and already paper-correct in the shipped YAML):
`trainer.train_batches: 20000` (`:6`), `val_interval: 1000` (`:12`), `test_interval: 1000` (`:13`),
`val_batches: 200` (`:7`), `test_batches: 200` (`:8`), `model.n_layer: 12` / `n_head: 6` /
`n_embd: 384` (`:33-35`), `dropout: 0.0`, `bias: false`, `mtp_horizon: 3` (`nextlat_…:42`),
`lambda_kl: 1.0` (`:41`), `lambda_mse: 1.0` (`:43`), `optimizer.optimizer_type: adam`,
`learning_rate: 5e-4`, `weight_decay: 0.1`, `beta1: 0.9`, `beta2: 0.95`, `grad_clip: 100`,
`lr_scheduler.schedule` (inherited `constant` from `defaults.yaml:132`), `warmup_iters: 0`,
`warmdown_iters: 0`, `data.stargraph_max_nodes: 100` (`:28`).

**Also required for H3 adaptation runs** (`lambda_mse: 0`, `lambda_kl: 0` per spec §6): both keys are
read straight into `NextLatConfig` at `core_train.py:91-92` and multiply the loss terms at
`model_nextlat.py:490-494`, so setting them to `0.0` cleanly reduces NextLat to next-token-only
training. Note `lambda_ce` also defaults to `0.0` (`defaults.yaml:116`) and the CE term is computed
regardless for logging (`model_nextlat.py:319-324`), costing throughput — `README.md:128` says so
explicitly.

### 4.4 Resolved model sizes (computed from the pinned code)

`n_embd=384`, `n_layer=12`, `n_head=6` (head dim 64), MLP hidden `= 128·round((8·384/3)/128) = 1024`
(`model_gpt.py:139-140`), SwiGLU gate+up so `gate_up` is `384 → 2048` (`model_base.py:843`),
`vocab_size=106`, `block_size=69`, RMSNorm (bias=False).

| | params (incl. embedding) | non-embedding |
|---|---|---|
| GPT | **21,324,672** | 21,283,968 |
| NextLat, `proj_factor=0.5` (dynamics hidden **384**) | **21,915,264** | 21,874,560 |
| NextLat, `proj_factor=1.0` (the silent fallback, hidden 768) | 22,800,000 | 22,759,296 |

The dynamics MLP at `proj_factor=0.5` is `Linear(768→384) → GELU → Linear(384→384) → GELU →
Linear(384→384)` plus `LayerNorm(768)` (`model_nextlat.py:52-67`) = 590,592 params — **three Linear
layers with hidden dimension 384**, which matches the paper's stated Path-Star dynamics model. This
confirms the spec §8 requirement ("Verify that the official NextLat YAML resolves to those values")
**only when `proj_factor=0.5` is actually in effect.**

---

## 5. Question 5 — every deviation between the pinned repo and the spec

### 5.1 Spec keys that DO exist (spec is correct)

| spec key | exists at | value |
|---|---|---|
| `trainer.train_batches: 20000` | `gpt_stargraph_5_5.yaml:6` | ✅ 20000 |
| `trainer.save_last_checkpoint: true` | `defaults.yaml:18` | ✅ default `true` |
| `trainer.save_best_checkpoint: true` | `defaults.yaml:20` | ✅ default `true` |
| `trainer.save_recovery_checkpoint: 250` | `defaults.yaml:24` | ✅ key exists, default **`-1` (disabled)** — must be set |
| `trainer.compile: false` | `defaults.yaml:40` | key exists, default `true`; 5_5 YAML sets `true` → **must override** |
| `data.dataset: stargraph` | `gpt_stargraph_5_5.yaml:21` | ✅ |
| `data.effective_batch_size: 512` | `gpt_stargraph_5_5.yaml:22` | ✅ |
| `data.stargraph_max_nodes: 100` | `gpt_stargraph_5_5.yaml:28` | ✅ |
| `model.n_layer/n_head/n_embd: 12/6/384` | `gpt_stargraph_5_5.yaml:33-35` | ✅ |
| `model.mtp_horizon: 3` | `nextlat_stargraph_5_5.yaml:42` | ✅ |
| `model.lambda_mse: 1.0`, `lambda_kl: 1.0` | `nextlat_stargraph_5_5.yaml:41,43` | ✅ |
| `model.proj_factor: 0.5` | **`nextlat_stargraph_5_5.yaml:61` — inside `sweep:` only** | ⚠️ not in `model:` |

### 5.2 Spec keys that DO NOT exist — bluntly, these are spec inventions

| spec YAML key | verdict | the real thing |
|---|---|---|
| `data.train_graphs: 200000` | **does not exist anywhere in the repo.** `grep -rn "train_graphs"` → zero hits. | `data.stargraph_train_data_path: "data/stargraph/graph_5_5_sample_200000.txt"` (`gpt_stargraph_5_5.yaml:26`). The count is encoded in the **filename**, and the file is produced by the generator's `--num_samples`. |
| `data.heldout_graphs: 20000` | **does not exist.** Zero hits. | `data.stargraph_test_data_path: "data/stargraph/graph_5_5_test_20000.txt"` (`gpt_stargraph_5_5.yaml:27`), from `--num_test_samples`. |
| `optimizer.name: AdamW` | does not exist | `optimizer.optimizer_type: adam` (`defaults.yaml:121`, `gpt_stargraph_5_5.yaml:44`, `nextlat_stargraph_5_5.yaml:45`); `"adam"` dispatches to **`torch.optim.AdamW`** (`model_base.py:70-77` → `_configure_adamw`, `model_base.py:233`). Spec's intent is met, the key name is wrong. |
| `optimizer.betas: [0.9, 0.95]` | does not exist as a list | two scalars `beta1: 0.9`, `beta2: 0.95` (`gpt_stargraph_5_5.yaml:47-48`), combined at `core_train.py:227`. |
| `optimizer.schedule: constant` | wrong section | `lr_scheduler.schedule` (`defaults.yaml:132`), read at `core_train.py:985`. Default is already `constant` → `lambda _: 1.0` (`core_train.py:992-994`). |
| `optimizer.clip_gradient_norm: 100` | does not exist | `optimizer.grad_clip: 100` (`gpt_stargraph_5_5.yaml:49`), used at `core_train.py:505-506`. |
| `sweep: - seed: [1234, 1235, 1236]` | key exists (`nextlat_stargraph_5_5.yaml:55`) | works, but see §5.4 — do not use it for confirmatory runs. |
| `data: dataset: hmm_belief` (§12) | **no such datamodule** | `DATAMODULES` (`train.py:34-42`) = tinystories, stargraph, fineweb10B, fineweb100B, finewebedu, countdown, manhattan. `train.py:176-178` asserts membership. Experiment B requires **writing a new datamodule and registering it in that dict**. |
| `data.train_sequences` / `sequence_length` (§12) | do not exist | new keys for the new datamodule. |

### 5.3 Behavioural deviations between repo and spec

1. **`compile: true` in the shipped Path-Star YAMLs contradicts the repo's own README.**
   `gpt_stargraph_5_5.yaml:17` and `nextlat_stargraph_5_5.yaml:16` set `compile: true`, while
   `README.md:117-122` says: *"We observe that `torch.compile()` produces inconsistent results on
   numerically sensitive benchmarks like Path-Star and A₅ … we recommend setting `trainer: compile:
   false`."* The spec is right; the shipped config is the bug.

2. **Node ids are `0…99`, not `1…100`** (spec §5). `prepare.py:11`.

3. **The paper/spec's "five seeds" are `1234…1238`** (`nextlat_stargraph_5_5.yaml:56`); the spec's three
   confirmatory seeds `1234,1235,1236` are the first three. Consistent.

4. **Accuracy is measured with multinomial sampling.** `data/stargraph.py:122-127` calls
   `model.generate(prefix, max_new_tokens=5, temperature=getattr(config.trainer,"temperature",1.0),
   top_k=getattr(config.trainer,"top_k",None))`. Neither key is in the 5_5 YAML, so
   `temperature=1.0, top_k=None`, and `generate` uses `torch.multinomial`
   (`model_gpt.py:555-557`, `model_nextlat.py:734-736`) — **not argmax**. The 90% competence gate is
   therefore a sampled exact-path accuracy. Our evaluator should also compute greedy accuracy and
   report both; adding `trainer.top_k: 1` to the config makes the in-training metric greedy without
   touching code.

5. **`trainer.deterministic` is dead.** Present at `defaults.yaml:38` with the comment *"Enable
   deterministic CUDA algorithms for reproducibility (slower)"*, but a repo-wide grep for
   `deterministic` / `use_deterministic_algorithms` finds **no reader**. Setting it does nothing.

6. **`data.stargraph_data_path` (`defaults.yaml:80`) is dead** — the code reads
   `stargraph_train_data_path` / `stargraph_test_data_path`, which are absent from `defaults.yaml`.

7. **The spec's "final prompt delimiter `=`" state predicts the source, not the branch.** See §2.4.
   Not a repo bug — a measurement-design issue the spec should acknowledge before freezing H1/H2.

8. **`epochs: -1`, `val_printsamples`, `test_batches`, `test_interval`, `pair_accum_steps` appear in
   the 5_5 YAMLs but not in `defaults.yaml`.** OmegaConf merge is non-struct so they are simply added.
   `test_interval` **is** read (`core_train.py:671`) and gates the accuracy eval — so it must be
   present in our config or training crashes at the first validation. `epochs` and `pair_accum_steps`
   are never read for GPT/NextLat.

9. **A completed run leaves a stale `recovery_ckpt` pointer to a deleted file**, which makes the next
   `init_from: resume` hard-fail at `core_train.py:148-150`. See §3.4.

10. **NextLat's KL term detaches `lm_head`** (`model_nextlat.py:316`:
    `lm_head_weight = self.model.lm_head.weight.detach()`) and detaches the teacher logits
    (`model_nextlat.py:327`). The MSE target is also detached (`model_nextlat.py:303`,
    `h_t_next.detach()`), and the loss is **Smooth L1**, not squared error, despite the variable being
    named `MSE`. Worth recording: spec §12 says "Smooth L1", so this matches — but the metric logged as
    `mse_loss` is a Huber loss.

11. **NextLat's backward is a two-stage manual graph split** (`model_nextlat.py:503-525`): the losses
    are computed on detached copies, then gradients are injected back into the trunk via
    `fabric.backward(combined_emb, gradient=combined_grad)`. Any hook that changes the hidden-state
    tensor in place during training would silently corrupt this. **Use read-only hooks, or extract
    offline.**

### 5.4 The `sweep:` mechanism — a reproducibility hazard

`train.py:273-339` pops the `sweep` block, expands every list-valued leaf into a Cartesian product
(`grid_to_list`, `train.py:57-84`), and runs the configs **sequentially in one process**. Two problems:

- `all_sweep_param_names` is a **`set`** (`train.py:280`) and is iterated at `train.py:322` to build
  `sweep_name`. Python randomizes `str` hashing per process unless `PYTHONHASHSEED` is fixed, so the
  **experiment directory name can differ between invocations of the same config** — which means a
  resume can start writing checkpoints into a different subdirectory than the pre-crash run (resume
  itself still works, because the pointer holds a full path).
- All sweep entries share one `out_dir`, hence one `latest_ckpt`/`recovery_ckpt` pair (§3.3).

**Do not use `sweep:` for the confirmatory matrix.** Emit one materialized YAML per (model, seed,
phase, branch) with an explicit `seed`, `experiment_name`, and `out_dir`, exactly as spec §9 requires.

---

## 6. Extra reporting items

### 6.1 Vocabulary / tokenizer construction

Covered in §1.5. Summary: a hand-written character/integer tokenizer local to
`data/stargraph.py:9-57`, **not** a HuggingFace tokenizer (`data.tokenizer_name_or_path` in
`defaults.yaml:75` is unused for stargraph). Vocabulary is `maxNodes + 5 + 1 = 106` for
`stargraph_max_nodes: 100`; ids `0-99` nodes, `100='|'`, `101='='`, `102='/'`, `103='$'`,
`104=EOS`, `105` unused. Commas are dropped by the encoder. `vocab_size` is injected into the model
config at runtime by `update_config` (`data/stargraph.py:250`).

### 6.2 `effective_batch_size` → physical batch and gradient accumulation

`train.py:139-166`:

```python
139	    # Calculate per device batch size
140	    assert (
141	        config.data.effective_batch_size % fabric.world_size == 0
142	    ), f"effective_batch_size {config.data.effective_batch_size} must be divisible by DDP world size {fabric.world_size}"
143	    config.data.device_batch_size = (
144	        config.data.effective_batch_size // fabric.world_size
145	    )
146	
147	    # Calculate micro batch size
148	    assert (
149	        config.data.device_batch_size % config.data.gradient_accum_steps == 0
150	    ), f"device_batch_size {config.data.device_batch_size} must be divisible by gradient_accum_steps {config.data.gradient_accum_steps}"
151	    config.data.micro_batch_size = (
152	        config.data.device_batch_size // config.data.gradient_accum_steps
153	    )
```

```
device_batch_size = effective_batch_size // world_size
micro_batch_size  = device_batch_size    // gradient_accum_steps
```

The **DataLoader's `batch_size` is `device_batch_size`**, not `micro_batch_size`
(`data/stargraph.py:172` → `data/stargraph.py:260`). Accumulation then **slices that already-loaded
batch** (`core_train.py:486-499`):

```python
486	                for accum_step in range(self.config.data.gradient_accum_steps):
487	                    start_idx = accum_step * self.config.data.micro_batch_size
488	                    end_idx = (accum_step + 1) * self.config.data.micro_batch_size
489	                    sub_batch = batch[start_idx:end_idx]
490	
491	                    # Only sync gradients for the last step
492	                    no_sync = accum_step < self.config.data.gradient_accum_steps - 1
493	                    losses_dict = self.model.compute_loss(
494	                        sub_batch,
...
497	                        loss_div=self.config.data.gradient_accum_steps,
498	                        no_sync=no_sync,
499	                    )
```

Consequences:
- Grad accumulation **does** reduce peak *activation* memory (each micro-batch does its own
  forward/backward), and **does not** reduce the memory of the input batch tensor (irrelevant here:
  512 × 69 int64 ≈ 283 KB).
- Loss is divided by `gradient_accum_steps` inside each `compute_loss`, so the effective gradient is a
  correct mean. One optimizer step per dataloader batch either way — the optimizer-update count is
  invariant, satisfying spec §11's requirement.
- Config → runtime for our single-GPU run: `512 / 1 = 512` per device, `/1 = 512` micro batch.
  **This is 2× the per-GPU batch of the shipped 2-GPU script (256).** That is the number the profiling
  gate must validate.
- The same accumulation loop runs in validation (`core_train.py:628-638`).

### 6.3 Seed plumbing

Single seeding site, `train.py:168-173`:

```python
168	    # Initialize PyTorch settings
169	    seed_offset = fabric.global_rank
170	    fabric.seed_everything(int(config.seed) + int(seed_offset))
171	    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
172	    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
173	    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles
```

- `config.seed` defaults to `1234` (`defaults.yaml:7`); the sweep overrides it per run.
- `fabric.seed_everything` seeds Python `random`, NumPy, torch CPU and CUDA, and sets
  `PL_GLOBAL_SEED`.
- **Rank offset:** each DDP rank gets `seed + global_rank`, so a 1-GPU run uses `seed + 0` and is
  *not* bit-identical to rank 0 of a 2-GPU run anyway (different sampler shard and different batch
  composition).
- It is called **once**, before the datamodule and the model are constructed. Everything downstream
  (weight init, `RandomSampler` permutations, `torch.multinomial` in `generate`) consumes the same
  global stream. **Nothing re-seeds on resume**, and the RNG state is not checkpointed (§3.1) — this
  is the root cause of the resume-divergence risk in §3.5 item 5.
- `torch.backends.cuda.matmul.allow_tf32 = True` (line 171) is unconditional. TF32 matmuls are
  non-deterministic in accumulation order across kernel selections; combined with the dead
  `trainer.deterministic` key (§5.3 item 5), **bit-exact reproducibility is not available out of the
  box.** The spec's "deterministic tolerance" language is the right framing; pick and record a
  tolerance rather than expecting exactness.
- The data generator's seed is completely independent (§1.1) — hard-coded `0`, no CLI flag.

### 6.4 W&B dependency — what blocks an offline Colab run

Module-level, unavoidable imports at `train.py:15,17,24`:

```python
15	import wandb
...
17	import wandb
...
24	from wandb.integration.lightning.fabric import WandbLogger
```

Also `core_train.py:9` (`import wandb`) and `train_probe.py`. `requirements.txt:5` lists `wandb`.

So: **the `wandb` package must be installed** even for a fully offline run — the import is not guarded.
(`eval/eval_checkpoints.py:26-29` does guard it with `try/except`, but that is not on our path.)

What is *runtime*-gated:
- `defaults.yaml:34` `log_to_wandb: true`, and **none of the five `config/stargraph/5_5/*.yaml`
  override it** — so a naive run of the shipped config *will* construct a `WandbLogger`
  (`train.py:110-123`) and attempt to `wandb.init` on the first `fabric.log_dict`.
- `train.py:196-206` calls `wandb.login(...)` only if `WANDB_API_KEY` is in the environment.
- `core_train.py:373-408` uploads the final checkpoint as a W&B artifact and calls `wandb.finish()` —
  gated on `log_to_wandb`.

**Two clean ways to go offline:**
1. `trainer.log_to_wandb=false` as a CLI override (dotlist, `train.py:265,349`). This disables the
   logger, the login, and the artifact upload. CSV logging still works
   (`log_to_file: true`, `defaults.yaml:33` → `CSVLogger` at `train.py:102-109`, writing
   `{out_dir}/{experiment_name}/version_N/metrics.csv`).
2. `export WANDB_MODE=offline` — keeps the logger but writes to a local run directory. Useful if you
   want the W&B schema without network. Note the artifact upload at `core_train.py:398-402` will then
   also be local-only.

**Recommended:** option 1 for confirmatory runs (metrics go to CSV, which the runner parses into
`results/metrics/step_{step}.json` per spec §9.3), plus `pip install wandb` to satisfy the import.

Other network dependencies on our path: **none**. `requirements.txt` pulls `lm_eval`, `trl`,
`datasets==4.6.1`, `boto3`, `osmnx`, `folium`, `selenium`, `gdown` — all for other benchmarks.
`train.py` imports `data.tinystories`, `data.fineweb`, `data.countdown`, `data.manhattan_dataset`
(`train.py:28-32`) at module level, so those modules must be **importable** (they import `datasets`,
`transformers`, `networkx`, etc.) even for a stargraph run. Budget for a full
`pip install -r requirements.txt` in the Colab image, or vendor a trimmed import path.

`README.md:40`: **PyTorch ≥ 2.6 is required** (`torch.distributed.fsdp` API used at
`model_nextlat.py:7`); ≥ 2.9 only for Muon, which we do not use.

---

## 7. Direct answers to spec §16, one line each

1. **Generator:** `python data/stargraph/prepare.py --num_samples 200000 --num_test_samples 20000
   --num_paths 5 --path_length 5 --max_nodes 100 --data_dir data/stargraph --generate_test_data`.
   Default `--max_nodes` is **50** (`prepare.py:133`) — override confirmed necessary. Format is
   `e1|e2|…|e20/src,goal=p0,p1,p2,p3,p4\n` with `,` inside pairs; tokenized to 69 ids with `=`→101 at
   index 62 and one EOS (104) appended. G(5,5) config consumes it by filename
   (`gpt_stargraph_5_5.yaml:26-27`), parsing `d` and `l` out of the name
   (`data/stargraph.py:187-190`) and overwriting `vocab_size=106`, `context_length=62`,
   `block_size=69` at runtime (`data/stargraph.py:249-252`).
2. **Final post-norm state:** `model_gpt.py:276` (`x = self.transformer.norm(x)`, returned second at
   `:290-291`) and `model_nextlat.py:197` (`text_embd = self.transformer.norm(x)`, returned second at
   `:199-200` — **before** `lm_head`, so NextLat's `return_hidden_states` path yields **no logits**).
   Minimal capture: call the inner transformer with `return_hidden_states=True`, or hook
   `model.model.transformer.norm`. Zero training-code changes either way.
3. **Checkpoints:** written by `model_base.py:404-417` to
   `{out_dir}/{experiment_name}/ckpt_iter_{step}_{loss}.pt` and
   `{out_dir}/{experiment_name}/recovery_ckpt_iter_{step}.pt`; pointers `latest_ckpt` and
   `recovery_ckpt` are plain text files at **`{out_dir}`** (`core_train.py:945,971`). Payload is
   `{model, optimizer, training_steps, lr_scheduler_state}` — **no RNG, no data position, no epoch**.
   `init_from: resume` (`core_train.py:139-172`) prefers `recovery_ckpt` over `latest_ckpt`
   unconditionally, then replays the dataloader `training_steps` times to fast-forward
   (`core_train.py:432-452`).
4. **Single GPU:** `fabric run --devices 1 --precision bf16-mixed train.py --config <cfg>`
   (recommended: add `--strategy ddp` for a deterministic sampler). Must change:
   `compile→false`, unique absolute `out_dir`, explicit `experiment_name` + `seed`, delete `sweep:`
   **and hoist `proj_factor: 0.5` into `model:`**, `save_recovery_checkpoint: 250`,
   `log_to_wandb: false`, absolute data paths. `effective_batch_size` stays 512 but becomes 512/GPU.
5. **Deviations:** see §5. Headline: `train_graphs`/`heldout_graphs` are spec inventions;
   `save_*_checkpoint` are real; `compile:true` in the shipped YAML contradicts the repo's own README;
   `proj_factor: 0.5` is sweep-only and silently reverts to 1.0; four optimizer key names in the spec
   are wrong; `hmm_belief` requires a new datamodule registered in `train.py:34-42`; the `=` state
   predicts the source, not the branch.
