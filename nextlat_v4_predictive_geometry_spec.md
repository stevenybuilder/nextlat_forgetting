# NextLat × Predictive Geometry

## Weekend research specification: Lure-Star + HMM

**Primary paper:** [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/html/2511.05963)  
**Official code:** [JaydenTeoh/NextLat](https://github.com/JaydenTeoh/NextLat)

> **Binding pre-outcome amendment (2026-08-24):**
> `docs/PREREGISTRATION_AMENDMENT_2026-08-24.md` supersedes any conflicting analysis language in
> this document. In particular, centered cosine and held-out whitened Mahalanobis are co-primary;
> H3's prospectively frozen D40 feasibility rule failed for 4/5,000 pairs and permanently withdrew
> Lure-Star H3; no adaptation branch or interference/mechanism endpoint is confirmatory. The surviving program is H1/H2 plus a
> three-regime HMM calibration with one regime-aggregated primary endpoint. Confirmatory training
> is blocked until the amendment's revised pre-compute freeze gate passes. The later, separately
> numbered CFS-2 study is governed by `docs/DECISION_CFS2_STIMULUS_REPAIR.md`; its three-site
> activation-patching sweep is mandatory and is not the optional Lure-Star patching described below.

## Instructions for Claude

Treat this document as the working project specification. It is revisable: where a constraint below turns out to be wrong, impossible, or to cost more than it buys, revise it in writing with the reasoning and the compute consequence, rather than working around it silently.

Reuse the official NextLat implementation. Path-Star is the confirmatory benchmark and the HMM supplies exact ground truth. Adding further benchmarks is a budget and wall-clock decision, not a matter of principle - price it before proposing it, and say what it displaces. Manhattan was priced and declined on cost; see section 14. Two constraints are not budget decisions and hold throughout: keep the neuroscience an explicitly-labelled computational analogy rather than a biological claim, and keep every claim at the evidence rung it can actually earn.

Work in this order:

1. Pin and inspect the official repository.
2. Implement and validate the matched Lure-Star stimuli.
3. Harden checkpoint/resume and pass a forced-interruption test.
4. Profile 500 steps before scheduling the sweep.
5. Train GPT, NextLat, and BST for five seeds.
6. Run H1/H2 geometry and behavior evaluations.
7. Verify the permanent H3 block is excluded from every job and result schema.
8. Train and evaluate the required HMM belief-geometry experiment.
9. Execute CFS-2 under its separate outcome-blind protocol, including the required per-branch
   blocks-3/7/10 activation-patching sweep. The older Lure-Star patching proposal remains optional.

Record exact commits, configs, seeds, commands, GPU type, runtime, peak VRAM, checkpoint lineage, artifact hashes, results, and failures.

Publishing to a public GitHub repository is authorized, subject to no secrets, credentials, `.env` files, or service-account material leaving the machine. Renting paid compute outside the existing Colab balance, or exceeding an agreed compute budget, still needs explicit approval - price it and ask.

## 1. Executive decision

The weekend project asks two linked questions:

> **Does NextLat selectively separate similar histories when a matched change alters the correct
> future, keep equally changed but future-equivalent histories closer across two complementary
> metrics, and does that geometry predict planning behavior?**

> **When the exact Bayesian belief state is known, does NextLat geometry better respect predictive equivalence and belief divergence than standard next-token training?**

This addresses a gap left by the paper: it measures compression and predictive performance but does not study the structure of the learned representation space.

The scope, in priority order. Everything below the line competes for the same GPU budget, so the ordering is what matters:

1. **Required experiment A:** Lure-Star geometry and planning behavior (H1/H2).
2. **Required experiment B:** HMM ground-truth Bayesian belief geometry.
3. **Deferred, in order:** latent-dynamics bottleneck width; the original Lure-Star causal-patching
   proposal; Manhattan geometry (priced at 3.2x the available compute balance - see section 14).
   CFS-2 activation patching is separately required and is not included in this deferred list.
4. **Out of scope for this project, because they are different research programs rather than missing pieces:** sparse autoencoders and monosemantic feature search, attention-head and circuit reverse engineering, RL post-training, hierarchical latent architectures, speculative-decoding engineering. If one of these turns out to be the cheapest way to answer a question actually raised here, price it and make the case; the point is that none of them is on the critical path.

## 2. What the paper establishes

NextLat adds a latent transition model to next-token training:

```text
(h_t, x_{t+1}) -> p_psi -> h_hat_{t+1}
```

The transformer, output head, and latent dynamics model are jointly trained so that the hidden state predicts the next token and the transition model predicts subsequent hidden states.

Under the paper's idealized optimization assumptions, the hidden state becomes sufficient for predicting the future—a belief state. This does **not** imply a unique coordinate system, minimality, or biological equivalence.

Already established in the paper:

- near-perfect Path-Star solving under the full setup;
- stronger planning, state tracking, and length generalization;
- future-token decodability on TinyStories;
- Manhattan sequence compression of `0.71` for NextLat versus `0.65` for GPT;
- Manhattan effective latent rank of `52.7` for NextLat versus `160.1` for GPT;
- detour robustness and coherent latent rollouts;
- up to `3.3×` speculative-decoding speedup in language modeling.

Therefore, these are not standalone contributions:

- reproducing ordinary Path-Star accuracy;
- showing lower effective rank;
- PCA, t-SNE, or UMAP plots;
- saying NextLat is more compressed;
- demonstrating generic forgetting after more training.

The contribution must identify **what is separated, what is compressed together, and whether that
geometry predicts planning behavior**, then calibrate the measurement against exact HMM ground truth.

## 3. Alignment with Sharma's current research

The NextLat paper leaves learned representation structure open. Sharma's [OP-Mix](https://arxiv.org/html/2605.15220) studies acquisition and retention during continual midtraining and instruction tuning. Her [LoRA versus full fine-tuning](https://arxiv.org/abs/2410.21228) work analyzes spectral mechanisms that cause forgetting.

Interference remains aligned with Sharma's agenda, but the preregistered H3 feasibility rule failed
before confirmatory training. It is therefore motivation for a future, separately preregistered
study, not evidence or an endpoint in this project. The original novelty bar was:

> The claim cannot be “models forget.” It must be that **pre-existing pairwise geometry predicts which specific memories interfere, and that NextLat changes this relationship**.

## 4. Neuroscience transfer: precise and limited

### Pattern separation

Use this computational abstraction:

```text
similar histories + different futures -> greater separation
similar histories + same future       -> relative invariance
```

### Population/state-space geometry

Analyze the full hidden vector rather than searching for one neuron or attention head that “contains” the memory. This is a methodological analogy to population-level neuroscience, not a claim of biological homology.

### Similarity-dependent interference

Ask whether overlapping predictive states are more vulnerable to later learning. This makes geometry consequential rather than decorative.

Keep this an analogy and say so. Claims of a hippocampus, dentate gyrus, place cells, neural-manifold homology, or brain-like circuitry are not supported by anything measured here, and asserting them would be the fastest way to lose a reader who knows the neuroscience.

## 5. Core benchmark: Lure-Star

### Path-Star base task

A `G(d,l)` Path-Star graph has a source node and `d` disjoint arms of length `l-1`. The prompt contains a shuffled edge list plus source and goal; the model emits the correct source-to-goal path.

Use `G(5,5)` for the confirmatory Lure-Star experiment. Match the paper's full configuration:

- 200,000 fixed training graphs and 20,000 held-out tests;
- node IDs sampled from `1...100`;
- 20,000 training updates;
- batch size 512;
- five seeds, all confirmatory;
- a 12-layer, 6-head, 384-dimensional transformer;
- `mtp_horizon=3`, because `l-2=3`.

Hold the architecture, dataset, batch size, and training steps at paper scale for the confirmatory
Path-Star results. The recovered exact-scale A100 engineering profile supports a **training-only
planning figure of 79.651 GPU-hours including the frozen 20% interruption margin** for the
strengthened five-seed design; the arithmetic and exclusions are in section 11. This is not a
completion-time guarantee or scientific result. The strengthened evaluation path has not yet been
profiled, so its delta remains explicitly unmeasured. Where a benchmark cannot be afforded at
paper scale, either run it at a stated reduced scale behind a competence gate and label every
number with the scale it was produced at, or decline it on cost and record the price. Do not fake
parity. The paper reports five seeds, and this extension preregisters five confirmatory seeds; the
count matches, while the paper prose does not identify the seed IDs. A tiny run may be used solely
to test the data pipeline, checkpoint recovery, and metrics; it cannot contribute scientific
results.

### Exact matched stimuli

Represent each arm as a node sequence beginning at the common source. A suffix swap at depth `k` changes:

```text
(u -> a), (v -> b)
```

to:

```text
(u -> b), (v -> a)
```

Only two serialized endpoint tokens change.

For every base graph, generate:

| Condition | Construction | Correct future | Purpose |
|---|---|---|---|
| Repeat | Reshuffle edge-list order only | Same path | Serialization nuisance control |
| Near-safe | Swap equal-depth suffixes between two distractor arms | Same path | Matched future-irrelevant perturbation |
| Near-critical | Swap equal-depth suffixes between the goal arm and a distractor arm; source and goal stay fixed | Different first branch/path | Matched future-relevant perturbation |
| Far-critical | Repartition/reorder the same node multiset into a low-edge-overlap valid graph | Different path | Distance control for adaptation |

Near-safe and near-critical must match exactly on:

- two edited endpoint tokens;
- serialized edge positions;
- token edit distance;
- node multiset and frequency;
- degree sequence;
- prompt and answer length;
- source and goal;
- graph validity.

A graph solver computes and validates every answer.

### Data pools

Create two disjoint pools before training:

- `E_lure`: held-out quartets used only for H1/H2 behavior and geometry;
- `A_pair/B_adapt`: archived H3 feasibility inputs; they are not confirmatory training or
  inference inputs after the permanent D40 block.

The archived H3 design would have used the same `A_pair` items for every model and seed. They must
not now be used to construct a replacement confirmatory H3. No `E_lure` graph or lure may enter
base training.

### Generator acceptance tests

For at least 1,000 quartets, verify automatically:

- base, repeat, and near-safe answers are identical;
- near-critical changes the first branch and full path;
- both near lures change exactly two endpoint tokens at matched positions;
- source, goal, node multiset, degree sequence, prompt length, and answer length are preserved;
- every graph is valid and solver-verified;
- evaluation items are absent from training;
- generation is deterministic under a recorded seed.

If exact matching fails, stop and fix the generator. A stimulus-design error cannot be regressed away afterward: the covariate you would adjust for is the thing the design was supposed to hold fixed, so the adjustment is unverifiable.

## 6. Preregistered hypotheses

### H1 — future-sensitive separation

Extract the state at the final prompt delimiter `=`, before the first answer token.

```text
PSI = d(h_base, h_near-critical) - d(h_base, h_near-safe)
```

Co-primary distances: centered cosine and held-out whitened Mahalanobis distance. The 2,000
`E_lure` quartets are split deterministically by canonical-base SHA-256 into 400 whitener-only
quartets and 1,600 scored quartets. Both co-primary metrics score the identical 1,600 items. Also
report the scale diagnostic

```text
nPSI = 2 sum(d_critical - d_safe) / sum(d_critical + d_safe),
```

which cannot rescue a failed co-primary result.

Prediction:

```text
PSI_NextLat > PSI_GPT
```

The claim is selective separation, not global representational expansion. The primary paired-seed
contrast is NextLat minus BST. Metric-robust confirmatory support requires a positive mean and a
two-sided 95% paired-seed interval wholly above zero for **both** co-primary metrics. Positive means
with intervals crossing zero are directionally consistent but unresolved; disagreement is
metric-dependent evidence. Report all seeds, the exact sign-flip p-value and attainable floor, MDE,
and leave-one-seed-out sensitivity. See the binding amendment §2 for the complete decision rule.

### H2 — geometry predicts planning behavior

Test whether greater base-to-critical distance predicts:

- correct first-branch logit margin;
- first-branch accuracy;
- exact-path accuracy.

For each co-primary metric, compare the nested held-out models:

```text
M0: critical_correct_branch_margin ~ base_correct_branch_margin
M1: critical_correct_branch_margin
      ~ base_correct_branch_margin + base_critical_distance
```

Use the outcome-blind hash-fixed two-fold split from the amendment. Report held-out `R²` for both,
`Delta R²`, Spearman correlation, and coefficient direction in every fold. Geometry adds predictive
value only if `Delta R² > 0` and the distance coefficient is positive in both folds under both
co-primary metrics. Margin is primary because accuracy may be near ceiling. H2 is predictive, not
causal.

### H3 — similarity-dependent interference (permanently withdrawn)

**Terminal disposition, 2026-08-24:** D40 applied the single permitted, model-blind expansion
exactly once. Its unchanged selector left 4/5,000 pairs unmatched and created
`manifests/h3_selected/PERMANENT_H3_BLOCK.json` (SHA-256 `82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778`).
Under the prospectively frozen rule below, Lure-Star H3 has zero confirmatory adaptation jobs and zero
inferential endpoints. The remainder of this subsection is retained only as the preregistered
design record; it must not be executed, relaxed, or presented as a null H3 result. The observed
fact is nuisance-bank infeasibility, not an interference outcome.

Here and in the amendment, “H3” in the permanent block means only the **Lure-Star adaptation and
interference estimand**, including its gradient/update/Jacobian mechanism probes. The HMM
posterior-decoding and future-distribution diagnostics remain in scope. Their existing result keys
`h3_posterior_*` and `h3_future_*` are legacy schema names, not references to Lure-Star H3; keep
those names stable before outcomes.

For every frozen base checkpoint, construct three branches from the exact same parent checkpoint:

```text
near branch: adapt on B_near
mid branch:  adapt on B_mid
far branch:  adapt on B_far
```

Use full-parameter **next-token-only adaptation for GPT, NextLat, and BST**. Set `lambda_mse=0`
and `lambda_kl=0` for NextLat. For BST, disable its dense prefix-suffix objective and use
teacher-forced cross-entropy through the generation-time forward-encoder/TextHead path with the
item-independent lone-EOS backward state. The current ordinary `use_bst=true` training loss is not
an admissible H3 adaptation path. This common adaptation estimand isolates the base representation
from ongoing auxiliary-objective differences.

Match near/mid/far branches for:

- adaptation examples and updates;
- initial loss quantiles;
- target-path distribution;
- paired item order;
- optimizer and scheduler state;
- learning rate and batch size.

Generate the candidate banks before confirmatory training. Decision D39 freezes the sole
nonconfirmatory nuisance-selection pilot to the already-computed BST seed-1234 checkpoint at exact
step 500 (checkpoint SHA-256 `1f5f00611e33…`); it is never a scientific replicate and cannot be
replaced by a confirmatory checkpoint. `B_far` uses the exact loss-rank mapping `3r+1` after sorting
the 5,000 near and 15,000 far candidates by `(pilot loss, prompt SHA-256)`. A `B_mid` candidate is
eligible only in the paired near item's pilot-loss decile, within absolute loss caliper 0.1, with
the same target-path length and a solver pass; selection minimizes distance to the global median
structural disagreement among all eligible candidates, with ascending SHA-256 as tie-break. Each
independent acquisition bank contains exactly 200 items per pilot-loss decile. Full 53,000-row
score coverage is mandatory, and infeasibility blocks H3 rather than permitting threshold tuning.
The frozen D39 pilot found the original 15,000-row middle pool infeasible for 1,115/5,000 pairs.
Prospective Decision D40 therefore permits one and only one model-blind support expansion: retain
the original three candidates and add nine unique candidates in each 1/2/3-rewire stratum, for
exactly 30 per near item and 150,000 total. The checkpoint, scientific loss, deciles, absolute
caliper 0.1, path/solver constraints, global eligible structural median, and SHA tie-break do not
change. All 5,000 must match after this expansion or H3 is permanently blocked with no further
matching amendment.
The one-shot expansion was scored under job SHA-256
`393c933e9e616cd24a4b7a9b408203b0c22002c39cf97f2d72b03176fe45482a`; its new-row loss SHA-256 is
`f84c73b81d7b9e8cab44e32d89cd272d320d420583bd7badf76f3c0dade7f537`, durable-state SHA-256 is
`e1ed1d814ea190b1602c31ef82bee86bcd0937dc26dec5963a31d692f8faa0c2`, and state-last generation
is `1787563059069047`. The create-only provenance and exact remote checkpoint generations are in
`docs/DECISION_D39_h3_pilot.md` and `manifests/h3_precompute/`. Freeze all mappings for every
confirmatory arm and seed. These are structural labels: if realized hidden distances do not order
near < mid < far, retain the branches, report the failed manipulation check, and make no distance
dose-response claim. Choosing controls after inspecting any final model is prohibited.

Primary outcome:

```text
erosion_near = original correct-branch margin before - after near adaptation
erosion_far  = original correct-branch margin before - after far adaptation

similarity-dependent interference = erosion_near - erosion_far
```

Also report:

1. `A_pair` cross-entropy increase and exact-path retention.
2. Acquisition on adaptation examples and independent near/mid/far validation sets.
3. Performance change on an untouched base control set.
4. Cosine drift of original `A_pair` states.
5. Whether smaller pre-adaptation distance predicts greater margin erosion after controlling for
   initial margin, lure loss, acquisition, gradient dot product/cosine, and hidden-state
   parameter-Jacobian overlap.

Before adaptation, freeze for each item/branch the full-parameter gradient dot product and cosine
against the exact first effective adaptation batch. On a disposable shadow copy, also compute the
actual first AdamW update (including clipping, moments, weight decay, and schedule) and its
first-order projection onto margin erosion. Estimate the index-63 hidden-state Jacobian overlap with
16 shared Rademacher output projections, seed `20260824`, after validating the sketch against exact
tiny-model Jacobians.

The primary incremental analysis uses outcome-blind five-fold cross-fitting. Its baseline includes
parent, branch, initial margin, lure loss, acquisition, gradient-dot/cosine, actual-update
projection, and Jacobian overlap;
the augmented model adds distance and a quadratic distance term. The quadratic is always present,
so intermediate/nonmonotonic behavior cannot be discovered by post-hoc binning. Report held-out
`Delta R²`, the continuous curve over the 5th–95th percentile, and any interior turning point.
Metric-robust incremental support requires positive held-out improvement and the predicted
closer-means-more-erosion slope across both co-primary metrics. See amendment §4.

Compute near-minus-far within each parent checkpoint, report the preregistered mid branch, bootstrap
paired items for conditional 95% intervals, report every seed, and compare models only after
aggregating within paired seeds. Items do not substitute for independent training seeds.

Path-Star has one consistent algorithm, so valid adaptation may produce little forgetting. Creating contradictory labels or reaching for an extreme learning rate would manufacture the interference rather than measure it. A clean null is a publishable outcome here and is reported as one.

## 7. Representation endpoint and original optional Lure-Star causal test

### Primary state

Use the **final post-normalization hidden state** returned by the official transformer. NextLat directly applies its latent objective to final-layer hidden states. Intermediate-layer trajectories are descriptive only.

### Causal stretch goal

Patch the penultimate-layer final-prompt state from a near-critical lure into its base prompt, then propagate through the remaining layer and output head.

Measure the change in correct-branch log odds. Controls:

- near-safe state patch;
- random-graph patch;
- norm-matched patch.

Patching the final pre-logit state and calling the resulting output change a discovered circuit would be circular - the output head consumes that state directly, so the effect is guaranteed by construction.

Unless this intervention and all controls are completed successfully, the writeup uses only
predictive/associational language for geometry (`predicts`, `tracks`, `is associated with`). Even a
successful patch licenses only a local causal statement about the patched activation at that site;
it does not show that global distance mediates retention or that the NextLat objective alone caused
the geometry.

This subsection applies only to the original Lure-Star representation study. The later CFS-2
protocol freezes a different intervention—matching parent-to-adapted states at blocks 3, 7, and 10
on the retention probes—as a required endpoint. See
`docs/DECISION_CFS2_STIMULUS_REPAIR.md`; completion of core H1/H2/HMM is not a prerequisite for
executing that already-committed CFS-2 endpoint once its branch checkpoints exist.

## 8. Models and training

### Conditions

Three width/depth-matched arms, all 12-layer / 6-head / 384-dim on `G(5,5)`:

1. **GPT** - the official repository's implementation with standard next-token training.
2. **NextLat** - the official NextLat objective.
3. **BST** - the official Belief State Transformer objective (`use_bst: true`,
   `bst_pair_minimum_gap: 2`; the config differs from the GPT config by those two keys alone).

BST is the **competence-matched control**. It is not parameter- or training-architecture-matched:
the objective adds a second transformer, substantially more parameters, and O(T²) prefix-suffix
gradient signals. The paper's digitized Figure 6 puts GPT on `G(5,5)` at ~18.6%, which is 1/d
chance, so a
GPT-versus-NextLat geometry difference admits a trivial reading: NextLat organises the space
because NextLat solved the task. BST solves `G(5,5)` at ~99.9% *without* a latent-transition
objective. A PSI gap between NextLat and BST is not competence-confounded, but it still mixes the
objective with BST's extra training architecture, parameter count, and gradient structure. Report
that limitation with the contrast rather than claiming objective-only identification.

In the recovered A100 profile projection, the five BST base jobs contribute 59.8729 GPU-hours
before the global interruption margin and dominate the training budget. Do not substitute an
unmeasured optimized-BST estimate. Preregistered contrasts, in priority order: NextLat vs BST
(competence-matched, primary), NextLat vs GPT (secondary, confounded by competence and reported as
such), BST vs GPT (shows how much of any effect is competence alone).

### Configuration authority

At the pinned repository commit, copy the official Path-Star `G(5,5)` GPT and NextLat YAML configurations. Reconstructing an approximate configuration from this document silently drops keys the trainer requires - this already cost one failed run in this project, where a hand-written config died on a missing `test_generalization` key. Permissible changes are limited to:

- the five preregistered confirmatory seeds;
- output paths and experiment names;
- additional checkpoint/recovery frequency;
- model-output hooks needed to save hidden states;
- paths for the immutable lure and adaptation manifests.

Hold model width/depth, optimizer, learning-rate schedule, loss coefficients, effective batch size, base dataset size, and base training steps fixed across the confirmatory runs. The reason is specific rather than ceremonial: this is a preregistered comparison, so a parameter tuned after seeing a result stops being a measurement. Changing one is permitted when it is written down in advance with its justification, applied identically to both models, and reported as a deviation. The expected paper-scale values, which the materialized YAML must verify before launch, are:

```yaml
trainer:
  train_batches: 20000
  save_last_checkpoint: true
  save_best_checkpoint: true
  save_recovery_checkpoint: 250
  compile: false
data:
  dataset: stargraph
  train_graphs: 200000
  heldout_graphs: 20000
  effective_batch_size: 512
  stargraph_max_nodes: 100
model:
  n_layer: 12
  n_head: 6
  n_embd: 384
  mtp_horizon: 3             # NextLat only
  lambda_mse: 1.0            # NextLat only
  lambda_kl: 1.0             # NextLat only
  proj_factor: 0.5           # NextLat only
optimizer:
  name: AdamW
  learning_rate: 5.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  schedule: constant
  clip_gradient_norm: 100
sweep:
  - seed: [1234, 1235, 1236, 1237, 1238]
```

**All five extension seeds are confirmatory** (decided 2026-08-23, before any model was trained;
the paper reports five seeds but does not name their IDs in the prose). Nothing here is being
chosen after seeing a result. Seeds are the inferential unit for every
cross-model contrast, so under the former three-seed plan the headline comparison would have rested
on three numbers against three, far weaker than the item-level analyses backed by 2,000
quartets. Five seeds improve the seed-level design on the one comparison the central claim depends
on. Their cost is included in section 11's recovered 15-base-job projection; no obsolete
marginal-cost estimate is decision authority. Full parity with the paper also removes an obvious
reviewer objection before it is raised.

The paper reports a three-layer latent-dynamics MLP with hidden dimension 384 for Path-Star. Verify that the official NextLat YAML resolves to those values and save the fully materialized configuration with every run.

Generate the paper's 200,000 fixed base-training graphs and 20,000 held-out graphs with `max_nodes=100`. The original generator CLI defaults to 50 nodes, so override it explicitly.

The withdrawn H3 design specified 5,000 adaptation items per near/mid/far branch and 500 updates.
Those values are retained for provenance, not as authorization to run a branch.

**Executed pre-compute stop (2026-08-24, outcome-blind):** D39 explicitly designated the sole
nonconfirmatory pilot as the A100 BST profile checkpoint at exact step 500, seed 1234, SHA-256
`1f5f00611e33ada0ac0a778f9d45bef9e174f1bbeedfaaa3491018a9bf400176`. Its complete 53,000-row
score exposed model-overlap infeasibility for 1,115 middle pairs. D40 prospectively froze the
single 150,000-row support expansion described above before scoring any new row. The 135,000 new
rows were scored completely, and the unchanged rule left 4/5,000 pairs unmatched. The permanent
H3 block is now terminal. Pilot substitution, another matching amendment, H3 adaptation, and H3
inference are prohibited.

Use the same fixed 20,000-step base schedule for GPT and NextLat.

The 90% exact-path competence gate applies to **NextLat and BST**. The paper's own Figure 6 puts GPT on `G(5,5)` at roughly 18.6%, which is 1/d chance: GPT failing Path-Star is the paper's headline result, not a bug to debug. Applying the gate to GPT would halt the project on a correct run. GPT exact-path accuracy is still evaluated, hash-bound, and reported; it is simply exempt from the halt threshold. A GPT run materially *above* chance is the outcome that needs investigating. See `docs/DECISION_D20_competence_gate.md` for what this costs the cross-model contrast and how BST repairs it.

Before H1/H2 evaluation, every base parent must be scientifically evaluated (`DONE`, not merely
`TRAINED`). Its immutable competence receipt must bind model, seed, final-checkpoint path
and SHA-256, evaluator and raw-output SHA-256, held-out dataset and manifest SHA-256, deterministic
greedy decoding (`top_k=1`, `temperature=0`), and integer correct/total exact-path counts; the receipt and
its SHA sidecar must themselves match the append-only ledger artifact hashes. Missing, mismatched,
or tampered receipts are refusals. This gate is executed independently for all five confirmatory
seeds. No adaptation plan may be emitted.

If NextLat or BST misses its gate, debug data, configuration, numerical precision, and repository parity. Rescuing the run by changing architecture, losses, step count, or graph topology would convert a failed replication into an unfalsifiable one - the configuration is the hypothesis here.

Set `compile:false`; the official README reports inconsistent Path-Star/A5 results with `torch.compile`, especially on Hopper GPUs.

Single-GPU mixed-precision launch:

```bash
fabric run --devices 1 --precision bf16-mixed train.py --config <config.yaml>
```

Use `16-mixed` only when the assigned GPU lacks stable BF16 support. Run a 500-step profile before the sweep.

Leave MTP, LoRA, OP-Mix, replay, memory modules, NextLat-depth sweeps, and every adaptation follow-up
out of the confirmatory path. Any interference experiment is now a future, separately
preregistered project; it cannot reuse D40 as a route around the permanent H3 stop.

## 9. Colab interruption and recovery contract

Assume the GPU runtime will disconnect. No required artifact may exist only under `/content` or notebook memory.

### Durable layout

```text
MyDrive/lurestar/
  source_snapshot/
  manifests/
  runs/{model}/{seed}/{phase}/{condition}/
  results/
  run_ledger.json
```

Use deterministic job IDs such as `nextlat-s1234-base`. Give every base and HMM job a separate
output root; the official resume pointer lives at the output-root level, and a pointer that crosses
jobs would silently corrupt lineage without any visible error. H3 job IDs are forbidden.

Before training, persist:

- pinned upstream commit;
- source archive or Git bundle plus uncommitted diff;
- resolved config and exact command;
- environment package list, PyTorch/CUDA versions, and GPU name;
- immutable dataset/lure JSONL manifests and SHA-256 hashes.

Regenerating stimuli during a resume would break the one guarantee the manifests exist to provide, namely that every seed and every branch saw byte-identical items.

### Atomic checkpoints

The official checkpoint includes model, optimizer, scheduler, and training step and supports `init_from: resume`. Extend it minimally:

1. Save every 250 steps initially, then choose an interval representing at most 10 minutes of work.
2. Save to `.partial`, flush, then atomically rename.
3. Update the recovery pointer using the same temporary-file pattern.
4. Keep two verified recovery checkpoints; delete the oldest only after loading and hashing the newest.
5. On a catchable interruption, attempt an emergency checkpoint and persist the traceback.
6. If Drive writes fail, save under `/content/lurestar_emergency/{run_id}`, mark `NEEDS_SYNC`, and retry with bounded backoff. Optional W&B artifacts are a secondary copy, never the sole record.

### Idempotent runner

`scripts/run_matrix.py` must:

1. Read `run_ledger.json`.
2. Skip `DONE` jobs only when hashes verify.
3. Resume incomplete jobs from the newest valid checkpoint; roll back one if corrupt.
4. Preserve config, seed, manifest, and output root.
5. Write atomic `metrics/step_{step}.json` files keyed by `(run_id, step)`.
6. Mark `DONE` only after final evaluation artifacts exist and verify.

Near and far branches must store the same `parent_checkpoint_sha256`.

### Mandatory recovery test

- Run 300 steps uninterrupted.
- Separately run 150 steps, terminate, resume, and finish at 300.
- Verify step, optimizer/scheduler state, data position, metrics, and final logits/weights within the chosen deterministic tolerance.
- If the trajectories materially diverge, add Python, NumPy, CPU, and CUDA RNG states to the checkpoint.

## 10. Evaluation and success criteria

### Behavioral

- near-critical first-branch accuracy;
- exact-path accuracy by lure condition;
- safe-lure invariance;
- repeat/serialization consistency;
- correct-branch logit margin;

### Geometry

- final-state PSI under centered cosine and held-out whitened Mahalanobis on identical scored items,
  with mandatory nuisance-normalized PSI and paired 95% bootstrap intervals;
- cross-model PSI contrasts across five seeds;
- distance–margin relationship on held-out items;
- HMM predictive-equivalence pair distance;
- hidden-distance versus exact posterior JS divergence;
- held-out posterior/future-distribution decoding and length-64 generalization.

### Novelty threshold

A strong result requires:

1. exactly matched safe/critical lures;
2. selective NextLat geometry relative to competence-matched BST under both co-primary metrics;
3. geometry predicting planning beyond baseline confidence under both co-primary metrics;
4. replication across five seeds;
5. three HMM regimes, thresholds, pair selection, and `TE` certificates frozen without inspecting
   model representations;
6. the preregistered equal-regime HMM primary aggregate passes under both co-primary metrics.

The strongest defensible claim is:

> **NextLat does more than lower global representational rank: it preferentially organizes hidden
> states around distinctions that matter for future prediction, that geometry predicts planning
> behavior, and it better respects exact predictive-state relationships across controlled HMMs.**

Four claims this design cannot support, whatever the numbers say: biological pattern separation, minimal belief states, a unique manifold, and a solution to catastrophic forgetting. The first is out of domain, the second and third are ruled out by the non-uniqueness of sufficient statistics, and the fourth is a scope claim this five-seed study still cannot earn.

### Stop conditions

Stop and document the result if:

- safe and critical lures cannot be exactly matched;
- interrupted training cannot resume reproducibly enough for the stated analysis;
- NextLat or BST remains below 90% exact-path accuracy; GPT is the preregistered reported chance arm;
- the permanent H3 block receipt is missing, altered, or not enforced against job/result schemas;
- a geometry effect exists only under one co-primary metric, a post-hoc metric, layer, binning, or
  HMM regime; report it as metric/regime-dependent rather than confirmatory support;
- the result depends on one seed;
- the available hardware cannot run the exact paper-scale configuration reliably. In that case, pause for a compute decision rather than shrinking the confirmatory model.

## 11. Hardware, compute, and schedule

### Hardware

- Train on one NVIDIA GPU.
- Use CPU for generation, solver checks, bootstrapping, and plots.
- Keep the confirmatory path on CUDA. The official implementation is PyTorch/Lightning, the paper's reported runs used NVIDIA GPUs, and moving to TPU would introduce a PyTorch/XLA or JAX port as an additional experimental variable.
- Prefer an A100 40/80 GB, L40S, H100, A6000, or another reliable 24+ GB NVIDIA GPU. The paper reports runs on RTX A5000, H100 NVL, and B200 GPUs.
- Treat free-Colab T4/L4 allocations as smoke-test or fallback resources unless the exact configuration passes memory and throughput profiling. Reducing model scale to fit a runtime is a scientific decision, not an operational one: if it happens, it is stated in the results table next to every number it produced.

TPU is not the preferred Colab accelerator for this project. TPU availability does not remove Colab runtime limits, and the exact paper-scale Path-Star model fits on one suitable GPU without model or optimizer sharding. Do not spend the weekend reimplementing NextLat in JAX or validating PyTorch/XLA parity. Reconsider TPU only as a separate engineering follow-up after the CUDA experiment is complete.

### How to use the JAX Scaling Book

Use [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) as a conceptual profiling and budgeting reference, not as a code dependency:

- use the roofline framework to distinguish compute, HBM-bandwidth, host-input, and checkpoint-I/O bottlenecks;
- use the transformer parameter/FLOP and training-memory accounting to sanity-check measured runtime and peak VRAM;
- use the GPU chapter to reason about arithmetic intensity, batch size, and utilization;
- use the profiling workflow's principle of measuring warm, steady-state execution rather than extrapolating from startup or compilation.

Use `jax-ml/scaling-book` as a reference, not a dependency - there is no reason to fork or import it. Its repository contains the textbook source and embedded worked problems/quizzes, not a drop-in trainer for NextLat. Its JAX/XLA, TPU topology, FSDP, tensor-parallel, pipeline-parallel, multi-pod, LLaMA-70B, and large-scale inference exercises are outside scope. Our roughly tens-of-millions-parameter Path-Star transformer fits on one accelerator, so sharding would add communication and implementation risk without addressing a binding constraint.

### Profiling gate

Profile Lure-Star GPT/NextLat/BST for 500 steps and HMM GPT/NextLat for 300 steps. For the 500-step Lure-Star profile, treat the first 100 steps as warmup and summarize the final 400. Record:

- median and p95 seconds per step;
- examples and tokens per second;
- peak allocated and reserved VRAM;
- GPU utilization and host-input wait;
- checkpoint-write duration and bytes;
- GPT versus NextLat throughput and memory overhead;
- validation accuracy and projected end-to-end runtime.

Profile the paper's physical batch first. If it does not fit, test gradient accumulation only as an execution fallback while preserving effective batch size 512 and optimizer-update count; document that deviation. Do not change width, depth, sequence construction, precision, loss, or training examples to improve throughput. Estimate standard-GPT training FLOPs from the materialized parameter count and processed tokens, but use measured throughput for NextLat because the latent-dynamics and KL passes add work beyond the usual transformer approximation.

Calculate the compute budget directly from measured exact-scale profiles. For the strengthened
design the job-count identities are:

```text
base GPU-hours = sum_model(seconds_per_step_model * 20,000 * 5 seeds / 3,600)
HMM GPU-hours = sum_model(seconds_per_step_model * 3,000 * 15 cells/model / 3,600)
```

The count surface is therefore 15 base jobs, zero adaptation branches, and exactly 30 HMM cells.
Add measured checkpoint overhead and a 20% interruption margin.

`results/profile_summary_recovered.json` is the current nonconfirmatory A100 engineering evidence.
Its SHA-256 is `ccccea44a5f4ce321b21827a2c8276ac23d906fd538c2e9893f78696b72d0d1a`.
The recovery receipt states that no model was retrained and no paid compute was used to reconstruct
the summary. Its pre-amendment projection contains:

```text
15 base jobs                                      66.0096047653 GPU-h
30 near/far adaptation branches                    3.3215172269 GPU-h
10 one-regime HMM cells                            0.1219404929 GPU-h
```

Holding the measured per-arm timing and checkpoint treatment fixed, removing all adaptation
branches, and scaling the HMM family from one to three regimes gives:

```text
66.0096047653 + 0.1219404929 * (30/10)
    = 66.3754262440 GPU-h subtotal
with 20% interruption margin = 79.6505114928 GPU-h
at the measured 5.3 CU/h     = 422.147710912 CU
```

Round only for planning: **79.651 GPU-hours / 422.148 CU**. This excludes an unknown
strengthened-design evaluation delta: dual-metric extraction and whitening; H1/H2 cross-fitting;
and complete three-regime evaluation/cache work. Target-profile that path before launch and record
the delta, but do not change endpoints, thresholds, regimes, or seeds in response. The BST base
arm dominates the recovered projection; an optimized BST implementation may replace this budget
input only after exact parity tests and a new measured profile. If the result exceeds one weekend,
preserve the exact model and continue across resumable sessions; reduce optional analyses before
reducing scientific fidelity. Renting compute or exceeding an agreed budget requires explicit
approval.

### Friday

- Pin repository and environment.
- Implement solver-verified quartet generation.
- Implement the HMM generator, exact forward algorithm, and posterior tests.
- Pass 1,000 quartet tests.
- Implement atomic recovery and runner ledger.
- Pass forced interruption/resume.
- Profile both objectives on both tasks.

### Saturday

- Train all five preregistered GPT/NextLat/BST seed triplets without inspecting outcomes to choose seeds.
- Evaluate every base checkpoint, persist its hash-bound receipt, and pass the 90% NextLat/BST competence gate before H1/H2 (GPT is reported but threshold-exempt).
- Freeze `E_lure` and extract behavior/final states.
- Train all 30 small HMM calibration runs (three frozen regimes × five preregistered seeds ×
  GPT/NextLat) and freeze every regime-specific pair bank.

### Sunday

- Compute PSI and geometry–behavior results.
- Run HMM predictive-equivalence, belief-divergence, posterior-decoding, and length-generalization analyses.
- Produce Lure-Star behavior/PSI and HMM belief-geometry figures plus one results table per task.
- Drop only the original optional Lure-Star causal patching before dropping any H1/H2/HMM
  endpoint. Do not drop the separately committed CFS-2 activation-patching sweep.

## 12. Required experiment B: HMM belief geometry

The HMM is part of the build as a **calibration of the measurement framework**, not a second
discovery surface. It provides ground truth that Path-Star cannot: exact Bayesian beliefs and exact
next-observation predictive distributions for every history. It does not identify a unique
posterior coordinate system.

### Generative process

Use three preregistered 4-state, 4-observation stationary HMM regimes with overlapping emissions.
Choose them from the existing deterministic model-blind grid under its unchanged acceptance box:

The launch verifier must read each threshold artifact's frozen matrix binding from
`thresholds.hmm_sha256`. A former top-level lookup was a verifier bug, not a scientific-schema
change. The exact 30-cell `--family --print-plan` must pass before any HMM compute.

1. the currently frozen persistent/moderate-aliasing matrix;
2. the passing candidate with lowest mean dwell time (fast-mixing/moderate-aliasing);
3. among passing candidates at or above median dwell, the candidate with greatest mean posterior
   entropy (persistent/high-aliasing).

Break ties lexicographically by the existing candidate tuple. Freeze all three into one family
manifest before training. For each regime require `rank(T @ E) = 4` and
`sigma_min(T @ E) > 0.05`; otherwise take the next candidate under the same fixed ordering and
record the rejected candidate. The model-blind generator test still requires that:

- state persistence is neither trivial nor nearly random;
- every observation has nonzero probability under at least two hidden states;
- posterior entropy spans a broad range;
- next-observation accuracy is meaningfully above chance but below determinism.

Freeze the matrices, candidate rankings, diagnostics, `T @ E` singular values, and hashes before
training. For each regime generate:

- 100,000 training sequences of length 32;
- 10,000 validation sequences of length 32;
- 10,000 length-generalization sequences of length 64.

For every prefix, compute the exact Bayesian belief state with the normalized forward algorithm:

```text
b_t(s) = P(S_t=s | X_1:t)
```

Store posterior vectors and exact next-observation distributions with the immutable evaluation manifest.

### HMM models

Train GPT and NextLat from scratch for the same five seeds in every regime:

```yaml
data:
  dataset: hmm_belief
  train_sequences: 100000
  sequence_length: 32
  effective_batch_size: 256
model:
  n_layer: 4
  n_head: 4
  n_embd: 128
  mtp_horizon: 1    # NextLat only
  lambda_mse: 1.0   # NextLat only
  lambda_kl: 0.0    # NextLat only
  proj_factor: 0.5  # NextLat only
trainer:
  train_batches: 3000
  val_interval: 300
  compile: false
sweep:
  - seed: [1234, 1235, 1236, 1237, 1238]
```

**All five extension seeds are confirmatory** (decided 2026-08-23, before any model was trained;
the paper reports five seeds but does not name their IDs in the prose). Nothing here is being
chosen after seeing a result. Seeds are the inferential unit for every
cross-model contrast, so under the former three-seed plan the headline comparison would have rested
on three numbers against three, far weaker than the item-level analyses backed by 2,000
quartets. Five seeds improve the seed-level design on the comparison the central claim depends on.
Their cost is included in section 11's recovered 30-cell HMM projection; no obsolete marginal-cost
estimate is decision authority. Full parity with the paper also removes an obvious reviewer
objection before it is raised.

Use `d=1` with Smooth L1 only because one-step transition consistency is sufficient for the paper's belief-state result, and the paper's ablation finds Smooth L1 alone strongest at `d=1` on its future-token probe. Confirm the small models learn next-observation prediction before running geometry analysis.

### Matched HMM pair bank

Construct two held-out pair types by searching exact one-step future distributions
`q_t = (b_t @ T) @ E`:

1. **Predictively equivalent pairs:** high observation-history edit distance but very low
   future-distribution Jensen–Shannon divergence.
2. **Predictively divergent near-lures:** low observation-history edit distance but high
   future-distribution Jensen–Shannon divergence.

Freeze future-divergence and edit-distance quantile thresholds from the validation pool, then apply
them unchanged to the final test pool, breaking ties by canonical pair SHA-256. Belief JS is a
mandatory secondary label but never selects the bank. This prevents selecting visually convenient
examples after seeing model states.

### HMM-H1 — predictive equivalence

Different histories with nearly identical posterior beliefs should produce relatively nearby final hidden states, especially under NextLat. Compare pair distance with a history-distance-matched control.

### HMM-H2 — relative predictive geometry

Test whether model-state distance tracks Jensen–Shannon divergence between exact one-step future
distributions `q_t = (b_t @ T) @ E` more strongly for NextLat. Report Spearman and partial Spearman
correlations plus held-out neighborhood retrieval; control for observation-history edit distance and
prefix length. Belief JS remains a mandatory secondary calibration because predictive sufficiency
does not require a unique belief-simplex coordinate system.

### HMM-C — posterior decodability (legacy result prefix `h3_`)

Fit a held-out linear probe from `h_t` to `b_t`. Evaluate posterior reconstruction, KL/JS error, calibration, and length-64 generalization. Also decode the exact next-observation distribution, which is invariant to non-unique posterior coordinates.

Crucial caveat: a sufficient predictive state is non-unique. An invertible transformation preserves predictive information while changing raw Euclidean geometry. Prioritize predictive equivalence, relative divergence, decodability, and future-distribution prediction—not literal coordinate alignment to the belief simplex.

### HMM primary aggregate and multiplicity

For each regime, seed, model, and co-primary hidden metric, compute partial Spearman correlation
between hidden distance and future-distribution JS, controlling edit distance and prefix length.
Fisher-transform the correlations, take NextLat minus GPT inside each regime/seed, then average the
three regimes equally **inside the seed**. This is the one primary HMM aggregate. It passes only if
the paired-seed 95% interval is above zero for both centered cosine and held-out whitened
Mahalanobis. Regimes are not independent replications.

Secondary regime-aggregated endpoints are predictive-equivalence contrast; future-distribution
probe JS at lengths 32 and 64; and posterior-probe JS at lengths 32 and 64. Apply Holm correction
across those five named endpoints. Report all regime/model/seed cells, exact sign-flip p-values and
their attainable floor, MDE at 80% power, leave-one-seed-out sensitivity, and every null. A null is
`not resolved at the detectable effect size`, not evidence of equivalence. No secondary endpoint or
favored regime can rescue a failed primary aggregate.

## 13. Later bottleneck-width ablation

Only after Lure-Star and the HMM show a signal, sweep `proj_factor` or the corresponding latent-dynamics hidden width while fixing the transformer.

Ask whether capacity changes:

- future-sensitive separation;
- predictive-equivalence collapse;
- HMM posterior decodability;
- Path-Star accuracy;
- effective rank.

This directly addresses the paper's stated uncertainty about the dynamics MLP bottleneck - section 6 of the paper says outright that the width 'effectively acts as a bottleneck that constrains belief-state capacity' and that they did not study it. It is deferred behind experiments A, B and C only because ablating a bottleneck's effect on a geometry you have not yet measured is premature, not because it is out of scope.

## 14. Deferred work, and why each is deferred

Nothing here is forbidden. Each is deferred because it costs GPU time or wall-clock that the
confirmatory results need first, or because it answers a question that only becomes well-posed
after those results exist. If one of them turns out to be the cheapest route to a question this
project actually raises, price it and make the case.

| Deferred | Why it waits |
|---|---|
| Bottleneck-width (`proj_factor`) sweep | Answers a stated open question in the paper's section 6. Waits until experiments A/B/C show there is a geometry to ablate. First in the queue. |
| Original Lure-Star causal patching of the penultimate state | Stretch goal in section 7. This row does not apply to the mandatory CFS-2 blocks-3/7/10 state-restoration sweep. |
| **Manhattan taxi-trajectory geometry** | **Priced and declined on cost, 2026-08-23.** The strongest generalization test available - two histories reaching the same intersection with the same heading are predictively equivalent by construction, and it is where the paper's compression and effective-rank numbers come from, so a geometry result there would speak directly to their headline claim. But the config is 48 layers over 400,000 steps and roughly 26B tokens: measured against the profiled A100 rate that is 359 GPU-hours for a single seed pair (1,902 CU, 106% of the available balance) and 1,076 GPU-hours for three seeds (5,705 CU, 319%). No checkpoints were released, so there is no cheap path. A 15%-scale variant (60k steps, one seed pair) would cost 54 GPU-hours / 285 CU / 16%, and remains the fallback if the confirmatory results justify it. |
| TinyStories geometry | The natural-language generalization step after Manhattan. The paper already ran future-token probes there, so the harness is partly built. |
| Sparse autoencoders, monosemantic features | A different question: what individual directions mean, rather than how the space is organised. Large separate program. |
| Attention-head / circuit reverse engineering, induction heads, IOI | Mechanistic rather than representational. Section 4 deliberately analyses the full population vector instead. |
| RL post-training, hierarchical latents, speculative-decoding engineering | Method development, not representation analysis. |
| Deception / sycophancy feature detection | Unrelated to predictive geometry. |

Two things that are not deferred but *disqualified as evidence*, because they cannot support the
claims being made rather than because they are off-limits: t-SNE and UMAP plots as evidence of
structure (both distort global geometry by construction, and the claim here is about distances),
and rank-only analysis (the paper already established lower effective rank; the open question is
what got compressed together).

## 15. Deliverables

```text
configs/
  gpt_lurestar.yaml
  nextlat_lurestar.yaml
  adapt_near.yaml
  adapt_mid.yaml
  adapt_far.yaml
  gpt_hmm.yaml
  nextlat_hmm.yaml
src/lurestar/
  generate.py
  validate.py
  evaluate.py
  representations.py
  durable_checkpoint.py
src/hmm_geometry/
  generate.py
  forward.py
  pair_bank.py
  evaluate.py
tests/
  test_lure_generator.py
  test_hmm_forward.py
  test_hmm_pairs.py
  test_resume.py
scripts/
  profile.sh
  run_matrix.py
manifests/
  stimuli.jsonl
  stimuli.sha256
  hmm_family.json
  hmm_eval_pairs.jsonl
results/
  metrics/
  metrics.jsonl
  compute_log.csv
  run_ledger.json
  figures/
README.md
```

Required figures:

1. Safe versus critical behavior.
2. Final-state PSI for GPT versus NextLat.
3. Representation distance versus critical-branch margin.
4. HMM hidden distance versus exact belief JS divergence.
5. HMM predictive-equivalence collapse and posterior-decoding performance.

The README must distinguish paper results from pilot results, neuroscience inspiration from evidence, and positive findings from nulls.

## 16. First instruction to execute

Inspect the pinned official repository and report:

1. the exact Path-Star generator and `G(5,5)` config;
2. the final normalized hidden-state return used by NextLat;
3. the checkpoint/resume path and pointer locations;
4. the single-GPU training command;
5. any deviation between the pinned repository and this spec.

Then implement one solver-validated equal-depth suffix-swap quartet—base, repeat, near-safe, and near-critical—without changing training code. Stop until the quartet test passes. Next, implement the 4-state HMM generator and verify the normalized forward algorithm against brute-force enumeration on short sequences before adding either training loop.

## Sources

- [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/html/2511.05963)
- [Official NextLat repository](https://github.com/JaydenTeoh/NextLat)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Scaling Book repository](https://github.com/jax-ml/scaling-book)
- [OP-Mix](https://arxiv.org/html/2605.15220)
- [LoRA versus full fine-tuning](https://arxiv.org/abs/2410.21228)
- Yassa & Stark, *Pattern separation in the hippocampus* (2011)
- Ebitz & Hayden, *The population doctrine in cognitive neuroscience* (2021)
