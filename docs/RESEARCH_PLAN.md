# Research plan

## Central question

When a transformer is trained to predict tokens, several future tokens, future summaries, or
future latent states, does that change only its output accuracy, or does it change the predictive
structure encoded inside the model and its vulnerability to later interference?

The program separates four questions that the legacy design had partially conflated. Planner
training is first because the representation and forgetting studies are interpretable only when
the parent model actually learned the generalizing Path-Star solution.

## Comparison set and affordable active scope

The literature comparison set contains seven objectives where technically valid:

1. standard next-token prediction;
2. multi-token prediction;
3. joint-token prediction;
4. the Belief State Transformer;
5. NextLat;
6. bag-of-words future-summary prediction;
7. Hierarchical Latent Prediction.

The first five have public implementations in the pinned NextLat or BST repositories. The last two
currently require paper-based ports for these tasks. Until an author implementation is available
and matched, results from those ports are independent implementations, not exact reproductions.

The **active, budget-capped phase contains only NextLat**. Its purpose is to establish a reliable
generalizing solver population and characterize the optimization transition. Objective comparisons,
BST validation, and FSP/HiLP ports begin only after that gate passes. This keeps the state-of-the-art
design ready without spending on downstream questions before the required parent population exists.

Each method first receives a native-setting validation that follows its own paper. The common
comparison then fixes data, architecture where possible, update budget, evaluation, runtime, and
seed roster. Native results answer “can we reproduce the method?”; common results answer “what
happens when objectives are compared under the same geometry?” They must never be pooled.

## 1. Reliable planner training

**One-sentence question:** Can the public NextLat procedure reliably reach the generalizing
Path-Star basin, and what distinguishes its transition from runs that remain in the shortcut basin?

### Immediate basin gate

- Common task: fixed-data Path-Star `G(5,5)` with node vocabulary size 100.
- Data: the original 200,000 training and 20,000 held-out examples.
- Model: 12 layers, width 384, 6 heads, batch 512, 20,000 optimizer updates, AdamW,
  learning rate `5e-4`, weight decay `0.1`, and no attention dropout.
- NextLat prediction horizon: 3, matching `l - 2` in the public design.
- Runtime: one frozen container and one verified RTX 4090 if its profile is within 20% of the A100
  throughput anchor.
- Population: 24 predeclared NextLat seeds with separately recorded initialization, data-order, and
  sampler seeds.
- Cohort A: the public fixed-data recipe on one GPU, `compile: false`, with a standard random
  sampler and no post-hoc restarts.
- Cohort B: run only if Cohort A fails the reliability rule; it uses a separately frozen
  structured-target batching intervention motivated by the Path-Star literature while leaving the
  corpus, objective, model, update count, and evaluator fixed.

A checkpoint reaches the **generalizing solver basis** when frozen greedy evaluation produces at
least 95% exact-path accuracy on all 20,000 held-out examples and every path-dependent interior
token position is at least 95% accurate. The threshold is a downstream usability gate, not a way
to exclude failures from the method-level result.

### Hypotheses and estimands

- Null: the probability that a fresh run reaches the generalizing solver basis is at most 0.80.
- Alternative: that probability is greater than 0.80.
- Primary estimand: unconditional solver probability for the frozen NextLat procedure and seed
  population.
- Secondary estimands: exact-path accuracy, first decision-token logit margin, updates until the
  generalizing transition, and probability of remaining in the shortcut basin.

Use an exact one-sided binomial test for the threshold claim and interval-censored survival analysis
for time to the generalizing transition. There is no between-objective test in this phase.
At `n=24`, rejecting at 23 or more successes has size 0.0331 under `p=0.80`, 0.661 power at
`p=0.95`, and 0.839 power at `p=0.97`. This design tests a high-reliability claim; it is not a
general-purpose small-effect comparison between objectives.

### Interpretation

The original five NextLat seeds remain a separate, unconditional reproduction audit. Failed seeds
stay failed. A later predeclared stream may train additional models to build a competent-parent
cohort for Studies 2 and 3, but it reports its acceptance rate and cannot replace members of the
original five or the new 24-seed basin cohort.

Passing means at least 23 of 24 seeds reach the basis. Cohort A supports a public-recipe reliability
claim; Cohort B, if needed and successful, supports only a stabilized-training claim. If neither
passes, stop the program rather than selecting successful seeds or moving to downstream studies.

After this gate, a separately budgeted objective comparison may add GPT, MTP, JTP, BST, FSP, and
HiLP, followed by `G(2,8)` and binary-tree external validity.

## 2. Future-sensitive representation

**One-sentence question:** After controlling the visible graph edit, are internal representations
more sensitive to changes in the future than to changes that preserve the future?

### Stimuli

For every base prompt, construct exact-solver-verified matched variants:

1. serialization-only: reorder edges without changing the graph or future;
2. future-preserving: change graph structure while preserving the correct path and future-token
   distribution from the measured state;
3. immediate future-changing: change the correct next decision token;
4. delayed future-changing: keep the next token fixed but change a later correct path token.

Match node multiset, token frequencies, degree sequence, prompt and answer length, edit count, edit
positions, and base difficulty as tightly as combinatorially possible. Matching diagnostics are
part of the result; unmatched pairs are not silently discarded.

### Hypotheses and estimands

- Null: after matching, future-changing edits do not increase representational distance relative
  to future-preserving edits.
- Alternative: future-changing edits produce larger distances, with delayed changes providing the
  crucial test beyond immediate next-token sensitivity.
- Primary estimand: the paired difference in held-out-whitened Mahalanobis distance at the branch
  decision state, measured in the final pre-normalization backbone residual stream.
- Robustness metric: centered cosine distance.
- Behavioral link: change in the logit margin at the first output position where the exact futures
  diverge.

Fit a hierarchical model with edit class and objective as fixed effects and graph identity and
training seed as crossed random effects. Report every trained run unconditionally and a separately
labeled competent-checkpoint analysis. For HiLP, the backbone state is primary and the
higher-level latent is diagnostic; the latter cannot substitute for a failed backbone result.

### Success rule

Support requires a positive future-changing versus future-preserving contrast on untouched test
graphs, survival of the delayed-change subset, and no comparable effect in the serialization-only
control. A correlation with behavior is secondary and does not by itself establish a future-based
geometry.

## 3. Controlled forgetting

**One-sentence question:** Does later learning cause the most forgetting when it reuses familiar
structure but assigns a conflicting future to the same decision state?

### Design

Cross two independently manipulated factors:

- low versus high structural overlap with the parent task;
- same versus conflicting future at the parent decision state.

Every adaptation stream is exactly balanced on example count, token exposure, path length,
optimizer updates, and initial new-task learning difficulty. Full-parameter fine-tuning is primary.
LoRA is a separate robustness experiment because it follows a different optimization trajectory
and is not equivalent to full fine-tuning.

The controlled-forgetting matrix is not funded in the first tranche. If the planner and geometry
results justify it, begin with 16 competent NextLat/JTP parent pairs as an estimation study and
expand only under a prospectively funded confirmatory design. If JTP does not produce competent
parents, a later preregistration may use BST; a failed method is not dragged into the forgetting
study merely to preserve symmetry.

### Hypotheses and estimands

- Null: the structural-overlap by future-conflict interaction on parent retention is zero or less.
- Alternative: high structural overlap amplifies the damage caused by a conflicting future.
- Primary outcome: erosion of the parent first-decision logit margin from before to after
  adaptation.
- Secondary outcomes: exact-path retention, acquisition of the new task, representation movement,
  and recovery under activation patching.

Estimate the preregistered difference in differences with parent seed as the experimental unit.
New-task acquisition must be matched before retention is interpreted; otherwise a cell that did
not learn cannot count as evidence of protection from forgetting.

### Causal diagnostic

Patch the clean parent activation into the adapted model at fixed early, middle, and late layers
(currently layers 3, 7, and 10 for the 12-layer backbone). Compare with self patches, unrelated
donors, and norm-matched random patches. The endpoint is normalized recovery of the parent
decision margin.

Patching tests whether a retained state is locally sufficient to restore the decision. It does not
prove that the patched layer is the unique cause, and it cannot rescue a null behavioral
interaction.

## 4. Exact predictive-state geometry

**One-sentence question:** Which objectives make internal states recover the exact predictive
state of a known stochastic process, including distinctions invisible to the next token?

### Benchmarks

- **Mess3:** a three-state process with a nontrivial fractal mixed-state geometry.
- **RRXOR:** a five-state process with 36 predictive states, including distinct beliefs with the
  same immediate next-token distribution.
- **Controlled grid walkers:** a later external-validity family in which exact future occupancy or
  predictive vectors can be enumerated.

The immediate Mess3/RRXOR anchor reproduces the authors' saved-checkpoint analyses and exact oracle
at negligible training cost. A later training comparison would follow the published architecture
and optimizer: context 10, four layers, one head, model width 64, ReLU, MLP width 256, batch 64,
online sampling from the stationary distribution, SGD at `0.01`, no weight decay, and up to
1,000,000 updates.

### Hypotheses and estimands

- Null: adding future-token, future-summary, or latent-prediction supervision does not improve
  held-out recovery of the exact predictive state relative to next-token training at matched
  predictive competence.
- Alternative: at least one future-aware objective improves that recovery, especially on RRXOR
  distinctions that share the same next-token distribution.
- Primary outcome: held-out affine decoding error for the exact belief vector.
- Required secondary outcomes: pairwise geometry alignment, RRXOR separation within
  next-token-degenerate groups, horizon-1 through horizon-8 future-distribution error, and
  layerwise versus all-layer-concatenated recovery.

Use grouped splits by history/predictive-state family, not random rows that leak near-duplicate
histories. Match models both at the final update budget and at checkpoints with equivalent gap to
the analytic Bayes loss. Include shuffled belief labels, untrained networks, and capacity-matched
nonlinear probes as controls.

The seven-objective, 32-seed training matrix is explicitly deferred. If this study is funded later,
start with an eight-seed GPT/NextLat pilot on both processes and use its blinded variance and
convergence results to price a separately preregistered comparison. The present milestone makes
only a reproduction claim from author checkpoints and exact oracles, not a new objective-ranking
claim. FSP, HiLP, BST, and grid-walker training remain later work.

## Program-level claim boundary

The four studies can fail independently. Reliable planner training does not prove a predictive
geometry; predictive geometry does not prove causal resistance to forgetting; and an exact HMM
geometry does not prove ordinary-language generalization. Conversely, a Path-Star reproduction
failure is itself relevant to Study 1 but does not invalidate the exact-state study.

Natural-language pretraining is outside the present confirmatory program. If added later, it is an
external-validity extension without the exact counterfactual and belief-state oracle available
here.
