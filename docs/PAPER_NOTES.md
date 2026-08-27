# PAPER_NOTES.md — Grounding in the actual NextLat paper

Purpose: an exact, quotable record of what arXiv:2511.05963v4 already establishes, so that the
Lure-Star / HMM extension in `nextlat_v4_predictive_geometry_spec.md` never claims as novel
something the paper has already claimed, and never mis-states a paper number.

## 0. Provenance and retrieval method (read this before trusting any number below)

* Retrieved 2026-08-23 by direct HTTP fetch (no model summarization in the loop) of the LaTeXML
  HTML of the paper:
  * `https://arxiv.org/html/2511.05963v4` → archived at
    `<repo-root>/docs/paper_source/v4.html`
    (sha256 `572bb951092ef764db4838c4af880a3bf9b6bdf80ec395ed0dc9f6678bddcae2`)
  * `https://arxiv.org/abs/2511.05963` → archived at `docs/paper_source/abs.html`
    (sha256 `ccd869a2780c906f655e8c0e0c25f48f4fd51bddd5f0c1007a5559524546b4ff`)
* The HTML was converted to plain text (math preserved as LaTeX `alttext`) by
  `docs/paper_source/conv2.py` → `docs/paper_source/v4.txt`. Line numbers cited below as
  `v4.txt:N` refer to that file. **Every full section of the paper was retrieved**: Abstract,
  §1–§7, Appendices A–F, and the bibliography.
* Two figures are vector SVGs whose bar heights can be recovered exactly from path geometry
  (Fig. 5, Fig. 6). They were downloaded and digitized with `docs/paper_source/digitize_figs.py`
  (`star.svg` = Fig. 6, `cd.svg` = Fig. 5). Digitized values are flagged **[digitized]** below and
  carry roughly ±0.5 pp uncertainty; they are NOT printed numbers in the paper.
* **What could NOT be retrieved as numbers** (stated explicitly rather than guessed): Figures 3,
  8, 9, 10, 12–17 are raster PNGs. Their quantities exist in the paper only as pictures plus the
  prose claims quoted below. In particular there is **no printed Path-Star accuracy table** in the
  paper; §4.3's only textual claim is "close to 100% solve rate for all graphs".
* Inline citations render in the HTML as bare numbers (e.g. "6", "87"); resolved where used:
  [2] Ahn, Lamb, Langford 2025 (JTP); [6] Bachmann & Nagarajan 2024 (pitfalls of next-token
  prediction / Path-Star); [25] Gandhi et al. 2024 (Stream of Search); [28] Gloeckle et al. 2024
  (MTP); [40] Hu et al. 2025 (Belief State Transformer); [45] Kaelbling et al. 1998;
  [49] Leviathan et al. (speculative decoding); [55] DeepSeek-V3 tech report;
  [62] Merrill et al. 2025; [75] Roy & Vetterli 2007 (effective rank); [81] Striebel 1965;
  [87] Vafa et al. 2024 (Manhattan world model); [92] Ye et al. 2025.

## (a) Exact bibliographic record

| Field | Value |
|---|---|
| Title | **Next-Latent Prediction Transformers Learn Compact World Models** (the HTML `<title>` element renders it as "Next-Latent Prediction Transformers Learn CompactWorld Models" — a LaTeXML line-break artifact, not the real title) |
| Authors (exact order) | Jayden Teoh; Manan Tomar; Kwangjun Ahn; Edward S. Hu; Tim Pearce; Pratyusha Sharma; Akshay Krishnamurthy; Riashat Islam; Alex Lamb; John Langford |
| Correspondence | Jayden Teoh — `jayden_t[at]mit[dot]edu` (v4.txt:157) |
| Affiliation line | Microsoft Research (v4.txt:177); abs page comment field: "Microsoft Research Preprint" |
| arXiv id / version | arXiv:2511.05963v4 [cs.LG] |
| Header date on v4 | 15 Jun 2026 (v4.txt:150) |
| Submission history | v1 Sat, 8 Nov 2025 10:41:26 UTC; v2 Fri, 22 May 2026 06:33:12 UTC; v3 Mon, 25 May 2026 15:53:24 UTC; v4 Mon, 15 Jun 2026 08:56:56 UTC (last revised) |
| Primary category | Machine Learning (cs.LG) |
| Code | `https://github.com/JaydenTeoh/NextLat` (stated in the Abstract) |

Note for our project: the pinned upstream checkout
`<repo-root>/upstream/NextLat` is at commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`, "Initial public release", dated **Mon May 25
21:50:04 2026 -0700** — i.e. contemporaneous with paper **v3**, three weeks before **v4**. Treat
the repo as the v3-era code release; do not assume it contains anything added in v4.

## (b) Method definition, with the paper's own symbols

Terminology (footnote 1, §3, v4.txt:240) — verbatim:

> "In the sequence modeling literature, intermediate latent representations are often referred to
> as “hidden states”. To disambiguate, we use the term “latent state” to broadly refer to learned
> representations within the transformer’s residual stream, and “hidden state” to refer to a
> subset of this representation—specifically, the final layer’s output at each time step (i.e.,
> the pre-logit activations)."

**Definition 3.1 (Belief states in sequence modeling)** (v4.txt:242–248), verbatim:

> "Let $X_{1:T}$ denote a token sequence $X_{1},\dots,X_{T}$. A random variable
> $\mathbf{b}_{t}=g(X_{1:t})$ is a belief state for the history $X_{1:t}$ if, for every bounded
> measurable function $f$ of the future,
> $\mathbb{E}[f(X_{t+1:T})\mid\mathbf{b}_{t}]=\mathbb{E}[f(X_{t+1:T})\mid X_{1:t}]\quad\text{a.s.}$
> Equivalently, $\mathbf{b}_{t}$ is a sufficient statistic (81) of the history $X_{1:t}$ for
> predicting the future tokens, i.e., from which we can sample from the distribution
> $\mathbb{P}(X_{t+1:T}\mid X_{1:t})$."

**Theorem 3.2** (v4.txt:252–268), verbatim in substance: the joint learning of (1) a transformer
with parameters $\theta$ producing hidden states $\mathbf{h}_{t}$, (2) an output head $p_{\theta}$
modeling the next-token distribution, (3) a latent dynamics model $p_{\psi}$ modeling the
transition dynamics of the transformer's hidden states. "If NextLat successfully optimizes the
following objectives:"

* (Next-Token Consistency), Eq. (1): $p_{\theta}(X_{t+1}\mid\mathbf{h}_{t})=\mathbb{P}(X_{t+1}\mid X_{1:t})$
* (Transition Consistency), Eq. (2): $p_{\psi}(\mathbf{h}_{t+1}\mid\mathbf{h}_{t},X_{t+1})=\mathbb{P}(\mathbf{h}_{t+1}\mid X_{1:t+1})$

"then $\mathbf{h}_{t}$ must be a belief state for the sequence $X_{1:t}$." Formal proof by backward
induction in Appendix B (v4.txt:2069–2135). Remark (v4.txt:266): "Optimizing only next-token
consistency (i.e., Equation 1) in standard autoregressive transformers does not guarantee that
$\mathbf{h}_{t}$ forms a belief state (see Theorem 3 in 40)."

**Practical objective (§3.2).** Exact loss terms as written:

1. Next-token cross-entropy (v4.txt:311):
   `L_next-token(θ) = E_{t<T}[ -log p_θ(X_{t+1} | h_t) ]`
   LaTeX: `$\mathcal{L}_{\text{next-token}}(\theta)=\mathbb{E}_{t<T}\big[-\log p_{\theta}(X_{t+1}\mid\mathbf{h}_{t})\big]$`

2. Next-hidden (latent) regression, Eq. (3) (v4.txt:321), over the $d$-step teacher-forced rollout
   $\hat{\mathbf{h}}_{t+d}=p_{\psi}(\mathbf{h}_{t},X_{t+1:t+d})$:
   `L_next-h(θ,ψ;d) = E_t[ (1/d) Σ_{i=1..d} SmoothL1Loss( sg[h_{t+i}], ĥ_{t+i} ) ]`
   LaTeX: `$\mathcal{L}_{\text{next-h}}(\theta,\psi;d)=\mathbb{E}_{t}\Big[\frac{1}{d}\sum_{i=1}^{d}\mathrm{SmoothL1Loss}\big(\textbf{sg}[\mathbf{h}_{t+i}],\hat{\mathbf{h}}_{t+i}\big)\Big]$`
   `sg[·]` is stop-gradient, "used to prevent representational collapse in self-predictive learning (67)".

3. KL self-distillation in token space, Eq. (4) (v4.txt:327):
   `L_KL(θ,ψ;d) = E_t[ (1/d) Σ_{i=1..d} D_KL( p_θ^sg(· | sg[h_{t+i}]) ‖ p_θ^sg(· | ĥ_{t+i}) ) ]`
   LaTeX: `$\mathcal{L}_{\mathrm{KL}}(\theta,\psi;d)=\mathbb{E}_{t}\Big[\frac{1}{d}\sum_{i=1}^{d}D_{\mathrm{KL}}\!\left(p_{\theta}^{\textbf{sg}}(\cdot\mid{\textbf{sg}[\mathbf{h}_{t+i}]})\;\|\;p_{\theta}^{\textbf{sg}}(\cdot\mid{\hat{\mathbf{h}}_{t+i}})\right)\Big]$`
   "where the output head $p_{\theta}^{\textbf{sg}}(\cdot)$ is frozen so that gradients flow only
   through the latent dynamics model."

4. Overall objective, Eq. (5) (v4.txt:341):
   `L_NextLat(θ,ψ; d, λ_next-h, λ_KL) = L_next-token(θ) + λ_next-h · L_next-h(θ,ψ;d) + λ_KL · L_KL(θ,ψ;d)`,
   with `λ_next-h, λ_KL > 0`.

   **Symbol warning for our configs:** the paper's coefficient is `λ_next-h` (Smooth L1 on hidden
   states). The spec and the repo YAML call this `lambda_mse`. They denote the same term; the
   paper's loss is Smooth L1, not MSE (Appendix E lists "replacing Smooth L1 with MSE loss" as an
   *unsuccessful fix* attempted for a training quirk, v4.txt:2390).

Other exact statements that constrain our implementation:

* Belief-state convergence "already holds for $d=1$. Multi-step supervision serves only to provide
  richer learning signal." (v4.txt:323)
* "during inference, the learned transformer can decode independently; $p_{\psi}$ is only needed
  during training to shape the transformer representations." (v4.txt:343)
* Latent dynamics parameterization, Appendix C (v4.txt:2136–2140): "We parameterize the latent
  transition model $p_{\psi}$ with a three-layer MLP using GELU [38] activations. The latent
  transition model takes as input the layer-normalized [5] concatenation of the current hidden
  state $\mathbf{h}_{t}$ and next-token embedding $X_{t+1}$, and outputs a delta update applied via
  residual connection: $\hat{\mathbf{h}}_{t+1}=p_{\psi}(\mathbf{h}_{t},X_{t+1})=f_{\psi}(\mathbf{h}_{t},X_{t+1})+\mathbf{h}_{t}$" (Eq. 9).
* Masking policy, Appendix C (v4.txt:2142): "we mask token-level losses (i.e.,
  $\mathcal{L}_{\text{next-token}}$ and $\mathcal{L}_{\mathrm{KL}}$) corresponding to context or
  prompt tokens. However, we do not apply masking for $\mathcal{L}_{\text{next-h}}$ on context
  tokens, ensuring that belief state representations develop even during context processing."
  → Directly relevant to H1: the state we read at the final prompt delimiter `=` **is** trained by
  the NextLat regression term even though its token loss is masked.

## (c) Every quantitative result the paper establishes

### Table 1 — Manhattan taxi rides world modeling (v4.txt:369–412; Table 1 caption v4.txt:412)

| Model | Next-Token Test ↑ | Valid Trajectories ↑ | Sequence Compression ↑ | Effective Latent Rank ↓ | Detour Robustness ↑ |
|---|---|---|---|---|---|
| GPT | 100% | 97.0% | 0.65 | 160.1 | 85.0% |
| MTP | 100% | 98.1% | 0.64 | 57.7 | 95.0% |
| JTP | 100% | 97.1% | 0.32 | 215.8 | 87.0% |
| **NextLat** | 100% | **98.7%** | **0.71** | **52.7** | **95.0%** |
| True world model | 100% | 100% | 1.00 | — | 100% |

Prose (v4.txt:445): "NextLat achieves the highest sequence compression of 0.71"; "NextLat has the
lowest effective latent rank of 52.7—over 3x smaller than GPT’s." Dataset: 91M sequences / 4.7B
tokens, 6 epochs; BST excluded for cost; $d=8$ for JTP/MTP/NextLat.

### Figure 4 (rendered as a table in the HTML) — Countdown accuracy (v4.txt:453–481)

| Model | Horizon d | Accuracy (%) |
|---|---|---|
| GPT | – | 33.1 |
| BST | – | 42.3 |
| MTP | 1 / 4 / 8 | 39.2 / 49.7 / 57.3 |
| JTP | 1 / 4 / 8 | 39.0 / 49.4 / 55.0 |
| **NextLat** | 1 / 4 / 8 | **54.8 / 57.6 / 58.7** |

500k training problems, 10k test problems, 8 pause tokens, 3 seeds. Prose (v4.txt:545): NextLat at
$d=1$ "substantially surpasses MTP and JTP trained with the same horizon (>35.7% improvement)".

### Figure 5 — Countdown equation validity **[digitized from `cd.svg`]** (v4.txt:545)

| Model | Eq. 1 | Eq. 2 | Eq. 3 (final) |
|---|---|---|---|
| GPT | 99.0 | 92.2 | 33.1 |
| BST | 98.6 | 90.6 | 42.3 |
| MTP (d=1) | 98.6 | 90.8 | 39.2 |
| JTP (d=1) | 98.5 | 91.4 | 39.0 |
| NextLat (d=1) | 97.6 | 89.2 | 54.2 |

Paper's own prose for this figure (printed numbers, authoritative): NextLat at $d=1$ "achieves
substantially higher mean validity in the final equation ($54.8$%) compared to the next best
baseline ($42.3$%)". Our digitization reproduces the baselines to ≤0.1 pp but reads NextLat Eq. 3
as ≈54.2 rather than 54.8 — quote the paper's 54.8%, and treat the digitized column as
cross-check only.

### Figure 6 — Path-Star accuracy **[digitized from `star.svg`]** (v4.txt:551–561)

Percent solve rate; 200k fixed training graphs, node values from $N=100$, $d=\ell-2$, five seeds.
**These are the only Path-Star numbers obtainable from the paper; the paper prints none.**

| Model | G(2,10) | **G(5,5)** | G(7,7) |
|---|---|---|---|
| GPT | 49.4 | **18.6** | 12.6 |
| BST | 98.9 | **99.9** | 9.7 |
| MTP | 66.2 | **21.2** | 12.6 |
| JTP | 69.5 | **47.3** | 11.7 |
| **NextLat** | 97.1 | **99.8** | 94.3 |

Sanity check that supports the digitization: GPT sits at chance-per-arm (1/2 = 50%, 1/5 = 20%,
1/7 ≈ 14%), and the ordering matches the prose (v4.txt:561): "NextLat maintains close to 100%
solve rate for all topologies of the Path-Star graphs. BST, while able to solve $G_{2,10}$ and
$G_{5,5}$, begins to fail at the larger graph $G_{7,7}$."
**For our project: the paper's own G(5,5) NextLat number is ≈99.8% and GPT is ≈18.6%.** The spec's
90% base-competence gate is therefore below NextLat's paper level and far above GPT's — GPT is
*expected* to fail a 90% exact-path gate on G(5,5); that is the paper's result, not a bug.

### Figure 8 / Figure 16 — TinyStories future-token linear probes (v4.txt:575)

No numeric values are printed and the figure is a PNG (**not retrievable as numbers**). Textual
claims: BST/MTP/JTP "consistently cause significant degradation in next-token prediction (i.e.,
token offset = 1)"; "NextLat matches GPT’s next-token performance across both $d\in\{1,8\}$ and
exhibits the strongest long-range predictive capability (up to 20 tokens ahead) for both $d=1$ and
$d=8$." Setup: 2.7M stories, vocab 1,000, sequence length 256, 100k steps, 20 independent
one-layer probes at offsets 1..20, trained 20k steps on frozen models.

### Table 2 / Table 6 — 1.3B params, 100B FineWeb-Edu tokens (v4.txt:672–747, 2617–2760)

| Model | FW-Edu ppl ↓ | Wiki ppl ↓ | LAMB ppl ↓ | LAMB acc | PIQA | HellaS. | Wino. | ARC-e | ARC-c | SIQA | SciQ | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT | 10.52 | 17.93 | 20.26 | 42.07 | 73.45 | 58.79 | 60.46 | 68.18 | 39.16 | 42.32 | 86.10 | 58.82 |
| JTP (d=1) | 11.08 | 19.28 | 21.88 | 41.35 | 74.92 | 57.43 | 58.64 | 68.73 | 39.25 | 42.99 | 87.30 | 58.83 |
| JTP (d=2) | 11.18 | 19.60 | 22.11 | 41.37 | 73.34 | 56.84 | 59.98 | 68.86 | 38.57 | 43.35 | 86.70 | 58.63 |
| JTP (d=4) | 11.29 | 20.45 | 20.65 | 41.90 | 73.45 | 56.58 | 57.70 | 69.73 | 39.68 | 42.53 | 88.50 | 58.76 |
| MTP (d=1) | 10.90 | 18.82 | 20.23 | 41.26 | 74.32 | 58.05 | 60.54 | 68.52 | 38.91 | 42.84 | 85.40 | 58.76 |
| MTP (d=2) | 11.00 | 18.61 | 18.34 | 43.43 | 72.80 | 57.92 | 59.35 | 68.35 | 39.08 | 41.97 | 86.60 | 58.69 |
| MTP (d=4) | 11.10 | 18.97 | 22.75 | 40.69 | 73.72 | 57.39 | 58.33 | 70.20 | 39.51 | 42.12 | 85.90 | 58.48 |
| NextLat (d=1) | 10.83 | 18.39 | 19.77 | 41.08 | 73.07 | 58.35 | 59.27 | 69.65 | 39.68 | 43.24 | 86.00 | 58.79 |
| **NextLat (d=2)** | 10.88 | 18.44 | **17.83** | 43.86 | 73.61 | 57.79 | 59.20 | 69.74 | 40.10 | 41.91 | 87.50 | **59.21** |

Explicit hedge by the authors (v4.txt:843): "NextLat ($d=2$) does show a modest gain in average
accuracy over GPT (59.21 vs. 58.82), but these improvements are not consistent across tasks.
Larger model sizes might be necessary to see more significant improvements."

### Table 3 / Table 7 — self-speculative decoding speedup and accepted tokens (v4.txt:757–841, 2790–2900)

| Model | Wikipedia speedup / acc.tok | Books | Code | Math |
|---|---|---|---|---|
| JTP (d=1) | 1.46 / 0.96 | 1.47 / 0.97 | 1.47 / 0.98 | 1.46 / 0.97 |
| JTP (d=2) | 1.88 / 1.84 | 1.90 / 1.89 | 1.88 / 1.85 | 1.89 / 1.86 |
| JTP (d=4) | 2.58 / 3.32 | 2.62 / 3.43 | 2.61 / 3.43 | 2.42 / 3.02 |
| MTP (d=1) | 1.38 / 0.91 | 1.39 / 0.95 | 1.40 / 0.97 | 1.39 / 0.95 |
| MTP (d=2) | 1.68 / 1.72 | 1.72 / 1.83 | 1.75 / 1.91 | 1.72 / 1.84 |
| MTP (d=4) | 2.10 / 3.04 | 2.25 / 3.44 | 2.32 / 3.68 | 2.25 / 3.46 |
| NextLat (d=1) | 2.68 / 3.52 | 2.72 / 3.64 | 2.29 / 2.66 | 2.30 / 2.72 |
| **NextLat (d=2)** | **3.21 / 4.59** | **3.32 / 4.86** | 2.38 / 2.83 | **2.87 / 3.94** |

"Accepted Tokens" excludes the always-accepted next token. Measured on 8× NVIDIA B200, 1024
prompts of length 512, 512-token continuations, speculative sampling of [49]. The headline "up to
$3.3\times$" (Abstract, §1, and v4.txt:849) refers to Figure 9 on FineWeb-Edu validation:
"Speedup increases sublinearly with draft length, reaching up to $3.3\times$, and fully valid
(i.e., all tokens accepted) drafts persist even at length 10."

### Table 4 — cost comparison on FineWeb-Edu pretraining (v4.txt:860–893)

| | GPT | BST | MTP (d=1/2/8) | JTP | NextLat |
|---|---|---|---|---|---|
| Training params | 1.32B | 2.57B | 1.37B / 1.42B / 1.72B | 1.34B | 1.40B |
| Inference params | 1.32B | 1.32B / 2.57B | 1.32B | 1.34B | 1.32B |
| Training steps/s (d=1/2/8) | 3.09 | 0.89 | 2.80 / 2.58 / 1.70 | 3.15 / 2.92 / 2.02 | 3.09 / 2.79 / 1.73 |
| Gradients | O(T) | O(T²) | O(Td) | O(Td) | O(Td) |

Single NVIDIA B200, batch 33k tokens. Prose: "On FineWeb-Edu pretraining, BST is over $3\times$
slower than NextLat in training speed".

### Figure 10 — A5 state tracking / length generalization (v4.txt:936–941)

PNG; **no numbers retrievable**. Textual claims: 2-layer transformers trained on 12-token
sequences, tested at 36 tokens; "the transformer trained with NextLat exhibits better
state-tracking performance than GPT within the 12-token training horizon"; "while the transformer
itself fails to generalize beyond the 12-token horizon, the learned latent dynamics model
successfully generalizes to 36-token sequences ($>95$% accuracy)"; the RNN ($p_{\psi}$) has
2.62M parameters vs the transformer's 6.43M; a "GPT (36 tokens)" model trained directly on
36-token sequences "is unable to solve the task". Setup (F.5): 1M unique 12-token sequences,
~100k 36-token eval sequences, $d=1$, $\lambda_{\text{next-h}}=1$, $\lambda_{\mathrm{KL}}=0$, RoPE
(NoPE degraded performance).

### Table 5 — hyperparameters, all benchmarks (v4.txt:2408–2510). Path-Star column verbatim:

| Path-Star setting | Value |
|---|---|
| Steps | 20k (51 epochs) |
| Batch size | 512 |
| Learning rate | 5e-4 |
| LR schedule | Constant |
| Weight decay | 0.1 |
| Clip gradient norm | 100 |
| Optimizer | AdamW with β₁=0.9, β₂=0.95 |
| Layers / Heads / Hidden dim | 12 / 6 / 384 |
| λ_next-h | 1.0 |
| λ_KL | 1.0 |
| p_ψ MLP hidden dim | 384 |
| p_ψ MLP layers | 3 |

This matches the spec's expected YAML (`n_layer=12, n_head=6, n_embd=384, lambda_mse=1.0,
lambda_kl=1.0, lr=5e-4, batch 512, clip 100, 20k steps`) — with the one caveat that the paper does
not report a `proj_factor`; the 384 MLP hidden width is what the repo's `proj_factor` must
resolve to for Path-Star (verify in the pinned repo before launch). Hardware, all experiments
(F, v4.txt:2404): "NVIDIA RTX A5000, NVIDIA H100 NVL, and NVIDIA B200 GPUs". Path-Star data
prep/training/eval follow [6] "except that we increase the weight decay to 0.1"; evaluation is on
"20k held-out test instances" (F.3, v4.txt:2568).

Ablation results that our bottleneck-width ablation must respect (Appendix D, v4.txt:2327–2360):
at $d=1$, "using Smooth L1 loss alone achieves the strongest probe performance 20 tokens ahead
relative to GPT"; at $d=8$ "the combined KL + Smooth L1 objective (i.e., the full NextLat design)
performs best"; stop-gradients on both components "yields the best probing performance across all
token offsets"; "on tasks like Countdown and Path-Star graph, we also observe that the
stop-gradient (especially on the Smooth L1 loss) is essentially [sic] for high accuracy."

## (d) VERBATIM: §6 Limitations and Future Work

The following is the complete text of Section 6 of arXiv:2511.05963v4, transcribed exactly
(including the typo "only an preliminary study"). Inline citation "55" = DeepSeek-V3 [55]. Math is
given in the paper's LaTeX.

> **6 Limitations and Future Work**
>
> While NextLat shows strong empirical performance, several limitations remain in our work. First, we do not explore the design space of the latent dynamics model; all experiments use simple MLPs to isolate and demonstrate the effectiveness of the core NextLat approach. More expressive architectures may further improve performance. We also do not study how the width of the hidden layers in the latent dynamics MLP affects learning, even though it effectively acts as a bottleneck that constrains belief-state capacity and may influence performance across tasks. Empirically, we observe that using smaller latent dimensions is beneficial on tasks such as Path-Star graph and Countdown.
>
> Second, the design of the NextLat objective (e.g., stop-gradients, KL self-distillation, Smooth L1 loss) is guided largely by small-scale ablations in Appendix D and empirical intuition. It remains unclear whether multi-step supervision ($d>1$) and KL token-level supervision are even necessary at larger model and data scales. More systematic studies are needed to better understand how these components interact. Third, due to computational constraints, we did not evaluate against more recent or specialized MTP variants such as the one introduced in 55. Finally, we did not fully exploit the variable-length nature of NextLat’s speculative decoding. In our experiments, the draft length remained fixed throughout decoding for each prompt; we only varied the static draft length between 2 and 10 tokens to identify the configuration with the highest inference speedup. We leave the exploration of more creative adaptive-length speculative decoding strategies for NextLat to future work.
>
> On the analysis side, we do not study the structure of the learned representations under NextLat, leaving open questions about how the method shapes latent spaces. In Appendix E, we also highlight several quirks observed during pretraining with NextLat, such as increases in Smooth L1 loss ($\mathcal{L}_{\text{next-h}}$) during training and differing loss trajectories across optimizers. These observations suggest that NextLat can be sensitive to optimization dynamics. Better understanding of how to scale and parameterize the NextLat objective remains an important direction for future work.
>
> This work represents only an preliminary study of next-latent prediction, leaving many promising directions for future research. Since the method requires no architectural changes beyond a lightweight latent dynamics model for shaping representations, an interesting direction is to apply it as a post-hoc finetuning objective for pretrained transformers. This could potentially improve reasoning, planning, and world-modeling capabilities of existing models without retraining from scratch. Moreover, because NextLat effectively organizes latent representations with recurrent-like dynamics, an interesting question is whether transformers trained with NextLat are better suited for RL post-training, where value estimation (be it implicit or explicit) benefits from such recursive “Bellman-like" latent structure. Finally, it would be valuable to explore richer latent architectures, such as higher-dimensional or hierarchical belief states spanning multiple layers or tokens, which may further improve long-horizon reasoning and planning.

(Source: v4.txt:966–974; `docs/paper_source/v4.html`, section id `S6`.)

### Why this section is the load-bearing one for our project

* **"we do not study the structure of the learned representations under NextLat, leaving open
  questions about how the method shapes latent spaces"** — this is the explicit gap the Lure-Star
  and HMM experiments occupy. Cite this sentence, not a paraphrase, when arguing novelty.
* **"We also do not study how the width of the hidden layers in the latent dynamics MLP affects
  learning, even though it effectively acts as a bottleneck that constrains belief-state
  capacity"** — this is exactly the spec's §13 later ablation. Note that the paper already states
  the *direction* of its informal observation: "using smaller latent dimensions is beneficial on
  tasks such as Path-Star graph and Countdown." So a finding that "smaller bottleneck helps
  Path-Star accuracy" is NOT novel; only its effect on representational geometry would be.
* The paper explicitly lists as future work: post-hoc finetuning objective, RL post-training,
  hierarchical/higher-dimensional latents, adaptive-length speculative decoding, and MTP-variant
  comparisons. The spec already excludes all of these — that exclusion is now grounded in the
  paper's own stated agenda (they are the authors' declared next steps, not our contribution).

## (e) Claims the paper ALREADY makes — our extension may not present these as novel

1. NextLat's hidden states "provably converge towards belief states" under Theorem 3.2 (Abstract,
   §3.1). *Any* theoretical claim that the latent is a sufficient statistic is the paper's.
2. NextLat yields more compact representations: sequence compression 0.71 vs GPT 0.65 and
   effective latent rank 52.7 vs GPT 160.1, "over 3x smaller than GPT’s" (Table 1, §4.1).
   → Rank-only or compression-only results are already established.
3. Better world-model coherence on Manhattan: valid trajectories 98.7% vs 97.0%, detour robustness
   95.0% vs 85.0%, visually more coherent reconstructed maps (Table 1, Figure 3).
4. Near-perfect Path-Star solving across G(2,10), G(5,5), G(7,7), where GPT/MTP/JTP/BST fail
   (Figure 6, §4.3). → Reproducing Path-Star accuracy is replication, not contribution.
5. Better lookahead planning / less "regretful compromise" on Countdown, including at $d=1$
   (Figure 4, Figure 5, §4.2).
6. Future-token decodability of hidden states: NextLat probes beat GPT out to 20 tokens ahead
   while preserving offset-1 performance on TinyStories (Figure 8/16, §4.3). → "NextLat's states
   contain more information about the future" is already shown by linear probing.
7. Length generalization / state tracking on A5, including the co-trained RNN generalizing 12→36
   tokens at >95% while the transformer does not (Figure 10, §5.2).
8. Data efficiency from dense latent supervision, and the argument that latent supervision gives
   richer gradients than token supervision (§3.1 "Better Data Efficiency", §4.3).
9. Variable-length self-speculative decoding with up to 3.3× speedup; accepted tokens far
   exceeding the training horizon (Tables 3/7, Figure 9, §4.4). → Explicitly out of scope for us.
10. Efficiency vs BST (O(T) vs O(T²) gradients, >3× faster) and horizon-independence of belief
    learning vs JTP's $d\ge k$ requirement (§5.1, Prop. A.2).
11. Preservation of next-token perplexity relative to MTP/JTP (Table 2).
12. The design-choice ablations: Smooth L1 alone strongest at $d=1$, KL+SmoothL1 best at $d=8$,
    stop-gradients on both components best, KL > CE at shallow horizons (Appendix D).
13. The informal claim that "smaller latent dimensions [in $p_\psi$] is beneficial on tasks such
    as Path-Star graph and Countdown" (§6).
14. Optimization quirks: Smooth L1 loss rising during LR cooldown; AdamW vs Muon trajectories;
    "the rise in smooth L1 loss may reflect changes in the scale or geometry of the latent states"
    (Appendix E).

What remains genuinely open after all of the above: **which distinctions the geometry encodes**
(selectivity: matched future-relevant vs future-irrelevant perturbations), **whether that geometry
predicts behavior item-by-item**, **whether it predicts later interference**, and **whether the
states respect exact Bayesian predictive equivalence when ground truth is known (HMM)**. The paper
supplies no pairwise-distance, predictive-equivalence, belief-divergence, posterior-decoding, or
interference/forgetting result of any kind.

## (f) Everything the paper says about hidden-state geometry, belief states, sufficiency, and representation structure

Verbatim quotes, with location:

1. Abstract: "Theoretically, we show that these latents provably converge towards belief states,
   compressed information about the history necessary to predict the future." … "NextLat
   effectively encourages transformers to form compact internal world models with coherent belief
   states and transition dynamics—crucial properties not guaranteed by standard next-token
   prediction alone."
2. §1: "we establish a theoretical foundation showing that NextLat provably shapes transformer
   representations into belief states—compact summaries of past information sufficient for
   predicting future observations. Such representations are important for planning and
   generalization, yet are not guaranteed to emerge from next-token prediction alone."
3. §2 (Belief States), quoting Kaelbling et al. [45]: a belief state is "a sufficient statistic for
   the past history … no additional data about its past actions or observations would supply any
   further information about the current state of the world". "In stochastic control, the same
   notion of sufficient statistics appears as “information state” (81)." "While recurrent neural
   networks naturally enforce such compression, transformers have no such constraint—their
   internal state, or memory, grows linearly with sequence length."
4. Definition 3.1 (quoted in full in §(b) above) — sufficiency for *every bounded measurable
   function of the future*.
5. §3 footnote 1 — "hidden state" = final layer output at each time step = pre-logit activations.
   (This is the exact endpoint the spec designates as the primary representation.)
6. §3.1 Remark: "Optimizing only next-token consistency … does not guarantee that $\mathbf{h}_{t}$
   forms a belief state (see Theorem 3 in 40). Intuitively, self-attention enables ad-hoc lookup of
   past tokens, so there is no pressure to compress all necessary information about the past into
   compact latent summaries at every time step."
7. §3.1 Better Data Efficiency: "the model is trained to predict its own next hidden state
   $\mathbf{h}_{t+1}$, which parameterizes the full predictive distribution over $X_{t+2}$. This
   shifts supervision from individual one-hot token labels to distribution-level alignment.
   Moreover, because the latent dynamics compose recursively—each latent is trained to predict the
   next—$\mathbf{h}_{t+1}$ implicitly carries information about future states
   $\mathbf{h}_{t+2},\mathbf{h}_{t+3},\dots$."
8. §3.2: "Our NextLat implementation operates primarily on the hidden states (i.e., the final-layer
   outputs) as they provide compact, fixed-dimensional vectors through which gradients can be
   propagated through the entire transformer efficiently."
9. §4.1 (compression metrics): "A model that accurately captures the underlying states and
   transitions should assign identical continuations to trajectories that end in the same state
   (i.e., intersection in Manhattan). By this criterion, NextLat achieves the highest sequence
   compression of 0.71." … "The true Manhattan graph comprises only 4,580 intersections and 9,846
   edges, and therefore an effective world model should require only a modest latent
   dimensionality. Indeed, NextLat has the lowest effective latent rank of 52.7—over 3x smaller
   than GPT’s."
   *Note:* "Sequence Compression" is defined (§4.1, metric 3) as "Percentage of cases where the
   model produces identical continuations when prompted with two different traversals arriving at
   the same state and sharing the same destination" — a **behavioral output-identity** metric, not
   a representational-distance metric. This is the closest thing in the paper to our
   predictive-equivalence test, and it is explicitly measured on *generated continuations*, not on
   hidden-state geometry. Our HMM-H1 (distance between hidden states of predictively equivalent
   histories) is therefore not the same measurement.
   "Effective Latent Rank: Effective rank/dimension of hidden states measured as the exponentiated
   Shannon entropy of the normalized singular values (75); lower values indicate better
   compression." Computation (F.1): "we pass a batch of 256 sequences (each of length 256) through
   the model to obtain the hidden state matrix. Singular values smaller than $1\mathrm{e}{-12}$ are
   discarded … For GPT and NextLat, we use the final-layer hidden states."
10. §4.3 (TinyStories): "Generating such sequences therefore depends not only on next-token
    prediction, but on belief-state–like abstractions that encode information predictive of future
    story trajectories." … "These results suggest that NextLat’s latent-state objective induces
    belief-like representations that encode predictive information about future events".
11. §4.4: "the average accepted tokens per drafting steps for NextLat far exceeds its training
    horizon $d$, indicating that the learned latent dynamics remains coherent over extended
    rollouts. This further highlights the strong long-range predictive capability of the induced
    belief state representations."
12. §5.1 (Belief State Learning): "GPT and MTP lack any theoretical learning pressure to form
    belief-state representations, which means that they do not necessarily learn sufficient
    representations predictive of future observations. JTP can learn belief states but only under
    the restrictive condition that the prediction horizon satisfies $d\geq k$, where $k$ denotes the
    observability horizon of the underlying data-generating process… NextLat, on the other hand,
    learns belief-state representations independently of $d$".
13. §5.2 (Expressivity of the Recurrence): "NextLat does not explicitly perform recurrence in the
    forward computation. Instead, recurrent-like dynamics emerge implicitly through one-step or
    multi-step unrolling of the $p_{\psi}$ and aligning successive hidden states via regression.
    This induces temporal consistency within the latent space without altering the transformer
    architecture. However, it is important to note that NextLat modifies the learned
    representations, not the underlying circuit complexity."
14. §6: "we do not study the structure of the learned representations under NextLat, leaving open
    questions about how the method shapes latent spaces."
15. Appendix A, Definition A.1 ($k$-observability) and Proposition A.2 (JTP forms belief states in
    $k$-observable systems), plus: "Note that both next-token prediction and multi-token prediction
    do not guarantee belief state representations (see 40)."
16. Appendix B: full backward-induction proof that $\mathbf{h}_{k}$ is a belief state.
17. Appendix C: "we do not apply masking for $\mathcal{L}_{\text{next-h}}$ on context tokens,
    ensuring that belief state representations develop even during context processing."
18. Appendix E: "This suggests that the rise in smooth L1 loss may reflect changes in the scale or
    geometry of the latent states, rather than degradation in the token-level coherence of the
    latent transition dynamics." (The only sentence in the paper that uses "geometry" of latents at
    all — and it is a conjecture about scale, unmeasured.)

**Non-uniqueness caveat is ours, not the paper's.** Nothing in the paper claims the belief state is
minimal, unique, or coordinate-aligned to a posterior simplex; Definition 3.1 defines sufficiency
only up to any invertible re-encoding. The spec's HMM caveat ("a sufficient predictive state is
non-unique") is consistent with, and not contradicted by, the paper.

## (g) Direct implications for the spec

* The spec's summary of the paper is accurate on every number checked: compression 0.71 vs 0.65,
  effective rank 52.7 vs 160.1, up to 3.3× speculative decoding, near-perfect Path-Star,
  TinyStories future-token decodability, detour robustness. One correction: the spec says
  "mtp_horizon=3, because l-2=3" — the paper says $d=\ell-2$ and for $G_{5,5}$, $\ell=5$, so
  $d=3$. Consistent.
* The paper's Path-Star hyperparameters (Table 5) match the spec's expected YAML exactly on every
  field the paper reports. The paper reports no `train_graphs`/`heldout_graphs` YAML keys but does
  state 200k training samples, $N=100$, and 20k held-out test instances.
* The paper reports **five seeds** for Path-Star and three for Countdown; the spec's three
  confirmatory seeds are a documented reduction.
* λ naming: paper `λ_next-h` ↔ repo/spec `lambda_mse`; the loss is Smooth L1.
* The spec's HMM choice of "Smooth L1 alone at $d=1$" is directly supported by Appendix D.
