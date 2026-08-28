# Literature and source ledger

Search cutoff: **2026-08-27**. This is a living evidence map, not a general bibliography. It records
the primary sources that materially constrain the stimuli, implementation, evaluation, or claims.
Where a paper and public code disagree, record the discrepancy rather than silently choosing one.

## Planning tasks and predictive objectives

| Source | Public artifact | What this project adopts |
| --- | --- | --- |
| [Next-token prediction can be myopic](https://arxiv.org/abs/2403.06963) | [Official Path-Star repository](https://github.com/gregorbachmann/Next-Token-Failures) | Original fixed-data Path-Star generator, serialization, Clever Hans failure mode, and exact-path evaluation |
| [The Mystery of the Pathological Path-Star Task](https://arxiv.org/abs/2410.13779) | Paper appendices and released code links | Treat optimization as trial-level stochastic behavior; zero attention dropout; structured target pairing as a native diagnostic; five or eleven seeds are not enough for a population claim |
| [The Belief State Transformer](https://arxiv.org/abs/2410.23506) | [Official BST repository](https://github.com/microsoft/BST) and [author article](https://www.microsoft.com/en-us/research/articles/belief-state-transformers/) | Native `N=50`, 8-million-example, 100-epoch validation; exact path over held-out graphs; quadratic prefix-suffix training is a compute caveat |
| [Joint Multiple Token Prediction](https://arxiv.org/abs/2503.21801) | Implementation included by the NextLat authors | JTP as a distinct token-space belief-learning baseline, horizon matched in common comparisons |
| [Next-Latent Prediction Transformers Learn Compact World Models, v4](https://arxiv.org/abs/2511.05963) | [Official code](https://github.com/JaydenTeoh/NextLat) and [author blog](https://jaydenteoh.github.io/blog/2026/nextlat/) | Fixed 200,000/20,000 `N=100` Path-Star data; `G(5,5)`, 20,000 updates, 12x384/6 heads, horizon 3; compare GPT, MTP, JTP, BST, and NextLat |
| [How Transformers Learn to Plan via Multi-Token Prediction](https://arxiv.org/abs/2604.11912) | Paper's full-standard-architecture appendix | MTP as a serious planning baseline; include binary-tree external validity and attention/decision diagnostics; note three-replicate uncertainty |
| [Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](https://arxiv.org/abs/2510.14751) | Uses the original Path-Star generator; no method repository pinned here at cutoff | Implement FSP-BoW; reproduce `G(2,6)` and `G(2,8)` with `N=50`, 200k/20k, 12x384/6, batch 256, learning rate `3e-4`, weight decay `0.01`, and 500 epochs |
| [Hierarchical Latent Prediction for Language Models](https://arxiv.org/abs/2608.05806) | No author Path-Star/HMM implementation pinned at cutoff | Include HiLP because it directly targets compounding latent-rollout error; treat the small-task port as independent, use backbone state as primary, and higher latent as diagnostic |

Important incompatibilities are part of the design. BST's original paper uses fresh large data at
`N=50`; NextLat's comparison uses fixed smaller data at `N=100`; FSP uses two-arm longer paths and
500 epochs; the MTP planning paper uses different data/model scaling. These are first reproduced
natively, then reimplemented under the common `G(5,5)` design.

The NextLat v4 paper reports five seeds per Path-Star baseline and says NextLat is near perfect on
the tested topologies. It does not publish per-seed checkpoints, complete runtime locks, or the
private execution path needed to replay the seed-level trajectories exactly. The repository's
reproducibility finding therefore tests public procedure portability, not contradiction of an
available private checkpoint.

## Exact predictive geometry

| Source | Public artifact | What this project adopts |
| --- | --- | --- |
| [Transformers Represent Belief State Geometry in their Residual Stream](https://arxiv.org/abs/2405.15943) | [Official repository](https://github.com/adamimos/epsilon-transformers) and [author explanation](https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/) | Mess3 and RRXOR processes, exact mixed states, affine recovery, shuffled correspondence, pairwise geometry, RRXOR all-layer concatenation, and the exact 4-layer/64-width/SGD/1M-update anchor |
| [Predictive Statistics Shape Emergent World Representations of Grid Walkers](https://arxiv.org/abs/2603.16689) | Paper artifacts | Add exact predictive-vector grid environments only after Mess3/RRXOR; do not infer a robust seed population from environment diversity alone |
| [NextLat v4](https://arxiv.org/abs/2511.05963) | Official code above | Compare objective-specific predictive geometry and preserve the distinction between theorem assumptions and finite optimization outcomes |
| [HiLP](https://arxiv.org/abs/2608.05806) | Paper algorithm and efficiency appendix | Test whether the higher temporal scale improves long-horizon predictive-state error; do not call it a Path-Star or HMM reproduction |

The key RRXOR endpoint is not merely visually attractive geometry. It asks whether states with the
same immediate next-token distribution but different full futures remain distinguishable.

## Continual learning and causal diagnostics

| Source | What this project adopts |
| --- | --- |
| [Disentangling task similarity in continual learning](https://arxiv.org/abs/2405.20236) | Cross feature/structural similarity with readout/future similarity; test the interaction rather than a one-dimensional “near versus far” scale |
| [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) | Make full fine-tuning primary and treat LoRA as a distinct robustness condition |
| [Always Learning, Always Mixing](https://arxiv.org/abs/2605.15220) | If mitigation is studied later, compare replay/data mixing without changing the primary forgetting estimand |
| [Best practices for activation patching](https://arxiv.org/abs/2309.16042) and [causal abstraction caveats](https://arxiv.org/abs/2404.15255) | Use fixed sites, multiple donor controls, normalized recovery, and restrained “local sufficiency” language |

The forgetting study is designed to test a mechanistic interaction, not to benchmark every
continual-learning mitigation. Replay, regularization, gating, and data mixing are future follow-ups
only after the uncontrolled full-fine-tuning effect is established.

## Source-handling rules

1. Prefer the authors' repository and exact tagged/committed source over third-party rewrites.
2. Store commit hashes, config hashes, and retrieval dates in machine-readable manifests.
3. Read paper appendices, README warnings, issue discussions, and author blog posts for execution
   details, but label non-peer-reviewed guidance as such.
4. Do not copy code or supplemental artifacts into this public repository until the license permits
   redistribution; pin a hash and retrieval procedure instead.
5. Search GitHub and author pages again immediately before implementation freeze, especially for
   the very recent HiLP paper and any released FSP code.
6. A paper-derived port validates the experiment we implemented; it does not become an author-code
   reproduction through similarity of results.
