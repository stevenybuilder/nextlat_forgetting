# Literature anchors

Research snapshot: 2026-08-28.

## Adopted foundations

- [Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding
  Models](https://arxiv.org/abs/2602.24264) motivates additive factorization and cross-factor
  orthogonality. Its [official code](https://github.com/oshapio/necessary-compositionality) is the
  metric-level prior art; this project adds strict held-out-combination estimation and closed-loop
  action outcomes.
- [Adversarial Concept Search](https://arxiv.org/abs/2606.13934) shows that representational
  interference can prospectively predict compositional errors in language models. The present
  study tests the analogous claim in an embodied action policy.
- [VIMA](https://arxiv.org/abs/2210.03094), the
  [official model implementation](https://github.com/vimalabs/VIMA), and
  [VIMA-Bench](https://github.com/vimalabs/VimaBench) provide the model, checkpoints, factorial
  metadata, and four official OOD partitions.
- [Diagnosing Compositional Generalization in Sequential Robot Tasks](https://arxiv.org/abs/2607.29687)
  motivates the routing alternative: sparse-data failure can reflect instruction steering rather
  than absence of low-level skills.
- [SAFE: Multitask Failure Detection for Vision-Language-Action Models](https://arxiv.org/abs/2506.09937)
  is the closest behavioral-prediction baseline: it supervises a detector on successful and failed
  VLA rollouts and transfers across tasks. This rules out claiming that internal VLA features have
  never predicted failure. The narrower open question here is whether an outcome-blind,
  factor-derived interference measure predicts failure before any failure labels are used.

## Work that rules out weaker framings

- [Probing a VLA for Symbolic States](https://arxiv.org/abs/2502.04558) already demonstrates high
  linear decodability of object and action state in OpenVLA.
- [Mechanistic Interpretability for Steering VLA Models](https://arxiv.org/abs/2509.00328) already
  demonstrates semantic action directions and activation steering.
- [Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models](https://arxiv.org/abs/2603.19183)
  already distinguishes reusable features from episode-specific memorization and causally steers
  behavior.
- [VLA-Trace](https://arxiv.org/abs/2605.30117) already studies checkpoint CKA, multimodal routing,
  knockout interventions, and rollout behavior.
- [Don't Blind Your VLA](https://arxiv.org/abs/2510.25616) already studies degradation of visual
  representations during action fine-tuning.

Consequently, probe accuracy, CKA, PCA pictures, or generic steering cannot be the principal
contribution. The claim lives or dies on prospective held-out failure prediction and selective
causal intervention.

## Benchmark implications

The official VIMA result table reports 100% Task 1 L2 success for VIMA-92M, which the local
feasibility run reproduced at ceiling. Task 1 therefore cannot test a failure-prediction
hypothesis. Task 16 L2 is the preregistered replacement because the same table reports 42.0%
success and the task composes object grounding, a remembered spatial relation, and two sequential
actions. The adapter preserves the official Task 16 generator rather than constructing a synthetic
replacement.
