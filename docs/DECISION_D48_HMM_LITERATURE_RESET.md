# D48 — Literature-grounded HMM reset

**Decision date:** 2026-08-25  
**Status:** prospective and outcome-blind; no predecessor HMM geometry outcome had been opened  
**Supersedes:** the bespoke three-regime 4-state HMM family as confirmatory evidence

## Decision

The 30 completed checkpoints from the project-designed
`persistent_moderate`, `fast_mixing_moderate`, and `persistent_high_aliasing` family are retained
for provenance but reclassified as **HMM-0 predecessor work**. They must not be evaluated or
reported as confirmatory support for the project. Mechanical validation of a hand-designed
generator does not establish that its regimes, stimulus coverage, or endpoints are accepted
scientific benchmarks.

The replacement study, **HMM-LIT-1**, will use published data-generating processes and evaluation
controls from Shai et al., *Transformers Represent Belief State Geometry in their Residual Stream*
(NeurIPS 2024):

- [peer-reviewed paper and official supplemental artifact](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html);
- [arXiv manuscript](https://arxiv.org/abs/2405.15943);
- official supplemental archive SHA-256:
  `e9aa54c7098360529e6bad028b838a42533b941bcc48f650252fe802e736f07e`.

The canonical primary processes are:

1. **Mess3**, using the official parameters `x=0.15`, `a=0.6`. Its mixed-state presentation has a
   nontrivial fractal belief geometry and tests whether that geometry is recoverable from the
   residual stream.
2. **RRXOR**, using the official parameters `pR1=0.5`, `pR2=0.5`. Its belief states include
   distinctions that are degenerate for immediate next-token prediction, making it the stronger
   test of future-sensitive rather than merely local predictive structure.

These are stochastic-process benchmarks, not fixed finite corpora. New sequences must be sampled
from the exact published generators. Scientific provenance comes from fixing the authors'
transition matrices, parameters, initial-state rule, tokenization, and evaluation procedures—not
from preserving an arbitrary downloaded list of sampled rows.

## Required benchmark inheritance

HMM-LIT-1 inherits the following evaluation surface from the NeurIPS study before adding any
NextLat-specific comparison:

- next-token validation loss relative to the analytic optimum;
- held-out affine decoding of the exact mixed-state belief vector;
- the published shuffled input-to-belief correspondence null;
- pairwise preservation of ground-truth belief-state distances;
- comparison against distances between next-token predictive distributions, especially on RRXOR;
- layerwise analysis and the published across-layer representation test for RRXOR.

Full predictive-distribution KL/total-variation error and length generalization may be added only as
named secondary endpoints grounded in later published or public-preprint HMM inference work, such
as [Hu, Liu, and Jin (2024)](https://arxiv.org/abs/2406.04089) and
[Dai et al. (2026)](https://arxiv.org/abs/2607.22646). They may not replace a failed inherited
benchmark endpoint.

## Reproduction anchor before NextLat

Before training a NextLat comparison, the project must:

1. acquire the official NeurIPS supplemental archive by its pinned URL and verify the SHA-256;
2. reproduce the authors' Mess3 and RRXOR process outputs from their packaged code;
3. run the inherited evaluator against the packaged saved transformer checkpoints and reproduce
   the published qualitative geometry and quantitative controls within stated numerical tolerance;
4. freeze a matched GPT/NextLat architecture, optimizer, seed roster, and competence rule without
   examining new comparison outcomes.

The supplemental archive contains no explicit license file. Do not copy its source into this
repository until redistribution terms are clarified. It may be used as an external, hash-pinned
reproduction artifact; the process matrices printed in the paper can support an independently
implemented compatibility layer if needed.

## Claim boundary

HMM-LIT-1 can calibrate claims about predictive/belief-state geometry. It is not a causal-forgetting
experiment and does not establish language-domain generalization. A NextLat advantage can be
claimed only if both model arms meet the frozen predictive-competence rule and the result survives
the inherited benchmark controls across the predeclared seed population.

The HMM-0 checkpoints cannot be substituted into HMM-LIT-1, used as extra seeds, or inspected to
tune the replacement design.
