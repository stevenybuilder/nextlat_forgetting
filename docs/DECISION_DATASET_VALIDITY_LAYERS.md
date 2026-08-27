# Decision — dataset validity is layered, not implied by scale

**Recorded:** 2026-08-24, before any CFS adaptation branch was trained or any
CFS scientific outcome was opened.

## Decision

The project will not treat “synthetic” as either an automatic strength or an
automatic defect. Each dataset must be justified against the construct it is
supposed to identify:

| layer | data | what it can identify | what it cannot establish |
| --- | --- | --- | --- |
| controlled task | Path-Star from the pinned NextLat `G(5,5)` generator | future-equivalence geometry, branch margins, and—only with exactly balanced interventions—causal forgetting | behavior on ordinary language or real environments |
| oracle calibration | three model-blind HMM regimes with exact posteriors | whether representation distances track known predictive-state relationships | linguistic or ecological validity |
| external-validity extension | a separately numbered language study, if its stimuli pass their own audit | whether the qualitative geometry carries into language-form inputs | exact predictive equivalence unless an independent target construction supplies it |

The first two layers are the core study. They are intentionally controlled
because their graph and HMM states give us ground truth that FineWeb-Edu or
free-form stories do not. They must nevertheless pass a human-readable
construct audit; generator tests alone prove implementation consistency, not
that the intended causal contrast is unconfounded.

## Immediate repairs

1. Preserve CFS-1 as an immutable design record and block its confirmatory
   launch. Its low-overlap `same` and `different` cells share 8 versus 7 probe
   edges, so the overlap-by-conflict estimand is not cleanly identified.
2. Construct CFS-2 from fresh identifiers and seeds. Require exact equality of
   shared-edge counts inside the low pair and inside the high pair, while
   retaining the source, goal, graph degree, token multiset, lengths, stream
   positions, answer relationships, and contamination exclusions.
3. Treat index 62 and index 63 according to what they actually encode. Index
   62 is the delimiter state immediately before a copied source token; index
   63 is the first nontrivial branch decision. Preserve index 62 as the original
   control/legacy estimand and prospectively bind index 63 as the
   mechanistically relevant decision-state analysis before opening outcomes.
4. Produce a model-outcome-blind, human-readable audit across Path-Star,
   Lure-Star, HMM, CFS-1, and CFS-2. The audit must list examples and calculate
   the balance quantities rather than merely cite passing unit tests.

## Why natural-language corpora are not silently substituted

The Path-Star checkpoints use a task-specific vocabulary and objective. A
TinyStories or FineWeb-Edu file cannot simply be added as another evaluation
split; doing so requires training a separate language-model family and defining
an observable counterpart of “same predictive state.” Substituting language
data without that target would change the research question while adding no
valid causal control.

TinyStories is useful because the official NextLat code and paper already use
it and it exercises language-form sequences, but its stories are model
generated. It therefore supplies broader surface validity, not evidence on a
real-world web corpus. FineWeb-Edu is filtered web text, with greater ecological
breadth but no guarantee that every document is human-authored or free of
duplication/model-generated contamination; it is also much more expensive and
does not expose oracle future distributions. Manhattan supplies
structured real-world trajectories, not natural language, and the project has
already documented that author-scale retraining is outside the current compute
budget.

## Separately numbered language extension (NL-1)

NL-1 is a planned external-validity extension, not a rescue analysis for the
core results. Before compute, it needs its own protocol specifying:

- corpus provenance and license, immutable train/validation splits, tokenizer,
  and contamination checks;
- model/checkpoint provenance and a measured compute budget;
- a model-independent rule for forming history pairs and their future targets;
- frozen lexical, length, frequency, and topic controls;
- a primary layer, token position, distance metric, and seed-level estimator;
- negative controls and a rule that all results—including nulls—are reported.

The selected implementation candidate is a compute-profiled FineWeb-Edu study
aligned with the official 100M-class NextLat/GPT pipeline, explicitly labelled
“filtered web-text external validity” rather than “human-authored ground truth.”
It must use immutable document-hash-disjoint splits and disclose residual corpus
contamination risk. Its full-token and any fixed budget fallback are chosen by
a predeclared timing/cost rule before scientific outcomes, never by whichever
version produces a stronger effect. NL-1 does not alter the core Path-Star/HMM
estimands, and its result cannot upgrade a failed core confirmatory claim.

## Claim boundary

If CFS-2 and the HMM calibration succeed, the defensible contribution is a
controlled account of predictive-state geometry plus a causally identified
test of interference in that controlled task family. Without NL-1, the paper
must say that transfer to ordinary language is unknown. With NL-1, the claim is
only the exact qualitative generalization tested there; it is not a license to
generalize to language models at large.
