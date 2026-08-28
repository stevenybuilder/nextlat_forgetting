# Documentation index

The active documentation is intentionally small. Numbered decision chains, paper mirrors,
provider incident notes, and superseded drafts from the earlier program were removed from the
working tree so they do not look like current scientific authority. They remain recoverable from
the public Git history at commit `5c71e5f`.

## Active design

The current VLA follow-on is [Geometry of intended futures](../intended_futures/README.md), with a
separate [frozen engineering-pilot preregistration](../intended_futures/PREREGISTRATION.md). It does
not modify or retroactively reinterpret the completed VIMA pilot.

1. [BASIN_CASE_STUDY.md](BASIN_CASE_STUDY.md) — the completed retrospective study of one
   generalizing and one shortcut NextLat trajectory.
2. [BLOG_POST.md](BLOG_POST.md) — a concise research-facing version of the case-study result.
3. [FUTURE_SENSITIVE_GEOMETRY_PILOT.md](FUTURE_SENSITIVE_GEOMETRY_PILOT.md) — the recommended
   no-training follow-on that directly tests the second original hypothesis.
4. [RESEARCH_PLAN.md](RESEARCH_PLAN.md) — four longer-term questions, hypotheses, estimands, and success rules.
5. [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) — literature-parity tiers, stimuli,
   counterfactuals, power, seed handling, and statistics.
6. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — reusable assets, missing implementations,
   execution gates, and deliverables.
7. [COMPUTE_BUDGET.md](COMPUTE_BUDGET.md) — measured profiles, assumptions, uncertainty bands,
   present-day Vast pricing, and staged authorization.
8. [LITERATURE.md](LITERATURE.md) — paper, GitHub, benchmark, and author-source ledger with a
   search cutoff.
9. [REPRODUCIBILITY_FINDING.md](REPRODUCIBILITY_FINDING.md) — the seed-1235, GPU-topology,
   sampler, and nondeterminism result in plain language.
10. [CURRENT_STATUS.md](CURRENT_STATUS.md) — what exists now and the next concrete gate.
11. [SCIENTIFIC_HISTORY.md](SCIENTIFIC_HISTORY.md) — what the legacy phase contributed and why it
   is not the active protocol.

These files are a design slate, not a frozen preregistration. Before a confirmatory launch, create
a machine-readable manifest from the final design and record its hash without looking at the new
outcomes.

## Authority order

For new work, use the frozen study manifest first, then the active design above, then the pinned
primary sources in the literature ledger. Existing configs, scripts, and results describe
implemented legacy behavior unless the implementation plan explicitly promotes them into a new
study.

## Historical provenance

The removed material is not evidence for the active studies, but no historical outcome was erased.
Use `git show 5c71e5f:<path>` when a specific legacy receipt is needed. The cleaned
[scientific history](SCIENTIFIC_HISTORY.md) and
[reproducibility finding](REPRODUCIBILITY_FINDING.md) retain the conclusions that still matter.
New work must use descriptive study names rather than reviving the old decision-number shorthand.

## Public-repository boundary

Tracked source, tests, compact manifests, active protocols, and compact diagnostic evidence may be
public. Keep the following out of Git:

- API keys, credentials, `.env`, `.secrets/`, SSH material, and provider tokens;
- checkpoints, datasets, caches, virtual environments, runtime state, and recovery tarballs;
- account balances, private bucket names, rental instance identifiers, and machine-specific logs;
- copied third-party artifacts whose redistribution license has not been verified.

For a large public artifact, publish a checksum and a separate licensed artifact location rather
than committing it directly.
