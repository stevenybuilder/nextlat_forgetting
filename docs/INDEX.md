# Documentation index

This page is the navigation layer for a research repository that accumulated several prospective
amendments, failed designs, and infrastructure incidents. Historical files are preserved for
provenance; their presence does not make them current authority.

## Authority order

When documents conflict, use this order:

1. A later, explicitly outcome-blind numbered decision for the study in question. For HMM work,
   [D48](DECISION_D48_HMM_LITERATURE_RESET.md) supersedes the bespoke three-regime design.
2. The study-specific current protocol and its frozen manifests.
3. [PREREGISTRATION_AMENDMENT_2026-08-24.md](PREREGISTRATION_AMENDMENT_2026-08-24.md) for core
   Lure-Star H1/H2 and HMM analysis.
4. [FOUNDATIONS.md](FOUNDATIONS.md) for pinned-upstream technical facts and the deviation ledger.
5. The original [`nextlat_v4_predictive_geometry_spec.md`](../nextlat_v4_predictive_geometry_spec.md).

[CURRENT_STATUS.md](CURRENT_STATUS.md) reports lifecycle status but does not amend an estimand.
`RUNLOG.md`, `HANDOFF.md`, and `.agent_state/` are evidence/history, never scientific authority.

## Canonical scientific protocol

### Core H1/H2 and HMM studies

- [DECISION_D48_HMM_LITERATURE_RESET.md](DECISION_D48_HMM_LITERATURE_RESET.md) — withdraws the
  bespoke HMM-0 family from confirmatory use and declares a literature-grounded Mess3/RRXOR
  replacement based on the official NeurIPS 2024 supplemental artifact.

- [PREREGISTRATION_AMENDMENT_2026-08-24.md](PREREGISTRATION_AMENDMENT_2026-08-24.md) — binding
  core analysis, multiplicity, three-regime HMM design, and terminal H3 withdrawal.
- [FOUNDATIONS.md](FOUNDATIONS.md) — pinned-repository cartography and spec/code deviation ledger;
  use its technical facts, not its old lifecycle statements.
- [STIMULUS_DESIGN.md](STIMULUS_DESIGN.md) — Lure-Star construction and the declared edit-position
  limitation.
- [STIMULUS_VALIDITY_AUDIT.md](STIMULUS_VALIDITY_AUDIT.md) — independent construct audit across
  Path-Star, Lure-Star, HMM, CFS-1, and CFS-2.
- [EXTRACTION.md](EXTRACTION.md) — base-only representation and evidence contract after H3 retirement.
- Historical HMM-0 records: [DECISION_D38_HMM_CENTERING_AMENDMENT.md](DECISION_D38_HMM_CENTERING_AMENDMENT.md),
  [HMM_MATRIX.md](HMM_MATRIX.md), and [HMM_EVALUATION.md](HMM_EVALUATION.md) — frozen HMM family,
  execution matrix, and evaluation populations. Preserve them for provenance; D48 forbids opening
  their outcomes as confirmatory evidence.
- [BASE_COMPETENCE_EVALUATION.md](BASE_COMPETENCE_EVALUATION.md) and
  [DECISION_D20_competence_gate.md](DECISION_D20_competence_gate.md) — deterministic competence
  measurement and the GPT exemption/BST control.
- [DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md](DECISION_D42_COMPLETE_MEASUREMENT_SURFACE.md) —
  required measurement surface.
- [DECISION_D46_H1_BRANCH_DECISION_ANALYSIS.md](DECISION_D46_H1_BRANCH_DECISION_ANALYSIS.md) —
  prospective H1-BD-1 at the nontrivial h63 branch-decision state; it cannot rescue legacy H1.
- [CONFIG_DEVIATIONS.md](CONFIG_DEVIATIONS.md) — exact deviations from the pinned upstream configs.
- [EVAL-REVIEW.md](EVAL-REVIEW.md) and
  [INDEPENDENT_CONFIRMATORY_REVIEW.md](INDEPENDENT_CONFIRMATORY_REVIEW.md) — pre-outcome audit and
  launch-integrity review for the core base/HMM matrix only.

### Controlled causal forgetting

- [DECISION_CFS2_STIMULUS_REPAIR.md](DECISION_CFS2_STIMULUS_REPAIR.md) — current CFS-2 scientific
  construction: exact 18/18 and 8/8 overlap with independently checked decomposition, plus the
  binding per-branch blocks-3/7/10 activation-patching contract and named controls.
- [DECISION_DATASET_VALIDITY_LAYERS.md](DECISION_DATASET_VALIDITY_LAYERS.md) — separates controlled,
  oracle-calibration, and external-validity claims.
- CFS-2 manifests under `manifests/cfs2/` are the only repaired causal-study inputs. The frozen
  outcome-blind update manifest and execution envelope bind the exact 64-branch runner/evaluator
  contract; exact parent-checkpoint lineage and explicit compute authorization remain before launch.
- `src/cfs2/patching.py`, `scripts/run_cfs2_patching.py`, and
  `tests/test_cfs2_patching.py` implement the mandatory inference-only state-restoration sweep.
  CFS-2 is incomplete until every trained branch has a hash-bound patching artifact.

### Deferred natural-language extension

These outcome-blind declarations are preserved as possible future work, not as launch instructions
or part of the current milestone. No language data download, profile, or GPU run is queued.

- [TS1_PROTOCOL.md](TS1_PROTOCOL.md) — frozen TinyStories GPT/NextLat parity design.
- [NL1_PROTOCOL.md](NL1_PROTOCOL.md) — frozen FineWeb-Edu external-validity estimand.
- [NL1_FEASIBILITY.md](NL1_FEASIBILITY.md) — future dataset-audit, profile, and budget rule if the
  language program is explicitly resumed.

## Current execution and engineering

- [CURRENT_STATUS.md](CURRENT_STATUS.md) — concise project snapshot and immediate next actions.
- [RUNLOG.md](RUNLOG.md) — append-only factual execution history; later entries supersede earlier
  operational expectations.
- [`HANDOFF.md`](../HANDOFF.md) — cold-pickup narrative. Its opening correction is useful, but the
  older 10/30 HMM and CFS-2-pending sections are stale.
- [`PROGRAM.md`](../PROGRAM.md) — outcome-blind autonomous-engineering constraints.
- [AUTORESEARCH_ENGINEERING.md](AUTORESEARCH_ENGINEERING.md) — engineering-only autoresearch loop;
  scientific metrics cannot select changes.

## Historical and terminal scientific decisions

These files explain how the current scope was reached. They are not launch instructions.

- [DECISION_D39_h3_pilot.md](DECISION_D39_h3_pilot.md),
  [DECISION_D40_h3_overlap_expansion.md](DECISION_D40_h3_overlap_expansion.md), and
  [ADAPTATION_BANKS.md](ADAPTATION_BANKS.md) — H3 feasibility history. D40's permanent block is
  terminal; no adaptation bank is now missing or awaiting selection.
- [DECISION_CFS1_NEW_EXPERIMENT.md](DECISION_CFS1_NEW_EXPERIMENT.md),
  [CFS1_PROTOCOL.md](CFS1_PROTOCOL.md), and [PREREGISTRATION_CFS1.md](PREREGISTRATION_CFS1.md) —
  blocked 18/18/8/7 predecessor design, retained for provenance only.
- [DECISION_CFS1_STATE_INTERCHANGE_COMMITMENT.md](DECISION_CFS1_STATE_INTERCHANGE_COMMITMENT.md) —
  commitment scoped to CFS-1. It does not automatically authorize or bind CFS-2 patching.

## Operational amendments and incident records

Useful for forensic reproducibility, but not part of the public scientific narrative:

- [DECISION_D41_RUNTIME_RECOVERY_AMENDMENT.md](DECISION_D41_RUNTIME_RECOVERY_AMENDMENT.md),
  [DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md](DECISION_D43_MEASUREMENT_AMENDMENT_CONTINUATION.md),
  and [DECISION_D44_OPERATIONAL_DURABILITY_CONTINUATION.md](DECISION_D44_OPERATIONAL_DURABILITY_CONTINUATION.md).
- [RECOVERY_GATE.md](RECOVERY_GATE.md), [COLAB_TRANSPORT.md](COLAB_TRANSPORT.md), and
  [COLAB_PROFILING.md](COLAB_PROFILING.md).

## Reference and review material

- [UPSTREAM_REPORT.md](UPSTREAM_REPORT.md) — detailed source cartography at the pinned NextLat commit.
- [PAPER_NOTES.md](PAPER_NOTES.md) and `paper_source/` — local paper extraction and figure
  digitization; cite the arXiv paper itself in public writing.
- [HMM_DATASET_PLAN.md](HMM_DATASET_PLAN.md) — early single-HMM integration plan, superseded by the
  materialized three-regime family and current HMM documents.
- [STYLE_GUIDE.md](STYLE_GUIDE.md) — blog-writing reference, not a scientific contract.
- `review/` — internal adversarial code-review records.

## Known stale contradictions

| File/claim | Current resolution |
| --- | --- |
| `HANDOFF.md`: 10/30 HMM cells and CFS-2 audit still pending | Historical snapshot; use `CURRENT_STATUS.md` and durable ledgers. |
| `HMM_DATASET_PLAN.md`: integration “has not been executed” | Superseded by the materialized three-regime family, `HMM_MATRIX.md`, and `HMM_EVALUATION.md`. |
| `ADAPTATION_BANKS.md` / D39: more H3 inputs or action remain | D40 permanently retired H3 after four unmatched pairs. |
| CFS-1 preregistration and state-interchange commitment | CFS-1 is blocked. Neither document silently transfers to CFS-2. |
| `FOUNDATIONS.md`: old pre-compute gates remain lifecycle blockers | Retain its source facts; use later receipts/status for lifecycle state. |
| `COLAB_*`: Colab is the supported live execution path | These are Colab-era transport/incident records; current provider state must be queried separately. |
| `results/live_numbers.json`: three seeds, active H3 adaptation, causal patching as a stretch goal | Draft-era prose; five core seeds are frozen, H3 is retired, the original Lure-Star patch is optional, CFS-2 patching is separately mandatory, and TS-1/NL-1 are deferred. Do not publish it as results. |
| Independent confirmatory review says “PASS” | Its authorization covers core base-only H1/H2 and HMM, not later CFS-2 or NL-1 studies. |

## Public-repository boundary

For a clean public release, keep the scientific source, tests, configs, compact manifests,
current protocols, audit, and final result tables. Preserve the following privately or move them
to an explicitly labeled archival supplement rather than presenting them as current docs:

- `.agent_state/`, credentials, rental configuration, session identifiers, tarballs, checkpoints,
  local caches, `.DS_Store`, and recovery scratch artifacts;
- Colab/Vast incident logs, provider balances, bucket paths, and machine-specific absolute paths;
- historical H3/CFS-1 pilot payloads and large JSONL loss files—publish checksums or an artifact
  DOI if reproducibility requires them;
- internal code-review transcripts and autonomous-agent process receipts;
- draft `results/live_numbers.json`, recovered profiling reports, and the unfinished Word/blog
  outputs;
- mirrored arXiv HTML/figures unless redistribution is license-checked; a citation and retrieval
  script are usually sufficient.

Nothing should be deleted until its hashes and provenance role are accounted for. Archive is not
the same as concealment: terminal H3 and CFS-1 decisions should remain visible in a concise
scientific history even if their operational bulk moves out of the main repository.
