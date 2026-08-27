# Predictive geometry and controlled forgetting in NextLat

This repository studies how GPT, BST, and NextLat represent histories with the same or different
futures, and whether the resulting geometry predicts interference during later learning. The core
experiments use controlled symbolic tasks so that future equivalence and causal interventions can
be checked exactly. A separately declared TinyStories/FineWeb-Edu language program is preserved
as deferred future external-validity work; it receives no data or compute in this milestone.

The project is active research, not a finished result release. No claim should be taken from the
draft blog or partial runtime artifacts. See [current status](docs/CURRENT_STATUS.md) and the
[documentation index](docs/INDEX.md) before running or interpreting anything.

## Scientific scope

| Study | Data | Purpose | Current disposition |
| --- | --- | --- | --- |
| Lure-Star H1/H2 | Synthetic Path-Star `G(5,5)` | Test future-sensitive representation geometry and its relationship to branch decisions | Confirmatory base-only study |
| HMM-0 predecessor | Three project-designed 4-state HMM regimes | Historical calibration attempt | 30/30 trained but withdrawn outcome-blind from confirmatory use |
| HMM-LIT-1 | Published Mess3 and RRXOR processes | Compare learned geometry with established belief-state benchmarks and inherited controls | Prospectively declared; official benchmark reproduction precedes training |
| CFS-2 | Fresh synthetic Path-Star interventions | Test controlled causal forgetting under exactly balanced structural overlap and locally intervene on retained decision states | Stimuli, execution envelope, runner, evaluator, and required blocks-3/7/10 activation-patching runner implemented; parent lineage precedes launch |
| TS-1 / NL-1 | TinyStories / filtered FineWeb-Edu | Test implementation parity and transfer to language | Deferred future work; no data or compute in the current milestone |

Two predecessor analyses remain in the repository for provenance but are not runnable scientific
studies:

- Lure-Star H3 was permanently retired when its frozen one-shot matching rule failed.
- CFS-1 is blocked because its low-overlap cells shared 8 versus 7 probe edges, confounding the
  intended interaction. CFS-2 is the separately numbered repair; the two stream families must
  never be mixed.

## Start here

1. [CURRENT_STATUS.md](docs/CURRENT_STATUS.md) — what is complete, running, blocked, or planned.
2. [INDEX.md](docs/INDEX.md) — authority order and a map of current versus historical documents.
3. [PREREGISTRATION_AMENDMENT_2026-08-24.md](docs/PREREGISTRATION_AMENDMENT_2026-08-24.md) — binding
   H1/H2/HMM analysis contract and terminal H3 disposition.
4. [STIMULUS_VALIDITY_AUDIT.md](docs/STIMULUS_VALIDITY_AUDIT.md) — outcome-blind dataset and
   construct audit, including the CFS-1 failure and CFS-2 repair.
5. [DECISION_CFS2_STIMULUS_REPAIR.md](docs/DECISION_CFS2_STIMULUS_REPAIR.md) — the current causal
   study, including the mandatory per-branch activation-patching endpoint;
   [TS1_PROTOCOL.md](docs/TS1_PROTOCOL.md) and [NL1_PROTOCOL.md](docs/NL1_PROTOCOL.md) preserve the
   deferred language-extension designs.
6. [DECISION_D48_HMM_LITERATURE_RESET.md](docs/DECISION_D48_HMM_LITERATURE_RESET.md) — the
   outcome-blind replacement of the bespoke HMM family with literature-grounded Mess3/RRXOR.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Representation, HMM, CFS-1, and CFS-2 scientific code |
| `scripts/` | Materialization, validation, evaluation, and durable execution entry points |
| `configs/` | Explicit configurations derived from the pinned upstream NextLat release |
| `manifests/` | Frozen dataset identities, construction receipts, and preregistration evidence |
| `docs/` | Scientific protocols, decisions, audits, execution notes, and historical incident records |
| `tests/` | Scientific-invariant, provenance, evaluator, and recovery tests |
| `upstream/NextLat/` | Read-only checkout of NextLat commit `3770be6009cea2b3c455a9ce7f2ca88b504bb955` |
| `report/` | Draft writing; not an authoritative result source |

Large generated datasets, checkpoints, credentials, and runtime state are intentionally excluded
from version control. Their identities are represented by manifests and SHA-256 receipts.

## Reproducibility checks

The stimulus audit reads only datasets, manifests, exact solvers, and HMM ground truth—never model
losses or scientific outcomes:

```bash
.venv/bin/python scripts/audit_stimulus_validity.py --write
.venv/bin/pytest -q tests/test_stimulus_validity_audit.py tests/test_cfs2_generate.py
```

The full training and evaluation matrix requires GPU infrastructure and immutable remote inputs;
do not infer authorization from the presence of a launcher. Use the current status and the
study-specific protocol before spending compute.

## Claim boundary

The current project milestone supports claims only about controlled sequence models and planning
tasks; it makes no claim that the result generalizes to ordinary language. TS-1/NL-1 would test
that transfer, but FineWeb-Edu would still provide observational external validity rather than the
exact predictive-state oracle available in Path-Star and the HMM family.
