# D44 — operational durability repair after D43 clearance

**Date:** 2026-08-24  
**Lifecycle:** training began before this repair; exactly 10 of the 30 HMM training cells are complete, 20 are pending, and no new cell is started by this decision.  
**Outcome visibility:** no scientific evaluation has run or been opened; evaluator invocations and scientific-metric inspections remain zero.

## Decision

D44 is a narrow operational amendment to the Colab durability controller. It repairs only the receipt-pinned recovery synchronization path required to make an interrupted session recoverable without changing scientific execution. It is not a new measurement amendment, a training amendment, or permission to inspect outcomes.

The immutable predecessor is the exact D43 source archive:

`baf9fa4986956b4ab8aa3e07b6f1fe74e570a848ac2d550667177560b68b258d`

Its source archive and issued D43 receipt remain preserved, hash-bound inputs. D44 must not replace, reinterpret, or regenerate D43 evidence. A D44 successor may differ from that archive only at these five source paths:

- `scripts/colab_train_loop.py`
- `tests/test_colab_train_loop.py`
- `docs/DECISION_D44_OPERATIONAL_DURABILITY_CONTINUATION.md`
- `scripts/d44_operational_durability_gate.py`
- `tests/test_d44_operational_durability_gate.py`

The declaration binds every changed byte before and after. The gate compares every packaged successor file with the live tree, rejects all other additions, removals, and edits, and checks that the controller delta is confined to `RuntimeDurability.sync_job`, `RuntimeDurability._artifact_paths`, and narrowly named private `RuntimeDurability._d44_*` helpers. Imports, constants, dispatch, job specifications, model/data/configuration code, training commands, evaluators, and all other controller behavior must stay AST-identical.

## Scientific invariants

The frozen data, manifests, configurations, models, optimizer, update targets, seeds, objectives, checkpoint payloads, measurement/extraction implementation, evaluation implementation, and statistical decision rules are unchanged. In particular, this decision does not authorize a selector or clearance integration, a model launch, an optimizer update, a representation extraction, an evaluator invocation, aggregation, or reading a scientific metric.

The ten persistent-moderate exact-step-3,000 checkpoints remain predecessor-created, generation-pinned, read-only inputs. D44 neither opens their model payloads nor selects among them. The remaining twenty cells remain the only pending HMM training work.

## Evidence and continuation rule

Before any later integration can use D44, the D44 gate must have a source-bound full-test receipt, an independent outcome-blind operational review, and a metadata-only incident attestation. The review and attestation bind one another without a hash cycle. They must state all of the following exactly: training started; 10 HMM cells complete; 20 pending; zero new cells; scientific evaluation not started; zero evaluator invocations; and zero scientific-metric inspections.

The D44 receipt is an operational proof only. It does **not** itself select a continuation gate, change a clearance, or start a Colab runtime. Those integrations must separately bind the exact D44 receipt and are subject to their own review and launch checks.
