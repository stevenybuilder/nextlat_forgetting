# HANDOFF — NextLat predictive geometry

Updated **2026-08-24 19:30 EDT**. This is the cold-pickup entry point, not a scientific
amendment. Read [`README.md`](README.md), then [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
and [`docs/INDEX.md`](docs/INDEX.md). The remote ledgers remain authoritative for live execution.

## Current state

- **HMM:** all 30/30 cells are durably `TRAINED` at exactly 3,000 updates with checkpoint paths
  and SHA-256s. HMM scientific evaluation has not been opened.
- **Lure-Star bases:** GPT seeds 1234--1237 are durably `DONE` at 20,000 updates with competence
  receipts. GPT seed 1238 is training on Vast instance `48593365`; BST seed 1236 is training on
  `48598595`; and BST seed 1234 is training on the third worker, `48625433`. The allowlists are
  disjoint. The remaining eight base jobs are not verified complete.
- **CFS-1:** blocked permanently for production claims by the 8-versus-7 low-overlap confound.
- **CFS-2:** balanced 18/18/8/8 stimuli, independent construct audit, frozen 64-branch execution
  envelope, runner, recovery, evaluator preflight, and three-site activation-patching runner are
  implemented. No branch has launched. The blocks-3/7/10 patching sweep with all named controls is
  required for every completed branch; it is not an optional post-hoc analysis.
- **Lure-Star H3:** permanently retired by its prospectively frozen matching stop rule. It is not
  pending work.
- **H1-BD-1:** separately declared h63 branch-decision analysis is code-ready; no outcome opened.
- **TS-1/NL-1:** the TinyStories parity and FineWeb-Edu external-validity protocols remain frozen
  for provenance, but both are deferred outside the current milestone. No corpus download,
  timing profile, or language GPU run is in the active queue.

## Vast worker policy

Supervisor is retained so training survives laptop, SSH, and Codex disconnects. It is not allowed
to retry every nonzero exit:

- incomplete training and explicit transport failures may resume;
- a `TRAINED` checkpoint with evaluator identity, schema, or hash failure is quarantined and exits
  cleanly without an autorestart loop;
- unknown post-training failures also quarantine for review rather than repeatedly spending GPU;
- Colab re-exec/reconnection behavior is not inherited by the Vast adapter;
- a restart skips broad recovery download only when the worker ledger and every selected terminal
  artifact verify locally by exact path, step, and SHA-256. Otherwise scoped durable restore runs.

The GPT-1234 incident was caused by a provider-adapter path mismatch. Training froze
`/content/project/manifests/corpus.sha256`; evaluation was initially given an identical copy at
`/content/lurestar/manifests/corpus.sha256`. Absolute path is part of the frozen identity, so the
refusal was correct. The adapter now uses the original frozen path. No training was lost or
repeated.

## Minimal next actions

1. Let the three active base cells finish. For each: evaluate once against its frozen competence
   identity, promote to `DONE`, and verify the durable ledger/state receipt before the next cell.
2. Evaluate the 30 completed HMM cells under the frozen HMM contract, then aggregate by seed. Do
   not treat training loss as an HMM scientific result.
3. After all required bases are `DONE`, run Lure-Star extraction, H1/H2, and the separately named
   H1-BD-1 analysis.
4. Launch CFS-2 only after exact parent-checkpoint lineage is materialized and compute is
   explicitly authorized. Never use a CFS-1 stream in CFS-2. Run the required inference-only
   activation-patching sweep for every completed CFS-2 branch.

## Scientific boundaries

- The core controlled studies use synthetic Path-Star and exact HMM data for internal validity.
- CFS-2 is the controlled causal-forgetting study; it does not establish causal mediation in
  ordinary language.
- TS-1/NL-1 are deferred external-validity studies and cannot rescue a core null. The current
  project makes no ordinary-language generalization claim.
- Legacy H1 at h62 remains unchanged. H1-BD-1 at h63 cannot relabel or rescue it.
- Report all frozen endpoints and nulls. Do not tune generators, metrics, layers, thresholds, or
  exclusions after looking at outcomes.
- A checkpoint is not a result: `TRAINED`, competence `DONE`, evidence extraction, scientific
  evaluation, aggregation, and interpretation are distinct lifecycle states.

## Operational checks

Query provider state using safe field selection; never print raw instance JSON because it may
contain Jupyter tokens:

```bash
vastai show instances --raw \
  | jq '[.[] | {id,actual_status,intended_status,gpu_name,dph_total,label}]'
```

Use focused tests for the changed component during a run. The current cross-study focused check is:

```bash
.venv/bin/pytest -q \
  tests/test_cfs2_generate.py tests/test_run_cfs2_matrix.py \
  tests/test_evaluate_cfs2.py tests/test_stimulus_validity_audit.py \
  tests/test_run_cfs1_matrix.py tests/test_lurestar_branch_decision.py \
  tests/test_nl1_declaration.py tests/test_vast_run_base_matrix.py
```

Run the full suite only before freezing a new scientific source bundle or after a cross-cutting
code change—not on every checkpoint, reconnect, evaluation, or provider restart.

Credentials, Jupyter tokens, `.env`, local session state, and provider logs must never be committed.
The documentation index classifies historical amendments and incident records; their presence is
provenance, not an instruction to rerun old gates.
