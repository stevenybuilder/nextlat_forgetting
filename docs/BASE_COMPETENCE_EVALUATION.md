# Production base-competence evaluation

The binding base competence result is a separate offline evaluation. The upstream training
metric remains paper-comparable ancestral sampling (`temperature=1`, no top-k restriction), while
the binding gate is deterministic greedy decoding (`top_k=1`, semantic `temperature=0`). These are
reported separately and never substituted for one another.

The evaluation plan is frozen **before base training starts**. Every base `JobSpec` and every
append-only ledger state binds the resolved evaluator-source path/SHA-256, held-out dataset
path/SHA-256, every evaluation-manifest path/SHA-256, and the deterministic decoding contract in
`competence_identity`. `run_matrix.py` refuses a fresh launch if any input is absent or the manifest
does not exactly bind the dataset name and digest; it refuses a resume if any value changed.

## Contract

`scripts/evaluate_base_competence.py` evaluates the exact frozen held-out
`graph_5_5_test_20000.txt` corpus. It:

- accepts only the pinned upstream commit `3770be6009cea2b3c455a9ce7f2ca88b504bb955`;
- reconstructs GPT, NextLat, or BST from the run's own `materialized_config.yaml` and loads the
  ledger-hashed final checkpoint without an optimizer;
- replays the upstream StarGraph tokenizer and rejects any row that is not the exact 69-token
  G(5,5) geometry (`=` at index 62, five answer tokens, EOS at index 68);
- requires a sha256sum-style frozen manifest row that exactly binds the held-out dataset name and
  digest;
- performs five explicit `torch.argmax` steps. It never calls upstream `generate`, whose
  implementation divides by temperature and then samples;
- writes integer exact-path and per-token counts, plus checkpoint, evaluator, source config,
  materialized config, dataset, manifest, upstream-commit, and runtime provenance; and
- atomically checkpoints partial counts after every inference batch in
  `evaluation/exact_path_raw.json.progress.json`. A reconnect resumes at `next_index`; a provenance
  mismatch is refused. Thus a disconnect can lose at most the current inference batch once the
  evaluation directory is included in the regular GCS durability sync.

The raw output uses `nextlat_forgetting/exact_path_evaluation/1`, the schema consumed by
`scripts/materialize_base_competence.py`.

## Fail-closed lifecycle hook

After `scripts/run_matrix.py --phase base` returns successfully, and before any adaptation plan or
session terminal marker, the runtime driver must invoke:

```bash
python scripts/evaluate_trained_bases.py \
  --ledger /content/lurestar/run_ledger.json \
  --upstream /content/project/upstream/NextLat \
  --dataset /content/lurestar/data/stargraph/graph_5_5_test_20000.txt \
  --manifest /content/lurestar/manifests/corpus.sha256 \
  --precision bf16-mixed --devices 1
```

The orchestrator requires every requested base ledger job to exist. For each job it verifies the
checkpoint, source-config hash, materialized config, every CSV metrics artifact, and the training
completion receipt. Before launching inference it recomputes the requested competence identity and
requires exact equality with the one frozen before training. It then evaluates through Lightning
Fabric and calls the production materializer. A failed evaluator or receipt check leaves the
append-only ledger at `TRAINED`.
`run_matrix.py` already refuses every adaptation parent that is not `DONE` with a verified receipt,
so this is a hard gate rather than a warning.

The orchestrator is idempotent. It skips hash-verified `DONE` entries and, after a disconnect,
first attempts to promote an already atomic raw output without repeating inference.

Promotion repeats the same checks and fails closed: the raw output's checkpoint, evaluator,
dataset, manifest hashes, decoding, model, seed, and integer counts must agree; the canonical
receipt embeds the exact pre-training `competence_identity`; and every parent training artifact
must still hash to its ledger value. Adaptation later re-verifies that complete chain, rather than
trusting the final checkpoint alone.

The Colab durability collector must upload and restore `evaluation/*.json` and
`evaluation/*.sha256` before committing that run's `state.json`. In particular, it must include
the raw result, progress receipt, canonical `base_competence.json`, and its SHA sidecar. The driver
must count a base session complete only after all requested entries are `DONE`, not merely
`TRAINED`.
