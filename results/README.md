# Results and diagnostic evidence

Tracked files in this directory are compact legacy evidence, profiles, and reproducibility
diagnostics. They are not a curated final-results release and must not be aggregated with the new
program without an explicit migration receipt.

New outputs belong at:

```text
results/studies/<study>/<freeze_id>/
```

Each aggregate must be generated from terminal receipts, report attempted and failed seeds, and
separate paper-native, common-comparison, unconditional, and competent-only populations. Never edit
an aggregate by hand to remove a failed or inconvenient run.

Checkpoints, raw datasets, credentials, provider state, and recovery bundles do not belong in Git.
