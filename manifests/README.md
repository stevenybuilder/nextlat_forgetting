# Manifests and receipts

Existing root subdirectories preserve the identities and execution history of the legacy program.
They must not be edited to make the new design appear complete.

New freezes belong at:

```text
manifests/studies/<study>/<freeze_id>/
```

A freeze includes the source/container identities, data and split hashes, ordered seeds, resolved
configs, competence and stopping rules, endpoints, statistical code hash, profile receipt, cost cap,
and artifact-retention policy. A result is interpretable only when it resolves to exactly one such
freeze.

Large data and checkpoints remain outside Git; publish their hashes and licensed artifact location.
