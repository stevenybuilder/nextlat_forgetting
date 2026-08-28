# Scripts

This directory contains legacy materialization, training, evaluation, profiling, provider, and
recovery entry points. Their presence does not mean their associated experiment is active or
authorized.

New study entry points should live under `scripts/studies/<study>/` and must:

- require a frozen manifest and verify its hash;
- refuse unresolved secrets or untracked config overrides;
- distinguish paper-native reproduction from common comparison;
- emit terminal receipts for success, incompetence, interruption, and numerical failure;
- record occupied wall time as well as inner-step throughput;
- write only to the manifest's descriptive study namespace.

Provider launchers are operational tools, not scientific protocols. Read the active implementation
plan before using any training script.
