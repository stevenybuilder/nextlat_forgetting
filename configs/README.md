# Configurations

The YAML files currently at this directory root belong to the legacy GPT/NextLat/BST Path-Star,
custom-HMM, and controlled-forgetting program. They are retained because manifests, receipts, and
tests refer to their paths and hashes.

New configurations belong under:

```text
configs/studies/planner_training/
configs/studies/future_sensitive_geometry/
configs/studies/controlled_forgetting/
configs/studies/exact_predictive_states/
```

Every new config must record its evidence tier (paper-native or common comparison), source commit,
source config, intentional deviations, seed mapping, runtime topology, and frozen study manifest.
Do not modify a legacy config in place to create a new study.
