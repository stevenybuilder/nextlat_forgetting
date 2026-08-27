# H3 adaptation-bank gate

The checked-in `manifests/b_near.jsonl` is the frozen 5,000-item near bank. The checked-in
`manifests/b_far.jsonl` is a 15,000-item **candidate** bank, not the far training bank. Treating
its first 5,000 rows as `B_far` would violate the preregistered loss-quantile matching design.

`scripts/materialize_adaptation_banks.py` performs serialization and validation only. It does
not open checkpoints or metrics, calculate losses, rank candidates, optimize H3 outcomes, or
make a scientific selection.

## What can be produced now

To serialize the already-frozen near bank into the exact filename expected by the upstream
Path-Star loader:

```bash
.venv/bin/python scripts/materialize_adaptation_banks.py \
  --near-only \
  --output-dir manifests/adapt
```

This writes `graph_5_5_bnear_5000.txt`, its SHA-256 sidecar, and a materialization receipt.
The receipt explicitly says that no scientific selection was performed. It does not certify
that the full adaptation experiment is launchable.

## Inputs still required for full materialization

Full mode requires all three separately frozen inputs:

1. A non-confirmatory pilot loss-quantile selection JSON plus `.sha256` sidecar.
2. An independent 2,000-item near validation bank plus `.sha256` sidecar.
3. An independent 2,000-item far validation bank plus `.sha256` sidecar.

The pilot selection JSON has this contract (the `selection` array must contain exactly 5,000
one-to-one records and cover every near item exactly once):

```json
{
  "schema_version": 1,
  "purpose": "h3_far_loss_quantile_match",
  "selection_method": "non_confirmatory_pilot_loss_quantile_match",
  "near_bank_sha256": "<sha256 of manifests/b_near.jsonl>",
  "candidate_bank_sha256": "<sha256 of manifests/b_far.jsonl>",
  "pilot": {
    "role": "non_confirmatory",
    "checkpoint_sha256": "<sha256>",
    "loss_table_sha256": "<sha256>",
    "selector_code_sha256": "<sha256>",
    "created_at_utc": "2026-08-23T00:00:00Z",
    "frozen_before_confirmatory": true,
    "inspected_confirmatory_checkpoints": false,
    "inspected_confirmatory_results": false,
    "optimized_h3_outcomes": false
  },
  "selection": [
    {
      "near_prompt_sha256": "<sha256>",
      "far_prompt_sha256": "<sha256>",
      "near_loss_quantile": 0.0001,
      "far_loss_quantile": 0.0001
    }
  ]
}
```

Then run:

```bash
.venv/bin/python scripts/materialize_adaptation_banks.py \
  --far-selection manifests/pilot/b_far_loss_quantile_selection.json \
  --near-validation manifests/pilot/b_near_validation.jsonl \
  --far-validation manifests/pilot/b_far_validation.jsonl \
  --output-dir manifests/adapt
```

Before writing anything, the gate verifies every input sidecar, exact sizes, unique prompts and
graph keys, pairwise disjointness of all training and validation outputs, candidate membership,
same-quantile one-to-one pairing in frozen near-item order, and the pilot's non-confirmatory
attestations. It refuses the first 5,000 candidates even if that set is reordered. Confirmatory
checkpoints and result metrics are not accepted as inputs, so they cannot influence
materialization.
