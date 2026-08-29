# Compute budget

Estimate date: **2026-08-27**. The only active compute is deterministic evaluation of 20
already-trained NextLat checkpoints. The hard provider-spend cap is **$5**, and no new training is
authorized.

Status update, 2026-08-28: no project compute is active. The later π0.5 target-control manipulation
check completed for about $0.50 including startup and transfer, below its separate $5 cap. The
shared Vast instance is stopped and no learned-controller run is authorized.

## Active case study

The checkpoint roster contains ten periodic checkpoints from seed 1234 and ten from seed 1235.
Each is evaluated with greedy decoding on the same 20,000 held-out Path-Star graphs. One final
checkpoint is evaluated twice as a determinism check. Checkpoints occupy about 5.3 GB in total and
the corpus is under 3 MB.

| Work package | Scope | Expected occupied time |
| --- | --- | ---: |
| Environment capture | GPU identity/topology, versions, hashes, one smoke evaluation | 15–30 min |
| Frozen trajectory | 20 checkpoint evaluations plus one clean repeat | 1–4 h |
| Transfer and verification | Copy compact JSON/receipts, hash-check, destroy instance | 15–30 min |

A read-only Vast query on 2026-08-27 found verified full-host RTX 4090 offers around
$0.30–$0.45/hour. The expected charge is therefore roughly **$0.50–$2.25**. The $5 stop allows for
marketplace variance or one interrupted evaluation without authorizing training or a seed sweep.

The instance must be destroyed immediately after verified result transfer. The launch is refused
unless there is one visible full-host GPU, no DDP or distributed sampler, enough disk for all
checkpoints and the pinned container, and a resolved hourly rate consistent with the cap.

## Scientific stop rules

- Stop if checkpoint, config, dataset, evaluator, or upstream hashes do not match the frozen roster.
- Stop if deterministic inference cannot be enabled under the pinned runtime.
- Stop if the repeated checkpoint changes any scientific output.
- Stop before $5 of provider spend.
- Do not respond to an unexpected result by training replacement seeds or selecting a different
  checkpoint subset.

## Deferred larger program

The earlier proposal for a 24-seed public-recipe cohort and a possible 24-seed stabilization cohort
is scientifically separate and currently has **zero authorization**. The recovered A100 profile
suggested about 1.40 GPU-hours per 20,000-update NextLat run. At then-observed 4090 prices, the
two-cohort worst case was estimated at $35–$60 with a conservative $100 cap. That estimate is kept
only for future planning; it is not the active experiment.

No budget is allocated to GPT/MTP/JTP/BST/FSP/HiLP comparisons, representation geometry,
controlled forgetting, Mess3/RRXOR retraining, grid walkers, or natural-language data.
