# D41: Runtime recovery amendment after the first confirmatory HMM interruption

**Status:** operational amendment; scientific estimands, models, data, seeds, thresholds,
and decision rules unchanged

**Predecessor frozen source:**
`a962cdb94c865e16c2c7c86d5c18b9cc2d3bd301feeea12e42075751f52c9285`

## Incident

The first A100 training runtime completed the ten `persistent_moderate` cells and committed
their step-3,000 checkpoints per job.  The runtime then disconnected before the global HMM
ledger and the per-job commit records were observed to agree.  A replacement runtime restored
the checkpoint payloads but trusted the stale global ledger, relaunched completed GPT cells,
and produced protocol-invalid step-3,001 checkpoints for GPT seeds 1234–1237. The invalid retry
also entered GPT seed 1238 but was stopped with its durable checkpoint still at step 3,000. The
host controller was stopped as soon as this mismatch was identified. Two subsequent status/quota
pairs agreed that no runtime was active and burn rate was zero; settled balance was
1779.3814353793016 CU.

No scientific HMM evaluation receipt was opened or interpreted. Raw training-loss text was
visible in the operational stream but was not used to alter any hypothesis, endpoint, threshold,
seed, regime, model, or aggregation rule. Recovery selection uses only job ID, exact step,
generation, size, and cryptographic hashes—not loss or any scientific metric.

## Evidence retained

- All ten original step-3,000 checkpoint payloads and metadata sidecars remain in GCS.
- Every retained checkpoint is keyed by job, step, size, generation, and SHA-256.
- The step-3,001 objects are retained as incident evidence but are ineligible for analysis.
- Superseded state records are archived under
  `lurestar/recovery_audit/exact-target-3000/` before any live state repair.
- `.agent_state/hmm-exact-target-state-repair.json` records the outcome-blind recovery of any
  poisoned live state pointer.
- Original terminal state/config generations recovered through GCS soft-delete, when used, are
  recorded by generation and SHA-256.  Recovery never selects among scientific outcomes.

## Required repair

The successor source may continue from the predecessor only under all of these conditions:

1. The launch sidecar explicitly names the exact predecessor source SHA-256.  The field is
   included in the canonical job-spec hash and confirmatory clearance.
2. Runtime restore accepts either the current source or that one declared predecessor; any
   other source fails closed.  Every new commit is stamped with the successor source and logs
   the source migration. Durable state and completion receipts retain both the checkpoint-creation
   source and the successor source that terminalized it.
3. A verified checkpoint already at the exact absolute training target is terminalized without
   invoking the trainer.  A checkpoint beyond the target fails closed and is never resumed.
4. Per-job state remains the authoritative checkpoint commit.  A stale or missing global ledger
   may be reconstructed from exact-target artifacts but may never cause an additional optimizer
   update.
5. Before relaunch, an atomic clearance-bound barrier must require exactly all ten expected
   `persistent_moderate` cells, resolve each to its exact step-3,000 checkpoint, and cross-check
   filename step, metadata step, payload `training_steps`, sidecar, source identity, and training
   artifacts. Any missing, extra, or over-target state aborts before any remaining job launches.
6. Retry telemetry and logger versions not named by the original exact terminal state are
   quarantined as incident evidence and cannot enter a successor completion receipt.

## Evaluation durability discovered during the audit

The same audit found that HMM representation-cache chunks were runtime-local: the previous
telemetry glob captured top-level evaluation JSON but not recursive cache chunks.  Before any
evaluation run, the successor controller must:

- verify every cache chunk and SHA sidecar named by `progress.json`;
- upload immutable dependencies before the exact progress bytes;
- publish the telemetry receipt last;
- restore the progress pointer last and re-verify the complete cache; and
- retain the representation manifest and final evaluation receipt/sidecar.

This changes only failure recovery.  It does not change extraction, representations, metrics,
statistical tests, or the fixed thirty-cell aggregation.

## Relaunch gate

No additional paid runtime may start until the successor source passes the full test suite, an
independent outcome-blind scientific/operational review, revalidation of unchanged original gates,
and a truthful D41 continuation gate that states confirmatory compute has started while scientific
evaluation remains unopened. It must also bind a source-diff receipt proving that no scientific
code/input changed and require the successor runtime to match the predecessor environment:
`NVIDIA A100-SXM4-40GB`, torch `2.11.0+cu128`, CUDA `12.8`, BF16 support, and pinned upstream
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`. A new job-specific clearance must bind those
receipts. The recovered ten-cell inventory must be verified before the remaining twenty training
cells begin. Evaluation remains a separate job and may aggregate only after exactly thirty
verified `DONE` receipts exist.
