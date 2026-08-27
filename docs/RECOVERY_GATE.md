# T4 forced-interruption recovery gate

This gate is an engineering equivalence test, not a scientific experiment. It uses the
pinned, runtime-patched GPT trainer and one separately generated, non-confirmatory Path-Star
corpus. The source, seed (`910241`), generator range (starting at `1,000,000`), update counts,
checkpoint cadence, precision, and comparison tolerances are constants in
`scripts/colab_recovery_gate.py`; none is a launch parameter. Confirmatory seeds 1234–1238
and the frozen confirmatory corpus are not used.

## Recovery design (schema v2)

An August 23 live rehearsal exposed two confounds in the original two-scratch-run design:
`os.killpg` terminated Fabric/TCPStore but did not catch a rank-zero worker that had escaped
the launcher's process group, and separately initialized FP16 runs had already diverged
before the planned interruption. No tolerance was changed in response.

Schema v2 makes one step-150 lineage the parent of both arms:

1. Start one scratch run and wait for the atomic step-150 recovery pointer.
2. Discover the launcher and all recursive descendants by parent PID, regardless of process
   group. Send `SIGSTOP` to every captured PID, repeat discovery until the tree is stable and
   quiescent, then send `SIGKILL` to every captured PID identity. On Colab, an identity
   includes PID and the kernel's `/proc` start tick (the local test fallback uses the process
   start timestamp), preventing PID-reuse mistakes. Snapshotting is forbidden until
   every captured identity is absent from the process table.
3. Upload the two retained verified checkpoint generations, metadata, pointer, metrics,
   materialized config, job identity, and step-0 contract. Artifact objects are written first;
   the immutable `state.json` commit is written last. Every artifact upload uses the GCS
   create-only generation precondition (`if_generation_match=0`), as does the state record,
   so replaying a gate ID cannot replace bytes already named by an immutable commit. A
   partial upload is not a snapshot.
4. Resume the retained local step-150 snapshot at its original absolute path and finish at
   exactly step 300. This is the reference continuation.
5. Move that completed reference tree aside, restore only the committed GCS snapshot back to
   the identical original absolute paths, and finish the recovered continuation at step 300.

Thus the comparison is not between two independent initializations. Both continuations share
the same checkpoint hash, optimizer/scheduler state, RNG state, metric prefix, data cursor,
configuration, and absolute paths. The only arm-level difference is retained-local versus
GCS-restored state after the real process-tree kill.

The pass condition covers final optimizer step, optimizer and scheduler state, RNG state,
exact equality of the FP16 AMP GradScaler state (`lurestar_amp_scaler_state_v1`),
the deterministic distributed-sampler cursor and observed step-150 fast-forward in both
continuations, checkpoint lineage, model weights, and actual GPT logits on a fixed token
probe. It also requires the normalized reference and recovered metric histories to have
identical, consecutive, non-duplicated step sets and equal stable numeric metrics within the
original preregistered tolerances. Throughput telemetry (`steps_per_sec`, `tokens_per_sec`) is
excluded because a process restart changes its timing window.

## Durable value during execution

The runtime publishes append-only progress snapshots every 60 seconds. Each snapshot uploads
the current `metrics.csv` files first, records their SHA-256 hashes and sizes plus current
checkpoint pointers, writes that snapshot's immutable state object last, and then emits a
unique committed-progress event. A final progress snapshot is forced before the result. The
step-150 recovery snapshot and both exact-step-300 finals similarly upload verified artifacts
before immutable state records. Runtime phase events are unique append-only GCS objects.

If `colab exec` returns, fails, or reaches its timeout, the host writes the return code, full
bounded output tail, elapsed time, and immediate runtime status to both the append-only local
receipt log and a unique GCS host-diagnostic object before teardown. A stream return is never
treated as success; only a durable `result.json` with the exact gate ID and source hash can
pass. These records preserve measurements and failure evidence even when a terminal result is
unavailable.

## Local checks (no runtime)

```bash
.venv/bin/python -m pytest tests/test_colab_recovery_gate.py -q
.venv/bin/python scripts/colab_recovery_gate.py
```

The second command only builds the deterministic content-addressed archive and sidecar and
appends a `PREREGISTERED` record. It prints `PREPARED_ONLY=True` and spends no compute.
The focused suite behaviorally replays resume and final commits against an in-memory
create-only object store, exercises hash-identical step-150 lineage binding, kills a nested
three-process-group tree, checks GradScaler missing/mismatch failures, and simulates an exec
timeout through verified runtime teardown. It does not rely on source-substring assertions.

## Paid T4 execution

```bash
.venv/bin/python scripts/colab_recovery_gate.py --run
```

Execution fails closed unless two Colab status reads, 30 seconds apart, agree that no runtime
is active; the account remains above the project hard-stop balance; and local ADC exists with
mode `0600`. The runtime uses the Python Google Cloud Storage client only. A heartbeat is
printed every 30 seconds, child output is relayed line by line, and teardown uses `colab stop`
followed by two agreeing stopped-state reads and two settled zero-burn quota reads even if the
gate fails.

Host receipts append to `results/recovery_gate_receipts.jsonl`. Runtime events, progress
snapshots, terminal diagnostics, final checkpoint states, and the final result are unique
objects under
`gs://YOUR_PRIVATE_BUCKET/lurestar/recovery-gates/<gate_id>/`.
The gate-specific snapshot schema deliberately does not invoke `RuntimeDurability.sync_job`,
which requires a MatrixRunner ledger; inserting a non-confirmatory engineering run into the
confirmatory ledger would violate isolation.

## Posthoc engineering clearance

Gate `rg-d21e8fee468a-1787545664418856000-9899dbda` retains its immutable schema-v2
`result.json` with `passed: false`. A separate, content-addressed receipt may clear only the
engineering restore/replay question when a fail-closed mechanical audit proves all of the
following: both arms share the same step-150 parent; their durable step-300 checkpoint bytes are
identical; weights, optimizer, scheduler, RNG, AMP scaler, probe logits, and normalized metrics
all have `max_abs = max_rel = 0`; and the sole false check is marker observation from the bounded
resume-log tails. This does not reclassify the original gate, change a tolerance, or create a
scientific result.

The executed source archive used `collections.deque(maxlen=300)`. Both fast-forward messages were
emitted near the start of their respective 150-step continuations and fell out of those diagnostic
tails before comparison. The later 5,000-line tail prevents recurrence for the current workload,
but it remains a bounded log rather than a proof primitive. The current harness now latches each
fast-forward as a structured event, atomically persists it at observation time, and compares the
structured snapshots rather than any log text. Both arm snapshots and their schema are embedded in
the terminal result. A behavioral test emits enough later output to evict the marker from the
5,000-line tail and verifies that both the in-memory and on-disk latch still prove the event.

Generate or re-verify the immutable supplement directly from its GCS source:

```bash
.venv/bin/python scripts/create_recovery_clearance_receipt.py
.venv/bin/python -m pytest tests/test_recovery_clearance_receipt.py -q
```

The generator is pinned to the original result SHA-256 and GCS generation, the executed source
archive and harness hashes, the shared parent, both final checkpoint hashes, and the preregistration
hash. It refuses any other failed check, any nonzero comparison delta, any modified object
generation, or any attempt to overwrite different receipt bytes.
