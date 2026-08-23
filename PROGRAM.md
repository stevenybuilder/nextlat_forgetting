# PROGRAM.md — the autonomous loop's constitution

This file governs what the autonomous agents may change while the human is away. It is the
analog of the `program.md` in Karpathy's autoresearch harness, with one deliberate inversion:
**autoresearch lets the agent mutate `train.py` freely and keeps whatever lowers val_bpb.
Here, the training configuration is frozen and hillclimbing a scientific metric is forbidden.**

The project is a preregistered confirmatory study (spec §6, §10). An agent that tuned anything
toward a better PSI, a larger interference gap, or a stronger belief-geometry correlation would
be manufacturing the result, not measuring it.

## Frozen surface — never modified by any agent, for any reason

From spec §8 and §11:

- model width, depth, heads, `n_embd`, `proj_factor`, `mtp_horizon`
- optimizer, learning rate, schedule, betas, weight decay, gradient clipping
- `lambda_mse`, `lambda_kl`
- effective batch size (512 Lure-Star / 256 HMM) and the optimizer-update count
- base dataset size (200,000 train / 20,000 heldout graphs; 100,000 HMM sequences)
- base training steps (20,000 Lure-Star / 3,000 HMM)
- the three confirmatory seeds: 1234, 1235, 1236
- graph topology `G(5,5)`, `stargraph_max_nodes: 100`
- the preregistered hypotheses, their primary metrics, and their primary distance
  (centered cosine; whitened Euclidean is a declared robustness check, not an alternative)
- every threshold frozen into a manifest: HMM JS-divergence and edit-distance cuts, the
  `B_far` loss-quantile mapping, the `A_pair` item set

The one permitted execution fallback is gradient accumulation that preserves effective batch
size *and* update count. It must be recorded as a deviation in `docs/FOUNDATIONS.md`.

## Mutable surface — where iteration is legitimate and encouraged

- data generation and serialization throughput, provided output stays hash-identical
- dataloader workers, prefetch, pinning, and checkpoint I/O scheduling
- durable-checkpoint and resume machinery, the run ledger, and recovery logic
- GCS transfer batching and region placement
- analysis, plotting, and bootstrap code, provided the estimator and its inferential unit
  do not change
- the writeup in `report/` and the docs in `docs/`
- test coverage

## Loop invariants

1. **Append-only ledger.** Every run writes to `results/run_ledger.json` and
   `results/metrics.jsonl`. Nothing is ever rewritten in place. A wrong entry is corrected by
   appending a superseding entry with a reason.
2. **Fixed budget per iteration.** Profile before sweeping (spec §11). No job launches
   without a measured seconds-per-step and a projected compute-unit cost.
3. **Durability before speed.** No required artifact may exist only on a Colab runtime.
   A step that is not on GCS did not happen.
4. **Report all preregistered metrics.** Including the nulls. Spec §10 is explicit: report
   every preregistered metric even if only one is positive. Dropping a null is fabrication.
5. **Stop rather than shrink.** If the hardware cannot run the exact configuration, the loop
   pauses for a compute decision. It never quietly reduces the confirmatory model.
6. **Every claim carries its rung.** R0 speculation through R4 replicated-and-controlled, as
   defined in the `unslop` agent. The writeup states the rung.

## Standing agent cadence

- `run-qa` — sweeps the ledger, checkpoint lineage, tests, leakage hashes, Colab liveness, and
  rendered figures. Ranks findings by which downstream phase they kill.
- `data-throughput` — classifies the bottleneck before touching anything; only data-path
  levers are in scope.
- `unslop` — audits the writeup and the implementation against the actual arXiv:2511.05963v4
  paper and the spec. Holds a veto: `BLOCKED` means nothing is shared.

## Stop conditions (spec §10) — halt and document, do not work around

- safe and critical lures cannot be exactly matched
- interrupted training cannot resume reproducibly enough for the stated analysis
- either model stays below the 90% exact-path base competence gate
- near/far branches cannot achieve comparable acquisition or initial difficulty
- a geometry effect appears only under a post-hoc metric or layer
- the result depends on a single seed
- the available hardware cannot run the exact paper-scale configuration reliably
