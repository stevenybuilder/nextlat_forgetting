# Current project status

Snapshot: **2026-08-24 19:30 EDT**. This is a lifecycle summary, not a scientific amendment or a
live process monitor. Remote ledgers and provider state remain authoritative for execution; update
this page only from durable receipts, never from a progress estimate.

No H1/H2, HMM-geometry, or CFS confirmatory outcome has been opened or reported in this snapshot.

## Execution snapshot

| Workstream | Durable status | What remains |
| --- | --- | --- |
| HMM-0 predecessor | **30/30 trained; confirmatory use withdrawn outcome-blind.** These checkpoints use the project-designed three-regime family and remain unopened. | Preserve for provenance; do not evaluate or report as confirmatory evidence. |
| HMM-LIT-1 replacement | **Prospectively declared in D48.** Mess3 and RRXOR, their generators, saved checkpoints, and inherited controls come from the official NeurIPS 2024 supplemental artifact. | Reproduce the packaged benchmark/evaluator first; then freeze a matched GPT/NextLat study before new training. |
| Lure-Star base training | **4/15 `DONE`:** GPT seeds 1234--1237, each at step 20,000 with durable checkpoint and competence receipt. **3 active:** GPT seed 1238 on the primary Vast RTX 3090 Ti, BST seed 1236 on the secondary RTX 3090, and BST seed 1234 on the third RTX 3090. The other 8 are not verified complete. | Continue the frozen disjoint allowlists from the three worker ledgers. |
| GPT seed-1234 competence | **Complete and durably promoted.** Exact-path accuracy is 3,752/20,000 = 18.76%; GPT is the preregistered threshold-exempt chance-control arm. | Preserve and report the gate result; it is not an H1/H2 geometry result. |
| CFS-1 | **Blocked permanently for production claims.** The 8-versus-7 low-overlap confound failed construct validity. | None. Preserve as predecessor provenance; do not launch or mix with CFS-2. |
| CFS-2 | Stimuli, independent audit, 64-branch runner, crash recovery, pre-evaluation envelope, and the three-site activation-patching runner with named controls are implemented. **0/64 branches launched in this snapshot.** | Supply the exact eight parent checkpoints/lineage receipt, launch the matrix, then run the required inference-only patching sweep on the completed branches. |
| TS-1 / NL-1 | TinyStories and FineWeb-Edu protocols are frozen. **Deferred future work; no corpus download or GPU run.** | Nothing in the current milestone. A future milestone would require fresh explicit authorization without rewriting the frozen designs. |
| H1 branch-decision extension | H1-BD-1 at h63 is prospectively declared and code-ready. | No outcome has been opened; evaluate only under D46. Legacy h62 remains unchanged. |

The primary Vast RTX 3090 Ti is running `gpt-s1235`; the secondary RTX 3090 is running
`bst-s1236`. The earlier GPT-1234 evaluation refusal was an adapter-path bug: training froze
`/content/project/manifests/corpus.sha256`, while the evaluator was given a byte-identical copy at
`/content/lurestar/manifests/corpus.sha256`. The Vast adapter now uses the frozen path and
quarantines deterministic post-training identity/schema/hash failures instead of restarting them.
Large checkpoint transfers use bounded resumable uploads rather than the storage client's hidden
120-second aggregate retry deadline; remote size/hash verification remains mandatory.
These provider facts are time-sensitive and must be re-queried before acting.

## Scientific disposition

- Core Lure-Star H1/H2 remains governed by the
  [preregistration amendment](PREREGISTRATION_AMENDMENT_2026-08-24.md). The amendment's bespoke
  three-regime HMM study is superseded prospectively by the
  [D48 literature reset](DECISION_D48_HMM_LITERATURE_RESET.md).
- Lure-Star H3 is permanently retired after its frozen one-shot matching rule failed. It is not a
  pending experiment.
- CFS-1 is a failed predecessor design. CFS-2 is the separately numbered balanced repair and may
  support only the controlled causal-forgetting claim stated in its
  [decision record](DECISION_CFS2_STIMULUS_REPAIR.md).
- TS-1/NL-1 are deferred external-validity studies. They cannot rescue a core null and cannot
  provide Path-Star/HMM-style exact predictive-state ground truth. The current milestone makes no
  ordinary-language generalization claim.
- A training checkpoint is not a result. Completion, competence, evidence extraction, evaluation,
  aggregation, and claim interpretation are separate states.

## Immediate critical path

1. Let `gpt-s1235` and `bst-s1236` finish, evaluate each against its frozen competence identity,
   and verify its durable terminal receipt; continue from the two worker ledgers, not from memory.
2. Keep the Vast-only retry policy provider-specific: incomplete training and explicit transport
   failures may resume; deterministic scientific-integrity failures quarantine and stop cleanly.
3. Keep all 30 HMM-0 checkpoints unopened. Reproduce the official Mess3/RRXOR benchmark and
   inherited evaluator under D48 before deciding any HMM-LIT-1 compute matrix.
4. Run Lure-Star competence, evidence extraction, H1/H2, and the separately declared H1-BD-1 only
   after their identities match.
5. Launch CFS-2 only after exact parent/lineage preflight and explicit compute authorization; after branch completion, run `scripts/run_cfs2_patching.py` for every branch and retain all three fixed-layer named-control effects.

Do not use `HANDOFF.md`, `results/live_numbers.json`, historical H3/CFS-1 files, or Colab-era
transport notes as current progress evidence. See the [documentation index](INDEX.md) for their
proper roles.
