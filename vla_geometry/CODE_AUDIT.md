# Anti-sync implementation audit

Review date: 2026-08-28. Scope: the executable VIMA Task 16 pilot, not merely its narrative design.

## Verdict

**Public technical release: publishable after minor revisions. General VLA or causal paper claim:
major revisions and new evidence required.** Confidence is high on benchmark/inference fidelity
and high that the preregistered pilot gate failed. The completed VIMA result is a scoped negative
pilot, not evidence about modern VLAs generally.

The strongest legitimate contribution is a prospectively frozen test of whether outcome-blind,
cross-fitted factor interference adds held-out failure-prediction value beyond factor identity,
policy confidence, and physical layout. This is narrower than generic VLA feature probing,
steering, or supervised failure detection.

## Claim ledger

| Claim | Evidence in the artifact | Necessary assumptions / rival explanation | Status after pilot |
| --- | --- | --- | --- |
| The runner evaluates official VIMA-92M on official Task 16 L2. | Checkpoint byte count and SHA-256, pinned clean upstream commits, unchanged partition arguments, 64 real-reset adapter checks. | Pinned upstream behavior and checkpoint are the intended published artifacts. | Supported. |
| The selected cells do not use outcomes. | Seed resolution mirrors generator RNG and has no policy, action, reward, or success input; representation, behavior, and smoke ranges are disjoint. | RNG mirror must equal real resets. | Supported by 64/64 integration checks. |
| The extracted vector is action-facing. | It is `predicted_action_tokens[-1]`, immediately passed unchanged to the official action decoder; history padding and action-token recurrence match the official example. | First-decision geometry may not describe later memory routing. | Supported for the first decision only. |
| Geometry prospectively predicts failure. | Frozen cross-fitting, independent representation/outcome seeds, cell-level bootstrap/permutation, and LOOCV baseline comparison. | Factor difficulty, layout, confidence, or later routing may explain failures. | Not supported: rho = -0.081, 95% bootstrap interval [-0.453, 0.298], permutation p = 0.654. |
| Interference is causally used. | Conditional intervention stage with factor-specific and norm/rank-matched random controls. | Association can arise without causal use. | Unsupported until the gate passes and interventions run. |
| Factor identity is additively represented. | Five-fold held-out additive reconstruction R-squared is 0.788. | Reconstruction alone does not establish behavioral or causal use. | Supported as a secondary representation diagnostic in this population. |
| The result generalizes to VLAs. | One older object-centric simulated VLA/task. | Architecture and benchmark specificity are substantial. | Unsupported; must remain a proof-of-concept claim. |

## Strong implementation choices

- The official generator is conditioned by frozen seeds rather than rewritten.
- Representation and behavior layouts are independent, preventing same-layout leakage.
- Whole factorial cells, not frames or rollouts, are the inferential units.
- Factor directions for a held-out cell are fit without that cell.
- Resume logic rejects changed provenance, changed manifests, mismatched metadata, duplicate seeds,
  non-finite activations, and frozen failure records.
- Runtime enforcement fixes checkpoint, source, configuration, upstream commits, Python/library
  versions, one RTX 4090, float32, deterministic algorithms, cuBLAS workspace, and TF32 policy.
- Raw checkpoints, activations, credentials, and runtime logs are excluded from Git.

## Critical findings and repairs

| Severity | Finding | Why it mattered | Repair | Acceptance test |
| --- | --- | --- | --- | --- |
| Blocker, resolved | The first Task 16 draft narrowed target shapes/textures; all six grid objects could share the selected target shape. | This changed the benchmark and could make any result an adapter artifact. | Preserve the entire official generator and stratify seeds on sampled factors. | Official task kwargs compare equal and two real resets per 32 cells match: 64/64. |
| Blocker, resolved | Task 1 behavior was at ceiling. | A predictor cannot explain failures that do not occur. | Exclude every Task 1 record and preregister official Task 16 L2 (published 42.0%). | Smoke must be valid, multi-step, and non-ceiling before the pilot. |
| Major, resolved | Old outputs could be resumed after source/config changes. | Mixed-code datasets can look complete while being scientifically uninterpretable. | Exact provenance/manifest comparison and per-record identity validation. | Any hash, seed, factor, mode, or topology mismatch aborts. |
| Major, resolved | The baseline omitted factor difficulty and used reference-sensitive ridge coding. | Geometry could proxy an intrinsically hard object/direction or arbitrary coding choice. | Symmetric full one-hot factor effects plus confidence and initial layout under fixed ridge. | Column-permutation/reference choice cannot change fitted predictions. |
| Major, resolved | Constant endpoints and nonconvergent logistic fits were not hard failures. | Degenerate data could produce misleading inference. | Reject undefined Spearman/bootstrap inputs and require IRLS convergence. | Unit tests exercise both failure modes. |
| Major, residual | Thirty-two cells provide only about 80% power at an absolute correlation near 0.48. | The interval still permits a moderately negative population association. | Report the interval and failure of the joint gate, not “proof of no relationship.” | No broad null claim after this negative pilot. |

## Alternative explanations

1. **Intrinsic factor difficulty:** addressed with target/receptacle/direction main effects.
2. **Policy uncertainty:** addressed with initial action confidence.
3. **Physical layout:** addressed with target/receptacle distance and target/neighbor coordinates.
4. **Later memory routing:** not eliminated by first-token geometry; additive reconstruction without
   outcome prediction supports this alternative.
5. **Supervised failure features:** SAFE already establishes this weaker result. Novelty requires
   success without failure-label training and, eventually, selective causal intervention.
6. **Benchmark/model specificity:** only a modern benchmark replication can distinguish a VIMA
   curiosity from a more general VLA phenomenon.

## Acceptance-gate outcome

1. Passed: four frozen smoke rollouts were valid and one exercised the complete two-action path.
2. Passed: exactly 256 representation and 640 behavior records completed with zero failures.
3. Passed: local and remote source hashes, configuration hash, seed manifest, and independent
   SciPy Spearman recomputation match.
4. Failed scientifically: rho, interval, permutation, and predictive-improvement thresholds did
   not pass; no causal intervention is run.
5. Passed for the VLA subtree: the secret scan reports zero findings. A final staged-diff scan is
   still required immediately before push.

The final local acceptance suite contains 17 passing tests, Python bytecode compilation, shell
syntax checks, critical Ruff undefined-name/syntax checks, JSON validation, and a zero-finding
staged-diff secret scan.

## Prior-art boundary

Closest work includes [SAFE](https://arxiv.org/abs/2506.09937),
[symbolic VLA probing](https://arxiv.org/abs/2502.04558),
[mechanistic VLA steering](https://arxiv.org/abs/2509.00328),
[VLA sparse-autoencoder features](https://arxiv.org/abs/2603.19183), and
[VLA-Trace](https://arxiv.org/abs/2605.30117). The geometric prior comes from
[linear/orthogonal compositional vision representations](https://arxiv.org/abs/2602.24264), but
that theorem is not treated as established for VLA policies.

## Candid publication answer

Would I let a respected colleague publish this as written? **Yes as a transparent technical report
or research blog saying that additive factor structure was present but this interference metric did
not predict closed-loop success. No as a general VLA, causal, or “orthogonality does not matter”
paper.** A paper-strength causal claim would need a successful prospective association,
factor-selective interventions, and modern-model replication.
