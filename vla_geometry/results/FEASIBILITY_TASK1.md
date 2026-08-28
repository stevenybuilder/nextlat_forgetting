# Task 1 feasibility result: valid pipeline, invalid primary benchmark

On 2026-08-28, the official VIMA-92M checkpoint completed all four smoke episodes and all 384
planned representation extractions for Task 1 (`visual_manipulation`) with zero runtime failures on
one RTX 4090. The behavior run was stopped for scientific futility after 134/134 valid rollouts
succeeded across the first seven cells.

This is not evidence for or against the geometry hypothesis. The VIMA paper's own Task 1 table
reports 100% success for VIMA-92M on the same L2 combinatorial-generalization level, so the binary
endpoint has no useful variance. Continuing to 960 rollouts could not support a failure-prediction
test.

The result is retained because it validates checkpoint loading, official environment integration,
factor enforcement, action decoding, provenance capture, and activation extraction. Task 1 is now
classified as an excluded feasibility benchmark. The replacement confirmatory pilot uses official
Task 16, where the published VIMA-92M L2 success rate is 42.0%.

No Task 1 activation or partial behavior record is used in Task 16 seed selection, model fitting,
analysis, or decision gates.
