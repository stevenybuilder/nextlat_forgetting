# A retrospective case study of two NextLat Path-Star trajectories

## Result in one paragraph

Two outcome-selected NextLat runs with the same scientific configuration but different seeds
reached sharply different solutions on Path-Star G(5,5). On a newly standardized frozen-corpus
evaluation, seed 1234 moved from 0/20,000 exact paths at update 1,000 to 19,803/20,000 (99.015%)
at update 3,000 and stayed near perfect through update 20,000. Seed 1235 reached 3,505/20,000
(17.525%) at update 4,000, then remained between 17.525% and 18.430% through update 20,000.
Its mean gold-token margin at the first nontrivial path decision became more negative, reaching
−4.232 logits at the final checkpoint. This is evidence about two selected trajectories, not an
estimate of how often NextLat reaches either solution.

## Figures

![Exact-path accuracy across retained checkpoints](../results/studies/basin_case_study/retrospective-2026-08-27/figures/exact_path_accuracy.svg)

Seed 1234's generalizing transition is bounded by the retained checkpoints as **(1,000, 3,000]**.
The archived in-training validation trace places it by update 2,000 (96.264% logged exact-path
accuracy), but that older trace is supporting evidence from a different evaluator path, not a
replacement for the frozen-corpus bound.

![First-decision gold-token margin](../results/studies/basin_case_study/retrospective-2026-08-27/figures/first_decision_margin.svg)

The margin is the gold next-node logit minus the largest non-gold logit after conditioning on the
correct source node. Seed 1234 moves from −1.368 at update 1,000 to +7.709 at update 3,000 and
+11.680 at update 20,000. Seed 1235 moves from −1.405 at update 2,000 to −2.515 at update 4,000
and −4.232 at update 20,000. The shortcut therefore is not merely an undertrained version of the
solver at the retained checkpoints; it becomes more confident in the wrong branch on average.

![Training and validation loss](../results/studies/basin_case_study/retrospective-2026-08-27/figures/loss_trajectory.svg)

Training loss falls in both runs. For seed 1235, the median total training loss falls from 1.589
in the second ordered 1,000-row window to 0.086 in the final window, while the reconstructed
validation total loss is 1.659 at update 2,000 and remains around 2.0–2.5 later. This is why
held-out path behavior, not training loss alone, is the appropriate basin diagnostic.

## Final-checkpoint phenotype

| Measure | Seed 1234: selected solver | Seed 1235: selected shortcut |
| --- | ---: | ---: |
| Exact five-token path | 99.955% | 18.255% |
| First nontrivial decision | 99.980% | 18.255% |
| Mean gold-token margin | +11.680 | −4.232 |
| Position 1 (source) | 99.995% | 99.995% |
| Position 2 | 99.980% | 18.255% |
| Position 3 | 99.995% | 18.340% |
| Position 4 | 99.975% | 18.635% |
| Position 5 (destination) | 100.000% | 93.600% |

The seed-1235 boundary-token accuracy alongside near-18% interior-token accuracy is consistent
with a shortcut that learns source/destination and serialization regularities without learning a
general path rule. This table is descriptive; assigning a detailed internal algorithm requires
additional representation or intervention evidence.

## Design and methods

This was frozen on 2026-08-27 as a retrospective, outcome-selected case study. Seed 1234 was
chosen because its final checkpoint was already known to generalize, and seed 1235 because its
final checkpoint was already known to fail the competence gate. The analysis includes every
retained ordinary periodic checkpoint and excludes recovery checkpoints, which duplicate training
states for restart purposes.

Both runs used the pinned public NextLat source at commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`, the NextLat objective, a 12-layer/6-head transformer
with width 384, effective batch size 512, Adam at a constant 5e-4 learning rate, BF16 mixed
precision, compilation disabled, 20,000 updates, the same 200,000-example G(5,5) training corpus,
and the same 20,000-example test corpus. Their materialized scientific settings differ by seed;
run names and output paths also differ.

All new checkpoint evaluation used one full-host RTX 4090 with exactly one visible GPU, no DDP or
distributed sampler, PyTorch 2.11.0+cu128, deterministic inference algorithms, TF32 disabled,
explicit greedy autoregressive decoding, and batch size 256. Every checkpoint was scored on all
20,000 frozen test examples. Repeating the seed-1234 final evaluation produced byte-identical
scientific JSON.

The primary outcome is exact greedy five-token path accuracy. Secondary descriptive outcomes are
per-position accuracy, teacher-forced accuracy and gold-token margin at the first nontrivial
decision, and saved training/validation losses. The 20,000 test graphs give precise conditional
measurements of a fixed checkpoint, but they do not turn two training runs into 40,000 independent
experimental units. No seed-population hypothesis test is reported.

The historical metrics CSVs contain 19,999 ordered training records and 19 validation records, but
their `step` column is incorrectly zero throughout. The validation cadence is reconstructed in
1,000-update increments and checked against nine retained checkpoint filenames per run, each of
which embeds the corresponding validation loss. Training curves use medians of ordered 1,000-row
windows. The behavioral checkpoint JSON, whose update identity is explicit and hash-bound, remains
the primary record.

## Historical-evaluator discrepancy

The original frozen evaluator recorded seed 1235's final checkpoint as 3,663/20,000 (18.315%).
The new standardized evaluator records the same checkpoint as 3,651/20,000 (18.255%), a difference
of 12 examples or 0.06 percentage points. The discrepancy does not change the basin classification,
but its exact source was not isolated. The historical and new evaluator paths are therefore
reported side by side and are not silently pooled. The historical value remains the authority for
the original five-seed competence cohort; the new value is the authority for this trajectory
case study.

## Relation to the earlier execution diagnostics

Separate diagnostic runs showed that nominally identical two-GPU jobs can diverge despite matching
initial weights, first batches, and recorded sampler permutations. Their first measurable
difference was a very small CUDA `token_embedding.weight` gradient difference; deterministic CUDA
settings removed that first-update difference. Deterministic training, however, still did not
guarantee the generalizing solution, and no single tested change—GPU count, sampler seed, compile
setting, or deterministic algorithms—explained the public-versus-paper reproduction gap.

Those diagnostics establish one source of within-environment trajectory variation and show that
the optimization is sensitive. They do **not** prove that the observed gradient difference caused
the seed-1234/seed-1235 contrast plotted here, nor do they identify the authors' private execution
path.

## Claim boundary

The case study supports this narrow statement:

> For these two outcome-selected NextLat artifacts, held-out behavior separates by update 3,000
> and remains in distinct regimes through update 20,000; the shortcut trajectory becomes
> confidently wrong at the first branch decision even while its training loss falls.

It does not establish:

- the probability that a fresh NextLat run reaches the generalizing basin;
- that seed 1235 is intrinsically defective;
- a comparison against GPT, BST, multi-token prediction, joint-token prediction, or hierarchical
  latent prediction;
- a unique causal mechanism for basin selection; or
- that the NextLat paper's successful runs did not occur.

The scientifically stronger next step is the program's original second hypothesis: test whether
the solver's branch-state representation is selectively sensitive to matched changes in the
future. The [future-sensitive geometry pilot](FUTURE_SENSITIVE_GEOMETRY_PILOT.md) compares
serialization-only, future-preserving, immediate future-changing, and delayed future-changing
edits across the archived transition, then uses controlled activation patching if the geometry
gate passes. It requires no new training. Checkpoint-rescue experiments remain a later option,
after this cheaper test of an original project hypothesis.

## Reproducibility and cost

The new work performed no training. The Vast evaluation session cost approximately **$0.382** by
provider balance decrement, below the frozen $5 stop. This is a session-level estimate, not an
itemized per-checkpoint bill.

The compact tracked release contains the
[machine-readable summary](../results/studies/basin_case_study/retrospective-2026-08-27/summary.json),
[checkpoint table](../results/studies/basin_case_study/retrospective-2026-08-27/checkpoint_summary.csv),
[loss table](../results/studies/basin_case_study/retrospective-2026-08-27/loss_summary.csv), and SVG
figures. Checkpoints and raw private-storage paths are not committed.

Given the local extracted evaluation bundle, rebuild the release with:

```bash
python3 scripts/case_study/build_basin_report.py \
  --input-root .agent_state/basin_case_study/extracted \
  --project-root . \
  --output-root results/studies/basin_case_study/retrospective-2026-08-27
```

The frozen artifact roster and hashes are in
[`manifests/case_study/basin/artifacts.json`](../manifests/case_study/basin/artifacts.json).
