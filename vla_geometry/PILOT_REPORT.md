# Additive VLA geometry did not predict compositional success

## Result in one sentence

On official VIMA-Bench Task 16 L2, target/receptacle/direction factors formed a strongly additive
action-token geometry, but the preregistered cross-factor interference score did not predict
closed-loop success and did not pass the causal-stage gate.

## What ran

- Model: official VIMA-92M checkpoint, SHA-256
  `858b4f6a811009285bd7969dc999daa228225ad65b2e887ff1523dd7546282b5`.
- Benchmark: official `manipulate_old_neighbor` combinatorial-generalization partition, with the
  generator and all object/texture supports unchanged.
- Cells: four target shapes × two receptacle shapes × four remembered directions = 32.
- Independent samples: eight representation resets and twenty behavior rollouts per cell.
- Total: 256/256 representation resets and 640/640 behavior episodes, with zero runtime, factor,
  or terminal-outcome failures.
- Runtime: one RTX 4090, float32, deterministic categorical actions, no distributed execution.

The observed overall behavior was 315/640 success, or 49.2% (Wilson 95% interval 45.4%–53.1%).
Cell success rates ranged from 30% to 75%, so the endpoint was neither at floor nor ceiling. This
conditioned 32-cell sample is not a direct replication estimate of the paper's full Task 16 L2
population, though its difficulty is in the expected range.

## Preregistered primary result

| Endpoint | Result | Gate |
| --- | ---: | --- |
| Spearman interference vs. success | -0.081 | Required ≤ -0.50 |
| Cell-bootstrap 95% interval | [-0.453, 0.298] | Required upper bound < 0 |
| Two-sided permutation p-value | 0.654 | Required < 0.05 |
| Control-only LOOCV binomial log loss | 0.6930 | Baseline |
| Control + interference LOOCV log loss | 0.6963 | Lower required |
| Relative held-out improvement | -0.48% | Required ≥ 5% |
| Advance to causal intervention | **No** | All gates required |

An independent SciPy recomputation exactly matched rho. The preregistered permutation p-value
differs slightly from SciPy's asymptotic p-value, as expected, but both are plainly non-significant.

## The interesting part

The five-fold held-out additive reconstruction R-squared was 0.788 (fold range 0.705–0.826). In
plain language, the first action-facing token contained a fairly systematic, reusable geometry for
target shape, receptacle shape, and remembered direction. Yet the angle-based interference among
those factor directions did not explain which combinations the policy solved.

That separates two claims that are often blurred together:

1. a policy can represent task factors in an approximately compositional linear geometry; and
2. failures need not be caused—or even prospectively marked—by those factor directions becoming
   less orthogonal.

For this model and task, the result favors a routing/execution account over the specific
first-decision interference account: information can be geometrically available while later
memory use, action selection, or execution still determines success. This is a scoped inference,
not proof that representation geometry never matters.

## What we do not claim

- The confidence interval still permits a moderately negative association; this pilot cannot prove
  an exact null.
- Reconstruction is not causal use, so the R-squared result is a representation diagnostic.
- One older simulated, object-centric VLA does not establish a result for modern raw-pixel VLAs or
  real robots.
- The failed association gate means factor-subspace interventions would be post hoc here; they were
  not run.
- SAFE and other prior work already show that supervised internal features can detect VLA failure.
  This result concerns one outcome-blind geometric score.

## Compute and reproducibility

Measured episode-stage wall time was 4.7 seconds for smoke, 76.5 seconds for representations, and
673.3 seconds for behavior. At the instance rate of $0.361/hour, those stages cost about $0.076;
including validation and model-loading overhead, marginal compute remained below $0.10.

The frozen configuration is [config/memory_pilot.json](config/memory_pilot.json). Compact public
artifacts include the [analysis](results/task16_pilot_analysis.json),
[provenance](results/task16_pilot_provenance.json),
[seed manifest](results/task16_seed_manifest.json), and
[64-reset adapter validation](results/task16_adapter_validation.json). Raw activations, episode
records, checkpoints, logs, and credentials are intentionally excluded.

The run used source-tree SHA-256
`802288abba0dc4d0d7228692fe6de531839bf617b20b0a55351e467530e1d616` and
configuration SHA-256
`b6d3f37959c7075b082e5fcfc62f94bc11c8c2feec0f931cee83e25041b0f686`.

## Bottom line

This is worth releasing as a careful negative pilot: VIMA's action representation looked
factorized, but the proposed orthogonality/interference metric was not the performance-limiting
quantity on Task 16. The next experiment should not be an automatic larger rerun. A materially
different hypothesis—such as where direction information is routed between the first and second
action—would need its own preregistration and justification.
