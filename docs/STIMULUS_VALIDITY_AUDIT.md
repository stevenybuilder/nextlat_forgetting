# Outcome-blind stimulus-validity audit

**Audit date:** 2026-08-24  
**Inputs:** stimulus files, generators, manifests, independent solvers, and exact HMM ground truth only.  
**Explicitly not opened:** checkpoints, losses, training logs, hidden states, evaluation metrics, or scientific outcomes.

## Executive decision

| Dataset | Coded invariants | Construct assessment | Decision |
| --- | --- | --- | --- |
| Path-Star | PASS | FIT_FOR_CONTROLLED_PATH_PLANNING | Continue |
| Lure-Star H1/H2 | PASS | FIT_WITH_DECLARED_POSITION_LIMITATION | Continue with declared limitation |
| HMM family | PASS | FIT_FOR_CALIBRATION_NOT_FORGETTING | Continue as calibration |
| CFS-1 | PASS | **FAIL_REDESIGN_REQUIRED** | **Pause and replace with CFS-2** |
| CFS-2 | PASS | **FIT_FOR_CONTROLLED_CAUSAL_FORGETTING** | Continue after downstream hash binding |

> A coded-invariant PASS means that the files match the implemented specification. It does not prove that the specification isolates the intended scientific construct.

## Critical finding: CFS-1 is mechanically correct but scientifically confounded

CFS-1 low/same has 8 shared edges while low/different has 7. The nuisance difference is perfectly confounded with future relation inside the low-overlap arm and can bias the difference-in-differences toward the hypothesized interaction. A regression cannot identify the two effects from these cells.

The exact overlap table is:

| Condition | Shared edges per update | Count |
| --- | ---: | ---: |
| high/same | 18 | 5000 |
| high/different | 18 | 5000 |
| low/same | 8 | 5000 |
| low/different | 7 | 5000 |

Because the low-arm nuisance difference is deterministic, a regression cannot recover the causal interaction from CFS-1. No CFS-1 adaptation branch should run. Preserve it as a documented design attempt and build CFS-2 with exact overlap balance.

## CFS-2 independently verifies the repair

CFS-2 removes CFS-1's differential total-overlap nuisance. The 4-versus-3 answer-edge difference is the intended future intervention; it is equal across high/low within relation, while the non-answer high-minus-low contrast is 10 in both relations.

| Condition | total shared | update-answer shared | other shared | Count |
| --- | ---: | ---: | ---: | ---: |
| high/same | 18 | 4 | 14 | 5000 |
| high/different | 18 | 3 | 15 | 5000 |
| low/same | 8 | 4 | 4 | 5000 |
| low/different | 8 | 3 | 5 | 5000 |

- Full independent bundle validator passed: True.
- Cross-dataset collisions with Path-Star, Lure-Star, and CFS-1: 0 prompts and 0 graphs.
- All eight materialized episode/cell streams exactly match independently reconstructed codebook order and contents: True.
- The non-answer high-minus-low contrast is 10 for both same and different future relations.

## Path-Star

Strong for algorithm/config fidelity, competence, and controlled predictive-state tests; does not by itself support claims about natural-language semantics.

- Train: 200,000 lines; 200,000 solver-valid; 0 duplicate graph identities.
- Test: 20,000 lines; 20,000 solver-valid; 0 duplicate graph identities.
- Train/test collisions: 0 prompts and 0 canonical graphs.
- Fixed prompt-length histogram: `{'63': 200000}`; answer-length histogram: `{'5': 200000}`.
- Train source coverage: `{'unique': 100, 'min_count': 1909, 'max_count': 2107}`; goal coverage: `{'unique': 100, 'min_count': 1848, 'max_count': 2122}`; source-goal pair coverage: `{'unique': 9900, 'min_count': 5, 'max_count': 39}`.
- All-node coverage: `{'unique': 100, 'min_count': 41345, 'max_count': 42429}`; duplicate prompts: 0.

Limitations:

- Same generator family is used for training and held-out test data.
- The corpus reproduces the pinned upstream distribution and seeds, not a claim of byte identity to every dataset used in paper v4.
- All examples have one fixed G(5,5) topology and fixed token/answer lengths, limiting external validity.

## Lure-Star H1/H2

Strong for minimal, solver-certified future-preserving versus future-changing edits. LS-1 shares the anchor but only exchangeably balances absolute edit position; LS-2 matches position using a different serialization anchor.

- E_lure: 2,000 paired quartets and 0 independent-checker violations.
- Training leakage: 0 prompt and 0 graph collisions.
- Far-control edge overlap: `{'0': 288, '1': 820, '2': 892}`.
- Suffix-depth coverage: `{'1': 665, '2': 615, '3': 720}`.
- Edit-position diversity (unique position pairs): `{'near_safe': 189, 'near_critical': 188, 'near_safe_aligned': 188}`; full position histograms are in the JSON receipt.

Per-condition balance and identity summary:

| Condition | n | prompt lengths | answer lengths | sources | goals | duplicate prompts/graphs |
| --- | ---: | --- | --- | --- | --- | ---: |
| base | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |
| far_critical | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |
| near_critical | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |
| near_safe | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |
| near_safe_aligned | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |
| repeat | 2000 | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 9, 'max_count': 32}` | `{'unique': 100, 'min_count': 9, 'max_count': 36}` | 0/0 |

Legacy pool disclosure:

| Pool | n | collision with Path-Star train (prompt/graph) | duplicate prompts/graphs |
| --- | ---: | ---: | ---: |
| a_pair.jsonl | 1000 | 1000/1000 | 0/0 |
| b_near.jsonl | 5000 | 0/0 | 0/0 |
| b_far.jsonl | 15000 | 0/0 | 0/0 |

Limitations:

- Safe and critical edits cannot occupy identical absolute positions in one common serialized anchor; this is an algebraic design limitation, not a failed check.
- Far controls are rejection-sampled to edge overlap <=2 and therefore represent a deliberately conditioned distribution.
- A_pair is intentionally drawn from training; it is not an independent evaluation set. Legacy H3 adaptation is retired and these pools must not be reinterpreted as current causal evidence.
- Synthetic graph perturbations isolate a future-change construct but do not establish natural-language generalization.

## HMM family

Excellent for exact posterior/future ground truth and robustness across mixing/aliasing regimes; it calibrates predictive geometry but is not itself a causal-forgetting or language task.

- `persistent_moderate`: train [100000, 32], val [10000, 32], length-generalization [10000, 64]; train/val exact collisions 0; posterior spot checks pass=True; train symbol histogram `{'0': 740068, '1': 775893, '2': 815704, '3': 868335}`; duplicate train/val/length-generalization sequences 0/0/0.
- `fast_mixing_moderate`: train [100000, 32], val [10000, 32], length-generalization [10000, 64]; train/val exact collisions 0; posterior spot checks pass=True; train symbol histogram `{'0': 762783, '1': 770112, '2': 810139, '3': 856966}`; duplicate train/val/length-generalization sequences 0/0/0.
- `persistent_high_aliasing`: train [100000, 32], val [10000, 32], length-generalization [10000, 64]; train/val exact collisions 0; posterior spot checks pass=True; train symbol histogram `{'0': 798444, '1': 800960, '2': 799782, '3': 800814}`; duplicate train/val/length-generalization sequences 0/0/0.

Limitations:

- Only 4 hidden states, 4 observation symbols, and sequence lengths 32/64 are represented.
- Exact duplicate sequences are possible samples from the same stochastic process; counts are disclosed rather than treated automatically as leakage.
- Only 64 deterministic rows per stored evaluation split are recomputed in this audit; all array ranges and probability normalizations are checked in full.

## CFS-1 construction details

The files faithfully implement their coded 18/18/8/7 construction, but that construction does not cleanly identify the intended overlap-by-conflicting-future interaction. Mechanical correctness is not construct validity.

- Independent full-bundle validation passed: True.
- Cross-dataset identity collisions: 0 prompts and 0 graphs.
- Retention probes: 2,000; untouched global controls: 2,000; each update bank: 5,000.
- The independent validator confirms identical global node-token counts, answer-length distributions, and probe reuse codebooks across the four update banks.

| Artifact/cell | n | overlap | prompt lengths | answer lengths | sources | goals | duplicate prompts/graphs |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| retention | 2000 | `n/a` | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 12, 'max_count': 31}` | `{'unique': 100, 'min_count': 7, 'max_count': 35}` | 0/0 |
| global_controls | 2000 | `n/a` | `{'63': 2000}` | `{'5': 2000}` | `{'unique': 100, 'min_count': 8, 'max_count': 30}` | `{'unique': 100, 'min_count': 11, 'max_count': 31}` | 0/0 |
| high/same | 5000 | `{'18': 5000}` | `{'63': 5000}` | `{'5': 5000}` | `{'unique': 100, 'min_count': 28, 'max_count': 77}` | `{'unique': 100, 'min_count': 15, 'max_count': 89}` | 0/0 |
| high/different | 5000 | `{'18': 5000}` | `{'63': 5000}` | `{'5': 5000}` | `{'unique': 100, 'min_count': 28, 'max_count': 77}` | `{'unique': 100, 'min_count': 15, 'max_count': 89}` | 0/0 |
| low/same | 5000 | `{'8': 5000}` | `{'63': 5000}` | `{'5': 5000}` | `{'unique': 100, 'min_count': 28, 'max_count': 77}` | `{'unique': 100, 'min_count': 15, 'max_count': 89}` | 0/0 |
| low/different | 5000 | `{'7': 5000}` | `{'63': 5000}` | `{'5': 5000}` | `{'unique': 100, 'min_count': 28, 'max_count': 77}` | `{'unique': 100, 'min_count': 15, 'max_count': 89}` | 0/0 |

Limitations and required action:

- CFS-1 low/same has 8 shared edges while low/different has 7. The nuisance difference is perfectly confounded with future relation inside the low-overlap arm and can bias the difference-in-differences toward the hypothesized interaction. A regression cannot identify the two effects from these cells.
- The fixed G(5,5) topology may make exact 18/18 and 8/8 construction infeasible without a new construction family.
- CFS-1 should be retained as an auditable design attempt, not silently rewritten or used for the strongest causal claim.
- **Do not start CFS-1 adaptation branches. Construct and re-audit CFS-2 with exact factorial overlap balance.**

## CFS-2 construction details

CFS-2 removes CFS-1's differential total-overlap nuisance. The 4-versus-3 answer-edge difference is the intended future intervention; it is equal across high/low within relation, while the non-answer high-minus-low contrast is 10 in both relations.

- Retention probes: 2,000; untouched global controls: 2,000; each update bank: 5,000.
- All overlap totals and decompositions below were recomputed from serialized probe/update edges; stored overlap fields were cross-checked only afterward.

| Artifact/cell | n | total overlap | answer overlap | other overlap | prompt lengths | sources | goals | duplicate prompts/graphs |
| --- | ---: | --- | --- | --- | --- | --- | --- | ---: |
| retention | 2000 | `n/a` | `n/a` | `n/a` | `{'63': 2000}` | `{'unique': 100, 'min_count': 11, 'max_count': 33}` | `{'unique': 100, 'min_count': 10, 'max_count': 32}` | 0/0 |
| global_controls | 2000 | `n/a` | `n/a` | `n/a` | `{'63': 2000}` | `{'unique': 100, 'min_count': 11, 'max_count': 31}` | `{'unique': 100, 'min_count': 9, 'max_count': 29}` | 0/0 |
| high/same | 5000 | `{'18': 5000}` | `{'4': 5000}` | `{'14': 5000}` | `{'63': 5000}` | `{'unique': 100, 'min_count': 27, 'max_count': 87}` | `{'unique': 100, 'min_count': 25, 'max_count': 79}` | 0/0 |
| high/different | 5000 | `{'18': 5000}` | `{'3': 5000}` | `{'15': 5000}` | `{'63': 5000}` | `{'unique': 100, 'min_count': 27, 'max_count': 87}` | `{'unique': 100, 'min_count': 25, 'max_count': 79}` | 0/0 |
| low/same | 5000 | `{'8': 5000}` | `{'4': 5000}` | `{'4': 5000}` | `{'63': 5000}` | `{'unique': 100, 'min_count': 27, 'max_count': 87}` | `{'unique': 100, 'min_count': 25, 'max_count': 79}` | 0/0 |
| low/different | 5000 | `{'8': 5000}` | `{'3': 5000}` | `{'5': 5000}` | `{'63': 5000}` | `{'unique': 100, 'min_count': 27, 'max_count': 87}` | `{'unique': 100, 'min_count': 25, 'max_count': 79}` | 0/0 |

Limitations and required action:

- CFS-2 identifies a controlled symbolic full-parameter adaptation effect, not a natural-language mechanism.
- Different-future cells necessarily retain one fewer update-answer edge than same-future cells; the design controls this by matching it across high/low and equalizing the non-answer high-low contrast.
- The construction still uses one fixed G(5,5) topology and 500-update adaptation regime.
- **Use only CFS-2 streams for the repaired causal study; never mix CFS-1 and CFS-2. Bind downstream runner/evaluator inputs to these hashes before branch execution.**

## What this audit does not establish

It does not show that a model learned the task, that any hypothesis is true, or that effects generalize to natural language. Those are outcome questions and were intentionally excluded. This audit establishes artifact provenance, mechanical validity, balance/disjointness facts, and a pre-outcome judgment about construct fit.

## Reproduction

```bash
.venv/bin/python scripts/audit_stimulus_validity.py --write
```

The command deterministically regenerates this document and `manifests/stimulus_validity_audit.json` from the frozen stimulus artifacts.
