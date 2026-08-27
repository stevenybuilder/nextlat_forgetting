# Evaluation review — pre-confirmatory scientific and agentic gates

**System type:** Hybrid (research code + autonomous paid-compute controller)
**Provider/framework:** model-agnostic PyTorch/Lightning experiment; hardened Colab/GCS driver
**Reviewed:** 2026-08-24, confirmatory-outcome-blind; D40 nuisance feasibility known
**Binding scientific contract:** `docs/PREREGISTRATION_AMENDMENT_2026-08-24.md`

The requested GSD reference file
`/Users/stevenyang/.Codex/get-shit-done/references/ai-evals.md` was not present anywhere under
`.Codex`, `.codex`, or `.agents` when this review began. This document therefore applies the
supplied evaluation requirements directly and records that missing framework input rather than
silently pretending it was read.

## 5. Evaluation strategy

### Critical failure modes

1. **Outcome-dependent flexibility:** a metric, layer, HMM matrix, seed, bank, or analysis form is
   selected after observing a confirmatory result.
2. **Metric-coordinate artifact:** a geometry claim succeeds under centered cosine or whitened
   Mahalanobis alone and is presented as metric-invariant.
3. **Stopping-rule circumvention:** D40's four unmatched pairs are restricted away, the caliper or
   bank is changed again, or an H3 adaptation/result is relabeled confirmatory after the frozen
   one-shot feasibility rule permanently withdrew H3.
4. **Pseudo-replication:** item counts or HMM regimes are treated as independent training
   replications instead of aggregating within the five paired seeds.
5. **Incomplete/null suppression:** a failed manipulation check, missing cell, null endpoint,
   metric disagreement, or MDE is omitted from the terminal receipt or report.
6. **Autonomous-compute failure:** the controller spends paid compute on a source/config/manifest
   identity that cannot produce confirmatory evidence, or loses a completed checkpoint/artifact.

### Rubrics and measurements

| Dimension | Priority | PASS | FAIL | Measurement |
|---|---|---|---|---|
| Scientific task completion | Critical | Every pre-compute freeze gate passes; all frozen cells and endpoints have hash-bound receipts; no confirmatory run starts earlier. | Training starts with a missing metric, bank, HMM regime, test, or receipt schema. | Code + human audit |
| Metric robustness | Critical | H1 and HMM primary claims obey the two-metric intersection-union rules on identical scored items. | One favorable metric promotes a claim, covariance is fit on scored items, or metrics score different item sets. | Code |
| Lure-Star H3 retirement integrity | Critical | The exact D40 lineage and 4/5,000 permanent-block receipt are bound; confirmatory jobs/results contain zero Lure-Star adaptation/interference/mechanism cells while retaining frozen HMM `h3_posterior_*`/`h3_future_*` diagnostics. | Any unmatched item is restricted away, matching is amended, Lure-Star H3 is promoted, or legacy HMM diagnostics are removed/renamed. | Code + human audit |
| Inferential validity | Critical | Model contrasts aggregate items/regimes inside each paired seed; five seed values, intervals, sign-flip p floor, MDE, and leave-one-out are reported. | Items/regimes inflate `n`, a missing seed is imputed, or a null is stated as equivalence without an equivalence design. | Code + human audit |
| Identification language | High | NextLat–BST is labeled competence-matched but architecture/parameter-confounded; NextLat–GPT is labeled competence-confounded; predictive language is used without intervention. | Any objective-only, mediation, biological, or geometry-causes-forgetting claim appears without qualifying evidence. | Code phrase lint + human review |
| HMM calibration validity | High | Three model-blind regimes are frozen with `TE` rank/singular-value certificates and combined equally within seed. | A regime is selected by model outcome, a fragile single HMM is generalized, or belief-coordinate alignment is treated as unique. | Code |
| Result completeness | Critical | Fixed schema requires every endpoint, invalid-cell marker, manipulation check, and multiplicity field; extra/missing keys refuse promotion. | A favorable subset can reach `DONE`, or terminal output lacks a preregistered null. | Code |
| Tool-use correctness | Critical | Controller verifies source/config/manifest/checkpoint identities, owns exactly one runtime, and commits artifacts before state. | Ambiguous ownership, state-before-artifact, wrong checkpoint lineage, or scientific evaluator invoked on an unverified model. | Code + integration test |
| Safety and cost adherence | Critical | No secrets enter snapshots/logs; circuit breakers and budget ceilings halt automatically; two agreeing status/quota reads confirm teardown. | Credential leakage, duplicate controller, continued burn without durable progress, or budget overrun. | Code + human audit |
| Test pass rate | Critical | Focused scientific tests and the complete suite pass from a clean command before launch. | Any unresolved P0/P1 failure or scientific test skip. | Code |
| Instruction following | High | The dated amendment, immutable inputs, no-p-hacking rules, and user-approved compute scope are respected. | Silent deviation, outcome-aware amendment labeled confirmatory, or unapproved compute expansion. | Human review |

### Reference datasets and fixtures

| Population | Minimum/frozen size | Composition | Labels and provenance |
|---|---:|---|---|
| Lure-Star evaluation | 2,000 quartets | Base, repeat, near-safe, near-critical, far-critical; 400 whitener-only and 1,600 scored | Solver-verified; canonical serialization hashes; deterministic hash split |
| H3 terminal feasibility record | 5,000 near identities; 150,000 frozen middle candidates; 188,000 combined pilot-loss rows | D39/D40 lineage and the four unmatched identities; no scientific outcome | Frozen model-blind pilot + create-only permanent-block receipt; excluded from confirmatory datasets |
| Gradient/Jacobian fixtures | At least 10 tiny deterministic examples plus exact tiny-model Jacobians | Zero gradient, aligned, orthogonal, opposed, rank-deficient and finite-precision cases | Automated analytic/numerical reference |
| HMM calibration family | 3 regimes; 100k train, 10k validation, 10k length-64 per regime | Persistent/moderate aliasing, fast mixing, persistent/high aliasing; exact beliefs and future distributions | Exact forward algorithm, brute-force short-sequence reference, family manifest |
| Recovery/controller fixtures | At least 20 scenarios | Clean start, exact resume, rollback, corrupt pointer, missing artifact, duplicate owner, timeout, quota lag, teardown | Automated fault injection and immutable receipts |

Dataset construction occurs before confirmatory implementation is declared complete. No LLM judge
labels scientific examples; graph solvers and exact Bayesian calculations are the authorities.
Human review samples at least 20 records per manifest, all schema failures, and every deviation.

### Gate 10 reduced Lure-Star schema contract

Gate 10 binds the implemented base-only H1/H2 chain, not the retired H3 interface. The extraction
job is `nextlat_forgetting/lurestar_evidence_extraction_job/2`; restart progress remains
`nextlat_forgetting/lurestar_evidence_progress/1`; evidence NPZs and their extraction receipts are
`nextlat_forgetting/lurestar_evidence/3`; and the evaluation manifest, confirmatory report, and
evaluation receipt are respectively `nextlat_forgetting/lurestar_evaluation_manifest/3`,
`nextlat_forgetting/lurestar_confirmatory_report/3`, and
`nextlat_forgetting/lurestar_evaluation_receipt/3`.

The gate preserves every H1/H2 completeness check—both co-primary metrics, strict missing/extra
metric refusal, invalid/null reporting, manipulation-failure reporting, and the frozen H1/H2
estimands. It additionally requires base checkpoints only, the exact permanent H3 block and
sidecar, and explicit absence of all Lure-Star H3, adaptation-checkpoint, mechanism-probe, and H3
analysis fields. These exclusions do not apply to the separately frozen HMM diagnostic names
`h3_posterior_*` and `h3_future_*`.

### Tooling

Existing scientific tooling is the default: `pytest`, NumPy/SciPy estimators, solver checks,
SHA-256 manifests, append-only ledgers, GCS generation preconditions, fixed-schema receipts, and
the Colab status/quota controller. No LangSmith, Langfuse, Braintrust, Promptfoo, RAGAS, or Phoenix
instrumentation was detected as an existing dependency.

Arize Phoenix is the default optional local observability layer for orchestration/tool spans—not
the source of scientific metrics and never required on the paid runtime:

```bash
pip install arize-phoenix opentelemetry-sdk
```

```python
import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

px.launch_app()  # local only: http://localhost:6006
provider = TracerProvider()
trace.set_tracer_provider(provider)
```

Because this is not a prompt/RAG product, RAGAS and Promptfoo do not measure the relevant failure
modes. Code-based receipts remain authoritative; Phoenix traces are disposable diagnostics and
must not contain credentials or model artifacts.

### CI and launch-blocking commands

The current minimum CI command is:

```bash
.venv/bin/python -m pytest tests/ -q
```

Before launch, implementation must add one deterministic command that validates the complete
scientific freeze and produces a receipt, conceptually:

```bash
.venv/bin/python scripts/validate_preregistration.py \
  --amendment docs/PREREGISTRATION_AMENDMENT_2026-08-24.md \
  --require-all --output results/preregistration_freeze_receipt.json
```

The paid-compute launcher must require that receipt and its SHA-256. A missing command or receipt
is presently a **BLOCK**, not a documentation-only warning.

## 6. Guardrails

### Online/pre-launch guardrails

| Guardrail | Scope | Failure action |
|---|---|---|
| Outcome-blind freeze receipt | Every confirmatory launch | Refuse launch if amendment/source/manifests/evaluator hashes differ. |
| Complete-cell contract | Every promotion to `DONE` | Reject missing/extra metric keys, invalid unacknowledged cells, or wrong seed/regime/arm set. |
| Metric split/leakage check | Every H1/H2/HMM evaluation | Refuse if `E_white` intersects scored IDs or metric item IDs differ. |
| Permanent H3 scope gate | Every job spec, result schema, and clearance | Require block receipt SHA `82d526ad…`; refuse Lure-Star H3 adaptation/interference/mechanism cells or D40 matching changes, but require unchanged HMM `h3_posterior_*`/`h3_future_*` keys. |
| HMM algebraic certificate | Every HMM train/eval | Read matrix identity from frozen `thresholds.hmm_sha256`; require the exact 30-job family print-plan; refuse unlisted regime, wrong corpus/pair bank, rank-deficient `TE`, or `sigma_min <= 0.05`. |
| Claim-language lint | Every generated report | Block prohibited causal/biological/objective-only phrases pending human adjudication. |
| Single-controller and durability gate | Every paid runtime | Refuse ambiguous ownership; artifact-first/state-last; stop after circuit-breaker conditions. |
| Secret and budget gate | Every package/launch | Refuse credential-containing archive; stop at per-job and cumulative compute ceilings. |

These checks are fast, deterministic, and belong on every relevant request/transition. They do not
call an LLM judge.

### Offline flywheel

| Signal | Sampling/cadence | Improvement use |
|---|---|---|
| Scientific receipt completeness | 100% of jobs and evaluations | Add schema/tests for every refusal or missing field; never relax a field to pass an observed result. |
| Estimator calibration | Every code change plus synthetic Monte Carlo batch | Track coverage, false-positive rate, leakage detection, sign-flip and Holm correctness. |
| Heterogeneity/null audit | 100% of terminal results | Check metric/regime/seed disagreements, MDE language, and absence of outcome suppression. |
| Recovery/failure value | Every interruption | Verify newest durable step/artifact and failure trace; convert each novel failure into a fault-injection test. |
| Claim audit | Every draft and figure caption | Human reviewer checks evidence rung, identification limits, causal wording, and related-work attribution. |
| Cost efficiency | Every profile/job | Compare compute spent with durable steps/checkpoints/evaluations recovered; adjust engineering only, not scientific outcomes. |

## 7. Production monitoring

“Production” here means the autonomous confirmatory experiment pipeline, not a user-facing model.
The authoritative telemetry is the append-only ledger plus immutable GCS receipts. Phoenix may
mirror controller spans locally but cannot determine scientific completion.

| Metric | Threshold | Action |
|---|---:|---|
| Freeze receipt validity | 100% | Any mismatch blocks launch. |
| Required scientific cells present | 100% | Any missing/extra cell blocks `DONE`. |
| Artifact hash verification | 100% | Roll back to newest verified generation or stop. |
| Runtime ownership | Exactly 1 controller/runtime | On ambiguity, switch to read-only monitoring and stop new provisioning. |
| Durable progress | Must advance inside circuit-breaker window | Two fast/no-progress returns stop the runtime and surface logs. |
| Checkpoint exposure | At most configured recovery interval | If exceeded, halt job and repair durability before continuing. |
| Source/config/manifest drift | 0 | Stop; new scientific changes require a dated amendment. |
| Invalid/nonfinite metric cells | 0 unacknowledged | Emit invalid receipt and block aggregate; never drop the cell. |
| Metric sign disagreement | Always surfaced | Downgrade to metric-dependent evidence automatically. |
| Regime/seed heterogeneity | All cells reported | Trigger mandatory leave-one-out/heterogeneity text, never selective rerun. |
| Compute spend | Per-job <= 3x profile; cumulative within approved stop-line | Stop and request a compute decision. |
| Secret findings | 0 | Stop packaging/launch, rotate if exposed, retain a redacted incident record. |

Sampling is 100% for launches, checkpoints, terminal scientific receipts, and budget/security
events. High-volume step telemetry may be reduced, but the latest durable step, checkpoint hash,
cost, and liveness are never sampled. Alert thresholds may be tightened for engineering reasons;
loosening a scientific threshold requires a new dated amendment and makes affected results
exploratory if outcomes were already visible.

## Verdict

**BLOCK confirmatory training** until the revised eleven pre-compute freeze gates in the amendment
are implemented and pass. H3's nuisance-feasibility gate has terminated the Lure-Star interference
hypothesis without a confirmatory outcome; its adaptation/mechanism fields must be absent from
training and inference while the legacy-named HMM diagnostics remain required. The remaining confirmatory
program is H1, H2, and exactly 30 three-regime HMM cells. The validator/evidence/job/result chain
must bind that scope, the permanent-block receipt, the aggregate/multiplicity rules, the complete
suite, and independent review before launch.
