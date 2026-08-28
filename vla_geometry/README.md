# Compositional action geometry in vision-language-action models

This study asks whether a VLA succeeds on unseen combinations because its action-facing hidden
state represents task factors as reusable, minimally interfering components.

The first stage is deliberately small and uses only public artifacts:

- the frozen VIMA 92M checkpoint;
- the official Task 16, `manipulate_old_neighbor`, in VIMA-Bench;
- the official combinatorial-generalization partition;
- a fixed single-GPU runtime with no distributed training or model updates.

The primary representation is the final transformer token consumed by VIMA's action decoder at
the first decision. The primary behavioral outcome is closed-loop task success on independent
simulator seeds. Target shape, receptacle shape, and remembered neighbor direction form 32
factorial cells. The official generator remains unchanged; outcome-blind seed stratification
selects those cells while all official textures and distractors remain nuisance variation.
Additive factor directions are estimated without behavioral outcomes, and their geometric
interference is used to predict success on held-out combinations.

This is not a generic probing study. A credible positive result requires all of the following:

1. factor directions and interference are estimated with combination-level cross-fitting;
2. geometry predicts success on simulator seeds not used to estimate the representation;
3. the prediction improves on preregistered controls such as action confidence and layout;
4. a later factor-specific intervention changes the intended action more than matched random
   subspace interventions.

Read [PREREGISTRATION.md](PREREGISTRATION.md) before running the confirmatory pilot. The frozen
machine-readable design is in [config/memory_pilot.json](config/memory_pilot.json). The saturated
Task 1 feasibility run is explicitly excluded and documented in
[results/FEASIBILITY_TASK1.md](results/FEASIBILITY_TASK1.md).

The completed pilot is reported in [PILOT_REPORT.md](PILOT_REPORT.md). Its main result is a useful
separation: factor identities reconstruct the held-out action-token geometry well, but the
preregistered cross-factor interference score does not predict which combinations succeed.

## Layout

| Path | Purpose |
| --- | --- |
| `config/memory_pilot.json` | Frozen factors, seed rule, endpoints, topology, and decision gates |
| `src/vla_geometry/` | Factorization, interference, splitting, and statistical analysis |
| `scripts/` | Pinned environment setup, checkpoint download, rollout, and analysis entry points |
| `tests/` | Scientific-invariant and leakage tests |
| `results/` | Compact result summaries and provenance; raw activations stay untracked |
| `CODE_AUDIT.md` | Anti-sycophancy implementation and publication-readiness review |

## Reproducibility boundary

Raw checkpoints, Hugging Face caches, rollout arrays, videos, and credentials are excluded from
Git. Every run writes a provenance record containing upstream commits, checkpoint SHA-256,
configuration SHA-256, package versions, GPU identity, and CUDA identity.

The VIMA discovery result is about a simulated, object-centric VLA. It does not establish that the
same geometry exists in modern raw-pixel VLAs or on real robots. SmolVLA on LIBERO is a subsequent
validation stage. Because the VIMA prediction gate did not pass, no causal intervention or modern
replication is automatically authorized by this pilot.
