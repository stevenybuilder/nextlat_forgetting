# Current status

Snapshot: **2026-08-29**. This page is a lifecycle summary, not a result report or compute
authorization.

## Where the project is now

The repository has been reset around four plainly stated studies. The active design incorporates
NextLat v4, the Belief State Transformer, multi-token and joint-token prediction, future-summary
prediction, the August 2026 Hierarchical Latent Prediction preprint, the published Mess3/RRXOR
geometry benchmark, and recent Path-Star optimization work.

No new four-study confirmatory matrix has been launched. The retrospective, evaluation-only case
study of two already-trained NextLat trajectories is complete. Existing configurations and result
files belong to the legacy replication/diagnostic phase unless explicitly promoted.

A separate low-compute VLA geometry pilot is also complete under [`vla_geometry/`](../vla_geometry/README.md).
It found additive factor structure in VIMA's first action-facing token but no preregistered evidence
that cross-factor interference predicts Task 16 success; its causal stage was not triggered.

The modern π0.5 [intended-futures pilot](../intended_futures/RESULTS.md) is now complete as well.
Across 120 validated same-scene prompt contrasts, action-expert geometry predicted intended target
displacement with leave-one-scene-out R² 0.850. A frozen 36-unit causal stage did not show selective
behavioral redirection: the learned rank-three patch's standardized scene effect was 0.125 and all
three causal gates failed.

A subsequent 36-unit all-call manipulation check found that PaliGemma layer-13 replay nearly
transplanted the clean donor policy's target behavior: donor first touch was 58.3%, versus 61.1%
for clean donor, 11.1% for clean recipient, and 13.9% for matched random perturbations. It passed
six of seven frozen checks but missed the required 0.10 absolute action-cosine margin with 0.0509.
The gate was honored and the learned target controller was not run.

The independently frozen follow-up used 240 new official states to test whether PaliGemma-13
encoded state-conditioned 3D target displacement beyond prompt identity. Its decoder reached
validation R² 0.953, but the exact prompt-pair baseline reached 0.995. The incremental-information
gate failed, so no controller was fitted and no causal-test or reserve rollout was opened.

| Area | Current state | Next gate |
| --- | --- | --- |
| Path-Star baseline | All 20 retained checkpoints were evaluated on the frozen 20,000-graph corpus; the deterministic repeat passed | Case-study release complete |
| Reproducibility | The selected solver crosses between retained updates 1,000 and 3,000; the selected shortcut plateaus near 18% and becomes confidently wrong | No population claim; the 24-seed reliability study remains deferred |
| Future-sensitive geometry | PaliGemma replay redirected target behavior, but the independent PaliGemma observer failed to beat prompt identity | Redesign stimuli for repeated prompts across varied layouts before any controller |
| Controlled forgetting | Designed, with reusable balanced interventions and patching machinery | **Deferred** |
| Exact predictive states | Mess3/RRXOR design and legacy HMM scaffolding exist | **Deferred** |
| New objectives | MTP/JTP are public; FSP/HiLP require later parity work | Literature context only in this phase; do not implement or train yet |
| VLA compositional geometry | Frozen Task 16 pilot completed 256 representation resets and 640 behavior episodes with zero failures; prediction gate failed | Public technical report only; no automatic causal stage |
| Compute | Path-Star evaluation cost approximately $0.382; VIMA episode stages cost approximately $0.076; prior π0.5 work cost approximately $2.67; TC1 added $4.13 | The Vast instance is stopped; no additional study is authorized |

## Existing result boundary

The old Path-Star evidence is scientifically useful but narrower than the new program:

- one original NextLat seed reached the shortcut basin despite completing all updates;
- nominally identical runs can diverge from CUDA embedding-backward nondeterminism;
- one versus two GPUs, sampler seeds, compiler use, and deterministic CUDA settings do not singly
  explain the public-versus-paper gap;
- the original five-seed cohort cannot be repaired by selecting replacement seeds;
- new downstream work may use a separately declared competent-parent cohort, while reporting the
  unconditional training success probability that produced it.

The legacy custom-HMM checkpoints and failed intervention designs remain preserved but are not
evidence for the new hypotheses.

## Completed release

The release includes every retained non-recovery checkpoint, a byte-identical repeat evaluation,
the one-GPU runtime record, deterministic report builder, three figures, CSV tables,
machine-readable summary, technical report, and blog post. The standardized seed-1235 final result
is 3,651/20,000 (18.255%); the historical evaluator's 3,663/20,000 result remains separately
reported. The evaluator discrepancy was not silently pooled.

The future-sensitive representation pilot has now been executed in a modern public VLA rather than
the archived NextLat checkpoints. The result supports accessible action-expert geometry and a
strong full-PaliGemma-replay effect, but the independent PaliGemma observer did not add geometric
information beyond prompt identity. Nothing in this release authorizes a revised controller study,
a multi-site follow-up, or any new model training.
