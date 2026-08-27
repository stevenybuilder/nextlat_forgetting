# Decision D-20 — GPT is supposed to fail Path-Star, and the spec's competence gate does not know that

**Status:** decided autonomously, reversible, flagged for the human to overturn.
**Date:** 2026-08-23. **Affects:** spec §8 competence gate, §10 stop conditions, H2 and H3 on Lure-Star.

## The problem

Spec §8 says: "If either model fails the base competence gate — initially 90% exact-path accuracy —
debug data, configuration, numerical precision, and repository parity." Spec §10 makes it a stop
condition: "either model remains below the base competence gate after a reasonable step increase."

But the NextLat paper reports GPT on `G(5,5)` at roughly **18.6% exact-path accuracy — which is 1/d,
i.e. chance** — against NextLat at roughly 99.8%. GPT failing Path-Star is not a bug to debug. It is
the paper's headline result, and it is the entire reason Path-Star is in the paper.

Read literally, the spec therefore halts the project on a *correct* GPT run, and instructs a debugging
campaign against a result we are supposed to reproduce.

## The deeper problem, which is scientific rather than procedural

If GPT sits at chance, the GPT-versus-NextLat contrast on Lure-Star is confounded with competence.
Any PSI difference admits a trivial reading: NextLat separates future-relevant histories because
NextLat *solved the task*, and GPT does not because GPT learned no task structure to organise. That is
a much weaker claim than the one this project set out to test.

It also makes two preregistered analyses degenerate for GPT specifically:

- **H2** asks whether representation distance predicts planning behaviour. A chance-level model has no
  planning behaviour to predict, and its correct-branch logit margin is noise around zero.
- **H3** measures erosion of the correct-branch margin after adaptation. You cannot erode a margin the
  model never had. `erosion_near - erosion_far` for GPT would be a difference of two noise terms.

## Decision

Run the full preregistered matrix exactly as specified. Change no configuration. Then:

1. **Apply the 90% competence gate to NextLat and the later-added BST control.** For GPT, the preregistered expectation is
   chance-level performance, and observing it is a successful replication of the paper's Figure 6
   rather than a stop condition. A GPT run *above* chance would be the surprising outcome and would
   itself require investigation.
2. **Demote the cross-model Lure-Star PSI contrast from primary to secondary**, and report it with the
   competence confound stated in the same breath as the number. It is still worth reporting — but it
   cannot carry the paper.
3. **Promote the within-NextLat chain to primary for Lure-Star:** does PSI vary across items and seeds,
   does larger base-to-critical distance predict a larger correct-branch margin under cross-fitting,
   and does smaller pre-adaptation distance predict greater margin erosion? Every one of those is an
   item-level question inside a model that actually solves the task, so none of them is confounded by
   competence. This is also the more novel claim: the paper already established that NextLat solves
   Path-Star and GPT does not.
4. **Let the HMM carry the clean cross-model contrast.** Both models can learn next-observation
   prediction on a 4-state HMM, so GPT is a competent baseline there rather than a floor. Experiment B
   was already required; this decision raises its weight.

## What this costs, stated plainly

The strongest framing in spec §10 — "NextLat does more than lower global representational rank" as a
claim *relative to GPT* — now rests mainly on the HMM rather than on Lure-Star. If the HMM comes out
null, the honest summary is that the within-NextLat geometry-behaviour-interference chain held while
the cross-model comparison was not cleanly identifiable on Path-Star. That is a weaker paper than the
one the spec imagined, and it is the one the evidence supports.

## Superseded in part, 2026-08-23: BST resolves the confound cheaply

After this decision was written, the competence confound turned out to have a direct fix already
sitting in the pinned repository. The paper's own Figure 6 reports **BST at ~99.9% on G(5,5)** -
a model that solves the task, is width/depth-matched to GPT and NextLat, and is trained without a
latent-transition objective. It is not parameter- or training-architecture-matched because its
objective adds a second transformer and O(T²) prefix-suffix supervision. The official BST config
differs from the GPT config by exactly two scientific settings, but the selected model class differs.

Adding BST as a third arm converts the primary cross-model contrast from
NextLat-versus-a-model-at-chance into
**NextLat versus BST**, where both arms solve the task. The contrast removes task competence as the
obvious explanation but still mixes the objective with BST's extra training architecture and
parameters. H2 and H3 stop being degenerate for the control arm, because BST has a real
correct-branch margin to predict and to erode.

Points 2 and 3 above are therefore **softened rather than withdrawn**: the within-NextLat chain
remains valuable and is still reported, but the cross-model Lure-Star contrast is no longer
demoted - it is re-pointed at BST. The operational form of point 1 is now: the 90% competence gate
applies to NextLat and BST, and GPT's chance-level accuracy remains a preregistered replication of Figure 6
rather than a stop condition. Point 4 stands: the HMM still carries an independent cross-model
contrast in a regime where all three arms are competent.

Approved by the user on 2026-08-23 alongside the later-ratified decision to run all five paper
seeds (1234--1238) as confirmatory seeds up front, before inspecting any outcome.

## Executable gate

No near/far adaptation branch may even be planned from a merely `TRAINED` parent. Each base parent
must be `DONE` with a ledger-hashed `evaluation/base_competence.json` and SHA sidecar. The receipt
binds the model, seed, exact final-checkpoint path and SHA-256, evaluator and raw evaluator-output
SHA-256, held-out dataset and manifest SHA-256, deterministic greedy decoding regime, and integer
correct/total exact-path counts. The runner recomputes every hash and the reported ratio.
NextLat and BST must each reach 0.90; GPT's result is always recorded and reported but does not halt
adaptation. A missing, mismatched, stale, or tampered receipt is a refusal, never an implicit pass.

Those evaluation inputs are not selected at promotion time. Before the first base update, the base
job and ledger freeze the exact evaluator source, held-out dataset, all evaluation manifests, their
resolved paths and SHA-256 digests, and the decoding contract as `competence_identity`. Evaluation
and promotion require exact equality to that parent identity and verify the materialized config,
every metrics log, the completion receipt, and the checkpoint. Thus a post-hoc evaluator/data swap
or a partially corrupted parent cannot be promoted even if its checkpoint alone still hashes.

The checkpoint-to-metric implementation is `scripts/evaluate_base_competence.py`, and the
idempotent production bridge for all TRAINED parents is `scripts/evaluate_trained_bases.py` (full
runtime and disconnect contract: `docs/BASE_COMPETENCE_EVALUATION.md`). The receipt path is
`scripts/materialize_base_competence.py`. It accepts the evaluator's
structured JSON output rather than an accuracy flag, atomically writes the receipt and sidecar,
validates the would-be `DONE` entry, and only then appends the TRAINED-to-DONE promotion. Thus
evaluation never relaunches paid training and tests cannot become a production gate bypass.

## How to overturn this

Two alternatives were considered and rejected, and either can be reinstated by the human:

- **Train GPT to competence on an easier topology** (a smaller `d`). Rejected because spec §8 forbids
  changing graph topology, and a `G(2,10)` GPT versus a `G(5,5)` NextLat is not an architecture-matched
  comparison — it changes the task, which is a larger confound than the one it fixes.
- **Train GPT longer until it passes the gate.** Rejected because spec §8 forbids changing step count
  and the paper's result is that GPT does not solve this task at all, not that it is slow to.
