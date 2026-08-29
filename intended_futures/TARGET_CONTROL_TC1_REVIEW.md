# Anti-sycophancy review of the TC1 design

Review date: 2026-08-29  
Verdict: **ready to execute as a narrowly scoped confirmatory pilot; not yet a publishable positive
claim**  
Confidence: **high**

## Strongest legitimate contribution

The completed work supplies an unusually clean representational/causal dissociation: the target is
decodable in one pathway, the original compact patch does not control behavior, and full replay at
a different, literature-motivated pathway nearly transfers target choice. TC1 asks the material
follow-up rather than repeating another layer probe: can the successful distributed state be
compressed into selective control?

## Claim ledger

| Claim | Required evidence | Current status before TC1 |
| --- | --- | --- |
| PaliGemma-13 carries causally sufficient target-routing information | Full replay redirects behavior beyond matched random | Supported descriptively by M0; independent confirmation required |
| A linear observer measures scene-conditioned target displacement at this site | Untouched-state prediction beyond exact prompt-pair means | Not yet tested at PaliGemma-13; mandatory pre-causal gate |
| A compact controller selectively changes target choice | Learned intervention beats clean recipient and norm-matched random while full replay succeeds | Not yet testable |
| The result demonstrates intended futures or a world model | Temporally extended, future-specific counterfactuals and causal use | Unsupported by this design and explicitly disallowed |

## Major risks and repairs

1. **State reuse and adaptive follow-up.** The original 0–9 states contributed to discovery.
   Repair: TC1 uses disjoint official states 10–39 and preserves 40–49 as a one-way replication
   reserve.
2. **Prompt identity masquerading as geometry.** Most displacement variance is structured by the
   instruction pair. Repair: the PaliGemma observer must improve at least 10% over a training-only
   exact-prompt-pair baseline on untouched validation states.
3. **Invalid absolute cosine threshold.** Clean donor and recipient actions were already highly
   similar in M0. Repair: behavioral first touch is primary; action recovery is normalized by the
   clean donor-recipient distance and remains secondary.
4. **Layer and strength fishing.** Repair: layer 13, the inverse damping, norm cap, condition set,
   and endpoints are fixed before outcome collection. No behavioral strength tuning is allowed.
5. **Broad replay is not selective control.** Repair: full replay is only the manipulation check;
   the learned controller must also beat a norm-matched random direction.
6. **Privileged simulator information.** The intervention receives oracle object displacement.
   Repair: narrow the claim to oracle-coordinate-assisted controllability. Do not call this an
   autonomous steering method.
7. **Limited scene-level power.** Twelve scenes cannot rule out modest effects reliably. Repair:
   report that M0-based approximate power is only 30% at 15 points and about 81% at 30 points,
   require intervals, and never convert failure to pass by consuming the reserve split.

## Acceptance tests

- All 480 candidate states exist, contain both intended objects, remain inside the workspace, and
  exceed the 5 cm separation contract before the protocol is frozen.
- Observer files contain only fit/validation splits; fitting code never loads causal/reserve rows.
- Every observer and causal record binds the pre-outcome single-GPU runtime receipt.
- The observer gate is computed before a controller clearance can exist.
- Every intervention receipt binds the controller hash and patches every matching call.
- Primary and reserve decisions are generated mechanically from the frozen JSON thresholds.
- Reporting says “instruction-selected pickup target,” not “future,” “planning,” or “world model.”

## Prior-art boundary

Action Atlas motivates pathway and layer selection through π0.5 goal decoding and activation
injection, but its results do not prove our target controller. LIBERO-CF supplies official
counterfactual scenes and evaluation machinery, but our paired prompt contrasts are a derived
dataset and must be described as such.

- Action Atlas: https://arxiv.org/abs/2603.19233
- Official Action Atlas code: https://github.com/CWRU-AISM/action-atlas
- LIBERO-CF: https://github.com/yuffish/libero-cf

## Publication judgment

Would I let a respected colleague publish a positive claim now? **No.** I would let them execute
this protocol. A publishable positive claim requires a passed observer gate, a passed full-replay
check, selective learned-control effects on the untouched causal test, and an unchanged reserve
replication. A negative result is still reportable, but only with the controller- and
population-specific boundary above.
