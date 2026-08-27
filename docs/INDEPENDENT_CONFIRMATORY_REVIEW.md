# Independent confirmatory review

Reviewer: Claude Opus 5 through the tool-less `gstack-claude` review path  
Review mode: outcome-blind; no filesystem, shell, write, web, or subagent tools  
Full re-review session: `ae93717c-16a1-44f6-af32-87a7fd2e10c8`  
Final validator-delta review session: `0511f988-ecac-4157-8355-80f918ea433b`

## Final verdict

**PASS**

Unresolved P0 scientific or execution-safety issues: **none**.  
Unresolved P1 issues: **none**.

This verdict authorizes the pre-compute design freeze and launch-integrity sequence for:

- Lure-Star base-only H1/H2: GPT, NextLat, and BST at seeds 1234–1238 (15 base jobs).
- The complete HMM family: three frozen regimes by GPT/NextLat by seeds 1234–1238
  (30 jobs), training followed by evaluation and complete aggregation.

It does not authorize Lure-Star H3, adaptation, interference, mechanism probes, scientific
outcome selection, or any relaxation of the frozen intersection-union and Holm rules.

## Review history and remediation

The first review returned **BLOCK** and therefore authorized zero confirmatory compute. It found
one guaranteed zero-progress runtime failure and three proof/hardening gaps. No GPU was active
while they were repaired.

1. The runtime unconditionally required the superseded single-HMM corpus even though the reduced
   immutable bundle intentionally contains only `hmm_family`. The runtime now branches on the
   semantically validated runner: HMM train/evaluate downloads only the complete family, while
   Lure-Star base downloads no HMM arrays. Missing required family inputs fail before child
   dispatch.
2. The clearance path already required a source-bound PASS review; explicit BLOCK/FAIL issuance
   and launch regressions now prove that condition at both boundaries.
3. Unreachable retired Lure-Star H3 implementations and legacy bindings were physically removed.
   Structural tests require their absence while preserving the separate legacy-named HMM
   posterior/future-state diagnostics.
4. The HMM aggregate now refuses anything other than all 30 exact receipts. Operational recovery
   subsets are marked aggregate-ineligible, and legacy-primary identities cannot enter a family
   selection.
5. Every new `colab start` now requires two agreeing quota reads, hard-floor authorization, and
   two fresh agreeing no-runtime status reads. This check occurs inside the retry loop, immediately
   before each provisioning attempt.
6. The first all-eleven candidate build exposed a path-root derivation defect: a nested candidate
   evidence directory was incorrectly treated as the project root for canonical H3 binding. The
   validator now anchors scientific subject paths to the authoritative spec's parent while using
   the evidence directory only for archive-excluded containment. The outcome-blind delta review
   returned PASS with zero unresolved P0/P1 and confirmed this is a semantic no-op on any input
   capable of producing a published PASS receipt.

The reviewer accepted the reported local evidence of 1,019 whole-suite tests passing before these
repairs and 256 targeted adversarial tests passing afterward, while correctly noting that it did
not execute those tests itself.

## Sequenced conditions

The PASS is conditional on the enforced launch sequence, not a waiver of it:

1. Include this report in the deterministic source archive.
2. Run the complete suite against that exact source and bind its PASS receipt to the archive SHA.
3. Build and validate the all-eleven-gate preregistration receipt, with gate 11 recording this
   review as PASS and `confirmatory_compute_launched: false`.
4. Issue GO clearance for the exact source archive, semantic job spec, protocol files, and the
   generation-pinned 46-object input bundle whose inventory SHA is
   `33fbf4358b7c7def932fb96c1f4a5c04cb8713925dccbff5d385982e910a5c43`.
5. Keep GPU compute off until all preceding conditions pass. Any source, protocol, job-spec, or
   input-bundle change invalidates clearance and requires a fresh review.

No confirmatory outcome was inspected or inferred during either review.
