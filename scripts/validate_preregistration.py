#!/usr/bin/env python
"""Deterministic all-eleven-gate preregistration freeze validator.

This command never launches training and never interprets scientific outcomes.  It validates a
hash-bound evidence index whose eleven blocks correspond one-for-one to amendment section 8.  Both
missing and unexpected blocks/fields are refusals.  PASS means only that the pre-compute design is
completely frozen; it does not mean any hypothesis is supported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Any, Mapping

SCHEMA = "nextlat_forgetting/preregistration_evidence/1"
RECEIPT_SCHEMA = "nextlat_forgetting/preregistration_freeze_receipt/1"
ATTESTATION_SCHEMA = "nextlat_forgetting/preregistration_artifact_attestation/1"
TEST_EVIDENCE_SCHEMA = "nextlat_forgetting/preregistration_test_evidence/1"
AMENDMENT_SCHEMA = "text/markdown-preregistration-amendment-2026-08-24"
SPEC_SCHEMA = "text/markdown-authoritative-spec"
REGIMES = ("persistent_moderate", "fast_mixing_moderate", "persistent_high_aliasing")
ARMS = ("gpt", "nextlat", "bst")
BRANCHES = ("near", "mid", "far")
METRICS = ("centered_cosine", "whitened_mahalanobis")
SEEDS = (1234, 1235, 1236, 1237, 1238)
LURESTAR_SCHEMA_CONTRACT = {
    "extraction_job": "nextlat_forgetting/lurestar_evidence_extraction_job/3",
    "extraction_progress": "nextlat_forgetting/lurestar_evidence_progress/1",
    "evidence_npz": "nextlat_forgetting/lurestar_evidence/4",
    "evidence_receipt": "nextlat_forgetting/lurestar_evidence/4",
    "evaluation_manifest": "nextlat_forgetting/lurestar_evaluation_manifest/4",
    "confirmatory_report": "nextlat_forgetting/lurestar_confirmatory_report/4",
    "evaluation_receipt": "nextlat_forgetting/lurestar_evaluation_receipt/4",
}
LURESTAR_SEMANTIC_TEST_NODES = (
    "tests/test_lurestar_evidence_extractor.py::test_assembly_emits_npsi_whitener_audit_and_labeled_secondaries",
    "tests/test_lurestar_evidence_extractor.py::test_job_binds_only_base_and_exact_canonical_permanent_block",
    "tests/test_lurestar_checkpoint_evaluator.py",
    "tests/test_representations.py::test_npsi_formula_and_fail_closed_denominator",
    "tests/test_representations.py::test_seed_interval_is_student_t_with_loso_not_seed_bootstrap",
    "tests/test_representations.py::test_h2_base_id_folds_are_exact_sha_parity_and_seed_free",
    "tests/test_representations.py::test_h2_is_nested_incremental_on_identical_folds",
    "tests/test_representations.py::test_intermediate_hooks_return_fixed_12x2_stack_verify_parity_and_remove",
    "tests/test_representations.py::test_bst_intermediate_hooks_capture_forward_stack_only",
    "tests/test_representations.py::test_whitened_euclidean_equals_mahalanobis_under_the_same_covariance",
    "tests/test_representations.py::test_reported_whitened_metric_demands_a_checkable_heldout_claim",
    "tests/test_materialize_lurestar_evaluation.py::test_dry_run_preflights_exact_15_and_writes_nothing",
    "tests/test_materialize_lurestar_evaluation.py::test_invalid_fifteenth_parent_permits_zero_subprocess_invocations",
    "tests/test_materialize_lurestar_evaluation.py::test_stale_fifteenth_evidence_permits_zero_subprocess_invocations",
    "tests/test_materialize_lurestar_evaluation.py::test_missing_or_extra_base_cell_is_refused_during_preflight",
    "tests/test_run_hmm_matrix.py::test_evaluator_preflight_accepts_exact_canonical_30_before_any_invocation",
    "tests/test_run_hmm_matrix.py::test_atomic_30_job_evaluation_preflight_refuses_before_first_cell",
    "tests/test_hmm_family.py::test_fisher_z_uses_exact_atanh_and_refuses_boundary_correlations",
    "tests/test_hmm_family.py::test_exact_sign_flip_discreteness_is_explicit",
    "tests/test_hmm_family.py::test_aggregate_requires_all_regimes_models_seeds_and_metrics",
    "tests/test_hmm_family.py::test_null_language_mde_and_regime_sign_reversal_are_mandatory_report_only",
)
LURESTAR_SEMANTIC_MODULES = (
    "scripts/build_preregistration_evidence.py",
    "scripts/extract_lurestar_evidence.py",
    "scripts/evaluate_lurestar_checkpoints.py",
    "scripts/materialize_lurestar_evaluation.py",
    "scripts/run_hmm_matrix.py",
    "src/lurestar/evaluate.py",
    "src/lurestar/representations.py",
    "src/hmm_geometry/aggregate.py",
    "tests/test_lurestar_evidence_extractor.py",
    "tests/test_lurestar_checkpoint_evaluator.py",
    "tests/test_representations.py",
    "tests/test_materialize_lurestar_evaluation.py",
    "tests/test_run_hmm_matrix.py",
    "tests/test_hmm_family.py",
)
LURESTAR_MANIPULATION_FAILURE_CONTRACT = {
    "applicable": False,
    "reason": "H3_PERMANENTLY_DROPPED_AFTER_D40_FEASIBILITY_GATE",
    "interpretation": "no surviving Lure-Star manipulation endpoint; not an H1 outcome check",
}
# Every entry names an executed pytest function and tokens that must occur in an assert or
# pytest.raises statement.  Thus a test name left behind after deleting its scientific assertion
# cannot witness the contract.
LURESTAR_SEMANTIC_WITNESS_SPECS: dict[str, dict[str, object]] = {
    "npsi_formula_and_denominator": {
        "node": "tests/test_representations.py::test_npsi_formula_and_fail_closed_denominator",
        "assertion_tokens": ("normalized_psi", "npsi", "strictly positive and finite"),
    },
    "paired_student_t_and_loso": {
        "node": "tests/test_representations.py::test_seed_interval_is_student_t_with_loso_not_seed_bootstrap",
        "assertion_tokens": ("leave_one_seed_out", "seed bootstrap RNG"),
        "function_tokens": ("stats.t.ppf",),
    },
    "exact_sha_base_id_folds": {
        "node": "tests/test_representations.py::test_h2_base_id_folds_are_exact_sha_parity_and_seed_free",
        "assertion_tokens": ("base_id_folds", "int(base_id, 16) % 2", "lowercase SHA-256"),
        "function_tokens": ('hashlib.sha256(base_id.encode("utf-8")).hexdigest()',),
    },
    "nested_h2_m0_delta_r2_identical_folds": {
        "node": "tests/test_representations.py::test_h2_is_nested_incremental_on_identical_folds",
        "assertion_tokens": ("M0", "M1", "delta_r2_heldout", "fold_index"),
    },
    "all_12_hooks_parity_and_cleanup": {
        "node": "tests/test_representations.py::test_intermediate_hooks_return_fixed_12x2_stack_verify_parity_and_remove",
        "assertion_tokens": ("intermediate_hidden", "(4, 12, 2, 6)", "_forward_hooks", "block 11 plus final norm"),
        "function_tokens": ("capture_blocks=True",),
    },
    "bst_forward_only_all_12_hooks": {
        "node": "tests/test_representations.py::test_bst_intermediate_hooks_capture_forward_stack_only",
        "assertion_tokens": ("intermediate_hidden", "(2, 12, 2, 6)", "transformer_f.blocks", "transformer_b.blocks"),
        "function_tokens": ('architecture="bst"', "capture_blocks=True"),
    },
    "whitener_exact_mahalanobis_parity": {
        "node": "tests/test_representations.py::test_whitened_euclidean_equals_mahalanobis_under_the_same_covariance",
        "assertion_tokens": ("np.allclose(got, want", "np.linalg.inv(w.covariance)"),
        "function_tokens": ("np.linalg.solve(w.covariance",),
    },
    "whitener_heldout_claim_fail_closed": {
        "node": "tests/test_representations.py::test_reported_whitened_metric_demands_a_checkable_heldout_claim",
        "assertion_tokens": ("pool_is_heldout", "fit without item_ids", "item_ids is required"),
    },
    "atomic_lurestar_exact_15_dry_run": {
        "node": "tests/test_materialize_lurestar_evaluation.py::test_dry_run_preflights_exact_15_and_writes_nothing",
        "assertion_tokens": ('result["status"] == "DRY_RUN"', 'result["plan"]["cell_count"] == 15', "len(validated) == 15", "not project[\"evaluation_root\"].exists()"),
    },
    "atomic_lurestar_invalid_fifteenth_zero_invocations": {
        "node": "tests/test_materialize_lurestar_evaluation.py::test_invalid_fifteenth_parent_permits_zero_subprocess_invocations",
        "assertion_tokens": ('checked[-1] == "bst-s1238-base"', "len(checked) == 15", "invocations == []", "not project[\"evaluation_root\"].exists()"),
    },
    "atomic_lurestar_stale_fifteenth_zero_invocations": {
        "node": "tests/test_materialize_lurestar_evaluation.py::test_stale_fifteenth_evidence_permits_zero_subprocess_invocations",
        "assertion_tokens": ("evidence/receipt pair is incomplete", "invocations == []"),
    },
    "atomic_lurestar_exact_cell_set": {
        "node": "tests/test_materialize_lurestar_evaluation.py::test_missing_or_extra_base_cell_is_refused_during_preflight",
        "assertion_tokens": ("missing=.*bst-s1238-base.*extra",),
    },
    "atomic_hmm_exact_30_acceptance": {
        "node": "tests/test_run_hmm_matrix.py::test_evaluator_preflight_accepts_exact_canonical_30_before_any_invocation",
        "assertion_tokens": ("(TRAINED,) * 30", "invoked == [job.job_id for job in jobs]"),
    },
    "atomic_hmm_exact_30_refusal": {
        "node": "tests/test_run_hmm_matrix.py::test_atomic_30_job_evaluation_preflight_refuses_before_first_cell",
        "assertion_tokens": ("canonical 30|TRAINED|checkpoint|provenance", "invoked == []"),
    },
    "hmm_fisher_z_exact_and_boundary_fail_closed": {
        "node": "tests/test_hmm_family.py::test_fisher_z_uses_exact_atanh_and_refuses_boundary_correlations",
        "assertion_tokens": ("pytest.approx(expected)", "pytest.raises(HMMAggregationError, match=error)"),
        "function_tokens": ("np.nextafter(1.0, 0.0)", "math.atanh(valid_rho) - math.atanh(0.25)", "Fisher-z.*open interval", "nonfinite h2_partial_spearman"),
    },
    "hmm_two_sided_sign_flip_floor": {
        "node": "tests/test_hmm_family.py::test_exact_sign_flip_discreteness_is_explicit",
        "assertion_tokens": ("exact_sign_flip_two_sided(np.ones(5)) == 2 / 32", "exact_sign_flip_two_sided(-np.ones(5)) == 2 / 32"),
    },
    "hmm_two_sided_mde_and_exact_family": {
        "node": "tests/test_hmm_family.py::test_aggregate_requires_all_regimes_models_seeds_and_metrics",
        "assertion_tokens": ("exact_two_sided_sign_flip_p", "standardized_mde_80pct_power_two_sided_alpha_0.05", '"one_sided" not in json.dumps(result)'),
    },
    "hmm_null_and_heterogeneity_report_only": {
        "node": "tests/test_hmm_family.py::test_null_language_mde_and_regime_sign_reversal_are_mandatory_report_only",
        "assertion_tokens": ("not resolved at the detectable effect size", "raw_scale_mde_80pct_power_two_sided_alpha_0.05", "equivalence_claim_permitted", "sign_reversal_across_regimes", "report_only_cannot_promote_or_alter_primary"),
    },
    "extractor_npsi_and_audit": {
        "node": "tests/test_lurestar_evidence_extractor.py::test_assembly_emits_npsi_whitener_audit_and_labeled_secondaries",
        "assertion_tokens": ("npsi", "normalized_psi"),
        "function_tokens": ("whitener_fit_source_sha256",),
    },
    "report_schema_and_required_statistics": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_report_emits_npsi_student_t_iut_and_nested_h2",
        "assertion_tokens": ("npsi", "M0", "delta_r2_heldout", "folds_reused_exactly", "two-sided paired Student-t", "leave_one_seed_out"),
    },
    "h1_four_state_classifier": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_h1_intersection_union_classifier_has_all_four_frozen_states",
        "assertion_tokens": ("metric-robust confirmatory support", "directionally consistent but unresolved evidence", "metric-dependent evidence", "no support"),
    },
    "tampered_field_invalid_emission": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_evaluator_refuses_tampered_npsi_or_whitener_audit_fields",
        "assertion_tokens": ("INVALID_INCOMPLETE", "invalid_cells", "nextlat"),
    },
    "base_only_checkpoint_scope": {
        "node": "tests/test_lurestar_evidence_extractor.py::test_job_binds_only_base_and_exact_canonical_permanent_block",
        "assertion_tokens": ("checkpoints", "base", "H3_BLOCK_SHA256"),
    },
    "adaptation_checkpoint_refusal": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_refuses_every_adaptation_checkpoint_field",
        "assertion_tokens": ("INVALID_INCOMPLETE", "invalid_cells", "only arm, seed, base checkpoint"),
    },
    "h3_mechanism_array_refusal": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_refuses_any_h3_adaptation_or_mechanism_array",
        "assertion_tokens": ("INVALID_INCOMPLETE", "invalid_cells", "extra="),
    },
    "h3_analysis_and_old_schema_refusal": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_refuses_h3_analysis_control_or_old_schema",
        "assertion_tokens": ("pytest.raises", "H3 analysis controls are forbidden", "manifest schema"),
    },
    "invalid_cells_terminal_schema": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_invalid_cell_emits_terminal_incomplete_report_and_receipt",
        "assertion_tokens": ("INVALID_INCOMPLETE", "invalid_cells", "reason_code", "NPSI_INVALID"),
    },
    "non_equivalence_nulls_and_manipulation_failures": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_terminal_report_emits_non_equivalence_null_interpretations",
        "assertion_tokens": ("nulls", "not resolved at the detectable effect size", "never evidence of equivalence", "manipulation_failures"),
        "source_literal": {
            "path": "scripts/evaluate_lurestar_checkpoints.py",
            "name": "MANIPULATION_FAILURES",
            "value": LURESTAR_MANIPULATION_FAILURE_CONTRACT,
        },
    },
    "terminal_required_fields_fail_closed": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_terminal_contract_rejects_removed_or_changed_nulls_manipulation_and_invalid_cells",
        "assertion_tokens": ("pytest.raises", "terminal report"),
        "function_tokens": ("invalid_cells", "nulls", "manipulation_failures"),
    },
    "binary_h2_secondary_ceiling_status": {
        "node": "tests/test_lurestar_checkpoint_evaluator.py::test_binary_h2_secondary_reports_ceiling_without_invalidating_primary",
        "assertion_tokens": ("first_branch_accuracy_secondary", "exact_path_accuracy_secondary", "both_metric_classification"),
        "function_tokens": ("not_estimable_due_to_ceiling/constant training-fold predictor",),
    },
}
H3_BLOCK_PATH = pathlib.Path("manifests/h3_selected/PERMANENT_H3_BLOCK.json")
H3_BLOCK_SHA256 = "82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778"
H3_BLOCK_SCHEMA = "nextlat_forgetting/h3_mid_expansion/1"
H3_BLOCK_ROLES = frozenset({
    "h3_permanent_block_receipt",
    "h3_adaptation_exclusion_receipt",
    "h3_mechanism_exclusion_receipt",
    "h3_analysis_exclusion_receipt",
})
H3_BLOCK_DOCUMENT: dict[str, Any] = {
    "combined_loss_sha256": "814058a162e12fde36c7204dd30798b63bfbf02294fce768046070672e5afece",
    "expanded_manifest_sha256": "2effd4e13d384786546c71cc61b4138dc97f082e3992bf3cdf398e6bf93264f1",
    "forbidden": [
        "candidate_expansion", "caliper_change", "weighting", "unmatched_restriction",
        "pilot_substitution", "matching_amendment",
    ],
    "no_further_amendments_permitted": True,
    "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
    "schema": H3_BLOCK_SCHEMA,
    "status": "PERMANENT_H3_BLOCK",
    "unmatched_count": 4,
    "unmatched_identity_sha256":
        "ab4fb10a1e049912fb3e24046cf1498b1027e489864e076d91c10044cef82bf6",
}
BOUND_RAW_ROLES = frozenset({
    "hmm_family_manifest", "hmm_materialization_receipt", *H3_BLOCK_ROLES,
})

ARTIFACT_SCHEMAS: dict[str, dict[str, str]] = {
    "1": {
        "amendment": AMENDMENT_SCHEMA,
        "authoritative_spec": SPEC_SCHEMA,
        "source_snapshot": "binary/source-snapshot",
    },
    "2": {
        "split_receipt": "nextlat_forgetting/eval_split_receipt/1",
        "five_condition_manifest": "nextlat_forgetting/e_lure_conditions/1",
        "disjointness_receipt": "nextlat_forgetting/pool_disjointness_receipt/1",
    },
    "3": {
        "whitener_fixture_receipt": "nextlat_forgetting/whitener_fixture_receipt/1",
        "metric_fixture_receipt": "nextlat_forgetting/co_primary_metric_fixture_receipt/1",
    },
    "4": {
        "h3_permanent_block_receipt": "nextlat_forgetting/h3_permanent_block_attestation/1",
    },
    "5": {
        "h3_adaptation_exclusion_receipt":
            "nextlat_forgetting/h3_adaptation_exclusion_attestation/1",
    },
    "6": {
        "h3_mechanism_exclusion_receipt":
            "nextlat_forgetting/h3_mechanism_exclusion_attestation/1",
    },
    "7": {
        "h3_analysis_exclusion_receipt":
            "nextlat_forgetting/h3_analysis_exclusion_attestation/1",
    },
    "8": {
        "hmm_family_manifest": "nextlat_forgetting/hmm_family/1",
        "hmm_materialization_receipt": "nextlat_forgetting/hmm_family_materialization/1",
        "hmm_te_receipt": "nextlat_forgetting/hmm_te_certificates/1",
    },
    "9": {
        "aggregate_fixture_receipt": "nextlat_forgetting/aggregate_fixture_receipt/1",
        "multiplicity_fixture_receipt": "nextlat_forgetting/multiplicity_fixture_receipt/1",
    },
    "10": {
        "lurestar_schema_receipt": "nextlat_forgetting/lurestar_schema_fixture/1",
        "hmm_schema_receipt": "nextlat_forgetting/hmm_schema_fixture/1",
    },
    "11": {
        "full_suite_receipt": "nextlat_forgetting/full_test_suite_receipt/1",
        "independent_review_receipt": "nextlat_forgetting/independent_scientific_review/1",
    },
}

EXPECTED_CHECKS: dict[str, dict[str, Any]] = {
    "1": {"frozen_before_outcomes": True, "confirmatory_training_started": False},
    "2": {
        "e_white_count": 400, "e_score_count": 1600, "overlap_count": 0,
        "conditions": ["base", "repeat", "near_safe", "near_critical", "far_critical"],
        "training_overlap_count": 0, "adaptation_overlap_count": 0,
        "membership_rule": "ascending_sha256_of_canonical_base_serialization",
    },
    "3": {
        "metrics": list(METRICS), "same_e_score_ids": True,
        "whitener_fit_population": "E_white_only", "synthetic_fixtures_pass": True,
        "nonfinite_or_zero_denominator_refused": True,
    },
    "4": {
        "h3_status": "PERMANENTLY_DROPPED_AFTER_D40_FEASIBILITY_GATE",
        "d40_unmatched_count": 4,
        "no_further_h3_amendments_permitted": True,
        "confirmatory_h3_included": False,
    },
    "5": {
        "confirmatory_h3_adaptation_included": False,
        "h3_exclusion_bound_to_permanent_block": True,
    },
    "6": {
        "confirmatory_h3_mechanism_probes_included": False,
        "h3_exclusion_bound_to_permanent_block": True,
    },
    "7": {
        "confirmatory_h3_analysis_included": False,
        "h3_exclusion_bound_to_permanent_block": True,
    },
    "8": {
        "regimes": list(REGIMES), "family_hash_bound": True,
        "corpora_frozen": True, "pair_banks_frozen": True,
        "thresholds_frozen_from_validation": True,
        "pair_selection_distance": "future_distribution_js",
        "common_seed_formula": "1105963+regime_index*100000",
    },
    "9": {
        "procedures": ["within_seed_equal_regime_aggregate", "exact_sign_flip", "mde_80",
                       "leave_one_seed_out", "intersection_union", "holm_five_endpoints"],
        "seeds": list(SEEDS), "regimes_aggregated_inside_seed": True,
        "items_as_replications": False, "deterministic_fixtures_pass": True,
    },
    "10": {
        "schemas": ["nextlat_forgetting/lurestar_evidence_extraction_job/3",
                    "nextlat_forgetting/lurestar_evidence_progress/1",
                    "nextlat_forgetting/lurestar_evidence/4",
                    "nextlat_forgetting/lurestar_evaluation_manifest/4",
                    "nextlat_forgetting/lurestar_confirmatory_report/4",
                    "nextlat_forgetting/lurestar_evaluation_receipt/4",
                    "nextlat_forgetting/hmm_geometry/1",
                    "nextlat_forgetting/hmm_cross_seed_aggregate/3"],
        "lurestar_schema_contract": LURESTAR_SCHEMA_CONTRACT,
        "lurestar_confirmatory_scope": "base_only_h1_h2",
        "h1_h2_metrics_preserved": True,
        "permanent_h3_exclusion_required": True,
        "h3_fields_refused": True,
        "adaptation_fields_refused": True,
        "mechanism_fields_refused": True,
        "missing_metrics_refused": True, "extra_metrics_refused": True,
        "invalid_cells_emitted": True, "nulls_emitted": True,
        "manipulation_failures_emitted": True, "multiplicity_fields_emitted": True,
    },
    "11": {
        "full_suite_pass": True, "unresolved_p0_scientific": 0,
        "unresolved_p1_scientific": 0, "independent_review_pass": True,
        "confirmatory_compute_launched": False,
    },
}

# Each role must independently attest the subset of gate claims it is competent to establish.
# The union for every gate covers every top-level check, preventing a truthful gate block from
# hiding a contradictory or placeholder artifact.
ROLE_CHECK_KEYS: dict[str, tuple[str, ...]] = {
    "split_receipt": ("e_white_count", "e_score_count", "overlap_count", "membership_rule"),
    "five_condition_manifest": ("conditions", "e_score_count"),
    "disjointness_receipt": ("overlap_count", "training_overlap_count",
                             "adaptation_overlap_count"),
    "whitener_fixture_receipt": ("whitener_fit_population", "synthetic_fixtures_pass",
                                  "nonfinite_or_zero_denominator_refused"),
    "metric_fixture_receipt": ("metrics", "same_e_score_ids", "synthetic_fixtures_pass",
                               "nonfinite_or_zero_denominator_refused"),
    "h3_permanent_block_receipt": (
        "h3_status", "d40_unmatched_count", "no_further_h3_amendments_permitted",
        "confirmatory_h3_included",
    ),
    "h3_adaptation_exclusion_receipt": (
        "confirmatory_h3_adaptation_included", "h3_exclusion_bound_to_permanent_block",
    ),
    "h3_mechanism_exclusion_receipt": (
        "confirmatory_h3_mechanism_probes_included", "h3_exclusion_bound_to_permanent_block",
    ),
    "h3_analysis_exclusion_receipt": (
        "confirmatory_h3_analysis_included", "h3_exclusion_bound_to_permanent_block",
    ),
    "hmm_family_manifest": ("regimes", "family_hash_bound", "pair_selection_distance",
                            "common_seed_formula"),
    "hmm_materialization_receipt": ("corpora_frozen", "pair_banks_frozen",
                                    "thresholds_frozen_from_validation"),
    "hmm_te_receipt": (),
    "aggregate_fixture_receipt": ("procedures", "seeds", "regimes_aggregated_inside_seed",
                                  "items_as_replications", "deterministic_fixtures_pass"),
    "multiplicity_fixture_receipt": ("procedures", "deterministic_fixtures_pass"),
    "lurestar_schema_receipt": ("schemas", "missing_metrics_refused", "extra_metrics_refused",
                                "invalid_cells_emitted", "nulls_emitted",
                                "manipulation_failures_emitted", "lurestar_schema_contract",
                                "lurestar_confirmatory_scope", "h1_h2_metrics_preserved",
                                "permanent_h3_exclusion_required", "h3_fields_refused",
                                "adaptation_fields_refused", "mechanism_fields_refused"),
    "hmm_schema_receipt": ("schemas", "missing_metrics_refused", "extra_metrics_refused",
                           "invalid_cells_emitted", "nulls_emitted",
                           "multiplicity_fields_emitted"),
    "full_suite_receipt": ("full_suite_pass", "confirmatory_compute_launched"),
    "independent_review_receipt": ("unresolved_p0_scientific", "unresolved_p1_scientific",
                                   "independent_review_pass", "confirmatory_compute_launched"),
}


def _assert_role_coverage() -> None:
    for gate, roles in ARTIFACT_SCHEMAS.items():
        if gate == "1":
            continue
        covered = {key for role in roles for key in ROLE_CHECK_KEYS[role]}
        if gate == "8":
            covered.add("te_certificates")
        expected = set(EXPECTED_CHECKS[gate]) | ({"te_certificates"} if gate == "8" else set())
        if covered != expected:
            raise RuntimeError(
                f"internal role/check coverage mismatch for gate {gate}: "
                f"missing={sorted(expected-covered)}, extra={sorted(covered-expected)}"
            )


_assert_role_coverage()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def derive_lurestar_semantic_witnesses(project_root: pathlib.Path) -> dict[str, dict[str, object]]:
    """Derive source hashes for the assertions that witness every gate-10 Lure semantic."""
    root = project_root.resolve()
    parsed: dict[pathlib.Path, tuple[str, ast.Module]] = {}
    witnesses: dict[str, dict[str, object]] = {}
    for feature, spec in LURESTAR_SEMANTIC_WITNESS_SPECS.items():
        node = str(spec["node"])
        try:
            relative, function_name = node.split("::", 1)
        except ValueError as exc:  # pragma: no cover - guarded by the frozen constant above
            raise ValueError(f"semantic witness {feature} has an invalid pytest node") from exc
        path = root / relative
        if path not in parsed:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"semantic witness module is missing/invalid: {relative}") from exc
            parsed[path] = (source, tree)
        source, tree = parsed[path]
        functions = [
            item for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == function_name
        ]
        if len(functions) != 1:
            raise ValueError(f"semantic witness function is absent/duplicate: {node}")
        function = functions[0]
        start_line = min(
            [function.lineno, *(item.lineno for item in function.decorator_list)]
        )
        function_source = "".join(
            source.splitlines(keepends=True)[start_line - 1:function.end_lineno]
        )
        if re.search(r"pytest\.(?:skip|xfail)|skipif", function_source):
            raise ValueError(f"semantic witness may not be skipped or xfailed: {node}")
        semantic_statements = [
            statement for statement in ast.walk(function)
            if isinstance(statement, ast.Assert)
            or (
                isinstance(statement, (ast.With, ast.AsyncWith))
                and "pytest.raises" in (ast.get_source_segment(source, statement) or "")
            )
        ]
        assertion_source = "\n".join(
            ast.get_source_segment(source, statement) or ast.dump(statement)
            for statement in semantic_statements
        )
        required_tokens = tuple(str(value) for value in spec["assertion_tokens"])
        missing = [token for token in required_tokens if token not in assertion_source]
        function_tokens = tuple(str(value) for value in spec.get("function_tokens", ()))
        missing_function = [token for token in function_tokens if token not in function_source]
        if not semantic_statements or missing or missing_function:
            raise ValueError(
                f"semantic witness {node} lacks required semantics: "
                f"assertions={missing}, function={missing_function}"
            )
        witnesses[feature] = {
            "pytest_node": node,
            "test_path": relative,
            "test_source_sha256": sha256_file(path),
            "test_function_ast_sha256": hashlib.sha256(
                ast.dump(function, annotate_fields=True, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "assertion_tokens": list(required_tokens),
            "function_tokens": list(function_tokens),
        }
        literal_spec = spec.get("source_literal")
        if literal_spec is not None:
            assert isinstance(literal_spec, dict)
            literal_path = root / str(literal_spec["path"])
            try:
                literal_tree = ast.parse(literal_path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"semantic literal source is missing/invalid: {literal_path}") from exc
            assignments = []
            for statement in literal_tree.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                if any(isinstance(target, ast.Name) and target.id == literal_spec["name"]
                       for target in targets):
                    assignments.append(statement.value)
            try:
                actual_literal = ast.literal_eval(assignments[0]) if len(assignments) == 1 else None
            except (TypeError, ValueError):
                actual_literal = None
            if actual_literal != literal_spec["value"]:
                raise ValueError(
                    f"semantic source literal {literal_spec['name']} changed or is not literal"
                )
            witnesses[feature]["source_literal"] = {
                "path": str(literal_spec["path"]),
                "name": str(literal_spec["name"]),
                "value": actual_literal,
                "source_sha256": sha256_file(literal_path),
            }
    return witnesses


def atomic_write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(partial, "w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _exact_keys(value: object, expected: set[str], label: str, issues: list[str]) -> bool:
    if not isinstance(value, dict):
        issues.append(f"{label} must be an object")
        return False
    got = set(value)
    if got != expected:
        issues.append(f"{label} keys mismatch: missing={sorted(expected-got)}, extra={sorted(got-expected)}")
        return False
    return True


def _binding(record: object, label: str, issues: list[str]) -> pathlib.Path | None:
    if not _exact_keys(record, {"path", "sha256"}, label, issues):
        return None
    assert isinstance(record, dict)
    path = pathlib.Path(str(record["path"])).resolve()
    digest = str(record["sha256"])
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(path) != digest:
        issues.append(f"{label} SHA-256 mismatch")
        return None
    return path


def _binding_list(value: object, label: str, issues: list[str]) -> list[tuple[pathlib.Path, str]]:
    if not isinstance(value, list) or not value:
        issues.append(f"{label} must be a nonempty list")
        return []
    resolved: list[tuple[pathlib.Path, str]] = []
    seen: set[pathlib.Path] = set()
    for index, record in enumerate(value):
        path = _binding(record, f"{label}[{index}]", issues)
        if path is None or not isinstance(record, dict):
            continue
        if path in seen:
            issues.append(f"{label} contains duplicate path: {path}")
            continue
        seen.add(path)
        resolved.append((path, str(record["sha256"])))
    return resolved


def _validate_role_payload(
    role: str, payload: object, *, gate_checks: object, issues: list[str],
    project_root: pathlib.Path,
) -> None:
    label = f"artifact[{role}] payload"
    keys = set(ROLE_CHECK_KEYS[role]) | {"claim"}
    if role in BOUND_RAW_ROLES:
        keys.add("subject")
    if role == "lurestar_schema_receipt":
        keys.add("semantic_witnesses")
    if role == "hmm_te_receipt":
        keys |= {"te_certificates", "rank_required", "sigma_min_exclusive_threshold"}
    elif role == "full_suite_receipt":
        keys |= {"exit_code", "tests_passed"}
    elif role == "independent_review_receipt":
        keys.add("reviewer")
    if not _exact_keys(payload, keys, label, issues):
        return
    assert isinstance(payload, dict)
    if payload["claim"] != role:
        issues.append(f"{label} claim mismatch")
    if not isinstance(gate_checks, dict):
        issues.append(f"{label} cannot bind invalid gate checks")
        return
    for key in ROLE_CHECK_KEYS[role]:
        if payload[key] != gate_checks.get(key):
            issues.append(f"{label} contradicts gate check {key}")
    if role == "lurestar_schema_receipt":
        try:
            observed_witnesses = derive_lurestar_semantic_witnesses(project_root)
        except ValueError as exc:
            issues.append(f"{label} semantic witnesses are invalid: {exc}")
        else:
            if payload["semantic_witnesses"] != observed_witnesses:
                issues.append(f"{label} semantic witnesses differ from source assertions")
    if role in BOUND_RAW_ROLES:
        subject = payload["subject"]
        if not _exact_keys(subject, {"path", "sha256", "schema"},
                           f"{label} subject", issues):
            return
        assert isinstance(subject, dict)
        subject_path = _binding(
            {"path": subject["path"], "sha256": subject["sha256"]},
            f"{label} subject", issues,
        )
        expected_subject_schema = (
            H3_BLOCK_SCHEMA if role in H3_BLOCK_ROLES else ARTIFACT_SCHEMAS["8"][role]
        )
        if subject.get("schema") != expected_subject_schema or subject_path is None:
            issues.append(f"{label} subject schema/binding mismatch")
            return
        try:
            raw = json.loads(subject_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(f"{label} raw subject is not valid JSON")
            return
        if not isinstance(raw, dict) or raw.get("schema") != subject["schema"]:
            issues.append(f"{label} raw subject embedded schema mismatch")
        elif role in H3_BLOCK_ROLES:
            expected_path = (project_root / H3_BLOCK_PATH).resolve()
            if subject_path != expected_path:
                issues.append(f"{label} must bind the canonical permanent H3 block path")
            if sha256_file(subject_path) != H3_BLOCK_SHA256 or raw != H3_BLOCK_DOCUMENT:
                issues.append(f"{label} raw permanent H3 block is mutated")
        elif role == "hmm_family_manifest":
            blinding = raw.get("selection_blinding")
            if (raw.get("required_regimes") != list(REGIMES) or
                    raw.get("primary_regime") is not None or
                    not isinstance(blinding, dict) or
                    any(blinding.get(key) is not False for key in (
                        "model_checkpoints_inspected", "model_representations_inspected",
                        "model_outcomes_inspected",
                    ))):
                issues.append(f"{label} raw family manifest is incomplete or unblinded")
        else:
            if (raw.get("status") != "complete" or
                    raw.get("required_regimes") != list(REGIMES) or
                    raw.get("model_outcomes_inspected") is not False or
                    isinstance(raw.get("n_artifacts"), bool) or
                    not isinstance(raw.get("n_artifacts"), int) or raw["n_artifacts"] <= 0 or
                    not re.fullmatch(r"[0-9a-f]{64}", str(raw.get("family_sha256", ""))) or
                    not re.fullmatch(r"[0-9a-f]{64}", str(raw.get("inventory_sha256", "")))):
                issues.append(f"{label} raw materialization receipt is incomplete or unblinded")
    if role == "hmm_te_receipt":
        if (payload["te_certificates"] != gate_checks.get("te_certificates") or
                payload["rank_required"] != 4 or
                payload["sigma_min_exclusive_threshold"] != 0.05):
            issues.append(f"{label} TE certificate contract mismatch")
    elif role == "full_suite_receipt":
        tests_passed = payload["tests_passed"]
        if payload["exit_code"] != 0 or isinstance(tests_passed, bool) or \
                not isinstance(tests_passed, int) or tests_passed <= 0:
            issues.append(f"{label} does not attest a nonempty successful full suite")
    elif role == "independent_review_receipt":
        if not isinstance(payload["reviewer"], str) or not payload["reviewer"].strip():
            issues.append(f"{label} independent reviewer identity is missing")


def _artifact(
    record: object, *, role: str, schema: str, issues: list[str],
    source_snapshot: tuple[pathlib.Path, str] | None,
    evidence_root: pathlib.Path, project_root: pathlib.Path, gate_checks: object,
) -> None:
    label = f"artifact[{role}]"
    if not _exact_keys(record, {"role", "path", "sha256", "schema"}, label, issues):
        return
    assert isinstance(record, dict)
    if record["role"] != role or record["schema"] != schema:
        issues.append(f"{label} role/schema mismatch")
    path = pathlib.Path(str(record["path"])).resolve()
    digest = str(record["sha256"])
    if not path.is_file():
        issues.append(f"{label} is missing: {path}")
        return
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(path) != digest:
        issues.append(f"{label} SHA-256 mismatch")
        return
    if schema.startswith(("text/", "binary/")):
        return
    if source_snapshot is None:
        issues.append(f"{label} cannot attest without a valid gate-1 source snapshot")
        return
    try:
        path.relative_to(evidence_root)
    except ValueError:
        issues.append(f"{label} must live under the archive-excluded evidence directory")
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        issues.append(f"{label} is not valid JSON")
        return
    envelope_keys = {
        "schema", "attestation_schema", "status", "role", "source_archive_sha256",
        "payload_sha256", "payload", "producer", "source_bindings", "test_bindings",
    }
    if not _exact_keys(document, envelope_keys, f"{label} envelope", issues):
        return
    assert isinstance(document, dict)
    if document["schema"] != schema or document["attestation_schema"] != ATTESTATION_SCHEMA:
        issues.append(f"{label} embedded schema/attestation schema mismatch")
    if document["status"] != "PASS":
        issues.append(f"{label} status is not PASS")
    if document["role"] != role:
        issues.append(f"{label} embedded role mismatch")
    snapshot_path, snapshot_sha = source_snapshot
    if document["source_archive_sha256"] != snapshot_sha:
        issues.append(f"{label} source archive binding mismatch")
    payload = document["payload"]
    try:
        payload_sha = canonical_json_sha256(payload)
    except (TypeError, ValueError):
        issues.append(f"{label} payload is not canonical finite JSON")
    else:
        if document["payload_sha256"] != payload_sha:
            issues.append(f"{label} canonical payload SHA-256 mismatch")
    producer_path = _binding(document["producer"], f"{label} producer", issues)
    source_bindings = _binding_list(
        document["source_bindings"], f"{label} source_bindings", issues)
    test_bindings = _binding_list(
        document["test_bindings"], f"{label} test_bindings", issues)
    if producer_path in {None, path, snapshot_path}:
        issues.append(f"{label} producer must be a distinct bound source file")
    elif producer_path is not None:
        try:
            producer_path.relative_to(evidence_root)
        except ValueError:
            pass
        else:
            issues.append(f"{label} producer must live outside archive-excluded evidence state")
    if (snapshot_path, snapshot_sha) not in source_bindings:
        issues.append(f"{label} source_bindings omit the exact gate-1 archive")
    if role in BOUND_RAW_ROLES and isinstance(payload, dict) and \
            isinstance(payload.get("subject"), dict):
        subject = payload["subject"]
        subject_pair = (
            pathlib.Path(str(subject.get("path", ""))).resolve(),
            str(subject.get("sha256", "")),
        )
        if subject_pair not in source_bindings:
            issues.append(f"{label} source_bindings omit the wrapped raw subject")
    if any(bound_path in {path, snapshot_path} for bound_path, _ in test_bindings):
        issues.append(f"{label} test bindings must be independent files")
    if producer_path is not None and any(
            bound_path == producer_path for bound_path, _ in source_bindings + test_bindings):
        issues.append(f"{label} producer/source/test bindings must be distinct")
    if {bound_path for bound_path, _ in source_bindings} & \
            {bound_path for bound_path, _ in test_bindings}:
        issues.append(f"{label} source and test bindings must be disjoint")
    for test_path, _ in test_bindings:
        try:
            test_document = json.loads(test_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(f"{label} test binding is not valid JSON: {test_path}")
            continue
        test_keys = {
            "schema", "status", "role", "source_archive_sha256",
            "exit_code", "tests_passed",
        }
        if role == "lurestar_schema_receipt":
            test_keys |= {"pytest_nodes", "modules", "semantic_witnesses_sha256"}
        if not _exact_keys(test_document, test_keys, f"{label} test evidence", issues):
            continue
        assert isinstance(test_document, dict)
        tests_passed = test_document["tests_passed"]
        if (test_document["schema"] != TEST_EVIDENCE_SCHEMA or
                test_document["status"] != "PASS" or test_document["role"] != role or
                test_document["source_archive_sha256"] != snapshot_sha or
                test_document["exit_code"] != 0 or isinstance(tests_passed, bool) or
                not isinstance(tests_passed, int) or tests_passed <= 0):
            issues.append(f"{label} test evidence is not a passing source-bound receipt")
        if role == "lurestar_schema_receipt" and isinstance(payload, dict):
            if test_document.get("pytest_nodes") != list(LURESTAR_SEMANTIC_TEST_NODES):
                issues.append(f"{label} did not execute the complete frozen semantic test set")
            witness_hash = canonical_json_sha256(payload.get("semantic_witnesses"))
            if test_document.get("semantic_witnesses_sha256") != witness_hash:
                issues.append(f"{label} test evidence does not bind its semantic witnesses")
            module_bindings = _binding_list(
                test_document.get("modules"), f"{label} semantic modules", issues,
            )
            expected_module_paths = {
                (project_root / relative).resolve() for relative in LURESTAR_SEMANTIC_MODULES
            }
            if {path for path, _ in module_bindings} != expected_module_paths:
                issues.append(f"{label} semantic module bindings are incomplete or unexpected")
            source_paths = {path for path, _ in source_bindings}
            if not expected_module_paths.issubset(source_paths | {producer_path}):
                issues.append(f"{label} source bindings omit semantic implementation/tests")
    _validate_role_payload(
        role, payload, gate_checks=gate_checks, issues=issues,
        project_root=project_root,
    )


def _source_snapshot(block: object) -> tuple[pathlib.Path, str] | None:
    if not isinstance(block, dict) or not isinstance(block.get("artifacts"), list):
        return None
    candidates = [
        record for record in block["artifacts"]
        if isinstance(record, dict) and record.get("role") == "source_snapshot"
    ]
    if len(candidates) != 1:
        return None
    record = candidates[0]
    if set(record) != {"role", "path", "sha256", "schema"} or \
            record.get("schema") != "binary/source-snapshot":
        return None
    path = pathlib.Path(str(record.get("path", ""))).resolve()
    digest = str(record.get("sha256", ""))
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest) or \
            sha256_file(path) != digest:
        return None
    return path, digest


def _validate_gate(
    gate_id: str, block: object, *, amendment: pathlib.Path, spec: pathlib.Path,
    source_snapshot: tuple[pathlib.Path, str] | None, evidence_root: pathlib.Path,
) -> dict:
    issues: list[str] = []
    expected_schema = f"nextlat_forgetting/preregistration_gate_{gate_id}/1"
    if not _exact_keys(block, {"schema", "artifacts", "checks"}, f"gate {gate_id}", issues):
        return {"gate": int(gate_id), "status": "BLOCK", "issues": issues}
    assert isinstance(block, dict)
    if block["schema"] != expected_schema:
        issues.append(f"gate {gate_id} schema must be {expected_schema}")
    roles = ARTIFACT_SCHEMAS[gate_id]
    artifacts = block["artifacts"]
    if not isinstance(artifacts, list):
        issues.append("artifacts must be a list")
    else:
        by_role = {item.get("role"): item for item in artifacts if isinstance(item, dict)}
        if len(by_role) != len(artifacts) or set(by_role) != set(roles):
            issues.append(
                f"artifact roles mismatch: missing={sorted(set(roles)-set(by_role))}, "
                f"extra={sorted(set(by_role)-set(roles))}"
            )
        for role, schema in roles.items():
            if role in by_role:
                _artifact(
                    by_role[role], role=role, schema=schema, issues=issues,
                    source_snapshot=source_snapshot, evidence_root=evidence_root,
                    project_root=spec.parent, gate_checks=block.get("checks"),
                )
        if gate_id == "1":
            for role, expected_path in (("amendment", amendment), ("authoritative_spec", spec)):
                if role in by_role and pathlib.Path(str(by_role[role].get("path", ""))).resolve() != expected_path:
                    issues.append(f"gate 1 {role} path differs from CLI authority")
    checks = block["checks"]
    expected = EXPECTED_CHECKS[gate_id]
    if gate_id == "8":
        expected_keys = set(expected) | {"te_certificates"}
        if _exact_keys(checks, expected_keys, "gate 8 checks", issues):
            assert isinstance(checks, dict)
            for key, wanted in expected.items():
                if checks[key] != wanted:
                    issues.append(f"gate 8 check {key} mismatch")
            certs = checks["te_certificates"]
            if not _exact_keys(certs, set(REGIMES), "gate 8 TE certificates", issues):
                pass
            else:
                assert isinstance(certs, dict)
                for regime in REGIMES:
                    cert = certs[regime]
                    if not _exact_keys(cert, {"rank_te", "sigma_min_te"}, f"TE[{regime}]", issues):
                        continue
                    rank, sigma = cert["rank_te"], cert["sigma_min_te"]
                    if rank != 4 or isinstance(sigma, bool) or not isinstance(sigma, (int, float)) \
                            or not math.isfinite(float(sigma)) or float(sigma) <= 0.05:
                        issues.append(f"TE[{regime}] fails rank=4 and sigma_min>0.05")
    elif _exact_keys(checks, set(expected), f"gate {gate_id} checks", issues):
        assert isinstance(checks, dict)
        for key, wanted in expected.items():
            if checks[key] != wanted:
                issues.append(f"gate {gate_id} check {key} mismatch")
    return {"gate": int(gate_id), "status": "PASS" if not issues else "BLOCK", "issues": issues}


def validate(
    evidence_path: pathlib.Path, *, amendment: pathlib.Path, spec: pathlib.Path,
) -> dict:
    amendment, spec, evidence_path = amendment.resolve(), spec.resolve(), evidence_path.resolve()
    global_issues: list[str] = []
    expected_evidence_path = (spec.parent / ".agent_state" /
                              "preregistration-evidence.json").resolve()
    if evidence_path != expected_evidence_path:
        global_issues.append(
            "evidence index must live at the archive-excluded "
            f"{expected_evidence_path} path"
        )
    for label, path in (("amendment", amendment), ("authoritative spec", spec)):
        if not path.is_file():
            global_issues.append(f"{label} is missing: {path}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        evidence = {}
        global_issues.append(f"evidence index unreadable: {exc}")
    if not _exact_keys(evidence, {"schema", "gates"}, "evidence index", global_issues):
        gates: Mapping[str, object] = {}
    else:
        assert isinstance(evidence, dict)
        if evidence["schema"] != SCHEMA:
            global_issues.append(f"evidence schema must be {SCHEMA}")
        gates = evidence["gates"] if isinstance(evidence["gates"], dict) else {}
        if not isinstance(evidence["gates"], dict):
            global_issues.append("evidence gates must be an object")
    wanted = {str(index) for index in range(1, 12)}
    missing, extra = sorted(wanted - set(gates)), sorted(set(gates) - wanted)
    if missing or extra:
        global_issues.append(f"evidence blocks mismatch: missing={missing}, extra={extra}")
    source_snapshot = _source_snapshot(gates.get("1"))
    results = [
        _validate_gate(
            gate_id, gates.get(gate_id), amendment=amendment, spec=spec,
            source_snapshot=source_snapshot, evidence_root=evidence_path.parent,
        )
        for gate_id in sorted(wanted, key=int)
    ]
    passed = not global_issues and all(result["status"] == "PASS" for result in results)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS" if passed else "BLOCK",
        "all_eleven_gates_pass": passed,
        "authority": {
            "amendment": {"path": str(amendment),
                          "sha256": sha256_file(amendment) if amendment.is_file() else None},
            "spec": {"path": str(spec), "sha256": sha256_file(spec) if spec.is_file() else None},
            "evidence": {"path": str(evidence_path),
                         "sha256": sha256_file(evidence_path) if evidence_path.is_file() else None},
            "validator": {"path": str(pathlib.Path(__file__).resolve()),
                          "sha256": sha256_file(__file__)},
        },
        "missing_gate_blocks": missing,
        "extra_gate_blocks": extra,
        "global_issues": global_issues,
        "gates": results,
        "meaning": (
            "pre-compute design frozen; confirmatory outcomes unevaluated; "
            "H3 prospectively dropped after the immutable D40 feasibility gate"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", required=True, type=pathlib.Path)
    parser.add_argument(
        "--spec", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "nextlat_v4_predictive_geometry_spec.md",
    )
    parser.add_argument(
        "--evidence", type=pathlib.Path,
        default=(pathlib.Path(__file__).resolve().parents[1] /
                 ".agent_state/preregistration-evidence.json"),
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args(argv)
    receipt = validate(args.evidence, amendment=args.amendment, spec=args.spec)
    atomic_write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 2 if args.require_all and receipt["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
