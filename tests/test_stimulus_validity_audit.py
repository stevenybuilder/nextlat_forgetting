from __future__ import annotations

from audit_stimulus_validity import (
    _overlap_decomposition,
    cfs_construct_assessment,
    cfs2_construct_assessment,
    render_markdown,
)


def test_cfs_overlap_audit_distinguishes_mechanical_match_from_construct_validity() -> None:
    result = cfs_construct_assessment({
        "high/same": {"18": 5000},
        "high/different": {"18": 5000},
        "low/same": {"8": 5000},
        "low/different": {"7": 5000},
    })

    assert result["low_overlap_factorially_balanced"] is False
    assert result["disposition"] == "REDESIGN_BEFORE_CAUSAL_BRANCH_TRAINING"
    assert "regression cannot identify" in result["critical_finding"]


def test_balanced_cfs2_style_overlap_does_not_trigger_cfs1_confound() -> None:
    result = cfs_construct_assessment({
        "high/same": {"18": 5000},
        "high/different": {"18": 5000},
        "low/same": {"8": 5000},
        "low/different": {"8": 5000},
    })

    assert result["low_overlap_factorially_balanced"] is True
    assert result["disposition"] == "NO_OVERLAP_REDESIGN_REQUIRED"


def test_cfs2_repair_requires_total_and_decomposed_factorial_balance() -> None:
    total = {
        "high/same": {"18": 5000}, "high/different": {"18": 5000},
        "low/same": {"8": 5000}, "low/different": {"8": 5000},
    }
    answer = {
        "high/same": {"4": 5000}, "high/different": {"3": 5000},
        "low/same": {"4": 5000}, "low/different": {"3": 5000},
    }
    nonanswer = {
        "high/same": {"14": 5000}, "high/different": {"15": 5000},
        "low/same": {"4": 5000}, "low/different": {"5": 5000},
    }

    result = cfs2_construct_assessment(total, answer, nonanswer)
    assert result["exact_repaired_contract"] is True
    assert result["nonanswer_high_minus_low_contrast"] == {"same": 10, "different": 10}
    assert result["disposition"] == "FIT_FOR_CONTROLLED_CAUSAL_FORGETTING"

    broken = {**nonanswer, "low/different": {"4": 5000}}
    assert cfs2_construct_assessment(total, answer, broken)["exact_repaired_contract"] is False


def test_overlap_decomposition_is_recomputed_from_edges_not_recorded_fields() -> None:
    probe = "0,1|1,2|0,3|3,4/0,2=0,1,2"
    update = "0,1|1,2|0,3|3,5/0,2=0,1,2"
    assert _overlap_decomposition(probe, update) == (3, 2, 1)


def test_human_report_makes_outcome_blinding_and_cfs_disposition_unmissable() -> None:
    summary = {
        "count": 1, "solver_valid_count": 1, "duplicate_graph_count": 0,
        "duplicate_prompt_count": 0, "prompt_token_length_histogram": {"63": 1},
        "answer_length_histogram": {"5": 1},
        "source_balance": {"unique": 1, "min_count": 1, "max_count": 1},
        "goal_balance": {"unique": 1, "min_count": 1, "max_count": 1},
        "source_goal_pair_coverage": {"unique": 1, "min_count": 1, "max_count": 1},
        "node_id_coverage": {"unique": 21, "min_count": 1, "max_count": 1},
    }
    audit = {
        "audit_date": "2026-08-24",
        "datasets": {
            "pathstar": {
                "coded_invariant_status": "PASS", "construct_validity_status": "FIT",
                "construct_match": "path", "known_limitations": [],
                "train": summary,
                "test": {"count": 1, "solver_valid_count": 1, "duplicate_graph_count": 0},
                "train_test_identity_disjointness": {"prompt_collisions": 0, "graph_collisions": 0},
            },
            "lurestar": {
                "coded_invariant_status": "PASS", "construct_validity_status": "FIT",
                "construct_match": "lure", "known_limitations": [],
                "e_lure": {"quartet_count": 1, "quartet_invariant_violation_count": 0,
                           "collisions_with_pathstar_train": {"prompt_collisions": 0, "graph_collisions": 0},
                           "far_edge_overlap_histogram": {"1": 1}, "suffix_depth_histogram": {"2": 1},
                           "unique_edit_position_pairs": {"near_safe": 1},
                           "condition_summaries": {"base": summary}},
                "legacy_pools": {
                    "a_pair.jsonl": {**summary, "collisions_with_pathstar_train": {"prompts": 1, "graphs": 1}}
                },
            },
            "hmm_family": {
                "coded_invariant_status": "PASS", "construct_validity_status": "CALIBRATION",
                "construct_match": "hmm", "known_limitations": [], "regimes": {},
            },
            "cfs1": {
                "coded_invariant_status": "PASS", "construct_validity_status": "FAIL_REDESIGN_REQUIRED",
                "construct_match": "cfs", "known_limitations": [],
                "factorial_overlap_assessment": {
                    "critical_finding": "8-vs-7 confound",
                    "observed_overlap_histograms": {
                        "high/same": {"18": 1}, "high/different": {"18": 1},
                        "low/same": {"8": 1}, "low/different": {"7": 1},
                    },
                },
                "independent_bundle_validation": {"passed": True},
                "identity_disjointness_from_pathstar_and_lurestar": {"prompt_collisions": 0, "graph_collisions": 0},
                "artifact_summaries": {
                    name: {**summary, **({"edge_overlap_with_probe_histogram": {"18": 1}} if "/" in name else {})}
                    for name in ("retention", "global_controls", "high/same", "high/different", "low/same", "low/different")
                },
                "required_action": "Do not train CFS-1.",
            },
            "cfs2": {
                "coded_invariant_status": "PASS",
                "construct_validity_status": "FIT_FOR_CONTROLLED_CAUSAL_FORGETTING",
                "construct_match": "repaired cfs", "known_limitations": [],
                "factorial_overlap_assessment": {
                    "interpretation": "repair passes",
                    "observed_total_overlap_histograms": {
                        "high/same": {"18": 1}, "high/different": {"18": 1},
                        "low/same": {"8": 1}, "low/different": {"8": 1},
                    },
                    "observed_answer_overlap_histograms": {
                        "high/same": {"4": 1}, "high/different": {"3": 1},
                        "low/same": {"4": 1}, "low/different": {"3": 1},
                    },
                    "observed_nonanswer_overlap_histograms": {
                        "high/same": {"14": 1}, "high/different": {"15": 1},
                        "low/same": {"4": 1}, "low/different": {"5": 1},
                    },
                },
                "independent_bundle_validation": {"passed": True},
                "identity_disjointness_from_pathstar_lurestar_and_cfs1": {
                    "prompt_collisions": 0, "graph_collisions": 0,
                },
                "independent_stream_checks": {
                    "one": {"exact_codebook_order_and_content_match": True},
                },
                "artifact_summaries": {
                    name: {
                        **summary,
                        **({
                            "recomputed_total_overlap_histogram": {"8": 1},
                            "recomputed_answer_overlap_histogram": {"4": 1},
                            "recomputed_nonanswer_overlap_histogram": {"4": 1},
                        } if "/" in name else {}),
                    }
                    for name in (
                        "retention", "global_controls", "high/same", "high/different",
                        "low/same", "low/different",
                    )
                },
                "required_action": "Bind hashes.",
            },
        },
    }

    report = render_markdown(audit)
    assert "Explicitly not opened" in report
    assert "mechanically correct but scientifically confounded" in report
    assert "Pause and replace with CFS-2" in report
    assert "CFS-2 independently verifies the repair" in report
