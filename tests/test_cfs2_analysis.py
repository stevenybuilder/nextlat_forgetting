from __future__ import annotations

import numpy as np
import pytest

from cfs2 import analysis as A


def _matrix(*, optional: bool = True):
    branches = []
    patches = []
    ids = np.asarray([f"probe-{index}" for index in range(6)])
    erosion_by_arm = {
        "high_different": 0.50,
        "high_same": 0.20,
        "low_different": 0.15,
        "low_same": 0.05,
    }
    for parent_number in range(8):
        parent = f"parent-{parent_number}"
        for episode in A.CFS2_EPISODES:
            for arm in A.CFS2_ARMS:
                overlap, relation = arm.split("_", 1)
                job_id = f"{parent}-{episode}-{arm}"
                erosion = erosion_by_arm[arm]
                if arm == "high_different":
                    erosion += parent_number * 0.01 + episode * 0.02
                pre = np.full(ids.size, 2.0)
                arrays = {
                    "pre_correct_first_branch_margin": pre,
                    "post_correct_first_branch_margin": pre - erosion,
                }
                if optional:
                    arrays |= {
                        "pre_retention_cross_entropy": np.full(ids.size, 0.2),
                        "post_retention_cross_entropy": np.full(ids.size, 0.2 + erosion),
                        "pre_retention_exact_path_accuracy": np.full(ids.size, 0.9),
                        "post_retention_exact_path_accuracy": np.full(ids.size, 0.9 - erosion / 2),
                        "adaptation_acquisition": np.full(ids.size, 0.7 + erosion),
                        "pre_global_control_margin": pre,
                        "post_global_control_margin": pre - 0.03,
                        "penultimate_state_drift": np.full(ids.size, 0.1 + erosion),
                    }
                branches.append(A.BranchOutcome(
                    job_id=job_id,
                    parent_id=parent,
                    episode=episode,
                    overlap=overlap,
                    future_relation=relation,
                    item_ids=ids,
                    arrays=arrays,
                    pregeometry=float(parent_number),
                ))
                effects = {}
                for layer in A.PATCH_LAYERS:
                    effects[layer] = {
                        "matching_parent": np.full(ids.size, 0.30 + layer / 100 + parent_number / 1000),
                        "unrelated_anchor": np.full(ids.size, 0.05 + layer / 100),
                        "norm_matched_random_subspace": np.full(ids.size, 0.10 + layer / 100),
                    }
                patches.append(A.PatchOutcome(
                    job_id=job_id,
                    parent_id=parent,
                    episode=episode,
                    overlap=overlap,
                    future_relation=relation,
                    probe_ids=ids,
                    effects=effects,
                ))
    return branches, patches


def test_complete_analysis_uses_parent_episode_did_and_emits_all_cells_layers_and_controls():
    branches, patches = _matrix()
    report = A.analyze_complete_matrix(branches, patches, analysis_seed=17, n_boot=100)

    assert report["status"] == "COMPLETE"
    assert len(report["branch_cells"]) == 64
    primary = report["endpoints"][A.PRIMARY_ENDPOINT]
    assert primary["parent_episode_mean"]["estimate"] == pytest.approx(0.245)
    assert set(primary["episode_robustness"]["separate_episode_summaries"]) == {"0", "1"}
    assert primary["episode_robustness"]["episode_difference"]["estimate"] == pytest.approx(0.02)
    assert set(report["activation_patching"]["layers"]) == {"3", "7", "10"}
    assert set(report["activation_patching"]["layers"]["7"]["controls"]) == set(A.PATCH_CONTROLS)
    comparison = report["activation_patching"]["layers"]["7"]["matching_parent_control_comparisons"]
    assert comparison["matching_parent_minus_unrelated_anchor"]["estimate"] == pytest.approx(0.2535)
    assert report["nulls"]["CFS2_GLOBAL_CONTROL"]["status"] == "TESTED"


def test_optional_diagnostics_are_emitted_as_null_not_silently_dropped():
    branches, patches = _matrix(optional=False)
    report = A.analyze_complete_matrix(branches, patches, analysis_seed=1, n_boot=100)

    assert report["endpoints"]["adaptation_acquisition"] == {
        "status": "NOT_AVAILABLE", "estimate": None, "null": None
    }
    assert all(
        cell["endpoints"]["global_control_margin_erosion"] is None
        for cell in report["branch_cells"]
    )
    assert report["nulls"]["CFS2_GLOBAL_CONTROL"]["status"] == "NOT_AVAILABLE"


def test_analysis_refuses_missing_branches_or_incomplete_fixed_patch_layers():
    branches, patches = _matrix()
    with pytest.raises(A.CFS2AnalysisError, match="64 branch outcomes"):
        A.analyze_complete_matrix(branches[:-1], patches, analysis_seed=1, n_boot=100)

    first = patches[0]
    patches[0] = A.PatchOutcome(
        job_id=first.job_id,
        parent_id=first.parent_id,
        episode=first.episode,
        overlap=first.overlap,
        future_relation=first.future_relation,
        probe_ids=first.probe_ids,
        effects={layer: value for layer, value in first.effects.items() if layer != 10},
    )
    with pytest.raises(A.CFS2AnalysisError, match="layers 3, 7, and 10"):
        A.analyze_complete_matrix(branches, patches, analysis_seed=1, n_boot=100)


def test_partial_optional_endpoint_is_refused_instead_of_analyzed_selectively():
    branches, patches = _matrix()
    first = branches[0]
    arrays = dict(first.arrays)
    arrays.pop("adaptation_acquisition")
    branches[0] = A.BranchOutcome(
        job_id=first.job_id,
        parent_id=first.parent_id,
        episode=first.episode,
        overlap=first.overlap,
        future_relation=first.future_relation,
        item_ids=first.item_ids,
        arrays=arrays,
        pregeometry=first.pregeometry,
    )
    with pytest.raises(A.CFS2AnalysisError, match="present in only part"):
        A.analyze_complete_matrix(branches, patches, analysis_seed=1, n_boot=100)

