from __future__ import annotations

import numpy as np
import pytest

from lurestar import evaluate as E
from lurestar import gradient_controls as G


def test_gradient_baselines_are_exact_dot_and_cosine_and_bind_losses():
    item = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    adapt = np.array([1.0, 0.0])
    out = E.gradient_alignment_baselines(
        item,
        adapt,
        item_loss=E.EXACT_ITEM_LOSS,
        adaptation_loss=E.EXACT_ADAPTATION_LOSS,
    )
    assert out["gradient_dot"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert out["gradient_cosine"].tolist() == pytest.approx([1.0, 2 ** -0.5, 0.0])
    with pytest.raises(ValueError, match="frozen exact"):
        E.gradient_alignment_baselines(
            item, adapt, item_loss="surrogate", adaptation_loss=E.EXACT_ADAPTATION_LOSS
        )


def test_gradient_and_jacobian_baselines_refuse_mismatched_parameter_axes():
    with pytest.raises(ValueError, match="parameter vectors"):
        E.gradient_alignment_baselines(
            np.ones((4, 3)), np.ones(4),
            item_loss=E.EXACT_ITEM_LOSS, adaptation_loss=E.EXACT_ADAPTATION_LOSS,
        )
    out = E.jacobian_ntk_overlap(
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]]), np.array([[1.0, 0.0]])
    )
    assert out["ntk_frobenius_dot"].tolist() == [1.0, 0.0]
    assert out["ntk_frobenius_cosine"].tolist() == [1.0, 0.0]


def test_grouped_folds_are_deterministic_and_keep_repeated_groups_together():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    a = E.grouped_folds(groups, n_folds=2, seed=17)
    b = E.grouped_folds(groups, n_folds=2, seed=17)
    assert np.array_equal(a, b)
    for group in set(groups):
        assert len(set(a[np.asarray(groups) == group])) == 1


def _h3_fixture(n=240, seed=2):
    rng = np.random.default_rng(seed)
    distance = rng.normal(size=n)
    gdot = rng.normal(size=n)
    gcos = rng.normal(size=n)
    margin = rng.normal(size=n)
    lure = rng.normal(size=n)
    acquisition = rng.normal(size=n)
    erosion = (
        1.8 * distance + 0.4 * gdot - 0.2 * gcos + 0.3 * margin
        + 0.1 * lure - 0.1 * acquisition + rng.normal(scale=0.15, size=n)
    )
    folds = E.grouped_folds(np.arange(n), n_folds=2, seed=73021)
    return erosion, distance, gdot, gcos, margin, lure, acquisition, folds


def test_incremental_h3_reuses_folds_and_finds_only_heldout_improvement():
    values = _h3_fixture()
    out = E.fit_h3_incremental(*values[:-1], folds=values[-1])
    assert out["folds_reused_exactly"] is True
    assert out["delta_r2_heldout"] > 0.5
    assert out["delta_mse_heldout"] < 0.0
    assert len(out["fold_assignment_sha256"]) == 64
    assert "frozen_hidden_distance" in out["augmented"]["model"]


def test_shared_parent_did_checks_real_lineage_and_item_order():
    before = np.array([3.0, 4.0, 5.0])
    near = before - np.array([0.4, 0.5, 0.3])
    far = before - np.array([0.1, 0.1, 0.2])
    sha = "a" * 64
    out = E.similarity_dependent_interference(
        before, near, far,
        item_ids_near=[1, 2, 3], item_ids_far=[1, 2, 3],
        parent_sha256_near=sha, parent_sha256_far=sha,
        rng=np.random.default_rng(9), n_boot=50,
    )
    assert out["shared_parent_verified"] and out["paired_item_order_verified"]
    with pytest.raises(ValueError, match="identical A_pair ids"):
        E.similarity_dependent_interference(
            before, near, far, item_ids_near=[1, 2, 3], item_ids_far=[1, 3, 2],
            parent_sha256_near=sha, parent_sha256_far=sha,
            rng=np.random.default_rng(9), n_boot=50,
        )
    with pytest.raises(ValueError, match="same parent"):
        E.similarity_dependent_interference(
            before, near, far, item_ids_near=[1, 2, 3], item_ids_far=[1, 2, 3],
            parent_sha256_near=sha, parent_sha256_far="b" * 64,
            rng=np.random.default_rng(9), n_boot=50,
        )


def test_fixed_nonlinear_diagnostic_reports_every_quintile_and_refuses_rebinning():
    erosion, d, gdot, gcos, margin, lure, acquisition, folds = _h3_fixture()
    controls = np.column_stack([gdot, gcos, margin, lure, acquisition])
    names = ["gradient_dot", "gradient_cosine", "margin", "lure", "acquisition"]
    out = E.h3_distance_diagnostic(
        erosion + 0.8 * d * d, d, controls, folds=folds, control_names=names
    )
    assert len(out["distance_quantiles"]) == 5
    assert sum(cell["n"] for cell in out["distance_quantiles"]) == len(d)
    assert out["delta_r2_quadratic_over_linear"] > 0.1
    with pytest.raises(ValueError, match="frozen"):
        E.h3_distance_diagnostic(
            erosion, d, controls, folds=folds, control_names=names,
            quantiles=(0.0, 0.5, 1.0),
        )


def test_model_distance_interaction_uses_frozen_bst_reference_and_grouped_folds():
    rng = np.random.default_rng(4)
    n = 90
    labels = np.repeat(np.array(E.ARMS), n)
    d0 = rng.normal(size=n)
    distance = np.tile(d0, 3)
    erosion = distance.copy()
    erosion[labels == "nextlat"] += 1.5 * distance[labels == "nextlat"]
    erosion[labels == "gpt"] -= 0.5 * distance[labels == "gpt"]
    erosion += rng.normal(scale=0.05, size=erosion.size)
    group = np.tile(np.arange(n), 3)
    folds = E.grouped_folds(group, n_folds=2, seed=7)
    controls = rng.normal(size=(erosion.size, 1))
    out = E.fit_h3_model_distance_interaction(
        erosion, distance, labels, controls, folds=folds, control_names=["control"]
    )
    assert out["reference_model"] == "bst"
    assert out["interaction_terms"] == [
        "distance:model[nextlat]", "distance:model[gpt]"
    ]
    assert out["report"]["r2_heldout"] > 0.9


def test_torch_gradient_adapter_binds_loss_identity_before_touching_runtime():
    with pytest.raises(ValueError, match="frozen confirmatory"):
        G.exact_loss_gradient_controls(
            [], None, [], item_loss_name="proxy", adaptation_loss_name=E.EXACT_ADAPTATION_LOSS
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="require torch"):
            G.exact_scalar_jacobian_controls([object()], object(), [object()])
