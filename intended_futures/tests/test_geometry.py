import numpy as np
import pytest

from intended_futures.geometry import (
    FutureSubspace,
    ZeroInterceptTargetDecoder,
    construct_target_controller_delta,
    leave_one_group_out_predictions,
    r2_score,
    select_zero_intercept_ridge,
)


def test_future_subspace_recovers_low_rank_token_preserving_signal():
    rng = np.random.default_rng(9)
    n, tokens, width, target_dim = 80, 4, 12, 3
    x = rng.normal(size=(n, tokens, width))
    direction = rng.normal(size=(tokens * width, target_dim))
    y = x.reshape(n, -1) @ direction
    model = FutureSubspace.fit(x, y, rank=3, ridge=1e-6)
    assert model.input_shape == (tokens, width)
    assert r2_score(y, model.predict(x)) > 0.99


def test_projection_changes_only_fitted_subspace():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 2, 5))
    y = x.reshape(30, -1)[:, :2]
    model = FutureSubspace.fit(x, y, rank=2, ridge=1e-6)
    donor, recipient = x[0], x[1]
    patch = model.patch(donor, recipient) - recipient
    flat = patch.reshape(-1)
    residual = flat - model.basis @ (model.basis.T @ flat)
    assert np.linalg.norm(residual) < 1e-10


def test_leave_one_group_out_never_fits_held_out_group():
    rng = np.random.default_rng(4)
    groups = np.repeat(["a", "b", "c", "d"], 8)
    x = rng.normal(size=(32, 3, 4))
    y = x.reshape(32, -1)[:, :2]
    prediction = leave_one_group_out_predictions(x, y, groups, rank=2, ridge=1e-5)
    assert prediction.shape == y.shape
    assert np.all(np.isfinite(prediction))


def test_constant_target_r2_is_rejected():
    with pytest.raises(ValueError, match="constant"):
        r2_score(np.ones((4, 2)), np.ones((4, 2)))


def test_zero_intercept_decoder_predicts_and_inverts_target_change():
    rng = np.random.default_rng(25)
    activations = rng.normal(size=(18, 2, 4))
    true_beta = rng.normal(size=(8, 3))
    targets = activations.reshape(18, -1) @ true_beta
    decoder = ZeroInterceptTargetDecoder.fit(
        activations, targets, ridge_fraction=1e-8
    )
    assert r2_score(targets, decoder.predict(activations)) > 0.999

    desired = np.asarray([0.3, -0.2, 0.1])
    delta = decoder.minimum_norm_delta(
        desired, inverse_ridge_fraction=1e-8
    )
    recovered = delta.reshape(1, -1) @ decoder.beta
    assert np.allclose(recovered[0], desired, atol=1e-5)


def test_minimum_norm_decoder_respects_full_difference_norm_cap():
    rng = np.random.default_rng(26)
    activations = rng.normal(size=(12, 3, 2))
    targets = rng.normal(size=(12, 3))
    decoder = ZeroInterceptTargetDecoder.fit(
        activations, targets, ridge_fraction=0.01
    )
    delta = decoder.minimum_norm_delta(
        np.asarray([100.0, -50.0, 25.0]),
        inverse_ridge_fraction=1e-6,
        maximum_norm=0.5,
    )
    assert np.linalg.norm(delta) == pytest.approx(0.5)


def test_zero_intercept_ridge_selection_is_group_held_out():
    rng = np.random.default_rng(27)
    activations = rng.normal(size=(24, 2, 3))
    beta = rng.normal(size=(6, 3))
    targets = activations.reshape(24, -1) @ beta
    groups = [f"scene-{index // 4}" for index in range(24)]
    result = select_zero_intercept_ridge(
        activations,
        targets,
        groups,
        ridge_fractions=[1e-6, 1e-2, 1.0],
    )
    assert result["selected"]["ridge_fraction"] == 1e-6
    assert result["selected"]["r2"] > 0.99


def test_target_controller_conditions_share_frozen_norm_contract():
    rng = np.random.default_rng(28)
    beta = rng.normal(size=(20, 3))
    difference = rng.normal(size=(4, 5))
    desired = np.asarray([0.2, -0.1, 0.3])
    minimum = construct_target_controller_delta(
        beta,
        difference,
        desired,
        kind="minimum_norm_target",
        inverse_ridge_fraction=1e-6,
        maximum_norm_fraction=1.0,
    )
    random = construct_target_controller_delta(
        beta,
        difference,
        desired,
        kind="random_controller",
        inverse_ridge_fraction=1e-6,
        maximum_norm_fraction=1.0,
        random_seed=91,
    )
    projection = construct_target_controller_delta(
        beta,
        difference,
        desired,
        kind="target_projection",
        inverse_ridge_fraction=1e-6,
        maximum_norm_fraction=1.0,
    )
    assert np.linalg.norm(minimum) == pytest.approx(np.linalg.norm(random))
    assert np.linalg.norm(minimum) <= np.linalg.norm(difference) + 1e-12
    basis, _ = np.linalg.qr(beta)
    residual = projection.reshape(-1) - basis @ (basis.T @ projection.reshape(-1))
    assert np.linalg.norm(residual) < 1e-10
