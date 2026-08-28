import numpy as np
import pytest

from intended_futures.geometry import FutureSubspace, leave_one_group_out_predictions, r2_score


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
