import numpy as np

from vla_geometry.analysis import (
    binomial_log_loss,
    factor_main_effect_features,
    fit_binomial_logistic,
    leave_one_cell_out_log_loss,
    predict_binomial_logistic,
)


def test_factor_main_effect_features_symmetrically_codes_all_levels_without_outcomes():
    cells = [
        {"shape": shape, "direction": direction}
        for shape in ("a", "b")
        for direction in ("north", "south")
    ]
    features, names = factor_main_effect_features(cells, ("shape", "direction"))
    assert features.shape == (4, 4)
    assert names == [
        "shape=a",
        "shape=b",
        "direction=north",
        "direction=south",
    ]


def test_binomial_logistic_recovers_direction():
    x = np.linspace(-2, 2, 30)[:, None]
    trials = np.repeat(40, len(x))
    probability = 1 / (1 + np.exp(-(0.2 + 1.4 * x[:, 0])))
    successes = np.rint(probability * trials).astype(int)
    fit = fit_binomial_logistic(x, successes, trials)
    predicted = predict_binomial_logistic(x, fit)
    assert predicted[-1] > predicted[0]
    assert binomial_log_loss(predicted, successes, trials) < 0.6


def test_added_predictor_improves_leave_one_out_loss():
    rng = np.random.default_rng(3)
    signal = np.linspace(-2, 2, 35)
    nuisance = rng.normal(size=len(signal))
    trials = np.repeat(30, len(signal))
    probability = 1 / (1 + np.exp(-2.0 * signal))
    successes = np.rint(probability * trials).astype(int)
    baseline = leave_one_cell_out_log_loss(nuisance[:, None], successes, trials)
    full = leave_one_cell_out_log_loss(
        np.column_stack([nuisance, signal]), successes, trials
    )
    assert full < baseline


def test_ridge_predictions_are_invariant_to_control_column_order():
    cells = [
        {"shape": shape, "direction": direction}
        for shape in ("a", "b", "c")
        for direction in ("north", "south")
    ]
    features, _ = factor_main_effect_features(cells, ("shape", "direction"))
    successes = np.asarray([2, 7, 4, 9, 6, 11])
    trials = np.repeat(12, len(cells))
    fit = fit_binomial_logistic(features, successes, trials, ridge=1.0)
    order = np.asarray([4, 0, 2, 1, 3])
    permuted_fit = fit_binomial_logistic(
        features[:, order], successes, trials, ridge=1.0
    )
    original = predict_binomial_logistic(features, fit)
    permuted = predict_binomial_logistic(features[:, order], permuted_fit)
    np.testing.assert_allclose(original, permuted, atol=1e-12)
