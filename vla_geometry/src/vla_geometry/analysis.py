from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def factor_main_effect_features(
    cells: Sequence[Mapping[str, str]], factors: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Symmetrically one-hot code cell factors for a ridge difficulty baseline.

    All levels are retained. Ridge regularization makes the redundant intercept-plus-indicators
    system identifiable while avoiding dependence on an arbitrary reference level.
    """

    columns = []
    names = []
    for factor in factors:
        levels = sorted({str(cell[factor]) for cell in cells})
        if len(levels) < 2:
            raise ValueError(f"factor {factor!r} needs at least two levels")
        for level in levels:
            columns.append(
                np.asarray([str(cell[factor]) == level for cell in cells], dtype=float)
            )
            names.append(f"{factor}={level}")
    return np.column_stack(columns), names


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def fit_binomial_logistic(
    features: np.ndarray,
    successes: Sequence[int],
    trials: Sequence[int],
    *,
    ridge: float = 1e-6,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small cell-level binomial logistic model with standardized predictors."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or n.shape != (len(x),):
        raise ValueError("invalid feature/count shapes")
    if np.any(y < 0) or np.any(n <= 0) or np.any(y > n):
        raise ValueError("invalid binomial counts")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("constant control or predictor")
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    converged = False
    for _ in range(max_iter):
        probability = np.clip(_sigmoid(design @ beta), 1e-8, 1 - 1e-8)
        gradient = design.T @ (y - n * probability) - penalty @ beta
        weights = n * probability * (1 - probability)
        hessian_positive = design.T @ (weights[:, None] * design) + penalty
        step = np.linalg.solve(hessian_positive, gradient)
        beta += step
        if float(np.linalg.norm(step)) < 1e-8:
            converged = True
            break
    if not converged:
        raise RuntimeError("binomial logistic IRLS did not converge")
    return beta, mean, scale


def predict_binomial_logistic(
    features: np.ndarray,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    beta, mean, scale = fit
    x = np.asarray(features, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return np.clip(_sigmoid(design @ beta), 1e-8, 1 - 1e-8)


def binomial_log_loss(
    probability: Sequence[float], successes: Sequence[int], trials: Sequence[int]
) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1 - 1e-8)
    y = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    return float(-np.sum(y * np.log(p) + (n - y) * np.log1p(-p)) / np.sum(n))


def leave_one_cell_out_log_loss(
    features: np.ndarray,
    successes: Sequence[int],
    trials: Sequence[int],
    *,
    ridge: float = 1.0,
) -> float:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(successes, dtype=np.int64)
    n = np.asarray(trials, dtype=np.int64)
    predictions = np.full(len(x), np.nan, dtype=np.float64)
    for held_out in range(len(x)):
        train = np.arange(len(x)) != held_out
        fit = fit_binomial_logistic(x[train], y[train], n[train], ridge=ridge)
        predictions[held_out] = predict_binomial_logistic(x[[held_out]], fit)[0]
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("non-finite leave-one-cell-out predictions")
    return binomial_log_loss(predictions, y, n)
