from __future__ import annotations

from typing import Sequence

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.ndim != 1 or len(x_array) < 3:
        raise ValueError("x and y must be equal-length one-dimensional arrays of length >= 3")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must be finite")
    x_rank = _average_ranks(x_array)
    y_rank = _average_ranks(y_array)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    return float("nan") if denominator == 0 else float(x_rank @ y_rank / denominator)


def bootstrap_spearman_interval(
    x: Sequence[float],
    y: Sequence[float],
    *,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not np.isfinite(spearman_correlation(x_array, y_array)):
        raise ValueError("Spearman correlation is undefined for constant input")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        indices = rng.integers(0, len(x_array), size=len(x_array))
        estimate = spearman_correlation(x_array[indices], y_array[indices])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < 0.95 * replicates:
        raise ValueError("too many degenerate bootstrap samples")
    return tuple(
        float(value)
        for value in np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    )


def permutation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> float:
    """Two-sided permutation p-value with the observed assignment included."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if replicates < 100:
        raise ValueError("at least 100 permutation replicates are required")
    observed = abs(spearman_correlation(x_array, y_array))
    if not np.isfinite(observed):
        raise ValueError("Spearman correlation is undefined for constant input")
    rng = np.random.default_rng(seed)
    exceedances = 1
    for _ in range(replicates):
        permuted = rng.permutation(y_array)
        exceedances += abs(spearman_correlation(x_array, permuted)) >= observed
    return float(exceedances / (replicates + 1))
