from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def target_progress(start: np.ndarray, end: np.ndarray, target: np.ndarray) -> float:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if start_array.shape != (3,) or end_array.shape != (3,) or target_array.shape != (3,):
        raise ValueError("start, end, and target must be three-dimensional positions")
    if not all(np.all(np.isfinite(value)) for value in (start_array, end_array, target_array)):
        raise ValueError("positions must be finite")
    return float(np.linalg.norm(start_array - target_array) - np.linalg.norm(end_array - target_array))


def clustered_bootstrap_mean(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    repetitions: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    if array.ndim != 1 or len(array) != len(group_array) or not np.all(np.isfinite(array)):
        raise ValueError("values and groups must be finite one-dimensional arrays of equal length")
    unique = np.unique(group_array)
    if len(unique) < 2 or repetitions < 100:
        raise ValueError("cluster bootstrap requires at least two groups and 100 repetitions")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        resampled = np.concatenate([array[group_array == group] for group in sampled])
        estimates[index] = np.mean(resampled)
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
    }


def exact_group_sign_flip_pvalue(values: Sequence[float], groups: Sequence[str]) -> float:
    array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups)
    if array.ndim != 1 or len(array) != len(group_array) or not np.all(np.isfinite(array)):
        raise ValueError("values and groups must be finite one-dimensional arrays of equal length")
    unique = np.unique(group_array)
    group_means = np.asarray([np.mean(array[group_array == group]) for group in unique])
    observed = abs(float(np.mean(group_means)))
    null = [abs(float(np.mean(group_means * np.asarray(signs)))) for signs in product((-1.0, 1.0), repeat=len(unique))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))
