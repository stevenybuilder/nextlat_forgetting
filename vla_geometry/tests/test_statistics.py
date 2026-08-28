import numpy as np
import pytest

from vla_geometry.statistics import (
    bootstrap_spearman_interval,
    permutation_pvalue,
    spearman_correlation,
)


def test_spearman_handles_ties_and_direction():
    x = np.asarray([1, 1, 2, 3, 4, 5], dtype=float)
    y = np.asarray([8, 8, 7, 4, 2, 1], dtype=float)
    assert spearman_correlation(x, y) < -0.95


def test_bootstrap_and_permutation_detect_strong_relation():
    x = np.arange(30, dtype=float)
    y = -x + np.sin(x) * 0.01
    low, high = bootstrap_spearman_interval(x, y, replicates=300, seed=4)
    assert high < -0.95
    assert low <= high
    assert permutation_pvalue(x, y, replicates=300, seed=4) < 0.02


def test_resampling_rejects_a_constant_endpoint_instead_of_false_significance():
    x = np.arange(20, dtype=float)
    y = np.ones(20, dtype=float)
    with pytest.raises(ValueError, match="constant"):
        bootstrap_spearman_interval(x, y, replicates=100, seed=2)
    with pytest.raises(ValueError, match="constant"):
        permutation_pvalue(x, y, replicates=100, seed=2)
