import numpy as np

from intended_futures.statistics import clustered_bootstrap_mean, exact_group_sign_flip_pvalue, target_progress


def test_target_progress_is_positive_when_endpoint_moves_toward_target():
    assert target_progress(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.5


def test_cluster_bootstrap_samples_groups_not_frames():
    result = clustered_bootstrap_mean(
        [1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"], repetitions=1000, seed=7
    )
    assert result["estimate"] == 2.0
    assert result["lower"] <= 2.0 <= result["upper"]


def test_five_group_exact_two_sided_test_cannot_reach_below_one_sixteenth():
    pvalue = exact_group_sign_flip_pvalue([1, 1, 1, 1, 1], ["a", "b", "c", "d", "e"])
    assert pvalue == 2 / 32
