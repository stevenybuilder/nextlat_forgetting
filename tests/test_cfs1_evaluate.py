from __future__ import annotations

import numpy as np
import pytest

from cfs1 import evaluate as E


def _parents() -> list[str]:
    return [f"nextlat-parent-{index}" for index in range(8)]


def test_primary_did_averages_fixed_episodes_inside_parent_only():
    episode_values = {
        0: {"high_different": 1.0, "high_same": .3, "low_different": .4, "low_same": .2},
        1: {"high_different": .8, "high_same": .1, "low_different": .2, "low_same": .1},
    }
    mean, per_episode = E.parent_episode_mean_did(episode_values)
    assert per_episode == pytest.approx({0: .5, 1: .6})
    assert mean == pytest.approx(.55)


def test_parent_summary_uses_eight_parents_with_exact_sign_flip_mde_ci_and_loso():
    values = {parent: .2 + index / 100 for index, parent in enumerate(_parents())}
    report = E.paired_parent_summary(values)
    assert report["inferential_unit"] == "independently trained parent checkpoint"
    assert report["ci"]["n_units"] == 8
    assert report["minimum_detectable_effect"]["sign_flip_p_floor"] == pytest.approx(1 / 128)
    assert report["minimum_detectable_effect"]["randomization_test_can_reject"] is True
    assert report["exact_two_sided_sign_flip_p"] == pytest.approx(1 / 128)
    assert len(report["leave_one_parent_out"]) == 8


def test_items_are_conditional_and_cannot_replace_parent_analysis():
    rng = np.random.default_rng(3)
    interval = E.conditional_item_bootstrap(np.arange(20.0), rng=rng, n_boot=100)
    assert interval["n_items"] == 20
    assert "cannot replace eight-parent inference" in interval["scope"]
    with pytest.raises(E.CFS1EvaluationError, match="exactly 8"):
        E.paired_parent_summary({"one": 1.0, "two": 2.0})


def test_holm_and_complete_identity_lattice_are_strict():
    adjusted = E.holm_adjust({"first": .01, "second": .02, "third": .8})
    assert adjusted == {"first": .03, "second": .04, "third": .8}
    assert len(E.expected_branch_keys(_parents())) == 64
    with pytest.raises(E.CFS1EvaluationError, match="exactly 8"):
        E.expected_branch_keys(_parents()[:-1])
    with pytest.raises(E.CFS1EvaluationError, match="exactly high_different"):
        E.difference_in_differences({"high_different": 1.0})
