from __future__ import annotations

import numpy as np
import pytest

from intended_futures.stimuli import object_position, validate_subject_positions


CONTRACT = {
    "max_absolute_xy": 2.0,
    "z_min": 0.0,
    "z_max": 2.0,
    "minimum_subject_separation": 0.05,
}


def test_workspace_contract_rejects_hidden_libero_object() -> None:
    with pytest.raises(RuntimeError, match="outside"):
        object_position({"bowl_pos": np.array([0.0, -10.0, 0.1])}, "bowl", CONTRACT)


def test_all_subjects_must_be_present_and_separated() -> None:
    obs = {"a_pos": np.array([0.0, 0.0, 0.9]), "b_pos": np.array([0.2, 0.0, 0.9])}
    result = validate_subject_positions(obs, ["a", "b"], CONTRACT)
    assert set(result) == {"a", "b"}
    with pytest.raises(RuntimeError, match="insufficiently separated"):
        validate_subject_positions(
            {"a_pos": np.array([0.0, 0.0, 0.9]), "b_pos": np.array([0.01, 0.0, 0.9])},
            ["a", "b"],
            CONTRACT,
        )
