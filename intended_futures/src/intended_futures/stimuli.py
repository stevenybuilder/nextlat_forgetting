from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def object_position(obs: dict[str, Any], subject: str, contract: dict[str, float]) -> np.ndarray:
    key = f"{subject}_pos"
    if key not in obs:
        raise RuntimeError(f"observation lacks intended object position: {key}")
    position = np.asarray(obs[key], dtype=np.float64).copy()
    valid = (
        position.shape == (3,)
        and np.all(np.isfinite(position))
        and np.max(np.abs(position[:2])) <= float(contract["max_absolute_xy"])
        and float(contract["z_min"]) < position[2] < float(contract["z_max"])
    )
    if not valid:
        raise RuntimeError(f"intended object {subject} is outside the frozen workspace: {position}")
    return position


def validate_subject_positions(
    obs: dict[str, Any], subjects: Sequence[str], contract: dict[str, float]
) -> dict[str, np.ndarray]:
    if len(subjects) != len(set(subjects)):
        raise ValueError("subjects must be unique")
    positions = {subject: object_position(obs, subject, contract) for subject in subjects}
    minimum = float(contract["minimum_subject_separation"])
    for index, subject_a in enumerate(subjects):
        for subject_b in subjects[index + 1 :]:
            separation = float(np.linalg.norm(positions[subject_b] - positions[subject_a]))
            if separation < minimum:
                raise RuntimeError(
                    f"intended objects {subject_a} and {subject_b} are insufficiently separated: {separation}"
                )
    return positions
