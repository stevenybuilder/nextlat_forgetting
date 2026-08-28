from __future__ import annotations

import pytest

from intended_futures.gates import evaluate_baseline_gate


def _records(successes: int, *, total: int = 20, invalid: int = 0):
    return [
        {
            "episode_id": f"episode-{index}",
            "success": index < successes,
            "valid": index >= invalid,
        }
        for index in range(total)
    ]


def test_baseline_gate_passes_at_frozen_boundary() -> None:
    records = _records(16)
    result = evaluate_baseline_gate(
        records,
        expected_episode_ids=[f"episode-{index}" for index in range(20)],
        minimum_successes=16,
    )
    assert result["passed"] is True
    assert result["success_rate"] == pytest.approx(0.8)


@pytest.mark.parametrize("successes,invalid", [(15, 0), (20, 1)])
def test_baseline_gate_fails_low_success_or_invalid(successes: int, invalid: int) -> None:
    records = _records(successes, invalid=invalid)
    result = evaluate_baseline_gate(
        records,
        expected_episode_ids=[f"episode-{index}" for index in range(20)],
        minimum_successes=16,
    )
    assert result["passed"] is False


def test_baseline_gate_rejects_duplicate_observations() -> None:
    records = _records(16)
    records[-1]["episode_id"] = records[0]["episode_id"]
    with pytest.raises(ValueError, match="unique"):
        evaluate_baseline_gate(
            records,
            expected_episode_ids=[f"episode-{index}" for index in range(20)],
            minimum_successes=16,
        )
