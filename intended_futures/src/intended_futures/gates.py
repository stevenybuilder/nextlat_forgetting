from __future__ import annotations

from typing import Any, Sequence


def evaluate_baseline_gate(
    records: Sequence[dict[str, Any]],
    *,
    expected_episode_ids: Sequence[str],
    minimum_successes: int,
    invalid_episodes_allowed: int = 0,
) -> dict[str, Any]:
    """Evaluate a fixed-population behavioral gate without retries or replacements."""

    expected = list(expected_episode_ids)
    observed = [str(record["episode_id"]) for record in records]
    if len(expected) != len(set(expected)):
        raise ValueError("expected episode ids must be unique")
    if len(observed) != len(set(observed)):
        raise ValueError("observed episode ids must be unique")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    invalid = sum(not bool(record.get("valid", False)) for record in records)
    successes = sum(bool(record.get("success", False)) for record in records)
    complete = not missing and not unexpected and len(records) == len(expected)
    passed = complete and successes >= minimum_successes and invalid <= invalid_episodes_allowed
    return {
        "passed": passed,
        "complete": complete,
        "episodes_expected": len(expected),
        "episodes_observed": len(records),
        "successes": successes,
        "success_rate": successes / len(expected) if expected else 0.0,
        "invalid_episodes": invalid,
        "minimum_successes": int(minimum_successes),
        "invalid_episodes_allowed": int(invalid_episodes_allowed),
        "missing_episode_ids": missing,
        "unexpected_episode_ids": unexpected,
    }
