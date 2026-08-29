#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _cosine(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 1e-12:
        raise ValueError("action cosine is undefined for a zero-norm chunk")
    return float(np.dot(left_array, right_array) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.statistics import (
        clustered_bootstrap_mean,
        exact_group_sign_flip_pvalue,
    )

    protocol = json.loads(args.m0_config.read_text(encoding="utf-8"))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input.glob("scene-*-state-*.json"))
    ]
    if len(records) != int(protocol["population"]["expected_units"]):
        raise RuntimeError("post-hoc analysis requires the complete frozen M0 population")
    if any(
        record.get("study") != protocol["study"] or not record.get("valid", False)
        for record in records
    ):
        raise RuntimeError("post-hoc analysis requires valid records from the frozen M0 study")

    conditions = [
        "donor_clean",
        "recipient_clean",
        "paligemma_l13_full",
        "paligemma_l13_random",
    ]
    condition_summary: dict[str, Any] = {}
    for condition_id in conditions:
        first_touches = Counter(
            record["conditions"][condition_id]["first_touch"] or "none"
            for record in records
        )
        condition_summary[condition_id] = {
            "first_touch_counts": dict(sorted(first_touches.items())),
            "donor_first_touch_rate": first_touches["donor"] / len(records),
            "mean_donor_progress_meters": float(
                np.mean(
                    [
                        record["conditions"][condition_id]["donor_progress"]
                        for record in records
                    ]
                )
            ),
            "mean_recipient_progress_meters": float(
                np.mean(
                    [
                        record["conditions"][condition_id]["recipient_progress"]
                        for record in records
                    ]
                )
            ),
        }

    groups = [str(record["scene_id"]) for record in records]
    touch_contrasts: dict[str, Any] = {}
    for label, left, right, seed in (
        (
            "paligemma_full_minus_recipient_clean",
            "paligemma_l13_full",
            "recipient_clean",
            91173,
        ),
        (
            "paligemma_full_minus_random",
            "paligemma_l13_full",
            "paligemma_l13_random",
            91174,
        ),
        (
            "paligemma_full_minus_donor_clean",
            "paligemma_l13_full",
            "donor_clean",
            91175,
        ),
    ):
        values = [
            float(record["conditions"][left]["first_touch"] == "donor")
            - float(record["conditions"][right]["first_touch"] == "donor")
            for record in records
        ]
        interval = clustered_bootstrap_mean(
            values, groups, repetitions=10000, seed=seed
        )
        touch_contrasts[label] = {
            "difference": interval["estimate"],
            "scene_cluster_bootstrap_95ci": [
                interval["lower"],
                interval["upper"],
            ],
            "exact_scene_sign_flip_p_unadjusted": exact_group_sign_flip_pvalue(
                values, groups
            ),
        }

    donor_actions = [
        record["conditions"]["donor_clean"]["first_action_chunk"]
        for record in records
    ]
    recipient_actions = [
        record["conditions"]["recipient_clean"]["first_action_chunk"]
        for record in records
    ]
    full_actions = [
        record["conditions"]["paligemma_l13_full"]["first_action_chunk"]
        for record in records
    ]
    random_actions = [
        record["conditions"]["paligemma_l13_random"]["first_action_chunk"]
        for record in records
    ]
    action_similarity = {
        "clean_donor_to_recipient": float(
            np.mean(
                [
                    _cosine(donor, recipient)
                    for donor, recipient in zip(donor_actions, recipient_actions)
                ]
            )
        ),
        "full_to_donor": float(
            np.mean(
                [_cosine(full, donor) for full, donor in zip(full_actions, donor_actions)]
            )
        ),
        "full_to_recipient": float(
            np.mean(
                [
                    _cosine(full, recipient)
                    for full, recipient in zip(full_actions, recipient_actions)
                ]
            )
        ),
        "random_to_donor": float(
            np.mean(
                [
                    _cosine(random, donor)
                    for random, donor in zip(random_actions, donor_actions)
                ]
            )
        ),
        "random_to_recipient": float(
            np.mean(
                [
                    _cosine(random, recipient)
                    for random, recipient in zip(random_actions, recipient_actions)
                ]
            )
        ),
    }

    output = {
        "study": protocol["study"],
        "status": "posthoc_exploratory",
        "units": len(records),
        "condition_summary": condition_summary,
        "touch_contrasts": touch_contrasts,
        "action_similarity": action_similarity,
        "claim_boundary": (
            "These secondary estimates explain the frozen M0 failure and cannot "
            "override its failed 0.10 action-margin gate or authorize TC1."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
