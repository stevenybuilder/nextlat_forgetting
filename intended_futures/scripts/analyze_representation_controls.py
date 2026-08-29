#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _cosine_rows(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    result = np.full(len(target), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = np.sum(target[valid] * prediction[valid], axis=1) / denominator[valid]
    return result


def _group_mean_prediction(target: np.ndarray, scenes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    prediction = np.empty_like(target)
    for scene in np.unique(scenes):
        test = scenes == scene
        train = ~test
        label = labels[np.flatnonzero(test)[0]]
        matched = train & (labels == label)
        if not np.any(matched):
            matched = train
        prediction[test] = np.mean(target[matched], axis=0)
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.config import load_config
    from intended_futures.geometry import leave_one_group_out_predictions, r2_score

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    representation = json.loads(args.representation_summary.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    scenes = np.asarray([row["scene_id"] for row in rows])
    subject_pairs = np.asarray(
        [f"{row['task_a']['intended_subject']}->{row['task_b']['intended_subject']}" for row in rows]
    )
    prompt_pairs = np.asarray(
        [f"{row['task_a']['prompt_id']}->{row['task_b']['prompt_id']}" for row in rows]
    )
    target, actions, activations = [], [], []
    selected_layer = int(representation["selected_layer"])
    for row in rows:
        with np.load(args.input / f"{row['stimulus_id']}.npz", allow_pickle=False) as record:
            target.append(record["target_b_position"] - record["target_a_position"])
            actions.append(record["actions_b"] - record["actions_a"])
            activations.append(
                record[f"activation_b_layer_{selected_layer}"]
                - record[f"activation_a_layer_{selected_layer}"]
            )
    target_array = np.asarray(target, dtype=np.float64)
    controls = {}
    for name, labels in (
        ("global_intercept", np.repeat("all", len(rows))),
        ("ordered_subject_pair", subject_pairs),
        ("exact_prompt_pair", prompt_pairs),
    ):
        prediction = _group_mean_prediction(target_array, scenes, labels)
        controls[name] = {
            "crossvalidated_r2": r2_score(target_array, prediction),
            "mean_cosine": float(np.nanmean(_cosine_rows(target_array, prediction))),
        }
    action_prediction = leave_one_group_out_predictions(
        np.asarray(actions, dtype=np.float64),
        target_array,
        scenes,
        rank=int(config["analysis"]["subspace_rank"]),
        ridge=float(config["analysis"]["ridge"]),
    )
    activation_prediction = leave_one_group_out_predictions(
        np.asarray(activations, dtype=np.float64),
        target_array,
        scenes,
        rank=int(config["analysis"]["subspace_rank"]),
        ridge=float(config["analysis"]["ridge"]),
    )
    prompt_prediction = _group_mean_prediction(target_array, scenes, prompt_pairs)
    controls["action_chunk"] = {
        "crossvalidated_r2": r2_score(target_array, action_prediction),
        "mean_cosine": float(np.nanmean(_cosine_rows(target_array, action_prediction))),
    }
    activation_sse = float(np.sum((target_array - activation_prediction) ** 2))
    prompt_sse = float(np.sum((target_array - prompt_prediction) ** 2))
    summary = {
        "study": config["study"],
        "status": "post-hoc controls required to interpret the preregistered gate",
        "selected_layer": selected_layer,
        "activation": {
            "crossvalidated_r2": r2_score(target_array, activation_prediction),
            "mean_cosine": float(np.nanmean(_cosine_rows(target_array, activation_prediction))),
            "residual_sse_reduction_vs_exact_prompt_pair": 1.0 - activation_sse / prompt_sse,
        },
        "controls": controls,
        "target_mean_norm": float(np.mean(np.linalg.norm(target_array, axis=1))),
        "mean_within_scene_rms": float(
            np.mean(
                [
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (target_array[scenes == scene] - np.mean(target_array[scenes == scene], axis=0))
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                    for scene in np.unique(scenes)
                ]
            )
        ),
        "interpretation": "activation geometry adds scene-conditioned information beyond strong prompt-pair structure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
