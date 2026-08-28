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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subspace-output", type=Path, required=True)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.geometry import FutureSubspace, leave_one_group_out_predictions, r2_score

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    group_field = str(config["analysis"].get("cross_validation_group_field", "pair_id"))
    groups = np.asarray([row[group_field] for row in rows])
    future = []
    actions = []
    layer_differences: dict[int, list[np.ndarray]] = {
        int(layer): [] for layer in config["representation"]["candidate_layers"]
    }
    for row in rows:
        source = args.input / f"{row['stimulus_id']}.npz"
        if not source.exists():
            raise FileNotFoundError(f"missing frozen stimulus output: {source}")
        with np.load(source, allow_pickle=False) as record:
            metadata = json.loads(str(record["metadata_json"].item()))
            if metadata["stimulus_id"] != row["stimulus_id"]:
                raise RuntimeError(f"record identity mismatch: {source}")
            future.append(record["target_b_position"] - record["target_a_position"])
            actions.append(record["actions_b"] - record["actions_a"])
            for layer in layer_differences:
                layer_differences[layer].append(
                    record[f"activation_b_layer_{layer}"] - record[f"activation_a_layer_{layer}"]
                )

    target = np.asarray(future, dtype=np.float64)
    action_difference = np.asarray(actions, dtype=np.float64)
    rank = int(config["analysis"]["subspace_rank"])
    ridge = float(config["analysis"]["ridge"])
    layer_results = []
    fitted: dict[int, FutureSubspace] = {}
    for layer, differences in layer_differences.items():
        activation = np.asarray(differences, dtype=np.float64)
        prediction = leave_one_group_out_predictions(
            activation, target, groups, rank=rank, ridge=ridge
        )
        cosines = _cosine_rows(target, prediction)
        per_pair = {
            pair: {
                "mean_cosine": float(np.nanmean(cosines[groups == pair])),
                "positive": bool(np.nanmean(cosines[groups == pair]) > 0),
            }
            for pair in np.unique(groups)
        }
        score = r2_score(target, prediction)
        layer_results.append(
            {
                "layer": layer,
                "crossvalidated_r2": score,
                "mean_cosine": float(np.nanmean(cosines)),
                "positive_task_pairs": sum(value["positive"] for value in per_pair.values()),
                "per_pair": per_pair,
                "activation_shape": list(activation.shape[1:]),
            }
        )
        fitted[layer] = FutureSubspace.fit(activation, target, rank=rank, ridge=ridge)

    best = max(layer_results, key=lambda result: result["crossvalidated_r2"])
    gates = config["analysis"]["pilot_gates"]
    advance = (
        best["crossvalidated_r2"] >= gates["future_delta_crossvalidated_r2_min"]
        and best["positive_task_pairs"] >= gates["positive_task_pairs_min"]
    )
    selected = fitted[int(best["layer"])]
    args.subspace_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.subspace_output,
        input_shape=np.asarray(selected.input_shape, dtype=np.int64),
        mean_activation=selected.mean_activation,
        mean_target=selected.mean_target,
        beta=selected.beta,
        basis=selected.basis,
        selected_layer=np.asarray(best["layer"], dtype=np.int64),
    )
    summary = {
        "study": config["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "n_matched_states": len(rows),
        "n_task_pair_clusters": len(np.unique(groups)),
        "cross_validation_group_field": group_field,
        "future_target_shape": list(target.shape[1:]),
        "action_difference_mean_norm": float(
            np.mean(np.linalg.norm(action_difference.reshape(len(rows), -1), axis=1))
        ),
        "layers": layer_results,
        "selected_layer": int(best["layer"]),
        "primary": best,
        "advance_to_causal_stage": advance,
        "claim_boundary": config["analysis"]["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["primary"], sort_keys=True))
    print(json.dumps({"advance_to_causal_stage": advance}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
