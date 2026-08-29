#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _standardized_scene_effect(values: np.ndarray, scenes: np.ndarray) -> tuple[float, np.ndarray]:
    scene_means = np.asarray([np.mean(values[scenes == scene]) for scene in np.unique(scenes)])
    standard_deviation = float(np.std(scene_means, ddof=1))
    effect = float(np.mean(scene_means) / standard_deviation) if standard_deviation > 0 else float("inf")
    return effect, scene_means


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.statistics import clustered_bootstrap_mean, exact_group_sign_flip_pvalue

    config = json.loads(args.causal_config.read_text(encoding="utf-8"))
    records = []
    for path in sorted(args.input.glob("scene-*-state-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record["valid"]:
            raise RuntimeError(f"invalid frozen causal unit: {path}")
        if set(record["conditions"]) != set(config["rollout"]["conditions"]):
            raise RuntimeError(f"condition set differs from the frozen protocol: {path}")
        records.append(record)
    if len(records) != int(config["sampling"]["expected_causal_units"]):
        raise RuntimeError("causal unit count differs from the frozen protocol")
    scenes = np.asarray([record["scene_id"] for record in records])

    definitions = {
        "learned_minus_none_donor": ("future_subspace", "none", "donor_progress"),
        "random_minus_none_donor": ("random_subspace", "none", "donor_progress"),
        "full_minus_none_donor": ("full_donor", "none", "donor_progress"),
        "learned_minus_random_donor": ("future_subspace", "random_subspace", "donor_progress"),
        "learned_minus_none_recipient": ("future_subspace", "none", "recipient_progress"),
    }
    metrics = {}
    for name, (condition, reference, endpoint) in definitions.items():
        values = np.asarray(
            [
                record["conditions"][condition][endpoint]
                - record["conditions"][reference][endpoint]
                for record in records
            ],
            dtype=np.float64,
        )
        standardized, scene_means = _standardized_scene_effect(values, scenes)
        interval = clustered_bootstrap_mean(
            values,
            scenes,
            repetitions=10000,
            seed=73129,
        )
        metrics[name] = {
            "mean_meters": float(np.mean(values)),
            "cluster_bootstrap_95ci_meters": [interval["lower"], interval["upper"]],
            "standardized_scene_effect": standardized,
            "positive_scenes": int(np.sum(scene_means > 0)),
            "exact_scene_sign_flip_pvalue": exact_group_sign_flip_pvalue(values, scenes),
            "scene_means_meters": {
                scene: float(value) for scene, value in zip(np.unique(scenes), scene_means)
            },
        }
    patch_norms = {}
    for condition in ("future_subspace", "random_subspace", "full_donor"):
        norms = [
            replan["patch_receipt"]["patch_norm"]
            for record in records
            for replan in record["conditions"][condition]["replans"]
        ]
        patch_norms[condition] = {"mean": float(np.mean(norms)), "sd": float(np.std(norms))}

    gates = config["endpoint"]["gates"]
    learned = metrics["learned_minus_none_donor"]
    random = metrics["random_minus_none_donor"]
    gate_results = {
        "learned_standardized_effect": learned["standardized_scene_effect"]
        >= float(gates["standardized_donor_progress_min"]),
        "positive_scenes": learned["positive_scenes"] >= int(gates["positive_scenes_min"]),
        "random_control": random["standardized_scene_effect"] <= float(gates["random_control_excess_max"]),
    }
    summary = {
        "study": config["study"],
        "causal_units": len(records),
        "scene_clusters": len(np.unique(scenes)),
        "metrics": metrics,
        "patch_norms": patch_norms,
        "gates": gate_results,
        "causal_gate_passed": all(gate_results.values()),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"causal_gate_passed": summary["causal_gate_passed"], "gates": gate_results}, sort_keys=True))
    print(json.dumps(metrics["learned_minus_none_donor"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
