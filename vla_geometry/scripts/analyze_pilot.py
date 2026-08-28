#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from vla_geometry.analysis import factor_main_effect_features, leave_one_cell_out_log_loss
    from vla_geometry.geometry import crossfit_interference
    from vla_geometry.grid import build_cells, get_factor_order, load_config
    from vla_geometry.io import (
        atomic_write_json,
        read_npz_record,
        sha256_file,
        sha256_source_tree,
    )
    from vla_geometry.statistics import (
        bootstrap_spearman_interval,
        permutation_pvalue,
        spearman_correlation,
    )
    from vla_geometry.seeds import resolve_all_seed_maps

    config = load_config(args.config)
    factor_order = get_factor_order(config)
    cells = build_cells(config["factors"], factor_order)
    seed_maps = resolve_all_seed_maps(config, cells)
    by_id = {cell["cell_id"]: cell for cell in cells}
    expected_config_sha = sha256_file(args.config)
    expected_source_sha = sha256_source_tree(Path(__file__).parents[1])
    provenance_path = args.input / "provenance.json"
    manifest_path = args.input / "seed_manifest.json"
    if not provenance_path.exists() or not manifest_path.exists():
        raise RuntimeError("run provenance or seed manifest is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("config_sha256") != expected_config_sha:
        raise RuntimeError("run configuration hash does not match analysis configuration")
    if provenance.get("source_tree_sha256") != expected_source_sha:
        raise RuntimeError("run source hash does not match analysis source")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != seed_maps:
        raise RuntimeError("run seed manifest does not match frozen configuration")

    representations: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    rep_metadata: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted((args.input / "representation").glob("*.npz")):
        activation, metadata = read_npz_record(path)
        cell_id = metadata["cell_id"]
        if cell_id not in by_id:
            raise ValueError(f"unexpected cell in {path}: {cell_id}")
        seed = int(metadata["seed"])
        expected_seeds = set(seed_maps["representation"][cell_id])
        if seed not in expected_seeds:
            raise ValueError(f"unexpected representation seed in {path}: {seed}")
        if seed in representations[cell_id]:
            raise ValueError(f"duplicate representation seed for {cell_id}: {seed}")
        expected_metadata = {
            "mode": "representation",
            "valid": True,
            "closed_loop": False,
            **{factor: by_id[cell_id][factor] for factor in factor_order},
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"representation metadata mismatch in {path}")
        representations[cell_id][seed] = activation
        rep_metadata[cell_id][seed] = metadata

    failure_files = sorted((args.input / "behavior").glob("*.failure.json"))
    if failure_files:
        raise RuntimeError(
            f"behavior contains {len(failure_files)} frozen-seed failure records"
        )
    behavior: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted((args.input / "behavior").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cell_id = record["cell_id"]
        if cell_id not in by_id:
            raise ValueError(f"unexpected behavior cell in {path}: {cell_id}")
        seed = int(record["seed"])
        if seed not in set(seed_maps["behavior"][cell_id]):
            raise ValueError(f"unexpected behavior seed in {path}: {seed}")
        if seed in behavior[cell_id]:
            raise ValueError(f"duplicate behavior seed for {cell_id}: {seed}")
        expected_metadata = {
            "mode": "behavior",
            "valid": True,
            "closed_loop": True,
            "terminal": True,
            **{factor: by_id[cell_id][factor] for factor in factor_order},
        }
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"behavior metadata mismatch in {path}")
        behavior[cell_id][seed] = record

    count_errors = []
    for cell in cells:
        cell_id = cell["cell_id"]
        expected_rep = len(seed_maps["representation"][cell_id])
        expected_behavior = len(seed_maps["behavior"][cell_id])
        if set(representations[cell_id]) != set(seed_maps["representation"][cell_id]):
            count_errors.append(
                f"{cell_id}: representation {len(representations[cell_id])}/{expected_rep}"
            )
        if set(behavior[cell_id]) != set(seed_maps["behavior"][cell_id]):
            count_errors.append(
                f"{cell_id}: behavior {len(behavior[cell_id])}/{expected_behavior}"
            )
    if count_errors:
        raise RuntimeError("incomplete pilot:\n" + "\n".join(count_errors))

    cell_embeddings = np.stack(
        [
            np.mean(list(representations[cell["cell_id"]].values()), axis=0)
            for cell in cells
        ]
    )
    geometry = crossfit_interference(
        cell_embeddings,
        cells,
        factor_order,
        n_folds=int(config["analysis"]["n_combination_folds"]),
        seed=int(config["analysis"]["fold_seed"]),
    )
    successes = np.asarray(
        [
            sum(
                bool(row["success"])
                for row in behavior[cell["cell_id"]].values()
            )
            for cell in cells
        ],
        dtype=np.int64,
    )
    trials = np.asarray(
        [len(behavior[cell["cell_id"]]) for cell in cells], dtype=np.int64
    )
    success_rate = successes / trials
    confidence = np.asarray(
        [
            np.mean(
                [
                    row["initial_action_confidence"]
                    for row in rep_metadata[cell["cell_id"]].values()
                ]
            )
            for cell in cells
        ]
    )
    layout_control_names = list(config["analysis"]["layout_controls"])
    layout_controls = np.column_stack(
        [
            np.asarray(
                [
                    np.mean(
                        [
                            row[control]
                            for row in rep_metadata[cell["cell_id"]].values()
                        ]
                    )
                    for cell in cells
                ],
                dtype=np.float64,
            )
            for control in layout_control_names
        ]
    )
    interference = np.asarray(geometry["interference"])
    correlation = spearman_correlation(interference, success_rate)
    analysis_seed = int(config["analysis"]["fold_seed"])
    interval = bootstrap_spearman_interval(
        interference,
        success_rate,
        replicates=int(config["analysis"]["bootstrap_replicates"]),
        seed=analysis_seed + 1,
        alpha=float(config["analysis"]["alpha"]),
    )
    p_value = permutation_pvalue(
        interference,
        success_rate,
        replicates=int(config["analysis"]["permutation_replicates"]),
        seed=analysis_seed + 2,
    )
    factor_controls, factor_control_names = factor_main_effect_features(
        cells, factor_order
    )
    controls = np.column_stack([factor_controls, confidence, layout_controls])
    full = np.column_stack([controls, interference])
    logistic_ridge = float(config["analysis"].get("logistic_ridge", 1.0))
    control_log_loss = leave_one_cell_out_log_loss(
        controls, successes, trials, ridge=logistic_ridge
    )
    full_log_loss = leave_one_cell_out_log_loss(
        full, successes, trials, ridge=logistic_ridge
    )
    improvement = (control_log_loss - full_log_loss) / control_log_loss
    minimum_rho = float(config["analysis"]["minimum_abs_spearman"])
    alpha = float(config["analysis"]["alpha"])
    gate = bool(
        correlation <= -minimum_rho
        and interval[1] < 0
        and p_value < alpha
        and improvement
        >= float(config["analysis"]["minimum_log_loss_improvement_fraction"])
    )

    cell_rows = []
    for index, cell in enumerate(cells):
        cell_rows.append(
            {
                **cell,
                "fold": int(geometry["folds"][index]),
                "interference": float(interference[index]),
                "successes": int(successes[index]),
                "trials": int(trials[index]),
                "success_rate": float(success_rate[index]),
                "mean_initial_action_confidence": float(confidence[index]),
                **{
                    f"mean_{name}": float(layout_controls[index, control_index])
                    for control_index, name in enumerate(layout_control_names)
                },
            }
        )
    result = {
        "study_id": config["study_id"],
        "complete": True,
        "n_cells": len(cells),
        "representation_seeds_per_cell": sorted(
            {len(rows) for rows in representations.values()}
        ),
        "behavior_trials_per_cell": sorted({len(rows) for rows in behavior.values()}),
        "primary": {
            "spearman_interference_vs_success": correlation,
            "bootstrap_95_interval": list(interval),
            "permutation_pvalue_two_sided": p_value,
            "control_loocv_log_loss": control_log_loss,
            "full_loocv_log_loss": full_log_loss,
            "log_loss_improvement_fraction": improvement,
            "advance_to_causal_stage": gate,
            "control_predictors": factor_control_names
            + ["initial_action_confidence"]
            + layout_control_names,
            "logistic_ridge": logistic_ridge,
        },
        "secondary": {
            "crossfit_additive_reconstruction_r2": float(geometry["overall_r2"]),
            "fold_reconstruction_r2": [float(value) for value in geometry["fold_r2"]],
        },
        "cells": cell_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result["primary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
