#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum(target**2))
    if denominator <= 1e-18:
        raise ValueError("residual R-squared target is constant")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


def _means(
    values: np.ndarray, families: np.ndarray, indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        str(family): np.mean(values[indices & (families == family)], axis=0)
        for family in sorted(set(families[indices]))
    }


def _residualize(
    values: np.ndarray, families: np.ndarray, means: dict[str, np.ndarray]
) -> np.ndarray:
    missing = sorted(set(families).difference(means))
    if missing:
        raise ValueError(f"families have no training mean: {missing}")
    return np.stack([value - means[str(family)] for value, family in zip(values, families)])


def _select_ridge_by_level(
    x: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    levels: np.ndarray,
    candidates: list[float],
    decoder_class: Any,
) -> dict[str, Any]:
    rows = []
    for ridge in sorted(set(candidates)):
        predictions = np.full_like(y, np.nan, dtype=np.float64)
        targets = np.full_like(y, np.nan, dtype=np.float64)
        for level in sorted(set(levels)):
            test = levels == level
            train = ~test
            x_means = _means(x, families, train)
            y_means = _means(y, families, train)
            usable_test = test & np.isin(families, list(x_means))
            x_train = _residualize(x[train], families[train], x_means)
            y_train = _residualize(y[train], families[train], y_means)
            decoder = decoder_class.fit(x_train, y_train, ridge_fraction=ridge)
            predictions[usable_test] = decoder.predict(
                _residualize(x[usable_test], families[usable_test], x_means)
            )
            targets[usable_test] = _residualize(
                y[usable_test], families[usable_test], y_means
            )
        valid = np.all(np.isfinite(predictions), axis=1) & np.all(
            np.isfinite(targets), axis=1
        )
        if int(np.sum(valid)) < len(y) * 0.8:
            raise RuntimeError("too few level-cross-fitted rows")
        rows.append(
            {
                "ridge_fraction": ridge,
                "r2": _r2(targets[valid], predictions[valid]),
                "rows": int(np.sum(valid)),
            }
        )
    return max(rows, key=lambda row: (row["r2"], row["ridge_fraction"])), rows


def _load_records(input_dir: Path, study: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive:
                continue
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("study") != study:
                continue
            records.append(
                {
                    "path": path,
                    "metadata": metadata,
                    "activation": np.asarray(
                        archive["activation_difference"], dtype=np.float64
                    ),
                    "action": np.asarray(archive["action_difference"], dtype=np.float64),
                    "target_xy": np.asarray(
                        archive["target_difference_xy"], dtype=np.float64
                    ),
                    "target_xyz": np.asarray(
                        archive["target_difference_xyz"], dtype=np.float64
                    ),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--clearance-output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.geometry import ZeroInterceptTargetDecoder

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    if args.controller_output.exists() or args.clearance_output.exists():
        raise FileExistsError("controller and clearance artifacts are create-only")
    if config.get("status") != "frozen":
        raise RuntimeError("controller fitting requires a frozen config")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("manifest is not bound to this config")
    if (
        receipt.get("study") != config["study"]
        or receipt.get("config_sha256") != _sha256(args.config)
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise RuntimeError("runtime receipt does not certify this fit")
    records = _load_records(args.input, config["study"])
    expected = sum(
        int(config["population"]["expected_split_units"][split])
        for split in ("observer_fit", "observer_validation")
    )
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} observer records, found {len(records)}")
    if any(record["metadata"].get("runtime_receipt_sha256") != _sha256(args.runtime_receipt) for record in records):
        raise RuntimeError("observer records do not share the runtime receipt")
    shapes = {tuple(record["activation"].shape) for record in records}
    if len(shapes) != 1:
        raise RuntimeError(f"activation shapes differ: {shapes}")

    fit_records = [record for record in records if record["metadata"]["split"] == "observer_fit"]
    validation_records = [
        record
        for record in records
        if record["metadata"]["split"] == "observer_validation"
    ]
    x_fit = np.stack([record["activation"] for record in fit_records])
    y_fit = np.stack([record["target_xy"] for record in fit_records])
    action_fit = np.stack([record["action"] for record in fit_records])
    families_fit = np.asarray([record["metadata"]["family_id"] for record in fit_records])
    levels_fit = np.asarray([int(record["metadata"]["level"]) for record in fit_records])
    selected, candidate_rows = _select_ridge_by_level(
        x_fit,
        y_fit,
        families_fit,
        levels_fit,
        [float(value) for value in config["observer"]["ridge_fraction_grid"]],
        ZeroInterceptTargetDecoder,
    )
    selected_fraction = float(selected["ridge_fraction"])
    x_means = _means(x_fit, families_fit, np.ones(len(x_fit), dtype=bool))
    y_means = _means(y_fit, families_fit, np.ones(len(y_fit), dtype=bool))
    action_means = _means(
        action_fit, families_fit, np.ones(len(action_fit), dtype=bool)
    )
    residual_decoder = ZeroInterceptTargetDecoder.fit(
        _residualize(x_fit, families_fit, x_means),
        _residualize(y_fit, families_fit, y_means),
        ridge_fraction=selected_fraction,
    )
    action_decoder = ZeroInterceptTargetDecoder.fit(
        _residualize(action_fit, families_fit, action_means),
        _residualize(y_fit, families_fit, y_means),
        ridge_fraction=selected_fraction,
    )
    x_validation = np.stack([record["activation"] for record in validation_records])
    y_validation = np.stack([record["target_xy"] for record in validation_records])
    action_validation = np.stack([record["action"] for record in validation_records])
    families_validation = np.asarray(
        [record["metadata"]["family_id"] for record in validation_records]
    )
    y_residual = _residualize(y_validation, families_validation, y_means)
    prediction = residual_decoder.predict(
        _residualize(x_validation, families_validation, x_means)
    )
    action_prediction = action_decoder.predict(
        _residualize(action_validation, families_validation, action_means)
    )
    residual_r2 = _r2(y_residual, prediction)
    action_r2 = _r2(y_residual, action_prediction)
    row_denominator = np.linalg.norm(y_residual, axis=1) * np.linalg.norm(
        prediction, axis=1
    )
    row_cosines = np.full(len(y_residual), -1.0, dtype=np.float64)
    valid_cosine = row_denominator > 1e-12
    row_cosines[valid_cosine] = (
        np.sum(y_residual * prediction, axis=1)[valid_cosine]
        / row_denominator[valid_cosine]
    )
    family_cosines = {
        str(family): float(np.mean(row_cosines[families_validation == family]))
        for family in sorted(set(families_validation))
    }
    gate = config["observer"]["validation_gate"]
    checks = {
        "minimum_layout_residual_r2": residual_r2
        >= float(gate["minimum_layout_residual_r2"]),
        "minimum_sse_reduction_over_exact_prompt_mean": residual_r2
        >= float(gate["minimum_sse_reduction_over_exact_prompt_mean"]),
        "minimum_positive_family_cosines": sum(
            value > 0 for value in family_cosines.values()
        )
        >= int(gate["minimum_positive_family_cosines"]),
        "minimum_action_chunk_positive_control_r2": action_r2
        >= float(gate["minimum_action_chunk_positive_control_r2"]),
    }
    passed = all(checks.values())
    analysis = {
        "study": config["study"],
        "status": "observer_gate_passed" if passed else "observer_gate_failed",
        "config_sha256": _sha256(args.config),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_receipt_sha256": _sha256(args.runtime_receipt),
        "fit_units": len(fit_records),
        "validation_units": len(validation_records),
        "families": len(set(families_validation)),
        "input_shape": list(next(iter(shapes))),
        "selected_ridge_fraction": selected_fraction,
        "fit_leave_one_level_out_candidates": candidate_rows,
        "validation": {
            "layout_residual_r2": residual_r2,
            "sse_reduction_over_exact_prompt_mean": residual_r2,
            "action_chunk_positive_control_r2": action_r2,
            "positive_family_cosines": sum(value > 0 for value in family_cosines.values()),
            "family_cosines": family_cosines,
        },
        "checks": checks,
        "advance_to_causal_test": passed,
        "claim_boundary": "untouched sample-3 layout validation; no sample-4 causal state was loaded",
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        print(json.dumps(analysis, sort_keys=True))
        return 2

    combined_records = fit_records + validation_records
    combined_x = np.stack([record["activation"] for record in combined_records])
    combined_y_xyz = np.stack([record["target_xyz"] for record in combined_records])
    combined_families = np.asarray(
        [record["metadata"]["family_id"] for record in combined_records]
    )
    full_decoder = ZeroInterceptTargetDecoder.fit(
        combined_x, combined_y_xyz, ridge_fraction=selected_fraction
    )
    family_mean_xyz = {
        str(family): np.mean(
            combined_y_xyz[combined_families == family], axis=0
        ).tolist()
        for family in sorted(set(combined_families))
    }
    args.controller_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.controller_output.with_suffix(args.controller_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pathway=np.asarray(config["site"]["pathway"]),
            layer=np.asarray(int(config["site"]["layer"])),
            input_shape=np.asarray(full_decoder.input_shape, dtype=np.int64),
            beta=np.asarray(full_decoder.beta, dtype=np.float32),
            selected_ridge_fraction=np.asarray(selected_fraction),
            inverse_ridge_fraction=np.asarray(
                float(config["controller"]["decoder_inverse_ridge_fraction"])
            ),
            maximum_norm_fraction_of_full_donor_delta=np.asarray(
                float(config["controller"]["maximum_norm_fraction_of_full_donor_delta"])
            ),
            family_mean_target_xyz_json=np.asarray(
                json.dumps(family_mean_xyz, sort_keys=True)
            ),
            config_sha256=np.asarray(_sha256(args.config)),
            manifest_sha256=np.asarray(manifest["manifest_sha256"]),
            runtime_receipt_sha256=np.asarray(_sha256(args.runtime_receipt)),
            fit_units=np.asarray(len(combined_x)),
        )
    temporary.replace(args.controller_output)
    clearance = {
        "study": config["study"],
        "authorization": "GO_CAUSAL_TEST",
        "config_sha256": _sha256(args.config),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_receipt_sha256": _sha256(args.runtime_receipt),
        "observer_analysis_sha256": _sha256(args.analysis_output),
        "controller_sha256": _sha256(args.controller_output),
        "observer_gate_checks": checks,
        "all_observer_gates_passed": True,
        "causal_test_states_loaded_during_fitting": False,
    }
    args.clearance_output.write_text(
        json.dumps(clearance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"advance_to_causal_test": True, "controller_sha256": _sha256(args.controller_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
