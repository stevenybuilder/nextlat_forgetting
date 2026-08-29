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
    total = float(np.sum((target - np.mean(target, axis=0, keepdims=True)) ** 2))
    if total <= 1e-18:
        raise ValueError("R-squared target is constant")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / total


def _mean_cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = np.sum(target * prediction, axis=1)
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    cosine = np.full(len(target), -1.0, dtype=np.float64)
    valid = denominator > 1e-12
    cosine[valid] = numerator[valid] / denominator[valid]
    return float(np.mean(cosine))


def _group_mean_predictions(
    train_targets: np.ndarray,
    train_keys: list[str],
    validation_keys: list[str],
) -> np.ndarray:
    global_mean = np.mean(train_targets, axis=0)
    mapping: dict[str, np.ndarray] = {}
    for key in sorted(set(train_keys)):
        mapping[key] = np.mean(
            train_targets[np.asarray(train_keys) == key], axis=0
        )
    return np.asarray([mapping.get(key, global_mean) for key in validation_keys])


def _load_records(input_dir: Path, expected_study: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(input_dir.glob("scene-*-state-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("study") != expected_study:
                continue
            records.append(
                {
                    "path": path,
                    "metadata": metadata,
                    "activation": np.asarray(
                        archive["activation_difference"], dtype=np.float64
                    ),
                    "target": np.asarray(archive["target_difference"], dtype=np.float64),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--clearance-output", type=Path, required=True)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.geometry import (
        ZeroInterceptTargetDecoder,
        select_zero_intercept_ridge,
    )

    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    if args.controller_output.exists() or args.clearance_output.exists():
        raise FileExistsError(
            "controller and clearance outputs are create-only; use a new study identity to refit"
        )
    if protocol.get("status") != "frozen":
        raise RuntimeError("controller fitting requires a frozen TC1 protocol")
    if manifest["protocol_sha256"] != _sha256(args.tc1_config):
        raise RuntimeError("TC1 manifest is not bound to this protocol")
    if (
        runtime_receipt.get("study") != protocol["study"]
        or runtime_receipt.get("tc1_config_sha256") != _sha256(args.tc1_config)
        or runtime_receipt.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise RuntimeError("runtime receipt does not certify this observer fit")
    records = _load_records(args.input, protocol["study"])
    expected = (
        int(protocol["population"]["expected_observer_fit_units"])
        + int(protocol["population"]["expected_observer_validation_units"])
    )
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} observer records, found {len(records)}")
    shapes = {tuple(record["activation"].shape) for record in records}
    if len(shapes) != 1:
        raise RuntimeError(f"observer activation shapes differ: {sorted(shapes)}")
    if any(record["target"].shape != (3,) for record in records):
        raise RuntimeError("observer target differences must all be three-vectors")
    receipt_sha = _sha256(args.runtime_receipt)
    if any(
        record["metadata"].get("runtime_receipt_sha256") != receipt_sha
        for record in records
    ):
        raise RuntimeError("observer records do not share the certified runtime receipt")

    fit = [record for record in records if record["metadata"]["split"] == "observer_fit"]
    validation = [
        record for record in records
        if record["metadata"]["split"] == "observer_validation"
    ]
    if (
        len(fit) != int(protocol["population"]["expected_observer_fit_units"])
        or len(validation)
        != int(protocol["population"]["expected_observer_validation_units"])
    ):
        raise RuntimeError("observer split counts differ from the protocol")
    x_fit = np.stack([record["activation"] for record in fit])
    y_fit = np.stack([record["target"] for record in fit])
    groups_fit = [str(record["metadata"]["scene_id"]) for record in fit]
    selection = select_zero_intercept_ridge(
        x_fit,
        y_fit,
        groups_fit,
        ridge_fractions=protocol["observer"]["ridge_fraction_grid"],
    )
    selected = selection["selected"]
    selected_fraction = float(selected["ridge_fraction"])
    fit_decoder = ZeroInterceptTargetDecoder.fit(
        x_fit, y_fit, ridge_fraction=selected_fraction
    )

    x_validation = np.stack([record["activation"] for record in validation])
    y_validation = np.stack([record["target"] for record in validation])
    prediction = fit_decoder.predict(x_validation)
    train_subject_keys = [
        f"{record['metadata']['subject_a']}->{record['metadata']['subject_b']}"
        for record in fit
    ]
    validation_subject_keys = [
        f"{record['metadata']['subject_a']}->{record['metadata']['subject_b']}"
        for record in validation
    ]
    train_prompt_keys = [
        f"{record['metadata']['prompt_a_id']}->{record['metadata']['prompt_b_id']}"
        for record in fit
    ]
    validation_prompt_keys = [
        f"{record['metadata']['prompt_a_id']}->{record['metadata']['prompt_b_id']}"
        for record in validation
    ]
    global_prediction = np.repeat(
        np.mean(y_fit, axis=0, keepdims=True), len(validation), axis=0
    )
    subject_prediction = _group_mean_predictions(
        y_fit, train_subject_keys, validation_subject_keys
    )
    prompt_prediction = _group_mean_predictions(
        y_fit, train_prompt_keys, validation_prompt_keys
    )
    prompt_sse = float(np.sum((y_validation - prompt_prediction) ** 2))
    activation_sse = float(np.sum((y_validation - prediction) ** 2))
    residual_reduction = (
        1.0 - activation_sse / prompt_sse if prompt_sse > 1e-18 else -1e300
    )
    cosine_denominator = np.linalg.norm(y_validation, axis=1) * np.linalg.norm(
        prediction, axis=1
    )
    row_cosines = np.full(len(validation), -1.0, dtype=np.float64)
    cosine_valid = cosine_denominator > 1e-12
    row_cosines[cosine_valid] = (
        np.sum(y_validation * prediction, axis=1)[cosine_valid]
        / cosine_denominator[cosine_valid]
    )
    validation_scenes = np.asarray(
        [str(record["metadata"]["scene_id"]) for record in validation]
    )
    scene_cosines = {
        scene: float(np.mean(row_cosines[validation_scenes == scene]))
        for scene in sorted(set(validation_scenes))
    }
    gate_spec = protocol["observer"]["validation_gate"]
    checks = {
        "minimum_r2": _r2(y_validation, prediction) >= float(gate_spec["minimum_r2"]),
        "residual_sse_reduction_over_exact_prompt_pair": residual_reduction
        >= float(gate_spec["minimum_residual_sse_reduction_over_exact_prompt_pair"]),
        "positive_scene_cosines": sum(value > 0 for value in scene_cosines.values())
        >= int(gate_spec["minimum_positive_scene_cosines"]),
    }
    passed = all(checks.values())
    analysis = {
        "study": protocol["study"],
        "status": "observer_gate_passed" if passed else "observer_gate_failed",
        "protocol_sha256": _sha256(args.tc1_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_receipt_sha256": receipt_sha,
        "fit_units": len(fit),
        "validation_units": len(validation),
        "input_shape": list(next(iter(shapes))),
        "selected_ridge_fraction": selected_fraction,
        "fit_leave_one_scene_out_candidates": [
            {
                "ridge_fraction": float(row["ridge_fraction"]),
                "r2": float(row["r2"]),
            }
            for row in selection["candidates"]
        ],
        "validation": {
            "activation_r2": _r2(y_validation, prediction),
            "activation_mean_cosine": _mean_cosine(y_validation, prediction),
            "global_mean_r2": _r2(y_validation, global_prediction),
            "ordered_subject_pair_mean_r2": _r2(y_validation, subject_prediction),
            "exact_prompt_pair_mean_r2": _r2(y_validation, prompt_prediction),
            "residual_sse_reduction_over_exact_prompt_pair": residual_reduction,
            "positive_scene_cosines": sum(value > 0 for value in scene_cosines.values()),
            "scene_cosines": scene_cosines,
        },
        "checks": checks,
        "advance_to_causal_test": passed,
        "claim_boundary": "observer validation only; no causal-test state was loaded",
    }
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        print(json.dumps(analysis, sort_keys=True))
        return 2

    combined_x = np.concatenate((x_fit, x_validation), axis=0)
    combined_y = np.concatenate((y_fit, y_validation), axis=0)
    final_decoder = ZeroInterceptTargetDecoder.fit(
        combined_x, combined_y, ridge_fraction=selected_fraction
    )
    args.controller_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.controller_output.with_suffix(args.controller_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pathway=np.asarray(protocol["site"]["pathway"]),
            layer=np.asarray(int(protocol["site"]["layer"])),
            input_shape=np.asarray(final_decoder.input_shape, dtype=np.int64),
            beta=np.asarray(final_decoder.beta, dtype=np.float32),
            selected_ridge_fraction=np.asarray(selected_fraction),
            inverse_ridge_fraction=np.asarray(
                float(protocol["controller"]["decoder_inverse_ridge_fraction"])
            ),
            maximum_norm_fraction_of_full_donor_delta=np.asarray(
                float(protocol["controller"]["maximum_norm_fraction_of_full_donor_delta"])
            ),
            protocol_sha256=np.asarray(_sha256(args.tc1_config)),
            manifest_sha256=np.asarray(manifest["manifest_sha256"]),
            runtime_receipt_sha256=np.asarray(receipt_sha),
            fit_units=np.asarray(len(combined_x)),
        )
    temporary.replace(args.controller_output)
    clearance = {
        "study": protocol["study"],
        "authorization": "GO_CAUSAL_TEST",
        "protocol_sha256": _sha256(args.tc1_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_receipt_sha256": receipt_sha,
        "observer_analysis_sha256": _sha256(args.analysis_output),
        "controller_sha256": _sha256(args.controller_output),
        "observer_gate_checks": checks,
        "all_observer_gates_passed": True,
        "causal_test_states_loaded_during_fitting": False,
    }
    args.clearance_output.parent.mkdir(parents=True, exist_ok=True)
    args.clearance_output.write_text(
        json.dumps(clearance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "advance_to_causal_test": True,
                "controller_sha256": _sha256(args.controller_output),
                "clearance_sha256": _sha256(args.clearance_output),
                "analysis": str(args.analysis_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
