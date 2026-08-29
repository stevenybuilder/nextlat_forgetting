#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(array**2, axis=1))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["study"] != config["study"]:
        raise RuntimeError("manifest belongs to a different study")
    if manifest["config_sha256"] != _sha256(args.config):
        raise RuntimeError("manifest is not bound to this config")
    rows = manifest["rows"]
    families = sorted({row["family_id"] for row in rows})
    thresholds = config["population"]["pre_model_geometry_gate"]
    target_field = str(thresholds.get("target_field", "bddl_target_difference_xy"))
    per_family: dict[str, dict[str, float | int]] = {}
    all_residuals: list[np.ndarray] = []
    validation_residuals: list[np.ndarray] = []
    minimum_separation = float("inf")
    for family in families:
        family_rows = [row for row in rows if row["family_id"] == family]
        fit = np.asarray(
            [
                row[target_field]
                for row in family_rows
                if row["split"] == "observer_fit"
            ],
            dtype=np.float64,
        )
        if len(fit) < 2:
            raise RuntimeError(f"{family} has fewer than two fit layouts")
        fit_mean = np.mean(fit, axis=0)
        target = np.asarray(
            [row[target_field] for row in family_rows],
            dtype=np.float64,
        )
        validation = np.asarray(
            [
                row[target_field]
                for row in family_rows
                if row["split"] == "observer_validation"
            ],
            dtype=np.float64,
        )
        residual = target - fit_mean
        validation_residual = validation - fit_mean
        all_residuals.extend(residual)
        validation_residuals.extend(validation_residual)
        separations = np.linalg.norm(target, axis=1)
        minimum_separation = min(minimum_separation, float(np.min(separations)))
        per_family[family] = {
            "rows": len(family_rows),
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "all_layout_residual_rms_meters": _rms(residual),
            "validation_residual_rms_meters": _rms(validation_residual),
            "target_difference_span_meters": float(
                np.linalg.norm(np.max(target, axis=0) - np.min(target, axis=0))
            ),
        }
    all_residual_array = np.asarray(all_residuals, dtype=np.float64)
    validation_residual_array = np.asarray(validation_residuals, dtype=np.float64)
    contract = config["stimulus"]["workspace_position_contract"]
    valid_simulator_states = True
    if target_field == "simulator_target_difference_xy":
        for row in rows:
            for field in ("simulator_target_xyz_a", "simulator_target_xyz_b"):
                target = np.asarray(row[field], dtype=np.float64)
                valid_simulator_states &= bool(
                    target.shape == (3,)
                    and np.all(np.isfinite(target))
                    and np.max(np.abs(target[:2]))
                    <= float(contract["max_absolute_xy"])
                    and float(contract["z_min"])
                    <= target[2]
                    <= float(contract["z_max"])
                )
    checks = {
        "expected_family_count": len(families)
        == int(config["population"]["expected_families"]),
        "minimum_target_separation": minimum_separation
        >= float(thresholds["minimum_target_separation_meters"]),
        "pooled_layout_residual_rms": _rms(all_residual_array)
        >= float(thresholds["minimum_pooled_layout_residual_rms_meters"]),
        "pooled_validation_residual_rms": _rms(validation_residual_array)
        >= float(thresholds["minimum_validation_residual_rms_meters"]),
        "families_with_required_span": sum(
            float(summary["target_difference_span_meters"])
            >= float(thresholds["minimum_family_target_span_meters"])
            for summary in per_family.values()
        )
        >= int(thresholds["minimum_families_with_required_span"]),
        "unique_stimulus_ids": len({row["stimulus_id"] for row in rows}) == len(rows),
        "unique_noise_seeds": len({row["noise_seed"] for row in rows}) == len(rows),
        "simulator_states_inside_workspace": valid_simulator_states,
    }
    payload = {
        "study": config["study"],
        "status": "passed" if all(checks.values()) else "failed",
        "written_before_model_outcomes": True,
        "config_sha256": _sha256(args.config),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "rows": len(rows),
        "families": len(families),
        "target_field": target_field,
        "minimum_target_separation_meters": minimum_separation,
        "pooled_layout_residual_rms_meters": _rms(all_residual_array),
        "pooled_validation_residual_rms_meters": _rms(validation_residual_array),
        "per_family": per_family,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "model_outcomes_observed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
