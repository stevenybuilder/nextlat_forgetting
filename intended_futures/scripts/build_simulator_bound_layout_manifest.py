#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _valid_state(record: dict[str, Any], contract: dict[str, float]) -> bool:
    target_a = [float(value) for value in record["simulator_target_xyz_a"]]
    target_b = [float(value) for value in record["simulator_target_xyz_b"]]
    return (
        all(float(contract["z_min"]) <= target[2] <= float(contract["z_max"])
            for target in (target_a, target_b))
        and max(abs(value) for target in (target_a, target_b) for value in target[:2])
        <= float(contract["max_absolute_xy"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if source_manifest["config_sha256"] != _sha256(args.source_config):
        raise RuntimeError("source manifest is not bound to its source config")
    source = config["source_design"]
    if (
        source["study"] != source_config["study"]
        or source["manifest_sha256"] != source_manifest["manifest_sha256"]
    ):
        raise RuntimeError("new design does not name the supplied frozen source design")

    source_rows = {row["stimulus_id"]: row for row in source_manifest["rows"]}
    records = []
    for path in sorted(args.measurements.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("study") != source_config["study"]
            or record.get("manifest_sha256") != source_manifest["manifest_sha256"]
            or record.get("model_outcomes_observed") is not False
        ):
            raise RuntimeError(f"measurement is not pre-model source geometry: {path}")
        record["measurement_file_sha256"] = _sha256(path)
        records.append(record)
    if len(records) != len(source_rows) or {r["stimulus_id"] for r in records} != set(
        source_rows
    ):
        raise RuntimeError("measurements do not cover the source manifest exactly")
    raw_records = [
        {key: value for key, value in record.items() if key != "measurement_file_sha256"}
        for record in records
    ]
    if _canonical_digest(raw_records) != source["simulator_measurements_sha256"]:
        raise RuntimeError("simulator measurement digest differs from the new design")

    contract = config["stimulus"]["workspace_position_contract"]
    rejected = [record for record in records if not _valid_state(record, contract)]
    rejected_ids = sorted(record["stimulus_id"] for record in rejected)
    if rejected_ids != sorted(config["population"]["excluded_invalid_state_ids"]):
        raise RuntimeError(
            f"measured invalid states {rejected_ids} differ from the frozen exclusions"
        )

    measurements = {record["stimulus_id"]: record for record in records}
    rows = []
    for stimulus_id, source_row in source_rows.items():
        record = measurements[stimulus_id]
        if stimulus_id in rejected_ids:
            continue
        if (
            record["bddl_sha256"] != source_row["bddl_sha256"]
            or record["init_sha256"] != source_row["init_sha256"]
            or record["family_id"] != source_row["family_id"]
            or record["split"] != source_row["split"]
        ):
            raise RuntimeError(f"measurement identity differs for {stimulus_id}")
        row = dict(source_row)
        row.update(
            {
                "simulator_geometry_record_sha256": record[
                    "measurement_file_sha256"
                ],
                "simulator_target_xyz_a": record["simulator_target_xyz_a"],
                "simulator_target_xyz_b": record["simulator_target_xyz_b"],
                "simulator_target_difference_xy": record[
                    "simulator_target_difference_xy"
                ],
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["family_id"], row["level"], row["sample"]))
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in config["population"]["split_samples"]
    }
    if counts != config["population"]["expected_split_units"]:
        raise RuntimeError(
            f"simulator-bound split counts {counts} differ from the new design"
        )
    if len(rows) != int(config["population"]["expected_manifest_rows"]):
        raise RuntimeError("simulator-bound row count differs from the new design")

    payload: dict[str, Any] = {
        "study": config["study"],
        "suite": source_manifest["suite"],
        "design": "libero_plus_simulator_bound_visible_target_displacement_v2",
        "config_sha256": _sha256(args.config),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "simulator_measurements_sha256": source["simulator_measurements_sha256"],
        "excluded_invalid_states": [
            {
                "stimulus_id": record["stimulus_id"],
                "measurement_file_sha256": record["measurement_file_sha256"],
                "simulator_target_xyz_a": record["simulator_target_xyz_a"],
                "simulator_target_xyz_b": record["simulator_target_xyz_b"],
                "reason": "instruction-selected subject left the frozen workspace after settling",
            }
            for record in sorted(rejected, key=lambda item: item["stimulus_id"])
        ],
        "rows": rows,
    }
    payload["manifest_sha256"] = _canonical_digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "split_counts": counts,
                "excluded_invalid_states": rejected_ids,
                "manifest_sha256": payload["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
