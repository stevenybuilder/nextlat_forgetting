#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _split_indices(protocol: dict[str, Any]) -> dict[int, str]:
    population = protocol["population"]
    fields = {
        "observer_fit": "observer_fit_state_indices",
        "observer_validation": "observer_validation_state_indices",
        "causal_test": "causal_test_state_indices",
        "reserve": "reserve_state_indices",
    }
    result: dict[int, str] = {}
    for split, field in fields.items():
        for raw_index in population[field]:
            index = int(raw_index)
            if index in result:
                raise ValueError(f"state index {index} occurs in multiple TC1 splits")
            result[index] = split
    previous = {int(index) for index in population["previously_observed_state_indices"]}
    overlap = previous.intersection(result)
    if overlap:
        raise ValueError(f"TC1 states overlap prior outcome-bearing states: {sorted(overlap)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.config import load_config

    pilot = load_config(args.pilot_config)
    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    if protocol["parent"]["pilot_study"] != pilot["study"]:
        raise ValueError("TC1 protocol references a different pilot")
    if protocol["population"]["suite"] != pilot["benchmark"]["suite"]:
        raise ValueError("TC1 protocol and pilot suites differ")
    split_by_index = _split_indices(protocol)
    suite = pilot["benchmark"]["suite"]
    bddl_dir = args.libero_root / "libero" / "bddl_files" / suite
    init_dir = args.libero_root / "libero" / "init_files" / suite
    bddl_files = sorted(bddl_dir.glob("*.bddl"))
    prompts = pilot["benchmark"]["prompts"]
    selected_scenes = set(protocol["population"]["scene_ids"])
    scenes = [
        scene for scene in pilot["benchmark"]["scenes"]
        if scene["scene_id"] in selected_scenes
    ]
    if {scene["scene_id"] for scene in scenes} != selected_scenes:
        raise ValueError("TC1 protocol references a scene absent from the frozen pilot")

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        task_index = int(scene["task_index"])
        bddl = bddl_files[task_index]
        init_file = init_dir / f"{bddl.stem}.pruned_init"
        if not init_file.is_file():
            raise FileNotFoundError(init_file)
        prompt_a = prompts[scene["contrast"][0]]
        prompt_b = prompts[scene["contrast"][1]]
        for state_index, split in sorted(split_by_index.items()):
            rows.append(
                {
                    "stimulus_id": f"{scene['scene_id']}-state-{state_index:02d}",
                    "scene_id": scene["scene_id"],
                    "task_index": task_index,
                    "task_file": f"{suite}/{bddl.name}",
                    "bddl_sha256": _sha256(bddl),
                    "init_sha256": _sha256(init_file),
                    "initial_state_index": state_index,
                    "split": split,
                    "noise_seed": 150000 + len(rows),
                    "task_a": {
                        "prompt_id": scene["contrast"][0],
                        "prompt": prompt_a["text"],
                        "intended_subject": prompt_a["subject"],
                    },
                    "task_b": {
                        "prompt_id": scene["contrast"][1],
                        "prompt": prompt_b["text"],
                        "intended_subject": prompt_b["subject"],
                    },
                }
            )
    expected = int(protocol["population"]["expected_manifest_rows"])
    if len(rows) != expected:
        raise RuntimeError(f"TC1 manifest has {len(rows)} rows, expected {expected}")
    payload = {
        "study": protocol["study"],
        "suite": suite,
        "design": "independent_state_target_control_v1",
        "protocol_sha256": _sha256(args.tc1_config),
        "rows": rows,
    }
    payload["manifest_sha256"] = _canonical_digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "manifest_sha256": payload["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
