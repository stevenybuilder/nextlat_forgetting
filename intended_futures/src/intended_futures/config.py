from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "study",
        "status",
        "upstream",
        "model",
        "benchmark",
        "sampling",
        "representation",
        "analysis",
        "runtime_contract",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing top-level configuration keys: {missing}")
    if config["status"] != "frozen":
        raise ValueError("scientific configuration must have status='frozen'")
    if config["benchmark"]["suite"] != "libero_cf_spatial_focused":
        raise ValueError("the pilot is frozen to LIBERO-CF Spatial-Focused")

    pairs = config["benchmark"]["matched_pairs"]
    if len(pairs) != 5:
        raise ValueError("the official suite must resolve to exactly five matched task pairs")
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("matched pair IDs must be unique")
    for pair in pairs:
        if len(pair["task_indices"]) != 2 or len(pair["intended_subjects"]) != 2:
            raise ValueError(f"pair {pair['pair_id']} must contain two tasks and two subjects")
        if pair["intended_subjects"][0] == pair["intended_subjects"][1]:
            raise ValueError(f"pair {pair['pair_id']} does not change the intended subject")

    indices = config["sampling"]["initial_state_indices"]
    if indices != list(range(10)):
        raise ValueError("pilot initial-state indices must be the frozen range 0..9")
    if config["sampling"]["expected_matched_pairs"] != len(pairs) * len(indices):
        raise ValueError("expected_matched_pairs is inconsistent with pairs and state indices")
    manifest_sha256 = config.get("manifest_sha256", "")
    if len(manifest_sha256) != 64 or any(character not in "0123456789abcdef" for character in manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    runtime = config["runtime_contract"]
    if runtime["num_gpus"] != 1 or runtime["distributed"]:
        raise ValueError("pilot runtime must be a single non-distributed GPU process")
    if config["model"]["action_horizon"] <= config["model"]["replan_steps"]:
        raise ValueError("action horizon must extend beyond each executed replan block")
