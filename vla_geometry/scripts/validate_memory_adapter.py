#!/usr/bin/env python3
"""Outcome-blind integration check for the pinned VIMA-Bench Task-16 adapter."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vimabench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds-per-cell", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seeds_per_cell < 1:
        raise ValueError("--seeds-per-cell must be positive")
    project_root = Path(__file__).parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(args.vimabench_root))

    from vima_bench import PARTITION_TO_SPECS, make

    from vla_geometry.grid import build_cells, get_factor_order, load_config
    from vla_geometry.io import atomic_write_json, sha256_file, sha256_source_tree
    from vla_geometry.runner import _actual_factors, _task_kwargs
    from vla_geometry.seeds import resolve_all_seed_maps

    config = load_config(args.config)
    benchmark = config["benchmark"]
    if benchmark["task"] != "manipulate_old_neighbor":
        raise ValueError("adapter validation is specific to Task 16")
    official = copy.deepcopy(
        PARTITION_TO_SPECS["test"][benchmark["partition"]][benchmark["task"]]
    )
    contract = config["seed_sampler_contract"]
    support_checks = {
        "target_shapes_ordered": official["possible_dragged_obj"]
        == contract["target_shapes_ordered"],
        "target_texture_count": len(official["possible_dragged_obj_texture"])
        == int(contract["target_texture_count"]),
        "receptacle_shapes_ordered": official["possible_base_obj"]
        == contract["receptacle_shapes_ordered"],
        "grid_rows": int(official.get("num_array_rows", 3)) == int(contract["rows"]),
        "grid_columns": int(official.get("num_array_columns", 2))
        == int(contract["columns"]),
        "null_distractors": int(official.get("num_null_distractors", 1))
        == int(contract["null_distractors"]),
    }
    failed_support = [name for name, passed in support_checks.items() if not passed]
    if failed_support:
        raise RuntimeError(f"seed-sampler contract differs from official spec: {failed_support}")

    cells = build_cells(config["factors"], get_factor_order(config))
    seed_maps = resolve_all_seed_maps(config, cells)
    checked = []
    for cell in cells:
        kwargs = _task_kwargs(config, cell)
        if kwargs != official:
            raise RuntimeError(
                f"Task-16 adapter changes the official generator for {cell['cell_id']}"
            )
        for seed in seed_maps["representation"][cell["cell_id"]][
            : args.seeds_per_cell
        ]:
            env = make(
                benchmark["task"],
                modalities=["segm", "rgb"],
                task_kwargs=kwargs,
                seed=int(seed),
                render_prompt=False,
                display_debug_window=False,
                hide_arm_rgb=bool(benchmark["hide_arm_rgb"]),
            )
            try:
                env.reset()
                actual = _actual_factors(env, benchmark["task"])
                expected = {
                    factor: str(cell[factor])
                    for factor in ("target_shape", "receptacle_shape", "direction")
                }
                if actual != expected:
                    raise RuntimeError(
                        f"seed mirror mismatch for seed {seed}: actual={actual}, "
                        f"expected={expected}"
                    )
                if int(env.unwrapped.meta_info["seed"]) != int(seed):
                    raise RuntimeError(f"seed substitution detected for requested seed {seed}")
                checked.append(
                    {"cell_id": cell["cell_id"], "seed": int(seed), **actual}
                )
            finally:
                env.close()

    result = {
        "passed": True,
        "policy_loaded": False,
        "actions_executed": False,
        "rewards_or_outcomes_observed": False,
        "config_sha256": sha256_file(args.config),
        "source_tree_sha256": sha256_source_tree(project_root),
        "support_checks": support_checks,
        "cells_checked": len(cells),
        "resets_checked": len(checked),
        "checks": checked,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("passed", "cells_checked", "resets_checked")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
