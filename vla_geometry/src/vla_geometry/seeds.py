from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def memory_factors_for_seed(
    seed: int, contract: Mapping[str, Any]
) -> dict[str, str]:
    """Mirror the pinned Task-16 factor draws without constructing an environment.

    The contract records the ordered support of the official L2 generator. Selection never loads
    a policy, executes an action, or observes a reward. An integration check compares this mirror
    against real resets from the pinned VIMA-Bench commit before the configuration is frozen.
    """

    rows = int(contract["rows"])
    columns = int(contract["columns"])
    null_distractors = int(contract["null_distractors"])
    target_shapes = tuple(map(str, contract["target_shapes_ordered"]))
    target_texture_count = int(contract["target_texture_count"])
    receptacle_shapes = tuple(map(str, contract["receptacle_shapes_ordered"]))
    if rows < 1 or columns < 1 or null_distractors < 0:
        raise ValueError("invalid Task-16 grid contract")
    if len(target_shapes) < 1 or target_texture_count < 1 or len(receptacle_shapes) < 1:
        raise ValueError("invalid Task-16 object contract")

    rng = np.random.default_rng(seed=seed)
    candidates = [(row, column) for row in range(rows) for column in range(columns)]
    target_flat_index = int(rng.choice(len(candidates)))
    target = candidates[target_flat_index]
    offsets = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }
    while True:
        direction = str(rng.choice(list(offsets)))
        offset = offsets[direction]
        neighbor = (target[0] + offset[0], target[1] + offset[1])
        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns:
            break

    null_indices: list[tuple[int, int]] = []
    while len(null_indices) < null_distractors:
        candidate = candidates[int(rng.choice(len(candidates)))]
        if candidate not in (target, neighbor) and candidate not in null_indices:
            null_indices.append(candidate)

    combination_count = len(target_shapes) * target_texture_count
    sampled_combinations = np.asarray(
        rng.choice(combination_count, replace=False, size=rows * columns),
        dtype=np.int64,
    )
    target_combination = int(sampled_combinations[target_flat_index])
    target_shape = target_shapes[target_combination // target_texture_count]

    # The official reset samples a three-vector object size for every array position before it
    # samples the receptacle. Values do not matter for stratification, but advancing the generator
    # by the same six three-vector draws does.
    for _ in range(rows * columns):
        rng.uniform(size=3)
    receptacle_shape = receptacle_shapes[int(rng.choice(len(receptacle_shapes)))]
    return {
        "target_shape": target_shape,
        "receptacle_shape": receptacle_shape,
        "direction": direction,
    }


def memory_direction_for_seed(
    seed: int, rows: int = 3, columns: int = 2
) -> str:
    """Return only the Task-16 direction draw (legacy diagnostic helper)."""

    rng = np.random.default_rng(seed=seed)
    candidates = [(row, column) for row in range(rows) for column in range(columns)]
    target = candidates[int(rng.choice(len(candidates)))]
    offsets = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }
    while True:
        direction = str(rng.choice(list(offsets)))
        offset = offsets[direction]
        neighbor = (target[0] + offset[0], target[1] + offset[1])
        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns:
            return direction


def resolve_seed_map(
    config: Mapping[str, Any],
    cells: Sequence[Mapping[str, str]],
    mode: str,
) -> dict[str, list[int]]:
    """Resolve legacy fixed seeds or the frozen outcome-blind Task-16 rule."""

    seed_spec = config["seeds"]
    if mode in seed_spec and isinstance(seed_spec[mode], list):
        return {
            str(cell["cell_id"]): [int(seed) for seed in seed_spec[mode]]
            for cell in cells
        }
    if seed_spec.get("strategy") != "official_task16_factor_stratified_v1":
        raise ValueError(f"unsupported seed specification for {mode!r}: {seed_spec}")
    selection = config["seed_selection"][mode]
    contract = config["seed_sampler_contract"]
    selected_cells = (
        [cells[0], cells[-1]] if mode == "smoke" else list(cells)
    )
    factor_to_cell_id = {
        (
            str(cell["target_shape"]),
            str(cell["receptacle_shape"]),
            str(cell["direction"]),
        ): str(cell["cell_id"])
        for cell in selected_cells
    }
    output = {cell_id: [] for cell_id in factor_to_cell_id.values()}
    count = int(selection["count_per_cell"])
    start = int(selection["start"])
    for seed in range(start, start + int(selection["scan_limit"])):
        factors = memory_factors_for_seed(seed, contract)
        key = (
            factors["target_shape"],
            factors["receptacle_shape"],
            factors["direction"],
        )
        cell_id = factor_to_cell_id.get(key)
        if cell_id is not None and len(output[cell_id]) < count:
            output[cell_id].append(seed)
        if all(len(seeds) == count for seeds in output.values()):
            break
    incomplete = {
        cell_id: len(seeds) for cell_id, seeds in output.items() if len(seeds) != count
    }
    if incomplete:
        raise RuntimeError(
            f"seed scan did not fill {mode} cells to {count}: {incomplete}"
        )
    return output


def resolve_all_seed_maps(
    config: Mapping[str, Any], cells: Sequence[Mapping[str, str]]
) -> dict[str, dict[str, list[int]]]:
    return {
        mode: resolve_seed_map(config, cells, mode)
        for mode in ("smoke", "representation", "behavior")
    }
