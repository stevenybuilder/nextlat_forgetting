from pathlib import Path

from vla_geometry.grid import build_cells, get_factor_order, load_config
from vla_geometry.seeds import memory_factors_for_seed, resolve_all_seed_maps


CONFIG = Path(__file__).parents[1] / "config" / "memory_pilot.json"


def test_factor_stratified_seed_manifest_is_complete_disjoint_and_outcome_blind():
    config = load_config(CONFIG)
    cells = build_cells(config["factors"], get_factor_order(config))
    by_id = {cell["cell_id"]: cell for cell in cells}
    manifest = resolve_all_seed_maps(config, cells)
    assert sum(map(len, manifest["representation"].values())) == 32 * 8
    assert sum(map(len, manifest["behavior"].values())) == 32 * 20
    assert sum(map(len, manifest["smoke"].values())) == 4
    all_seed_sets = [
        set(seed for rows in manifest[mode].values() for seed in rows)
        for mode in ("smoke", "representation", "behavior")
    ]
    assert all_seed_sets[0].isdisjoint(all_seed_sets[1])
    assert all_seed_sets[0].isdisjoint(all_seed_sets[2])
    assert all_seed_sets[1].isdisjoint(all_seed_sets[2])
    for mode, cell_rows in manifest.items():
        for cell_id, seeds in cell_rows.items():
            expected = {
                factor: by_id[cell_id][factor]
                for factor in ("target_shape", "receptacle_shape", "direction")
            }
            assert all(
                memory_factors_for_seed(seed, config["seed_sampler_contract"])
                == expected
                for seed in seeds
            )
