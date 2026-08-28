import json
from pathlib import Path

import pytest

from vla_geometry.grid import FACTOR_ORDER, build_cells, get_factor_order, load_config


CONFIG = Path(__file__).parents[1] / "config" / "pilot.json"


def test_frozen_grid_has_48_unique_cells():
    config = load_config(CONFIG)
    cells = build_cells(config["factors"])
    assert len(cells) == 48
    assert len({cell["cell_id"] for cell in cells}) == 48
    assert all(set(FACTOR_ORDER).issubset(cell) for cell in cells)


def test_grid_rejects_duplicate_factor_values():
    config = json.loads(CONFIG.read_text())
    config["factors"]["target_shape"] = ["block", "block"]
    with pytest.raises(ValueError, match="duplicate"):
        build_cells(config["factors"])


def test_schema_v2_supports_a_frozen_factor_order():
    config = load_config(Path(__file__).parents[1] / "config" / "memory_pilot.json")
    order = get_factor_order(config)
    cells = build_cells(config["factors"], order)
    assert order == ("target_shape", "receptacle_shape", "direction")
    assert len(cells) == 32
    assert len({cell["cell_id"] for cell in cells}) == 32
