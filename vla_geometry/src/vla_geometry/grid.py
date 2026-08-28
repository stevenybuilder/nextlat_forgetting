from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FACTOR_ORDER = (
    "target_shape",
    "target_texture",
    "receptacle_shape",
    "receptacle_texture",
)


def combination_id(
    cell: Mapping[str, str], factor_order: Iterable[str] = FACTOR_ORDER
) -> str:
    """Return a stable identifier that contains no filesystem-sensitive punctuation."""

    ordered = tuple(factor_order)
    missing = [factor for factor in ordered if factor not in cell]
    if missing:
        raise ValueError(f"cell is missing factors: {missing}")
    canonical = "\x1f".join(str(cell[factor]) for factor in ordered)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"cell-{digest}"


def build_cells(
    factors: Mapping[str, Iterable[str]],
    factor_order: Iterable[str] = FACTOR_ORDER,
) -> list[dict[str, str]]:
    """Materialize the preregistered factorial grid in a deterministic order."""

    ordered = tuple(factor_order)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("factor_order must contain at least two unique factors")
    unknown = sorted(set(factors) - set(ordered))
    missing = sorted(set(ordered) - set(factors))
    if unknown or missing:
        raise ValueError(f"factor schema mismatch; unknown={unknown}, missing={missing}")
    values = []
    for factor in ordered:
        factor_values = tuple(str(value) for value in factors[factor])
        if len(factor_values) < 2:
            raise ValueError(f"factor {factor!r} needs at least two values")
        if len(set(factor_values)) != len(factor_values):
            raise ValueError(f"factor {factor!r} contains duplicate values")
        values.append(factor_values)

    cells = []
    for product in itertools.product(*values):
        cell = dict(zip(ordered, product))
        cell["cell_id"] = combination_id(cell, ordered)
        cells.append(cell)
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise AssertionError("combination-id collision")
    return cells


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") not in (1, 2):
        raise ValueError(f"unsupported schema version: {config.get('schema_version')}")
    build_cells(config["factors"], get_factor_order(config))
    return config


def get_factor_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the frozen factor order, retaining schema-v1 compatibility."""

    return tuple(config.get("factor_order", FACTOR_ORDER))
