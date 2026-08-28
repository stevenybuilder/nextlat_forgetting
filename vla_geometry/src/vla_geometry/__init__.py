"""Compositional geometry tools for frozen VLA representations."""

from .geometry import AdditiveFactorModel, crossfit_interference
from .grid import build_cells, combination_id

__all__ = [
    "AdditiveFactorModel",
    "build_cells",
    "combination_id",
    "crossfit_interference",
]

