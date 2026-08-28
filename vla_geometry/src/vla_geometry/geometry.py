from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _as_2d_float(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n>=2, d>=1); got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _validate_labels(
    labels: Mapping[str, Sequence[str]], n_samples: int
) -> dict[str, np.ndarray]:
    if len(labels) < 2:
        raise ValueError("at least two factors are required")
    normalized: dict[str, np.ndarray] = {}
    for factor, values in labels.items():
        array = np.asarray(values, dtype=str)
        if array.shape != (n_samples,):
            raise ValueError(
                f"factor {factor!r} has shape {array.shape}, expected {(n_samples,)}"
            )
        if len(np.unique(array)) < 2:
            raise ValueError(f"factor {factor!r} needs at least two observed values")
        normalized[factor] = array
    return normalized


@dataclass(frozen=True)
class AdditiveFactorModel:
    """Least-squares additive model with centered, identifiable factor effects.

    Treatment coding is used for fitting. The fitted level vectors are then centered within each
    factor, which makes pairwise effect directions independent of the arbitrary reference level.
    """

    factors: tuple[str, ...]
    levels: Mapping[str, tuple[str, ...]]
    intercept: np.ndarray
    treatment_effects: Mapping[str, Mapping[str, np.ndarray]]
    centered_effects: Mapping[str, Mapping[str, np.ndarray]]

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: Mapping[str, Sequence[str]],
        *,
        ridge: float = 1e-8,
    ) -> "AdditiveFactorModel":
        x = _as_2d_float(embeddings, "embeddings")
        y = _validate_labels(labels, x.shape[0])
        factors = tuple(y.keys())
        levels = {
            factor: tuple(sorted(np.unique(values).tolist()))
            for factor, values in y.items()
        }

        columns = [np.ones(x.shape[0], dtype=np.float64)]
        column_keys: list[tuple[str, str]] = []
        for factor in factors:
            for level in levels[factor][1:]:
                columns.append((y[factor] == level).astype(np.float64))
                column_keys.append((factor, level))
        design = np.column_stack(columns)
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ x
        )

        intercept = coefficients[0]
        treatment: dict[str, dict[str, np.ndarray]] = {}
        cursor = 1
        for factor in factors:
            factor_effects = {
                levels[factor][0]: np.zeros(x.shape[1], dtype=np.float64)
            }
            for level in levels[factor][1:]:
                factor_effects[level] = coefficients[cursor]
                cursor += 1
            treatment[factor] = factor_effects

        centered: dict[str, dict[str, np.ndarray]] = {}
        for factor, factor_effects in treatment.items():
            mean_effect = np.mean(list(factor_effects.values()), axis=0)
            centered[factor] = {
                level: effect - mean_effect
                for level, effect in factor_effects.items()
            }
        return cls(factors, levels, intercept, treatment, centered)

    def _check_known(self, labels: Mapping[str, Sequence[str]]) -> int:
        if tuple(labels.keys()) != self.factors:
            raise ValueError(
                f"factor order/schema mismatch: got {tuple(labels)}, expected {self.factors}"
            )
        lengths = {len(values) for values in labels.values()}
        if len(lengths) != 1:
            raise ValueError("all factor label arrays must have equal length")
        n_samples = lengths.pop()
        for factor, values in labels.items():
            unknown = sorted(set(map(str, values)) - set(self.levels[factor]))
            if unknown:
                raise ValueError(f"unseen levels for {factor!r}: {unknown}")
        return n_samples

    def predict(self, labels: Mapping[str, Sequence[str]]) -> np.ndarray:
        n_samples = self._check_known(labels)
        prediction = np.repeat(self.intercept[None, :], n_samples, axis=0)
        for factor in self.factors:
            for row, level in enumerate(labels[factor]):
                prediction[row] += self.treatment_effects[factor][str(level)]
        return prediction

    def interference(self, cell: Mapping[str, str], *, eps: float = 1e-12) -> float:
        """Mean absolute cosine across the selected factor-value effect vectors."""

        vectors = []
        for factor in self.factors:
            level = str(cell[factor])
            try:
                vector = self.centered_effects[factor][level]
            except KeyError as error:
                raise ValueError(f"unknown level {level!r} for factor {factor!r}") from error
            norm = float(np.linalg.norm(vector))
            if norm <= eps:
                raise ValueError(f"near-zero effect for {factor}={level}; interference undefined")
            vectors.append(vector / norm)
        cosines = [
            abs(float(vectors[i] @ vectors[j]))
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        return float(np.mean(cosines))

    def reconstruction_r2(
        self,
        embeddings: np.ndarray,
        labels: Mapping[str, Sequence[str]],
    ) -> float:
        x = _as_2d_float(embeddings, "embeddings")
        prediction = self.predict(labels)
        residual = float(np.sum((x - prediction) ** 2))
        centered = x - np.mean(x, axis=0, keepdims=True)
        total = float(np.sum(centered**2))
        return float("nan") if total <= 0 else 1.0 - residual / total


def assign_combination_folds(
    cells: Sequence[Mapping[str, str]],
    factors: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Assign whole cells to deterministic folds and enforce train-level coverage."""

    if n_folds < 2 or n_folds >= len(cells):
        raise ValueError("n_folds must be between 2 and n_cells - 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    assignments = np.empty(len(cells), dtype=np.int64)
    assignments[order] = np.arange(len(cells)) % n_folds

    for fold in range(n_folds):
        train = assignments != fold
        for factor in factors:
            all_levels = {str(cell[factor]) for cell in cells}
            train_levels = {
                str(cell[factor]) for idx, cell in enumerate(cells) if train[idx]
            }
            if train_levels != all_levels:
                raise ValueError(
                    f"fold {fold} drops levels for {factor}: {sorted(all_levels - train_levels)}"
                )
    return assignments


def crossfit_interference(
    cell_embeddings: np.ndarray,
    cells: Sequence[Mapping[str, str]],
    factors: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 314159,
) -> dict[str, np.ndarray | float]:
    """Estimate each cell's geometry using an additive model that excluded that cell."""

    x = _as_2d_float(cell_embeddings, "cell_embeddings")
    if x.shape[0] != len(cells):
        raise ValueError("one embedding is required per combination cell")
    factor_tuple = tuple(factors)
    labels = {
        factor: np.asarray([str(cell[factor]) for cell in cells])
        for factor in factor_tuple
    }
    folds = assign_combination_folds(
        cells, factor_tuple, n_folds=n_folds, seed=seed
    )
    interference = np.full(len(cells), np.nan, dtype=np.float64)
    prediction = np.full_like(x, np.nan, dtype=np.float64)
    fold_r2 = np.full(n_folds, np.nan, dtype=np.float64)

    for fold in range(n_folds):
        train = folds != fold
        test = ~train
        train_labels = {factor: values[train] for factor, values in labels.items()}
        test_labels = {factor: values[test] for factor, values in labels.items()}
        model = AdditiveFactorModel.fit(x[train], train_labels)
        prediction[test] = model.predict(test_labels)
        fold_r2[fold] = model.reconstruction_r2(x[test], test_labels)
        for idx in np.flatnonzero(test):
            interference[idx] = model.interference(cells[idx])

    if not np.all(np.isfinite(interference)) or not np.all(np.isfinite(prediction)):
        raise AssertionError("cross-fitting left non-finite outputs")
    residual = float(np.sum((x - prediction) ** 2))
    total = float(np.sum((x - np.mean(x, axis=0, keepdims=True)) ** 2))
    overall_r2 = float("nan") if total <= 0 else 1.0 - residual / total
    return {
        "folds": folds,
        "interference": interference,
        "prediction": prediction,
        "fold_r2": fold_r2,
        "overall_r2": overall_r2,
    }

