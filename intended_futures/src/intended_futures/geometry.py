from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _finite_array(value: np.ndarray, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    y = _finite_array(target, "target", ndim=2)
    pred = _finite_array(prediction, "prediction", ndim=2)
    if pred.shape != y.shape:
        raise ValueError("target and prediction shapes differ")
    total = float(np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2))
    if total <= 0:
        raise ValueError("R-squared is undefined for a constant target")
    return 1.0 - float(np.sum((y - pred) ** 2)) / total


@dataclass(frozen=True)
class FutureSubspace:
    """Reduced-rank ridge map from a token-preserving activation grid to future deltas."""

    input_shape: tuple[int, ...]
    mean_activation: np.ndarray
    mean_target: np.ndarray
    beta: np.ndarray
    basis: np.ndarray

    @classmethod
    def fit(
        cls,
        activation_differences: np.ndarray,
        future_differences: np.ndarray,
        *,
        rank: int,
        ridge: float,
    ) -> "FutureSubspace":
        x_grid = _finite_array(activation_differences, "activation_differences")
        y = _finite_array(future_differences, "future_differences", ndim=2)
        if x_grid.ndim < 2 or x_grid.shape[0] != y.shape[0]:
            raise ValueError("activation and future arrays need the same nonzero sample axis")
        if x_grid.shape[0] < 2:
            raise ValueError("at least two matched stimuli are required")
        x = x_grid.reshape(x_grid.shape[0], -1)
        if not 1 <= rank <= min(x.shape[1], y.shape[1]):
            raise ValueError("rank exceeds the available activation or target dimensions")
        if ridge <= 0:
            raise ValueError("ridge must be positive")

        mean_x = np.mean(x, axis=0)
        mean_y = np.mean(y, axis=0)
        centered_x = x - mean_x
        centered_y = y - mean_y
        # Dual ridge avoids forming a prohibitively large feature-by-feature matrix.
        gram = centered_x @ centered_x.T
        beta = centered_x.T @ np.linalg.solve(
            gram + float(ridge) * np.eye(gram.shape[0]), centered_y
        )
        left, _, _ = np.linalg.svd(beta, full_matrices=False)
        basis = left[:, :rank]
        beta = basis @ (basis.T @ beta)
        return cls(tuple(x_grid.shape[1:]), mean_x, mean_y, beta, basis)

    def _flatten(self, activations: np.ndarray) -> np.ndarray:
        array = _finite_array(activations, "activations")
        if tuple(array.shape[1:]) != self.input_shape:
            raise ValueError(f"activation shape {array.shape[1:]} != fitted {self.input_shape}")
        return array.reshape(array.shape[0], -1)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        x = self._flatten(activations)
        return (x - self.mean_activation) @ self.beta + self.mean_target

    def project_difference(self, donor: np.ndarray, recipient: np.ndarray) -> np.ndarray:
        donor_array = _finite_array(donor, "donor")
        recipient_array = _finite_array(recipient, "recipient")
        if donor_array.shape != recipient_array.shape or tuple(donor_array.shape) != self.input_shape:
            raise ValueError("donor and recipient must each match the fitted activation grid")
        difference = (donor_array - recipient_array).reshape(-1)
        projected = self.basis @ (self.basis.T @ difference)
        return projected.reshape(self.input_shape)

    def patch(self, donor: np.ndarray, recipient: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        if not np.isfinite(strength):
            raise ValueError("patch strength must be finite")
        recipient_array = _finite_array(recipient, "recipient")
        return recipient_array + float(strength) * self.project_difference(donor, recipient)


def leave_one_group_out_predictions(
    activations: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    *,
    rank: int,
    ridge: float,
) -> np.ndarray:
    x = _finite_array(activations, "activations")
    y = _finite_array(targets, "targets", ndim=2)
    group_array = np.asarray(groups)
    if x.shape[0] != y.shape[0] or len(group_array) != x.shape[0]:
        raise ValueError("activations, targets, and groups must share the sample axis")
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 2:
        raise ValueError("leave-one-group-out requires at least two groups")
    predictions = np.full_like(y, np.nan, dtype=np.float64)
    for group in unique_groups:
        test = group_array == group
        train = ~test
        model = FutureSubspace.fit(x[train], y[train], rank=rank, ridge=ridge)
        predictions[test] = model.predict(x[test])
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("cross-validation left non-finite predictions")
    return predictions


def random_orthonormal_basis(feature_count: int, rank: int, *, seed: int) -> np.ndarray:
    if not 1 <= rank <= feature_count:
        raise ValueError("rank must be between one and feature_count")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(feature_count, rank)))
    return basis[:, :rank]
