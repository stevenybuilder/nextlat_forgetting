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


@dataclass(frozen=True)
class ZeroInterceptTargetDecoder:
    """Scale-aware ridge map for a minimum-norm target intervention.

    The map has no intercept so that zero activation change necessarily means zero decoded
    target change. That origin is required for a meaningful causal inverse.
    """

    input_shape: tuple[int, ...]
    beta: np.ndarray
    ridge_fraction: float
    ridge_value: float

    @classmethod
    def fit(
        cls,
        activation_differences: np.ndarray,
        target_differences: np.ndarray,
        *,
        ridge_fraction: float,
    ) -> "ZeroInterceptTargetDecoder":
        x_grid = _finite_array(activation_differences, "activation_differences")
        y = _finite_array(target_differences, "target_differences", ndim=2)
        if x_grid.ndim < 2 or x_grid.shape[0] != y.shape[0] or x_grid.shape[0] < 2:
            raise ValueError("activation and target arrays need the same sample axis of length >= 2")
        if ridge_fraction <= 0 or not np.isfinite(ridge_fraction):
            raise ValueError("ridge_fraction must be finite and positive")
        x = x_grid.reshape(x_grid.shape[0], -1)
        gram = x @ x.T
        scale = float(np.mean(np.diag(gram)))
        if scale <= 1e-12:
            raise ValueError("activation differences are degenerate")
        ridge_value = float(ridge_fraction) * scale
        beta = x.T @ np.linalg.solve(
            gram + ridge_value * np.eye(gram.shape[0]), y
        )
        return cls(tuple(x_grid.shape[1:]), beta, float(ridge_fraction), ridge_value)

    def _flatten(self, activations: np.ndarray) -> np.ndarray:
        array = _finite_array(activations, "activations")
        if tuple(array.shape[1:]) != self.input_shape:
            raise ValueError(f"activation shape {array.shape[1:]} != fitted {self.input_shape}")
        return array.reshape(array.shape[0], -1)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self._flatten(activations) @ self.beta

    def minimum_norm_delta(
        self,
        desired_target_delta: np.ndarray,
        *,
        inverse_ridge_fraction: float,
        maximum_norm: float | None = None,
    ) -> np.ndarray:
        desired = _finite_array(desired_target_delta, "desired_target_delta")
        if desired.shape != (self.beta.shape[1],):
            raise ValueError("desired target delta does not match decoder output dimension")
        if inverse_ridge_fraction <= 0 or not np.isfinite(inverse_ridge_fraction):
            raise ValueError("inverse_ridge_fraction must be finite and positive")
        output_gram = self.beta.T @ self.beta
        scale = float(np.mean(np.diag(output_gram)))
        if scale <= 1e-18:
            raise ValueError("decoder is degenerate")
        inverse_ridge = float(inverse_ridge_fraction) * scale
        coefficients = np.linalg.solve(
            output_gram + inverse_ridge * np.eye(output_gram.shape[0]), desired
        )
        delta = self.beta @ coefficients
        norm = float(np.linalg.norm(delta))
        if maximum_norm is not None:
            if maximum_norm <= 0 or not np.isfinite(maximum_norm):
                raise ValueError("maximum_norm must be finite and positive")
            if norm > maximum_norm:
                delta = delta * (float(maximum_norm) / norm)
        return delta.reshape(self.input_shape)

    def project_difference(self, difference: np.ndarray) -> np.ndarray:
        array = _finite_array(difference, "difference")
        if tuple(array.shape) != self.input_shape:
            raise ValueError("difference does not match decoder activation grid")
        basis, _ = np.linalg.qr(self.beta)
        flat = array.reshape(-1)
        projected = basis @ (basis.T @ flat)
        return projected.reshape(self.input_shape)


def select_zero_intercept_ridge(
    activations: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    *,
    ridge_fractions: Sequence[float],
) -> dict[str, object]:
    """Select ridge strength using leave-one-group-out predictions only."""

    x = _finite_array(activations, "activations")
    y = _finite_array(targets, "targets", ndim=2)
    group_array = np.asarray(groups)
    if x.shape[0] != y.shape[0] or len(group_array) != x.shape[0]:
        raise ValueError("activations, targets, and groups must share the sample axis")
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 2:
        raise ValueError("ridge selection requires at least two groups")
    candidates = sorted({float(value) for value in ridge_fractions})
    if not candidates or any(value <= 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("ridge_fractions must contain finite positive values")

    flat = x.reshape(x.shape[0], -1)
    gram = flat @ flat.T
    rows: list[dict[str, object]] = []
    for ridge_fraction in candidates:
        predictions = np.full_like(y, np.nan, dtype=np.float64)
        for group in unique_groups:
            test = group_array == group
            train_indices = np.flatnonzero(~test)
            test_indices = np.flatnonzero(test)
            train_gram = gram[np.ix_(train_indices, train_indices)]
            scale = float(np.mean(np.diag(train_gram)))
            if scale <= 1e-12:
                raise ValueError("activation differences are degenerate in a held-out fold")
            coefficients = np.linalg.solve(
                train_gram
                + ridge_fraction * scale * np.eye(len(train_indices)),
                y[train_indices],
            )
            predictions[test_indices] = (
                gram[np.ix_(test_indices, train_indices)] @ coefficients
            )
        score = r2_score(y, predictions)
        rows.append(
            {
                "ridge_fraction": ridge_fraction,
                "r2": score,
                "predictions": predictions,
            }
        )
    # The larger ridge wins exact ties, as frozen in the protocol.
    selected = max(rows, key=lambda row: (float(row["r2"]), float(row["ridge_fraction"])))
    return {"selected": selected, "candidates": rows}


def construct_target_controller_delta(
    beta: np.ndarray,
    donor_minus_recipient: np.ndarray,
    desired_target_delta: np.ndarray,
    *,
    kind: str,
    inverse_ridge_fraction: float,
    maximum_norm_fraction: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Construct one frozen TC1 intervention with a donor-difference norm cap."""

    decoder = _finite_array(beta, "beta", ndim=2)
    difference_grid = _finite_array(donor_minus_recipient, "donor_minus_recipient")
    difference = difference_grid.reshape(-1)
    desired = _finite_array(desired_target_delta, "desired_target_delta")
    if decoder.shape != (len(difference), 3) or desired.shape != (3,):
        raise ValueError("controller dimensions do not match the activation and target grids")
    if inverse_ridge_fraction <= 0 or not np.isfinite(inverse_ridge_fraction):
        raise ValueError("inverse_ridge_fraction must be finite and positive")
    if maximum_norm_fraction <= 0 or not np.isfinite(maximum_norm_fraction):
        raise ValueError("maximum_norm_fraction must be finite and positive")
    if kind not in {"minimum_norm_target", "target_projection", "random_controller"}:
        raise ValueError(f"unsupported target controller kind: {kind}")

    output_gram = decoder.T @ decoder
    output_scale = float(np.mean(np.diag(output_gram)))
    full_norm = float(np.linalg.norm(difference))
    if output_scale <= 1e-18 or full_norm <= 1e-12:
        raise ValueError("controller or donor difference is degenerate")
    coefficients = np.linalg.solve(
        output_gram
        + float(inverse_ridge_fraction) * output_scale * np.eye(output_gram.shape[0]),
        desired,
    )
    minimum_delta = decoder @ coefficients
    maximum_norm = float(maximum_norm_fraction) * full_norm
    minimum_norm = float(np.linalg.norm(minimum_delta))
    if minimum_norm <= 1e-12:
        raise ValueError("minimum-norm target delta is degenerate")
    if minimum_norm > maximum_norm:
        minimum_delta = minimum_delta * (maximum_norm / minimum_norm)
        minimum_norm = maximum_norm

    if kind == "minimum_norm_target":
        result = minimum_delta
    elif kind == "target_projection":
        basis, _ = np.linalg.qr(decoder)
        result = basis @ (basis.T @ difference)
        result_norm = float(np.linalg.norm(result))
        if result_norm > maximum_norm:
            result = result * (maximum_norm / result_norm)
    else:
        if random_seed is None:
            raise ValueError("random_controller requires random_seed")
        random = np.random.default_rng(int(random_seed)).normal(size=difference.shape)
        random_norm = float(np.linalg.norm(random))
        if random_norm <= 1e-12:
            raise ValueError("matched random direction is degenerate")
        result = random * (minimum_norm / random_norm)
    return result.reshape(difference_grid.shape)


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
