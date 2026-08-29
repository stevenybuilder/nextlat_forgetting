from __future__ import annotations

from contextlib import ExitStack
import copy
import time
from typing import Any

import numpy as np

from .openpi_hooks import CaptureCalls, FirstCallDeltaPatch, disable_compiled_sampling, resolve_action_expert_layer


_CONTROL_KEYS = {
    "__intended_futures_mode",
    "__prompt_a",
    "__prompt_b",
    "__noise_seed",
    "__recipient_prompt",
    "__donor_prompt",
    "__selected_layer",
    "__patch_kind",
    "__future_basis",
    "__random_basis",
}


class InstrumentedPairedPolicy:
    """OpenPI policy wrapper for matched prompts at an identical observation and noise draw."""

    def __init__(self, base_policy: Any, *, layer_indices: list[int], expected_denoising_calls: int):
        if not getattr(base_policy, "_is_pytorch_model", False):
            raise TypeError("instrumentation requires a converted PyTorch OpenPI checkpoint")
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must be nonempty and unique")
        if expected_denoising_calls <= 0:
            raise ValueError("expected_denoising_calls must be positive")
        self._base = base_policy
        self._model = base_policy._model
        disable_compiled_sampling(self._base)
        self._layers = {
            int(index): resolve_action_expert_layer(self._model, int(index))
            for index in layer_indices
        }
        self._expected_calls = int(expected_denoising_calls)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._base.metadata)
        metadata["intended_futures"] = {
            "instrumented": True,
            "layers": sorted(self._layers),
            "expected_denoising_calls": self._expected_calls,
            "token_pooling": "none",
        }
        return metadata

    def _capture(self, obs: dict[str, Any], noise: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, int]]:
        captures: dict[int, CaptureCalls] = {}
        with ExitStack() as stack:
            for index, layer in self._layers.items():
                captures[index] = stack.enter_context(CaptureCalls(layer))
            result = self._base.infer(obs, noise=noise)

        activations: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for index, capture in captures.items():
            if len(capture.calls) != self._expected_calls:
                raise RuntimeError(
                    f"layer {index} emitted {len(capture.calls)} calls, expected {self._expected_calls}"
                )
            first = capture.calls[0]
            if first.shape[0] != 1 or first.ndim != 3:
                raise RuntimeError(f"unexpected action-expert activation shape at layer {index}: {first.shape}")
            activations[str(index)] = first[0]
            counts[str(index)] = len(capture.calls)
        return result, activations, counts

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        mode = str(obs.get("__intended_futures_mode", "normal"))
        if mode == "normal":
            clean = {key: value for key, value in obs.items() if key not in _CONTROL_KEYS}
            return self._base.infer(clean)
        if mode == "causal_action":
            return self._causal_action(obs)
        if mode != "extract_pair":
            raise ValueError(f"unsupported intended-futures mode: {mode}")

        prompt_a = str(obs["__prompt_a"])
        prompt_b = str(obs["__prompt_b"])
        if not prompt_a or not prompt_b or prompt_a == prompt_b:
            raise ValueError("matched extraction requires two distinct nonempty prompts")
        noise_seed = int(obs["__noise_seed"])
        config = self._model.config
        noise = np.random.default_rng(noise_seed).standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        clean = {key: value for key, value in obs.items() if key not in _CONTROL_KEYS}
        obs_a = copy.deepcopy(clean)
        obs_b = copy.deepcopy(clean)
        obs_a["prompt"] = prompt_a
        obs_b["prompt"] = prompt_b

        started = time.monotonic()
        result_a, activations_a, calls_a = self._capture(obs_a, noise)
        result_b, activations_b, calls_b = self._capture(obs_b, noise)
        return {
            "actions_a": np.asarray(result_a["actions"], dtype=np.float32),
            "actions_b": np.asarray(result_b["actions"], dtype=np.float32),
            "activations_a": activations_a,
            "activations_b": activations_b,
            "denoising_calls_a": calls_a,
            "denoising_calls_b": calls_b,
            "noise_seed": noise_seed,
            "pair_infer_ms": (time.monotonic() - started) * 1000.0,
        }

    def _causal_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        recipient_prompt = str(obs["__recipient_prompt"])
        donor_prompt = str(obs["__donor_prompt"])
        patch_kind = str(obs["__patch_kind"])
        selected_layer = int(obs["__selected_layer"])
        if selected_layer not in self._layers:
            raise ValueError(f"selected layer {selected_layer} was not instrumented")
        if not recipient_prompt or not donor_prompt or recipient_prompt == donor_prompt:
            raise ValueError("causal action requires distinct donor and recipient prompts")
        if patch_kind not in {"none", "future_subspace", "random_subspace", "full_donor"}:
            raise ValueError(f"unsupported patch kind: {patch_kind}")
        noise_seed = int(obs["__noise_seed"])
        config = self._model.config
        noise = np.random.default_rng(noise_seed).standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        clean = {key: value for key, value in obs.items() if key not in _CONTROL_KEYS}
        recipient_obs = copy.deepcopy(clean)
        recipient_obs["prompt"] = recipient_prompt
        if patch_kind == "none":
            started = time.monotonic()
            result = self._base.infer(recipient_obs, noise=noise)
            return {
                "actions": np.asarray(result["actions"], dtype=np.float32),
                "patch_kind": patch_kind,
                "noise_seed": noise_seed,
                "causal_infer_ms": (time.monotonic() - started) * 1000.0,
                "patch_receipt": {"calls_seen": 0, "calls_patched": 0, "patch_norm": 0.0},
            }

        donor_obs = copy.deepcopy(clean)
        donor_obs["prompt"] = donor_prompt
        started = time.monotonic()
        _, donor_activations, donor_calls = self._capture(donor_obs, noise)
        _, recipient_activations, recipient_calls = self._capture(recipient_obs, noise)
        donor = donor_activations[str(selected_layer)]
        recipient = recipient_activations[str(selected_layer)]
        difference = (donor - recipient).reshape(-1).astype(np.float64)
        future_basis = np.asarray(obs["__future_basis"], dtype=np.float64)
        if future_basis.ndim != 2 or future_basis.shape[0] != len(difference):
            raise ValueError("future basis does not match the selected activation grid")
        future_delta = future_basis @ (future_basis.T @ difference)
        if patch_kind == "future_subspace":
            delta = future_delta
        elif patch_kind == "full_donor":
            delta = difference
        else:
            random_basis = np.asarray(obs["__random_basis"], dtype=np.float64)
            if random_basis.shape != future_basis.shape:
                raise ValueError("random basis must match the future basis")
            random_delta = random_basis @ (random_basis.T @ difference)
            random_norm = float(np.linalg.norm(random_delta))
            future_norm = float(np.linalg.norm(future_delta))
            if random_norm <= 1e-12 or future_norm <= 1e-12:
                raise RuntimeError("cannot norm-match a degenerate subspace intervention")
            delta = random_delta * (future_norm / random_norm)

        grid_delta = delta.reshape(recipient.shape)[None, ...].astype(np.float32)
        with FirstCallDeltaPatch(self._layers[selected_layer], grid_delta) as patch:
            result = self._base.infer(recipient_obs, noise=noise)
        if patch.receipt.calls_seen != self._expected_calls:
            raise RuntimeError(
                f"patched layer emitted {patch.receipt.calls_seen} calls, expected {self._expected_calls}"
            )
        return {
            "actions": np.asarray(result["actions"], dtype=np.float32),
            "patch_kind": patch_kind,
            "noise_seed": noise_seed,
            "causal_infer_ms": (time.monotonic() - started) * 1000.0,
            "donor_calls": donor_calls[str(selected_layer)],
            "recipient_calls": recipient_calls[str(selected_layer)],
            "patch_receipt": {
                "calls_seen": patch.receipt.calls_seen,
                "calls_patched": patch.receipt.calls_patched,
                "original_norm": patch.receipt.original_norm,
                "patch_norm": patch.receipt.patch_norm,
                "future_projection_norm": float(np.linalg.norm(future_delta)),
                "full_difference_norm": float(np.linalg.norm(difference)),
            },
        }
