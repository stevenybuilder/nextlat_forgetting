from __future__ import annotations

from contextlib import ExitStack
import copy
import time
from typing import Any

import numpy as np

from .openpi_hooks import CaptureCalls, disable_compiled_sampling, resolve_action_expert_layer


_CONTROL_KEYS = {
    "__intended_futures_mode",
    "__prompt_a",
    "__prompt_b",
    "__noise_seed",
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
