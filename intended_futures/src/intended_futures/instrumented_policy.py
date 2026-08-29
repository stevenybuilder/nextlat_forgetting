from __future__ import annotations

from contextlib import ExitStack
import copy
import time
from typing import Any

import numpy as np

from .openpi_hooks import (
    AllCallsDeltaPatch,
    CaptureCalls,
    FirstCallDeltaPatch,
    ReplayCallsPatch,
    disable_compiled_sampling,
    resolve_action_expert_layer,
    resolve_paligemma_layer,
)


_CONTROL_KEYS = {
    "__intended_futures_mode",
    "__prompt_a",
    "__prompt_b",
    "__noise_seed",
    "__recipient_prompt",
    "__donor_prompt",
    "__selected_layer",
    "__selected_pathway",
    "__patch_kind",
    "__patch_schedule",
    "__random_direction_seed",
    "__future_basis",
    "__random_basis",
}


class InstrumentedPairedPolicy:
    """OpenPI policy wrapper for matched prompts at an identical observation and noise draw."""

    def __init__(
        self,
        base_policy: Any,
        *,
        layer_indices: list[int],
        expected_denoising_calls: int,
        paligemma_layer_indices: list[int] | None = None,
    ):
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
        paligemma_indices = paligemma_layer_indices or []
        if len(set(paligemma_indices)) != len(paligemma_indices):
            raise ValueError("paligemma_layer_indices must be unique")
        self._paligemma_layers = {
            int(index): resolve_paligemma_layer(self._model, int(index))
            for index in paligemma_indices
        }
        self._pathway_layers = {
            "expert": self._layers,
            "paligemma": self._paligemma_layers,
        }
        self._expected_calls = int(expected_denoising_calls)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._base.metadata)
        metadata["intended_futures"] = {
            "instrumented": True,
            "layers": {
                "expert": sorted(self._layers),
                "paligemma": sorted(self._paligemma_layers),
            },
            "expected_denoising_calls": self._expected_calls,
            "token_pooling": "none",
        }
        return metadata

    def _capture(
        self, obs: dict[str, Any], noise: np.ndarray
    ) -> tuple[
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, int],
        dict[str, dict[str, list[np.ndarray]]],
        dict[str, dict[str, int]],
    ]:
        captures: dict[tuple[str, int], CaptureCalls] = {}
        with ExitStack() as stack:
            for pathway, pathway_layers in self._pathway_layers.items():
                for index, layer in pathway_layers.items():
                    captures[(pathway, index)] = stack.enter_context(CaptureCalls(layer))
            result = self._base.infer(obs, noise=noise)

        activations: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        all_calls: dict[str, dict[str, list[np.ndarray]]] = {
            pathway: {} for pathway in self._pathway_layers
        }
        pathway_counts: dict[str, dict[str, int]] = {
            pathway: {} for pathway in self._pathway_layers
        }
        for (pathway, index), capture in captures.items():
            if not capture.calls:
                raise RuntimeError(
                    f"{pathway} layer {index} emitted no calls"
                )
            if pathway == "expert" and len(capture.calls) != self._expected_calls:
                raise RuntimeError(
                    f"expert layer {index} emitted {len(capture.calls)} calls, "
                    f"expected {self._expected_calls}"
                )
            first = capture.calls[0]
            if first.shape[0] != 1 or first.ndim != 3:
                raise RuntimeError(
                    f"unexpected {pathway} activation shape at layer {index}: {first.shape}"
                )
            all_calls[pathway][str(index)] = capture.calls
            pathway_counts[pathway][str(index)] = len(capture.calls)
            if pathway == "expert":
                activations[str(index)] = first[0]
                counts[str(index)] = len(capture.calls)
        return result, activations, counts, all_calls, pathway_counts

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
        result_a, activations_a, calls_a, _, _ = self._capture(obs_a, noise)
        result_b, activations_b, calls_b, _, _ = self._capture(obs_b, noise)
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
        selected_pathway = str(obs.get("__selected_pathway", "expert"))
        patch_schedule = str(obs.get("__patch_schedule", "first_call"))
        selected_layer = int(obs["__selected_layer"])
        if selected_pathway not in self._pathway_layers:
            raise ValueError(f"unsupported selected pathway: {selected_pathway}")
        pathway_layers = self._pathway_layers[selected_pathway]
        if selected_layer not in pathway_layers:
            raise ValueError(
                f"selected {selected_pathway} layer {selected_layer} was not instrumented"
            )
        if not recipient_prompt or not donor_prompt or recipient_prompt == donor_prompt:
            raise ValueError("causal action requires distinct donor and recipient prompts")
        if patch_kind not in {
            "none",
            "future_subspace",
            "random_subspace",
            "full_donor",
            "random_direction",
        }:
            raise ValueError(f"unsupported patch kind: {patch_kind}")
        if patch_schedule not in {"first_call", "all_calls_replay", "all_calls_delta"}:
            raise ValueError(f"unsupported patch schedule: {patch_schedule}")
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
        _, _, _, donor_all, donor_pathway_counts = self._capture(donor_obs, noise)
        _, _, _, recipient_all, recipient_pathway_counts = self._capture(
            recipient_obs, noise
        )
        donor_sequence = donor_all[selected_pathway][str(selected_layer)]
        recipient_sequence = recipient_all[selected_pathway][str(selected_layer)]
        if len(donor_sequence) != len(recipient_sequence):
            raise RuntimeError(
                f"donor and recipient {selected_pathway} call counts differ: "
                f"{len(donor_sequence)} != {len(recipient_sequence)}"
            )
        donor = donor_sequence[0][0]
        recipient = recipient_sequence[0][0]
        difference = (donor - recipient).reshape(-1).astype(np.float64)
        future_delta: np.ndarray | None = None
        if patch_kind == "future_subspace":
            future_basis = np.asarray(obs["__future_basis"], dtype=np.float64)
            if future_basis.ndim != 2 or future_basis.shape[0] != len(difference):
                raise ValueError("future basis does not match the selected activation grid")
            future_delta = future_basis @ (future_basis.T @ difference)
            delta = future_delta
        elif patch_kind in {"full_donor", "random_direction"}:
            delta = difference
        else:
            future_basis = np.asarray(obs["__future_basis"], dtype=np.float64)
            if future_basis.ndim != 2 or future_basis.shape[0] != len(difference):
                raise ValueError("future basis does not match the selected activation grid")
            future_delta = future_basis @ (future_basis.T @ difference)
            random_basis = np.asarray(obs["__random_basis"], dtype=np.float64)
            if random_basis.shape != future_basis.shape:
                raise ValueError("random basis must match the future basis")
            random_delta = random_basis @ (random_basis.T @ difference)
            random_norm = float(np.linalg.norm(random_delta))
            future_norm = float(np.linalg.norm(future_delta))
            if random_norm <= 1e-12 or future_norm <= 1e-12:
                raise RuntimeError("cannot norm-match a degenerate subspace intervention")
            delta = random_delta * (future_norm / random_norm)

        layer_module = pathway_layers[selected_layer]
        if patch_schedule == "all_calls_replay":
            if patch_kind != "full_donor":
                raise ValueError("all_calls_replay is defined only for full_donor")
            with ReplayCallsPatch(layer_module, donor_sequence) as patch:
                result = self._base.infer(recipient_obs, noise=noise)
        elif patch_schedule == "all_calls_delta":
            if patch_kind == "random_direction":
                random_seed = int(obs["__random_direction_seed"])
                rng = np.random.default_rng(random_seed)
                call_deltas = []
                for donor_call, recipient_call in zip(donor_sequence, recipient_sequence):
                    call_difference = donor_call.astype(np.float64) - recipient_call.astype(np.float64)
                    random_delta = rng.normal(size=call_difference.shape)
                    random_norm = float(np.linalg.norm(random_delta))
                    difference_norm = float(np.linalg.norm(call_difference))
                    if random_norm <= 1e-12 or difference_norm <= 1e-12:
                        raise RuntimeError("cannot norm-match a degenerate all-call random direction")
                    call_deltas.append(
                        (random_delta * (difference_norm / random_norm)).astype(np.float32)
                    )
            elif patch_kind == "full_donor":
                call_deltas = [
                    (donor_call - recipient_call).astype(np.float32)
                    for donor_call, recipient_call in zip(donor_sequence, recipient_sequence)
                ]
            else:
                if future_delta is None:
                    raise RuntimeError("all-call learned patch requires a fitted future delta")
                first_grid_delta = delta.reshape(recipient.shape)[None, ...].astype(np.float32)
                call_deltas = [first_grid_delta.copy() for _ in recipient_sequence]
            with AllCallsDeltaPatch(layer_module, call_deltas) as patch:
                result = self._base.infer(recipient_obs, noise=noise)
        else:
            grid_delta = delta.reshape(recipient.shape)[None, ...].astype(np.float32)
            with FirstCallDeltaPatch(layer_module, grid_delta) as patch:
                result = self._base.infer(recipient_obs, noise=noise)
            if patch.receipt.calls_seen != len(recipient_sequence):
                raise RuntimeError(
                    f"first-call patch observed {patch.receipt.calls_seen} calls, "
                    f"expected {len(recipient_sequence)}"
                )

        patch_receipt = {
            "calls_seen": patch.receipt.calls_seen,
            "calls_patched": patch.receipt.calls_patched,
            "shape_mismatches": int(getattr(patch.receipt, "shape_mismatches", 0)),
            "original_norm": getattr(patch.receipt, "original_norm", None),
            "patch_norm": getattr(patch.receipt, "patch_norm", None),
            "original_norms": getattr(patch.receipt, "original_norms", None),
            "source_norms": getattr(patch.receipt, "source_norms", None),
            "patch_norms": getattr(patch.receipt, "patch_norms", None),
            "selected_pathway": selected_pathway,
            "selected_layer": selected_layer,
            "patch_schedule": patch_schedule,
        }
        return {
            "actions": np.asarray(result["actions"], dtype=np.float32),
            "patch_kind": patch_kind,
            "noise_seed": noise_seed,
            "causal_infer_ms": (time.monotonic() - started) * 1000.0,
            "donor_calls": donor_pathway_counts[selected_pathway][str(selected_layer)],
            "recipient_calls": recipient_pathway_counts[selected_pathway][str(selected_layer)],
            "patch_receipt": {
                **patch_receipt,
                "future_projection_norm": (
                    float(np.linalg.norm(future_delta)) if future_delta is not None else None
                ),
                "full_difference_norm": float(np.linalg.norm(difference)),
            },
        }
