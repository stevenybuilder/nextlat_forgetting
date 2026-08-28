from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np


def resolve_action_expert_layer(model: Any, layer_index: int) -> Any:
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"action-expert layer {layer_index} outside [0, {len(layers)})")
    return layers[layer_index]


def disable_compiled_sampling(policy: Any) -> None:
    """Replace both OpenPI's cached sampler and the model attribute with the eager callable."""

    sample = policy._sample_actions
    original = getattr(sample, "_torchdynamo_orig_callable", None)
    if original is not None:
        policy._sample_actions = original
        policy._model.sample_actions = original


def _activation_from_output(output: Any) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("PyTorch is required for OpenPI instrumentation") from error
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported action-expert layer output: {type(output)!r}")


def _replace_activation(output: Any, replacement: Any) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


class CaptureCalls(AbstractContextManager["CaptureCalls"]):
    def __init__(self, module: Any):
        self.module = module
        self.calls: list[np.ndarray] = []
        self._handle: Any = None

    def __enter__(self) -> "CaptureCalls":
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            activation = _activation_from_output(output)
            cpu_activation = activation.detach().float().cpu()
            try:
                captured = cpu_activation.numpy().copy()
            except RuntimeError as error:
                # Older PyTorch wheels can disable the NumPy bridge when imported with NumPy 2.
                if "Numpy is not available" not in str(error):
                    raise
                captured = np.asarray(cpu_activation.tolist(), dtype=np.float32)
            self.calls.append(captured)

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None


@dataclass
class PatchReceipt:
    calls_seen: int = 0
    calls_patched: int = 0
    original_norm: float | None = None
    patch_norm: float | None = None


class ProjectedFirstCallPatch(AbstractContextManager["ProjectedFirstCallPatch"]):
    def __init__(
        self,
        module: Any,
        *,
        donor_activation: np.ndarray,
        basis: np.ndarray,
        strength: float = 1.0,
    ):
        self.module = module
        self.donor_activation = np.asarray(donor_activation, dtype=np.float32)
        self.basis = np.asarray(basis, dtype=np.float32)
        self.strength = float(strength)
        self.receipt = PatchReceipt()
        self._handle: Any = None

    def __enter__(self) -> "ProjectedFirstCallPatch":
        import torch

        if self.basis.ndim != 2:
            raise ValueError("basis must be feature-by-rank")

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            self.receipt.calls_seen += 1
            if self.receipt.calls_patched:
                return None
            recipient = _activation_from_output(output)
            donor = torch.as_tensor(self.donor_activation, device=recipient.device, dtype=recipient.dtype)
            if donor.shape != recipient.shape:
                raise ValueError(f"donor shape {tuple(donor.shape)} != recipient {tuple(recipient.shape)}")
            flat = (donor - recipient).reshape(recipient.shape[0], -1)
            basis = torch.as_tensor(self.basis, device=recipient.device, dtype=recipient.dtype)
            if basis.shape[0] != flat.shape[1]:
                raise ValueError(f"basis features {basis.shape[0]} != activation features {flat.shape[1]}")
            delta = (flat @ basis @ basis.T).reshape_as(recipient) * self.strength
            replacement = recipient + delta
            self.receipt.calls_patched = 1
            self.receipt.original_norm = float(torch.linalg.vector_norm(recipient.float()).detach().cpu())
            self.receipt.patch_norm = float(torch.linalg.vector_norm(delta.float()).detach().cpu())
            return _replace_activation(output, replacement)

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        if exc_type is None and self.receipt.calls_patched != 1:
            raise RuntimeError("the action-expert intervention did not patch exactly one denoising call")
