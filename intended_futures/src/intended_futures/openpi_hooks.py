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


def resolve_paligemma_layer(model: Any, layer_index: int) -> Any:
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"PaliGemma layer {layer_index} outside [0, {len(layers)})")
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


@dataclass
class MultiCallPatchReceipt:
    calls_seen: int = 0
    calls_patched: int = 0
    shape_mismatches: int = 0
    original_norms: list[float] | None = None
    source_norms: list[float] | None = None
    patch_norms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.original_norms is None:
            self.original_norms = []
        if self.source_norms is None:
            self.source_norms = []
        if self.patch_norms is None:
            self.patch_norms = []


class ReplayCallsPatch(AbstractContextManager["ReplayCallsPatch"]):
    """Sequentially replace every layer call with its matched source activation.

    This follows the capture/replay convention used by Action Atlas rather than
    broadcasting a single first-call activation across the flow-denoising loop.
    """

    def __init__(self, module: Any, source_calls: list[np.ndarray]):
        if not source_calls:
            raise ValueError("source_calls must be nonempty")
        self.module = module
        self.source_calls = [np.asarray(call, dtype=np.float32) for call in source_calls]
        self.receipt = MultiCallPatchReceipt()
        self._handle: Any = None

    def __enter__(self) -> "ReplayCallsPatch":
        import torch

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            call_index = self.receipt.calls_seen
            self.receipt.calls_seen += 1
            if call_index >= len(self.source_calls):
                return None
            recipient = _activation_from_output(output)
            source = torch.as_tensor(
                self.source_calls[call_index], device=recipient.device, dtype=recipient.dtype
            )
            if source.shape != recipient.shape:
                self.receipt.shape_mismatches += 1
                return None
            delta = source - recipient
            self.receipt.calls_patched += 1
            assert self.receipt.original_norms is not None
            assert self.receipt.source_norms is not None
            assert self.receipt.patch_norms is not None
            self.receipt.original_norms.append(
                float(torch.linalg.vector_norm(recipient.float()).detach().cpu())
            )
            self.receipt.source_norms.append(
                float(torch.linalg.vector_norm(source.float()).detach().cpu())
            )
            self.receipt.patch_norms.append(
                float(torch.linalg.vector_norm(delta.float()).detach().cpu())
            )
            return _replace_activation(output, source)

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        if exc_type is None:
            expected = len(self.source_calls)
            if (
                self.receipt.calls_seen != expected
                or self.receipt.calls_patched != expected
                or self.receipt.shape_mismatches
            ):
                raise RuntimeError(
                    "sequential activation replay did not patch every matching call: "
                    f"seen={self.receipt.calls_seen}, patched={self.receipt.calls_patched}, "
                    f"expected={expected}, mismatches={self.receipt.shape_mismatches}"
                )


class AllCallsDeltaPatch(AbstractContextManager["AllCallsDeltaPatch"]):
    """Add one precomputed, call-matched delta at every layer invocation."""

    def __init__(self, module: Any, deltas: list[np.ndarray]):
        if not deltas:
            raise ValueError("deltas must be nonempty")
        self.module = module
        self.deltas = [np.asarray(delta, dtype=np.float32) for delta in deltas]
        self.receipt = MultiCallPatchReceipt()
        self._handle: Any = None

    def __enter__(self) -> "AllCallsDeltaPatch":
        import torch

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            call_index = self.receipt.calls_seen
            self.receipt.calls_seen += 1
            if call_index >= len(self.deltas):
                return None
            recipient = _activation_from_output(output)
            delta = torch.as_tensor(
                self.deltas[call_index], device=recipient.device, dtype=recipient.dtype
            )
            if delta.shape != recipient.shape:
                self.receipt.shape_mismatches += 1
                return None
            replacement = recipient + delta
            self.receipt.calls_patched += 1
            assert self.receipt.original_norms is not None
            assert self.receipt.source_norms is not None
            assert self.receipt.patch_norms is not None
            self.receipt.original_norms.append(
                float(torch.linalg.vector_norm(recipient.float()).detach().cpu())
            )
            self.receipt.source_norms.append(
                float(torch.linalg.vector_norm(replacement.float()).detach().cpu())
            )
            self.receipt.patch_norms.append(
                float(torch.linalg.vector_norm(delta.float()).detach().cpu())
            )
            return _replace_activation(output, replacement)

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        if exc_type is None:
            expected = len(self.deltas)
            if (
                self.receipt.calls_seen != expected
                or self.receipt.calls_patched != expected
                or self.receipt.shape_mismatches
            ):
                raise RuntimeError(
                    "all-call delta intervention did not patch every matching call: "
                    f"seen={self.receipt.calls_seen}, patched={self.receipt.calls_patched}, "
                    f"expected={expected}, mismatches={self.receipt.shape_mismatches}"
                )


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


class FirstCallDeltaPatch(AbstractContextManager["FirstCallDeltaPatch"]):
    """Add a precomputed token-preserving delta to exactly the first layer call."""

    def __init__(self, module: Any, delta: np.ndarray):
        self.module = module
        self.delta = np.asarray(delta, dtype=np.float32)
        self.receipt = PatchReceipt()
        self._handle: Any = None

    def __enter__(self) -> "FirstCallDeltaPatch":
        import torch

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            self.receipt.calls_seen += 1
            if self.receipt.calls_patched:
                return None
            recipient = _activation_from_output(output)
            delta = torch.as_tensor(self.delta, device=recipient.device, dtype=recipient.dtype)
            if delta.shape != recipient.shape:
                raise ValueError(f"delta shape {tuple(delta.shape)} != activation {tuple(recipient.shape)}")
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
