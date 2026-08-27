"""Activation patching primitives for the CFS-2 retention study.

The functions in this module operate on the pinned NextLat *inner* transformer
(``wrapper.model``).  They intentionally use ordinary PyTorch forward hooks so
the upstream model does not need to be edited.  A patch changes one token state
at one transformer block and lets the unchanged downstream computation produce
the counterfactual logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PATCH_POSITION = 63
DEFAULT_PATCH_LAYERS = (3, 7, 10)
PATCH_EFFECT_NAMES = (
    "patch_parent_state_effect",
    "patch_unrelated_anchor_effect",
    "patch_norm_matched_random_subspace_effect",
)


class CFS2PatchingError(RuntimeError):
    """The model, tensors, or intervention do not satisfy the patch contract."""


@dataclass(frozen=True)
class CapturedStates:
    """Decision-position states and logits captured in one model pass."""

    states: Mapping[int, Any]
    logits: Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on GPU runtime
        raise CFS2PatchingError("activation patching requires PyTorch") from exc
    return torch


def transformer_blocks(model: Any) -> tuple[Any, ...]:
    stack = getattr(model, "transformer", None)
    blocks = tuple(getattr(stack, "blocks", ()))
    if not blocks:
        raise CFS2PatchingError("model does not expose transformer.blocks")
    return blocks


def validate_patch_layers(model: Any, layers: Iterable[int]) -> tuple[int, ...]:
    blocks = transformer_blocks(model)
    selected = tuple(int(layer) for layer in layers)
    if not selected or len(set(selected)) != len(selected):
        raise CFS2PatchingError("patch layers must be nonempty and unique")
    if any(layer < 0 or layer >= len(blocks) for layer in selected):
        raise CFS2PatchingError(
            f"patch layer outside 0..{len(blocks) - 1}: {selected}"
        )
    return selected


def _decision_slice(value: Any, *, position: int) -> Any:
    tensor = value[0] if isinstance(value, (tuple, list)) else value
    if getattr(tensor, "ndim", None) != 3 or position < 0 or position >= tensor.shape[1]:
        raise CFS2PatchingError("block output is not a B x T x D decision-state tensor")
    return tensor[:, position, :]


def capture_states_and_logits(
    model: Any,
    tokens: Any,
    *,
    layers: Sequence[int] = DEFAULT_PATCH_LAYERS,
    position: int = PATCH_POSITION,
) -> CapturedStates:
    """Run one inference pass and capture block outputs at ``position``."""

    torch = _torch()
    selected = validate_patch_layers(model, layers)
    captured: dict[int, Any] = {}
    handles = []

    def hook(layer: int):
        def capture(_module, _inputs, output):
            captured[layer] = _decision_slice(output, position=position).detach().clone()

        return capture

    blocks = transformer_blocks(model)
    for layer in selected:
        handles.append(blocks[layer].register_forward_hook(hook(layer)))
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(tokens)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    if set(captured) != set(selected):
        raise CFS2PatchingError("not every requested transformer state was captured")
    if getattr(logits, "ndim", None) != 3 or position >= logits.shape[1]:
        raise CFS2PatchingError("model did not return B x T x V logits")
    return CapturedStates(captured, logits[:, position, :].detach().clone())


def forward_with_patch(
    model: Any,
    tokens: Any,
    replacement: Any,
    *,
    layer: int,
    position: int = PATCH_POSITION,
) -> Any:
    """Replace one block's decision state and return downstream decision logits."""

    torch = _torch()
    selected = validate_patch_layers(model, (layer,))
    target = transformer_blocks(model)[selected[0]]

    def patch(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if tensor.ndim != 3 or position < 0 or position >= tensor.shape[1]:
            raise CFS2PatchingError("cannot patch a non-B x T x D block output")
        value = replacement.to(device=tensor.device, dtype=tensor.dtype)
        if value.shape != (tensor.shape[0], tensor.shape[2]):
            raise CFS2PatchingError(
                f"replacement must be {(tensor.shape[0], tensor.shape[2])}, got {tuple(value.shape)}"
            )
        changed = tensor.clone()
        changed[:, position, :] = value
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed

    handle = target.register_forward_hook(patch)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(tokens)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if getattr(logits, "ndim", None) != 3 or position >= logits.shape[1]:
        raise CFS2PatchingError("patched model did not return B x T x V logits")
    return logits[:, position, :].detach().clone()


def fixed_derangement(size: int, seed: int) -> np.ndarray:
    """A deterministic no-fixed-point permutation for unrelated donors."""

    if size < 2:
        raise CFS2PatchingError("unrelated-anchor control requires at least two probes")
    rng = np.random.default_rng(int(seed))
    # Sattolo's algorithm samples a single-cycle permutation, which guarantees
    # that every probe receives a different donor without imposing a fixed
    # cyclic offset on an otherwise meaningful manifest order.
    result = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        donor = int(rng.integers(0, index))
        result[index], result[donor] = result[donor], result[index]
    if np.any(result == np.arange(size)):
        raise CFS2PatchingError("failed to construct an unrelated-anchor derangement")
    return result


def norm_matched_random_replacement(
    adapted: np.ndarray,
    parent: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Add an isotropic random direction with the real patch-delta norm.

    Each row spans a random one-dimensional subspace.  The magnitude is exactly
    ``||parent - adapted||`` for that row, so recovery cannot be attributed only
    to injecting a perturbation of a particular size.
    """

    adapted_value = np.asarray(adapted)
    parent_value = np.asarray(parent)
    if adapted_value.shape != parent_value.shape or adapted_value.ndim != 2:
        raise CFS2PatchingError("adapted and parent states must be matching N x D arrays")
    work_adapted = adapted_value.astype(np.float64, copy=False)
    delta = parent_value.astype(np.float64, copy=False) - work_adapted
    target_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(delta.shape)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction_norm = np.maximum(direction_norm, np.finfo(np.float64).tiny)
    replacement = work_adapted + direction * (target_norm / direction_norm)
    return replacement.astype(adapted_value.dtype, copy=False)


def tensor_to_numpy(value: Any) -> np.ndarray:
    tensor = value.detach().float().cpu()
    try:
        return tensor.numpy()
    except RuntimeError:  # old torch wheels built against NumPy 1.x
        return np.asarray(tensor.tolist(), dtype=np.float32)


def numpy_to_tensor(value: np.ndarray, *, device: Any) -> Any:
    torch = _torch()
    return torch.as_tensor(np.asarray(value), device=device)
