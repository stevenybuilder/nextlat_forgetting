"""Streaming exact autograd controls for Lure-Star H3 (torch imported lazily).

The upstream-specific extractor owns tokenization/model loading and must construct the two exact
losses named in :mod:`lurestar.evaluate`.  This module owns the easy-to-get-wrong autograd part:
one bank-mean adaptation gradient, one exact item gradient at a time, unused parameters represented
as zeros, and signed dot/cosine values accumulated without materializing an ``items × parameters``
matrix.  The same primitive can compute scalar-output Jacobian/empirical-NTK overlaps.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .evaluate import EXACT_ADAPTATION_LOSS, EXACT_ITEM_LOSS

__all__ = ["exact_loss_gradient_controls", "exact_scalar_jacobian_controls"]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the CPU analysis host
        raise RuntimeError("exact gradient controls require torch on the evaluation runtime") from exc
    return torch


def _parameters(parameters: Iterable) -> tuple:
    params = tuple(p for p in parameters if getattr(p, "requires_grad", False))
    if not params:
        raise ValueError("no trainable parameters were supplied")
    return params


def _grads(scalar, params: tuple, *, retain_graph: bool) -> tuple:
    torch = _torch()
    if getattr(scalar, "ndim", None) != 0:
        raise ValueError("each exact loss/output must be a scalar tensor")
    raw = torch.autograd.grad(
        scalar, params, retain_graph=retain_graph, create_graph=False, allow_unused=True
    )
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, raw))


def _alignment(item_scalars: Sequence, reference_scalar, parameters: Iterable) -> dict:
    torch = _torch()
    params = _parameters(parameters)
    if not item_scalars:
        raise ValueError("at least one item scalar is required")
    reference = _grads(reference_scalar, params, retain_graph=True)
    ref_norm2 = sum(torch.sum(g.double() * g.double()) for g in reference)
    dots, item_norms = [], []
    for index, scalar in enumerate(item_scalars):
        item = _grads(scalar, params, retain_graph=index < len(item_scalars) - 1)
        dot = sum(torch.sum(g.double() * r.double()) for g, r in zip(item, reference))
        norm2 = sum(torch.sum(g.double() * g.double()) for g in item)
        dots.append(float(dot.cpu()))
        item_norms.append(float(torch.sqrt(norm2).cpu()))
    ref_norm = float(torch.sqrt(ref_norm2).cpu())
    dot = np.asarray(dots, dtype=np.float64)
    item_norm = np.asarray(item_norms, dtype=np.float64)
    denom = item_norm * ref_norm
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-30)
    return {
        "dot": dot,
        "cosine": cosine,
        "item_norm": item_norm,
        "reference_norm": np.full(dot.shape, ref_norm),
        "parameter_count": int(sum(p.numel() for p in params)),
        "parameter_order": "caller tuple order; must be identical for item and reference",
        "precision": "dot products accumulated in float64",
    }


def exact_loss_gradient_controls(
    item_losses: Sequence,
    first_effective_adaptation_batch_mean_loss,
    parameters: Iterable,
    *,
    item_loss_name: str,
    adaptation_loss_name: str,
) -> dict:
    """Signed exact gradient alignment for the frozen H3 item and adaptation CE losses."""
    if item_loss_name != EXACT_ITEM_LOSS or adaptation_loss_name != EXACT_ADAPTATION_LOSS:
        raise ValueError("loss tensors do not carry the frozen confirmatory loss identities")
    out = _alignment(item_losses, first_effective_adaptation_batch_mean_loss, parameters)
    return {
        "gradient_dot": out.pop("dot"),
        "gradient_cosine": out.pop("cosine"),
        "item_gradient_norm": out.pop("item_norm"),
        "adaptation_gradient_norm": out.pop("reference_norm"),
        "item_loss": item_loss_name,
        "adaptation_loss": adaptation_loss_name,
        **out,
    }


def exact_scalar_jacobian_controls(
    item_outputs: Sequence,
    adaptation_bank_mean_output,
    parameters: Iterable,
    *,
    item_target: str = "correct_first_branch_logit_margin@index63",
    adaptation_target: str = "mean_correct_next_token_logit_on_adaptation_bank",
) -> dict:
    """Exact scalar-output NTK diagnostic.

    This helper is useful for validation, but is *not* the production H3 ``Jov`` estimand, which
    requires 16 shared Rademacher projections of the index-63 hidden state (seed 20260824).
    The production extractor must implement that vector-state sketch and may not relabel this
    scalar diagnostic as ``jacobian_overlap``.
    """
    out = _alignment(item_outputs, adaptation_bank_mean_output, parameters)
    return {
        "ntk_dot": out.pop("dot"),
        "ntk_cosine": out.pop("cosine"),
        "item_jacobian_norm": out.pop("item_norm"),
        "adaptation_jacobian_norm": out.pop("reference_norm"),
        "item_target": item_target,
        "adaptation_target": adaptation_target,
        "role": "preregistered diagnostic; exact loss-gradient alignment remains primary",
        **out,
    }
