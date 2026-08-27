"""Common confirmatory H3 adaptation objective.

The upstream GPT and NextLat wrappers already expose teacher-forced next-token cross entropy.
BST does not: its normal :meth:`compute_loss` enumerates dense forward/backward prefix-suffix
pairs.  H3 is intended to compare what three *already-trained* representations do under one
common intervention, so adaptation must not continue three different pretraining objectives.

This module is copied, byte-for-byte and SHA-bound, into the pinned runtime by
``scripts/runtime_bootstrap.py``.  It changes no base-training behavior.  For a job whose name
contains ``-adapt-`` it:

* leaves GPT on its native next-token loss;
* requires all NextLat auxiliary coefficients to be exactly zero; and
* replaces only BST's instance-level ``compute_loss`` with next-token cross entropy through
  the exact generation-time forward encoder and ``TextHead`` path.  The backward input is a
  lone EOS for every item, exactly as ``BST.generate`` constructs it.

All model parameters remain trainable.  In particular the constant lone-EOS backward state is
not detached, so gradients can update the backward encoder as part of the full-parameter
estimand even though its *input* is item-independent.
"""

from __future__ import annotations

import hashlib
import pathlib
import types
from contextlib import ExitStack
from typing import Any

import torch
import torch.nn.functional as F


CONTRACT_NAME = "h3_full_parameter_next_token_ce_v1"


class AdaptationContractError(RuntimeError):
    """An H3 job does not satisfy the frozen common-objective contract."""


def contract_sha256(path: str | pathlib.Path | None = None) -> str:
    """Hash the exact trainer implementation installed in the runtime."""
    source = pathlib.Path(path) if path is not None else pathlib.Path(__file__)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def is_adaptation_job(config: Any) -> bool:
    return "-adapt-" in str(config.trainer.experiment_name)


def _forward_positions(encoder: Any, batch: torch.Tensor) -> torch.Tensor:
    positions = encoder.create_position_indices(batch)
    return positions[0] if isinstance(positions, tuple) else positions


def _masked_targets(model: Any, batch: torch.Tensor) -> torch.Tensor:
    targets = batch[:, 1:].clone()
    context_length = int(getattr(model.config, "context_length", 0))
    if context_length > 0:
        positions = _forward_positions(model.encoder, batch)
        context_mask = (positions <= context_length + 1) & (positions != 0)
        targets = targets.masked_fill(context_mask[:, 1:], -100)
    if not torch.any(targets != -100):
        raise AdaptationContractError("next-token adaptation batch has no scored targets")
    return targets


def bst_lone_eos_backward_state(model: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return BST's differentiable, item-independent inference-time backward state."""
    eos = torch.full(
        (int(batch_size), 1), int(model.config.eos_token_id), dtype=torch.long, device=device
    )
    _, backward = model.encoder(eos, compute_forward=False, compute_backward=True)
    if backward is None or backward.shape[:2] != (batch_size, 1):
        raise AdaptationContractError("BST lone-EOS backward encoder returned an invalid shape")
    return backward


def bst_next_token_logits(model: Any, prefixes: torch.Tensor) -> torch.Tensor:
    """Logits from the same forward-encoder/TextHead route as ``BST.generate``.

    The return shape is ``(batch, sequence, vocabulary)``.  Upstream ``TextHead`` stacks
    next/previous at dimension 1, so its raw shape is ``(batch, 2, sequence, vocabulary)``.
    The final sequence position is
    exactly the one-step generation logit for the supplied prefix.
    """
    if prefixes.ndim != 2 or prefixes.shape[1] < 1:
        raise AdaptationContractError("BST prefixes must have shape (batch, nonempty sequence)")
    forward, _ = model.encoder(prefixes, compute_forward=True, compute_backward=False)
    backward = bst_lone_eos_backward_state(model, prefixes.shape[0], prefixes.device)
    backward = backward.expand(-1, forward.shape[1], -1)
    next_previous = model.text_head(forward, backward)
    if next_previous.ndim != 4 or next_previous.shape[1] != 2:
        raise AdaptationContractError("BST TextHead did not return next/previous logits")
    return next_previous[:, 0, :, :]


def bst_teacher_forced_next_token_loss(
    model: Any,
    batch: torch.Tensor,
    *,
    backpropagate: bool,
    no_sync: bool = False,
    loss_div: int = 1,
) -> dict[str, torch.Tensor]:
    """Full-parameter next-token CE for a BST wrapper already set up with Fabric."""
    model._assert_fabric_is_setup()
    if batch.ndim != 2 or batch.shape[1] < 2:
        raise AdaptationContractError("next-token adaptation needs at least two tokens")
    if isinstance(loss_div, bool) or int(loss_div) <= 0:
        raise AdaptationContractError("loss_div must be a positive integer")

    inputs = batch[:, :-1]
    targets = _masked_targets(model, batch)
    # H3 is preregistered for one device.  Use both Fabric module contexts so the function
    # remains correct if the wrapped implementation later adds synchronization hooks.
    with ExitStack() as stack:
        stack.enter_context(model.fabric.no_backward_sync(model.encoder, no_sync))
        stack.enter_context(model.fabric.no_backward_sync(model.text_head, no_sync))
        logits = bst_next_token_logits(model, inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
        ) / int(loss_div)
        if not torch.isfinite(loss):
            raise AdaptationContractError("BST next-token adaptation loss is nonfinite")
        if backpropagate:
            model.fabric.backward(loss)
    detached = loss.detach()
    return {"loss": detached, "next_token_loss": detached}


def _bst_compute_loss(self: Any, batch: torch.Tensor, backpropagate: bool, no_sync: bool = False,
                      loss_div: int = 1, **_: Any) -> dict[str, torch.Tensor]:
    # ``pair_batch_size`` is accepted in **_ solely because the pinned Trainer passes it to
    # every model family.  It is deliberately unused: invoking pair generation is forbidden.
    self._lurestar_adaptation_loss_calls += 1
    return bst_teacher_forced_next_token_loss(
        self, batch, backpropagate=backpropagate, no_sync=no_sync, loss_div=loss_div
    )


def install_common_adaptation(config: Any, model: Any, fabric: Any) -> dict[str, Any] | None:
    """Install/validate the frozen H3 objective after checkpoint loading.

    Returns a serializable contract record for adaptation jobs and ``None`` for base jobs.
    Re-installation is idempotent and refuses a model-family/configuration disagreement.
    """
    if not is_adaptation_job(config):
        return None
    if str(config.data.dataset) != "stargraph":
        raise AdaptationContractError("the common H3 adaptation objective is stargraph-only")

    use_bst = bool(config.use_bst)
    use_nextlat = bool(config.use_nextlat)
    family = "bst" if use_bst else ("nextlat" if use_nextlat else "gpt")
    if use_bst and use_nextlat:
        raise AdaptationContractError("BST and NextLat model flags cannot both be enabled")

    coefficients = {
        key: float(getattr(config.model, key, 0.0))
        for key in ("lambda_mse", "lambda_kl", "lambda_ce")
    }
    if family == "nextlat" and any(value != 0.0 for value in coefficients.values()):
        raise AdaptationContractError(
            "NextLat H3 adaptation requires lambda_mse=lambda_kl=lambda_ce=0"
        )

    if family == "bst":
        if getattr(model, "_lurestar_adaptation_contract", None) not in (None, CONTRACT_NAME):
            raise AdaptationContractError("BST already carries a different adaptation contract")
        if getattr(model, "_lurestar_dense_compute_loss", None) is None:
            model._lurestar_dense_compute_loss = model.compute_loss
            model._lurestar_adaptation_loss_calls = 0
            model.compute_loss = types.MethodType(_bst_compute_loss, model)
        # A poisoned/changed instance method must never be accepted as installed.
        bound = getattr(model.compute_loss, "__func__", None)
        if bound is not _bst_compute_loss:
            raise AdaptationContractError("BST dense objective was not replaced")
        model._lurestar_adaptation_contract = CONTRACT_NAME

    record = {
        "schema": "nextlat_forgetting/adaptation_trainer/1",
        "contract": CONTRACT_NAME,
        "contract_sha256": contract_sha256(),
        "family": family,
        "full_parameter": True,
        "loss": "teacher_forced_next_token_cross_entropy",
        "bst_backward_input": "item_independent_lone_eos" if family == "bst" else None,
        "bst_dense_prefix_suffix_objective": False if family == "bst" else None,
        "nextlat_auxiliary_coefficients": coefficients if family == "nextlat" else None,
    }
    fabric.print("LURESTAR_ADAPTATION=" + __import__("json").dumps(record, sort_keys=True))
    return record
