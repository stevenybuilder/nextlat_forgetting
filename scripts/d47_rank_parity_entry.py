#!/usr/bin/env python3
"""Trace the first NextLat optimizer update without changing the training code.

This wrapper is intentionally diagnostic-only.  It monkey-patches the public
NextLat classes before executing the upstream ``train.py`` entry point and
records, per rank:

* the initial parameter bytes;
* the local batch and the world-wide multiset of examples;
* gradients after NextLat's two-stage backward; and
* parameters after the optimizer step.

The trace answers a narrow question: does DDP synchronize the custom NextLat
backward path, and does changing world size change the first global batch?
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import runpy
import sys
from typing import Any

import torch
import torch.distributed as dist


UPSTREAM = pathlib.Path(os.environ["D47_TRACE_UPSTREAM"]).resolve()
TRACE_ROOT = pathlib.Path(os.environ["D47_TRACE_DIR"]).resolve()
LABEL = os.environ["D47_TRACE_LABEL"]
TRACE_DIR = TRACE_ROOT / LABEL
TRACE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UPSTREAM))

from models.model_base import ModelBase  # noqa: E402
from models.model_nextlat import NextLat  # noqa: E402


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _tensor_record(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    numeric = detached.float()
    return {
        "sha256": hashlib.sha256(_bytes(detached)).hexdigest(),
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "sum_fp64": float(numeric.double().sum().item()),
        "l2_fp64": float(torch.linalg.vector_norm(numeric.double()).item()),
        "max_abs_fp64": float(numeric.double().abs().max().item()),
    }


def _parameter_records(model: NextLat, *, gradients: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in model.model.named_parameters():
        result[name] = _tensor_record(parameter.grad if gradients else parameter)
    return result


def _batch_record(batch: torch.Tensor) -> dict[str, Any]:
    rows = [hashlib.sha256(_bytes(row)).hexdigest() for row in batch]
    return {
        "shape": list(batch.shape),
        "dtype": str(batch.dtype),
        "ordered_sha256": hashlib.sha256(_bytes(batch)).hexdigest(),
        "row_sha256": rows,
    }


def _rng_record() -> dict[str, str]:
    record = {"cpu": hashlib.sha256(_bytes(torch.get_rng_state())).hexdigest()}
    if torch.cuda.is_available():
        record["cuda"] = hashlib.sha256(_bytes(torch.cuda.get_rng_state())).hexdigest()
    return record


def _gather(local: dict[str, Any]) -> list[dict[str, Any]]:
    if _world_size() == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * _world_size()
    dist.all_gather_object(gathered, local)
    return [item for item in gathered if item is not None]


def _write(name: str, payload: dict[str, Any]) -> None:
    if _rank() != 0:
        return
    destination = TRACE_DIR / name
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


_original_compute_loss = NextLat.compute_loss
_original_optimizer_step = ModelBase.optimizer_step


def _traced_compute_loss(self: NextLat, batch: torch.Tensor, *args: Any, **kwargs: Any):
    should_trace = bool(kwargs.get("backpropagate")) and not getattr(self, "_d47_backward_traced", False)
    before = None
    if should_trace:
        before = {
            "rank": _rank(),
            "world_size": _world_size(),
            "rng": _rng_record(),
            "batch": _batch_record(batch),
            "parameters": _parameter_records(self, gradients=False),
        }
    result = _original_compute_loss(self, batch, *args, **kwargs)
    if should_trace:
        local = dict(before or {})
        local["gradients"] = _parameter_records(self, gradients=True)
        ranks = _gather(local)
        world_rows = sorted(
            row_hash
            for rank_record in ranks
            for row_hash in rank_record["batch"]["row_sha256"]
        )
        _write(
            "after_backward.json",
            {
                "schema": "nextlat_forgetting/d47_rank_parity/1",
                "label": LABEL,
                "world_size": _world_size(),
                "global_batch_multiset_sha256": hashlib.sha256(
                    "\n".join(world_rows).encode("ascii")
                ).hexdigest(),
                "ranks": ranks,
            },
        )
        self._d47_backward_traced = True
    return result


def _traced_optimizer_step(self: ModelBase, *args: Any, **kwargs: Any):
    result = _original_optimizer_step(self, *args, **kwargs)
    if isinstance(self, NextLat) and not getattr(self, "_d47_optimizer_traced", False):
        local = {
            "rank": _rank(),
            "world_size": _world_size(),
            "parameters": _parameter_records(self, gradients=False),
        }
        _write(
            "after_optimizer.json",
            {
                "schema": "nextlat_forgetting/d47_rank_parity/1",
                "label": LABEL,
                "world_size": _world_size(),
                "ranks": _gather(local),
            },
        )
        self._d47_optimizer_traced = True
    return result


NextLat.compute_loss = _traced_compute_loss
ModelBase.optimizer_step = _traced_optimizer_step

train_path = UPSTREAM / "train.py"
sys.argv[0] = str(train_path)
runpy.run_path(str(train_path), run_name="__main__")
