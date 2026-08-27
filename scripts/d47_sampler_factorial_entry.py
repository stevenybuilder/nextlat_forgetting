#!/usr/bin/env python3
"""Run public NextLat with an explicit, independently seeded Path-Star sampler.

This diagnostic wrapper leaves the pinned upstream tree untouched.  It replaces
only the Path-Star training dataloader so that model initialization and example
order can be varied independently.  It also makes the epoch behavior explicit:

* ``fixed`` repeats the same distributed permutation every dataloader epoch;
* ``reshuffle`` advances the distributed-sampler epoch after every pass.

Each rank records the exact sampled indices and a compact first-update trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import runpy
import sys
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


DETERMINISTIC = os.environ.get("D47_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


UPSTREAM = pathlib.Path(os.environ["D47_UPSTREAM"]).resolve()
TRACE_DIR = pathlib.Path(os.environ["D47_TRACE_DIR"]).resolve()
DATA_SEED = int(os.environ["D47_DATA_SEED"])
EPOCH_MODE = os.environ["D47_EPOCH_MODE"]
if EPOCH_MODE not in {"fixed", "reshuffle"}:
    raise SystemExit(f"unsupported D47_EPOCH_MODE={EPOCH_MODE!r}")

TRACE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(UPSTREAM))

from data.stargraph import StarGraphDataModule  # noqa: E402
from models.model_nextlat import NextLat  # noqa: E402


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _model_sha256(model: NextLat, *, gradients: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.model.named_parameters():
        tensor = parameter.grad if gradients else parameter
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if tensor is None:
            digest.update(b"<none>")
        else:
            digest.update(
                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
    return digest.hexdigest()


def _gradient_parameter_records(model: NextLat) -> dict[str, dict[str, Any] | None]:
    records: dict[str, dict[str, Any] | None] = {}
    for name, parameter in model.model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            records[name] = None
            continue
        numeric = gradient.detach().float()
        records[name] = {
            "sha256": _tensor_sha256(gradient),
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype),
            "sum_fp64": float(numeric.double().sum().item()),
            "l2_fp64": float(torch.linalg.vector_norm(numeric.double()).item()),
            "max_abs_fp64": float(numeric.double().abs().max().item()),
        }
    return records


class TracedDistributedSampler(DistributedSampler):
    """DistributedSampler that records and explicitly controls epoch advance."""

    def __iter__(self) -> Iterator[int]:
        sampled_epoch = int(self.epoch)
        indices = list(super().__iter__())
        packed = torch.tensor(indices, dtype=torch.int64).numpy().tobytes()
        record = {
            "schema": "nextlat_forgetting/d47_sampler_epoch/1",
            "rank": int(self.rank),
            "world_size": int(self.num_replicas),
            "data_seed": int(self.seed),
            "epoch_mode": EPOCH_MODE,
            "sampled_epoch": sampled_epoch,
            "num_indices": len(indices),
            "indices_sha256": hashlib.sha256(packed).hexdigest(),
            "first_indices": indices[:16],
        }
        with (TRACE_DIR / f"sampler.rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if EPOCH_MODE == "reshuffle":
            self.set_epoch(sampled_epoch + 1)
        return iter(indices)


def _explicit_train_dataloader(self: StarGraphDataModule):
    sampler = TracedDistributedSampler(
        self.train_dataset,
        num_replicas=self.fabric.world_size,
        rank=self.fabric.global_rank,
        shuffle=True,
        seed=DATA_SEED,
        drop_last=False,
    )
    dataloader = DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        collate_fn=self.collate_fn,
    )
    return self.fabric.setup_dataloaders(
        dataloader, use_distributed_sampler=False
    )


StarGraphDataModule.train_dataloader = _explicit_train_dataloader

_original_compute_loss = NextLat.compute_loss


def _traced_compute_loss(self: NextLat, batch: torch.Tensor, *args: Any, **kwargs: Any):
    should_trace = bool(kwargs.get("backpropagate")) and not getattr(
        self, "_d47_sampler_factorial_traced", False
    )
    before = None
    if should_trace:
        before = {
            "schema": "nextlat_forgetting/d47_sampler_first_update/1",
            "rank": _rank(),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "data_seed": DATA_SEED,
            "epoch_mode": EPOCH_MODE,
            "deterministic_algorithms": DETERMINISTIC,
            "initial_parameters_sha256": _model_sha256(self, gradients=False),
            "local_batch_sha256": _tensor_sha256(batch),
            "local_batch_row_sha256": [_tensor_sha256(row) for row in batch],
        }
    result = _original_compute_loss(self, batch, *args, **kwargs)
    if should_trace:
        assert before is not None
        before["gradients_sha256"] = _model_sha256(self, gradients=True)
        before["gradient_parameters"] = _gradient_parameter_records(self)
        _atomic_json(TRACE_DIR / f"first_update.rank{_rank()}.json", before)
        self._d47_sampler_factorial_traced = True
    return result


NextLat.compute_loss = _traced_compute_loss

train_path = UPSTREAM / "train.py"
sys.argv[0] = str(train_path)
runpy.run_path(str(train_path), run_name="__main__")
