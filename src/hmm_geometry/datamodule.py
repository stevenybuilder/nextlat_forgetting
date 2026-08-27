"""Additive HMM dataset integration for the pinned NextLat trainer.

This module deliberately does not import from or modify ``upstream/``.  It implements the
small datamodule protocol consumed by ``train.do_train`` and is registered at runtime by
``scripts/train_hmm.py``.
"""
from __future__ import annotations

import os
import pathlib
import typing as t

import numpy as np

__all__ = ["HMMTokenizer", "HMMSequenceDataset", "HMMBeliefDataModule"]


def _torch_module():
    """Import torch only when tensors/loaders are requested.

    Dataset generation and geometry tests run on CPU-only hosts where the training stack is
    intentionally absent.  Keeping this import lazy lets those hosts still validate the on-disk
    corpus and tokenizer contract; an actual trainer invocation fails here with a direct message.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only on a misconfigured runtime
        raise RuntimeError(
            "HMM training requires PyTorch; install the pinned NextLat runtime dependencies"
        ) from exc
    return torch


def _required(data_config, key: str):
    try:
        value = getattr(data_config, key)
    except (AttributeError, KeyError) as exc:
        try:
            stale_nested = getattr(data_config, "hmm")
        except (AttributeError, KeyError):
            stale_nested = None
        hint = (
            " (the obsolete nested data.hmm block is present; move its trainer inputs "
            "to flat data.hmm_* keys)"
            if stale_nested is not None
            else ""
        )
        raise ValueError(f"missing required config key data.{key}{hint}") from exc
    if value is None or value == "":
        raise ValueError(f"config key data.{key} must not be empty")
    return value


def _load_observations(path: os.PathLike | str, *, n_obs: int, split: str) -> np.ndarray:
    resolved = pathlib.Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"HMM {split} observations not found at {resolved}; paths are interpreted "
            "relative to the trainer working directory, so deployed configs should use "
            "absolute paths"
        )
    if resolved.suffix != ".npy":
        raise ValueError(
            f"HMM {split} data must be a raw 2-D .npy observation array, got {resolved}"
        )

    observations = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if observations.ndim != 2 or observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError(
            f"HMM {split} observations must have non-empty shape (N, L), got "
            f"{observations.shape} at {resolved}"
        )
    if not np.issubdtype(observations.dtype, np.integer):
        raise ValueError(
            f"HMM {split} observations must have an integer dtype, got "
            f"{observations.dtype} at {resolved}"
        )
    observed_min = int(observations.min())
    observed_max = int(observations.max())
    if observed_min < 0 or observed_max >= n_obs:
        raise ValueError(
            f"HMM {split} symbols must be in [0, {n_obs - 1}], got "
            f"[{observed_min}, {observed_max}] at {resolved}"
        )
    return observations


class HMMTokenizer:
    """Identity tokenizer for observation symbols, with one terminal EOS token."""

    def __init__(self, n_obs: int = 4):
        if int(n_obs) <= 0:
            raise ValueError(f"n_obs must be positive, got {n_obs}")
        self.n_obs = int(n_obs)
        self.eos_token_id = self.n_obs
        self.vocab_size = self.n_obs + 1

    def tokenize(self, symbols: t.Iterable[int]) -> np.ndarray:
        values = np.asarray(symbols, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError(f"one HMM sequence must be 1-D, got shape {values.shape}")
        if values.size and (int(values.min()) < 0 or int(values.max()) >= self.n_obs):
            raise ValueError(f"HMM symbols must be in [0, {self.n_obs - 1}]")
        return np.append(values, self.eos_token_id)


class HMMSequenceDataset:
    """Memory-mapped fixed-width observation sequences exposed as EOS-terminated tensors."""

    def __init__(self, observations: np.ndarray, tokenizer: HMMTokenizer):
        self.observations = observations
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, idx: int):
        torch = _torch_module()
        # ``torch.tensor`` copies out of the read-only memmap and performs the required long cast.
        return torch.tensor(
            self.tokenizer.tokenize(self.observations[idx]), dtype=torch.long
        )


class HMMBeliefDataModule:
    """Datamodule satisfying the exact interface used by pinned NextLat ``train.py``."""

    def __init__(self, fabric, config):
        self.fabric = fabric
        self.batch_size = int(_required(config.data, "device_batch_size"))
        self.num_workers = int(getattr(config.data, "num_workers", 0))
        self.test_generalization = bool(getattr(config.data, "test_generalization", False))

        n_obs = int(_required(config.data, "hmm_n_obs"))
        self.tokenizer = HMMTokenizer(n_obs)
        train = _load_observations(
            _required(config.data, "hmm_train_data_path"), n_obs=n_obs, split="train"
        )
        val = _load_observations(
            _required(config.data, "hmm_val_data_path"), n_obs=n_obs, split="validation"
        )
        if train.shape[1] != val.shape[1]:
            raise ValueError(
                "HMM train and validation sequence lengths must match, got "
                f"{train.shape[1]} and {val.shape[1]}"
            )

        self.train_dataset = HMMSequenceDataset(train, self.tokenizer)
        self.val_dataset = HMMSequenceDataset(val, self.tokenizer)
        self.generalization_datasets: list[HMMSequenceDataset] = []
        self.total_len = int(train.shape[1]) + 1

        if self.test_generalization:
            paths = _required(config.data, "hmm_generalization_data_path")
            if isinstance(paths, (str, os.PathLike)):
                raise ValueError(
                    "data.hmm_generalization_data_path must be a list of .npy paths"
                )
            if not paths:
                raise ValueError(
                    "data.test_generalization is true but "
                    "data.hmm_generalization_data_path is empty"
                )
            for index, path in enumerate(paths):
                observations = _load_observations(
                    path, n_obs=n_obs, split=f"generalization[{index}]"
                )
                self.generalization_datasets.append(
                    HMMSequenceDataset(observations, self.tokenizer)
                )
                self.total_len = max(self.total_len, int(observations.shape[1]) + 1)

    def update_config(self, config) -> None:
        config.model.vocab_size = self.tokenizer.vocab_size
        config.model.context_length = 0
        config.model.block_size = self.total_len

    def get_tokenizer(self) -> HMMTokenizer:
        return self.tokenizer

    def _dataloader(self, dataset: HMMSequenceDataset, *, shuffle: bool):
        torch = _torch_module()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def generalization_dataloader(self) -> list:
        return [self._dataloader(dataset, shuffle=False) for dataset in self.generalization_datasets]
