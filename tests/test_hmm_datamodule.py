from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hmm_geometry import datamodule as dm


class _FakeDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs


class _FakeTorch:
    long = np.int64
    utils = SimpleNamespace(data=SimpleNamespace(DataLoader=_FakeDataLoader))

    @staticmethod
    def tensor(values, dtype):
        return np.asarray(values, dtype=dtype)


class _Fabric:
    def setup_dataloaders(self, dataloader, **kwargs):
        dataloader.setup_kwargs = kwargs
        return dataloader


def _config(tmp_path, *, generalization=True):
    train = np.array([[0, 1, 2], [3, 2, 1], [1, 0, 3], [2, 3, 0]], dtype=np.int8)
    val = np.array([[1, 2, 3], [0, 0, 1]], dtype=np.int8)
    lengen = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int8)
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    lengen_path = tmp_path / "lengen.npy"
    np.save(train_path, train)
    np.save(val_path, val)
    np.save(lengen_path, lengen)
    return SimpleNamespace(
        data=SimpleNamespace(
            device_batch_size=2,
            num_workers=0,
            test_generalization=generalization,
            hmm_n_obs=4,
            hmm_train_data_path=str(train_path),
            hmm_val_data_path=str(val_path),
            hmm_generalization_data_path=[str(lengen_path)],
        ),
        model=SimpleNamespace(vocab_size=0, context_length=-1, block_size=1024),
    )


def test_tokenizer_appends_one_eos_and_rejects_invalid_symbols():
    tokenizer = dm.HMMTokenizer(4)
    assert tokenizer.eos_token_id == 4
    assert tokenizer.vocab_size == 5
    assert tokenizer.tokenize([3, 0, 2]).tolist() == [3, 0, 2, 4]
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        tokenizer.tokenize([0, 4])


def test_datamodule_contract_sets_model_shape_and_loader_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "_torch_module", lambda: _FakeTorch)
    config = _config(tmp_path)
    module = dm.HMMBeliefDataModule(_Fabric(), config)
    module.update_config(config)

    assert config.model.vocab_size == 5
    assert config.model.context_length == 0
    assert config.model.block_size == 5  # length-64 analogue (4 here) plus EOS
    assert module.train_dataset[0].tolist() == [0, 1, 2, 4]

    train_loader = module.train_dataloader()
    assert train_loader.kwargs == {
        "batch_size": 2,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 0,
    }
    assert train_loader.setup_kwargs == {"use_distributed_sampler": True}
    generalization = module.generalization_dataloader()
    assert len(generalization) == 1
    assert generalization[0].kwargs["shuffle"] is False


def test_datamodule_rejects_obsolete_nested_config_shape(tmp_path):
    config = SimpleNamespace(
        data=SimpleNamespace(device_batch_size=2, hmm=SimpleNamespace(hmm_n_obs=4)),
        model=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match=r"obsolete nested data\.hmm.*flat data\.hmm_\*"):
        dm.HMMBeliefDataModule(_Fabric(), config)


def test_datamodule_rejects_npz_and_out_of_vocabulary_data(tmp_path):
    npz_path = tmp_path / "stale.npz"
    np.savez(npz_path, observations=np.zeros((2, 3), dtype=np.int8))
    with pytest.raises(ValueError, match=r"raw 2-D \.npy"):
        dm._load_observations(npz_path, n_obs=4, split="train")

    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, np.array([[0, 4]], dtype=np.int8))
    with pytest.raises(ValueError, match=r"symbols must be in \[0, 3\]"):
        dm._load_observations(bad_path, n_obs=4, split="train")
