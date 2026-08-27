from __future__ import annotations

from types import SimpleNamespace

import pytest

import train_hmm
from hmm_geometry.datamodule import HMMBeliefDataModule


def test_registration_is_in_memory_only():
    upstream_train = SimpleNamespace(DATAMODULES={"stargraph": object()})
    train_hmm.register_hmm(upstream_train)
    assert upstream_train.DATAMODULES["hmm_belief"] is HMMBeliefDataModule


def test_upstream_layout_makes_working_directory_contract_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py and defaults.yaml"):
        train_hmm._upstream_layout(tmp_path)

    (tmp_path / "train.py").write_text("# pinned")
    (tmp_path / "defaults.yaml").write_text("# defaults")
    root, defaults = train_hmm._upstream_layout(tmp_path)
    assert root == tmp_path.resolve()
    assert defaults == root / "defaults.yaml"


def test_main_registers_then_delegates_all_supported_flags(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text("# pinned")
    (tmp_path / "defaults.yaml").write_text("# defaults")
    config_path = tmp_path / "hmm.yaml"
    config_path.write_text("data: {}")
    calls = []
    upstream_train = SimpleNamespace(DATAMODULES={})

    monkeypatch.setattr(train_hmm, "_import_upstream_train", lambda root: upstream_train)
    monkeypatch.setattr(
        train_hmm,
        "_load_merged_config",
        lambda config, defaults, overrides: calls.append((config, defaults, overrides)) or "CFG",
    )
    upstream_train.do_train = lambda config, **kwargs: calls.append((config, kwargs))

    assert train_hmm.main([
        "--upstream-root", str(tmp_path),
        "--config", str(config_path),
        "--no_pbar",
        "--shard",
        "--checkpoint_path", "/tmp/model.pt",
        "seed=1235",
    ]) == 0
    assert upstream_train.DATAMODULES["hmm_belief"] is HMMBeliefDataModule
    assert calls[0][2] == ["seed=1235"]
    assert calls[1] == (
        "CFG",
        {
            "hide_progress_bar": True,
            "use_sharding": True,
            "checkpoint_path": "/tmp/model.pt",
        },
    )
