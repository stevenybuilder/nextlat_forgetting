from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from lurestar.adaptation import (
    CONTRACT_NAME,
    bst_next_token_logits,
    contract_sha256,
    install_common_adaptation,
)


class _Fabric:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def no_backward_sync(self, _module, _enabled):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def print(self, message):
        self.messages.append(message)


class _Encoder(nn.Module):
    def __init__(self, vocab: int, width: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.forward_projection = nn.Linear(width, width, bias=False)
        self.backward_projection = nn.Linear(width, width, bias=False)

    def create_position_indices(self, batch):
        positions = torch.arange(batch.shape[1], device=batch.device).expand_as(batch)
        return positions, positions.flip(1)

    def forward(self, batch, compute_forward=True, compute_backward=True):
        embedded = self.token_embedding(batch)
        forward = self.forward_projection(embedded) if compute_forward else None
        backward = self.backward_projection(embedded) if compute_backward else None
        return forward, backward


class _TextHead(nn.Module):
    def __init__(self, width: int, vocab: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * width, 2 * vocab, bias=False)
        self.vocab = vocab

    def forward(self, forward, backward):
        logits = self.projection(torch.cat((forward, backward), -1))
        next_logits, previous_logits = logits.chunk(2, dim=-1)
        return torch.stack((next_logits, previous_logits), dim=1)


class _BST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_id=0, context_length=0)
        self.encoder = _Encoder(11, 7)
        self.text_head = _TextHead(7, 11)
        self.fabric = _Fabric()
        self.dense_calls = 0

    def _assert_fabric_is_setup(self):
        assert self.fabric is not None

    def compute_loss(self, *_args, **_kwargs):
        self.dense_calls += 1
        raise AssertionError("dense prefix-suffix BST objective was invoked")


def _config(*, family: str = "bst", name: str = "bst-seed1234-adapt-mid"):
    return SimpleNamespace(
        trainer=SimpleNamespace(experiment_name=name),
        data=SimpleNamespace(dataset="stargraph"),
        use_bst=family == "bst",
        use_nextlat=family == "nextlat",
        model=SimpleNamespace(lambda_mse=0.0, lambda_kl=0.0, lambda_ce=0.0),
    )


def _real_pinned_bst():
    """Import a tiny instance of the exact pinned BST without installing Lightning locally."""
    import torch.distributed.fsdp as fsdp
    import torch.nn.functional as functional

    if not hasattr(fsdp, "FSDPModule"):
        fsdp.FSDPModule = type("FSDPModule", (), {})
    if not hasattr(functional, "rms_norm"):
        functional.rms_norm = lambda value, _shape, weight, eps: (
            value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps) * weight
        )
    lightning = types.ModuleType("lightning")
    lightning.Fabric = object
    lightning_fabric = types.ModuleType("lightning.fabric")
    lightning_fabric.is_wrapped = lambda _module: False
    sys.modules.setdefault("lightning", lightning)
    sys.modules.setdefault("lightning.fabric", lightning_fabric)
    upstream = Path(__file__).resolve().parents[1] / "upstream" / "NextLat"
    sys.path.insert(0, str(upstream))
    from models.model_bst import BST, BSTConfig

    return BST(BSTConfig(
        block_size=16, vocab_size=17, eos_token_id=0,
        n_layer=1, n_head=1, n_embd=16, dropout=0.0,
    ))


def _bst_named_parameters(model):
    yield from ((f"encoder.{name}", value) for name, value in model.encoder.named_parameters())
    yield from ((f"text_head.{name}", value) for name, value in model.text_head.named_parameters())


def test_bst_teacher_forced_logits_are_identical_to_generation_path() -> None:
    torch.manual_seed(4)
    model = _BST()
    prefixes = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits = bst_next_token_logits(model, prefixes)[:, -1]

    # Literal one-step body of pinned BST.generate: forward last state + lone-EOS backward.
    eos = torch.zeros((2, 1), dtype=torch.long)
    forward, _ = model.encoder(prefixes, compute_forward=True, compute_backward=False)
    _, backward = model.encoder(eos, compute_forward=False, compute_backward=True)
    expected = model.text_head(forward[:, -1:], backward)[:, 0, 0, :]
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_exact_pinned_bst_logits_match_its_literal_generation_route() -> None:
    torch.manual_seed(11)
    model = _real_pinned_bst()
    model.eval()
    prefixes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    actual = bst_next_token_logits(model, prefixes)[:, -1, :]

    eos = torch.full((2, 1), model.config.eos_token_id, dtype=torch.long)
    _, backward = model.encoder(eos, compute_forward=False, compute_backward=True)
    forward, _ = model.encoder(prefixes, compute_forward=True, compute_backward=False)
    # Exact pinned BST.generate lines 803-812: TextHead stacks next at dimension 1.
    expected = model.text_head(forward[:, -1:, :], backward)[:, 0, 0, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exact_pinned_bst_common_loss_updates_both_encoders_and_texthead() -> None:
    torch.manual_seed(12)
    model = _real_pinned_bst()
    model.fabric = _Fabric()
    install_common_adaptation(_config(), model, model.fabric)
    optimizer = torch.optim.AdamW([value for _, value in _bst_named_parameters(model)], lr=1e-2)
    before = {name: value.detach().clone() for name, value in _bst_named_parameters(model)}
    batch = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 0]])
    result = model.compute_loss(batch, pair_batch_size=1, backpropagate=True)
    assert torch.isfinite(result["loss"])
    assert model._avg_valid_pairs is None and model._avg_sampled_pairs is None
    optimizer.step()
    changed = {
        name for name, value in _bst_named_parameters(model)
        if not torch.equal(before[name], value.detach())
    }
    assert any(name.startswith("encoder.transformer_f") for name in changed)
    assert any(name.startswith("encoder.transformer_b") for name in changed)
    assert any(name.startswith("text_head") for name in changed)


def test_bst_install_disables_dense_loss_and_updates_all_parameter_groups() -> None:
    torch.manual_seed(7)
    model = _BST()
    receipt = install_common_adaptation(_config(), model, model.fabric)
    assert receipt is not None
    assert receipt["contract"] == CONTRACT_NAME
    assert receipt["contract_sha256"] == contract_sha256()
    assert receipt["bst_dense_prefix_suffix_objective"] is False

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 0]])
    result = model.compute_loss(batch, pair_batch_size=1, backpropagate=True)
    assert torch.isfinite(result["loss"])
    assert model.dense_calls == 0
    assert model._lurestar_adaptation_loss_calls == 1
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    changed = {
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    }
    assert any(name.startswith("encoder.forward_projection") for name in changed)
    assert any(name.startswith("encoder.backward_projection") for name in changed)
    assert any(name.startswith("text_head") for name in changed)


def test_common_contract_preserves_base_and_rejects_nextlat_auxiliary_loss() -> None:
    model = _BST()
    assert install_common_adaptation(
        _config(name="bst-seed1234-base"), model, model.fabric
    ) is None
    try:
        model.compute_loss(torch.tensor([[1, 2]]), backpropagate=False)
    except AssertionError as exc:
        assert "dense prefix-suffix" in str(exc)
    else:  # pragma: no cover - the poison is the proof base behavior was untouched
        raise AssertionError("base BST compute_loss was unexpectedly replaced")

    config = _config(family="nextlat", name="nextlat-seed1234-adapt-near")
    config.model.lambda_mse = 0.1
    try:
        install_common_adaptation(config, model, model.fabric)
    except RuntimeError as exc:
        assert "lambda_mse=lambda_kl=lambda_ce=0" in str(exc)
    else:
        raise AssertionError("nonzero NextLat auxiliary loss was accepted")
