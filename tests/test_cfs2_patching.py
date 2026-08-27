from __future__ import annotations

import numpy as np
import pytest

from cfs2 import patching as P


def test_unrelated_anchor_derangement_is_deterministic_and_has_no_fixed_points() -> None:
    left = P.fixed_derangement(2000, 17)
    right = P.fixed_derangement(2000, 17)
    assert np.array_equal(left, right)
    assert sorted(left.tolist()) == list(range(2000))
    assert not np.any(left == np.arange(2000))


def test_norm_matched_random_control_preserves_each_real_patch_delta_norm() -> None:
    rng = np.random.default_rng(4)
    adapted = rng.normal(size=(32, 12)).astype(np.float32)
    parent = rng.normal(size=(32, 12)).astype(np.float32)
    replacement = P.norm_matched_random_replacement(adapted, parent, seed=19)
    expected = np.linalg.norm(parent - adapted, axis=1)
    observed = np.linalg.norm(replacement - adapted, axis=1)
    assert np.allclose(observed, expected, atol=2e-6, rtol=2e-6)
    assert not np.allclose(replacement, parent)


def test_invalid_controls_fail_loudly() -> None:
    with pytest.raises(P.CFS2PatchingError):
        P.fixed_derangement(1, 0)
    with pytest.raises(P.CFS2PatchingError):
        P.norm_matched_random_replacement(np.zeros((2, 3)), np.zeros((3, 2)), seed=0)


def test_forward_hook_capture_and_self_patch_are_identical() -> None:
    torch = pytest.importorskip("torch")

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False) for _ in range(3)]
            )
            self.readout = torch.nn.Linear(4, 6, bias=False)

        def forward(self, value):
            for block in self.transformer.blocks:
                value = torch.tanh(block(value))
            return self.readout(value)

    torch.manual_seed(3)
    model = ToyModel()
    tokens = torch.randn(5, 4, 4)
    captured = P.capture_states_and_logits(model, tokens, layers=(0, 2), position=2)
    patched = P.forward_with_patch(
        model, tokens, captured.states[2], layer=2, position=2
    )
    assert tuple(captured.states[0].shape) == (5, 4)
    assert torch.allclose(patched, captured.logits, atol=1e-7, rtol=1e-7)
