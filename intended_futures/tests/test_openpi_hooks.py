import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
from torch import nn

from intended_futures.openpi_hooks import (
    AllCallsDeltaPatch,
    CaptureCalls,
    FirstCallDeltaPatch,
    ProjectedFirstCallPatch,
    ReplayCallsPatch,
    disable_compiled_sampling,
)


def test_disable_compiled_sampling_replaces_policy_cache_and_model_reference():
    def eager():
        return "eager"

    def compiled():
        return "compiled"

    compiled._torchdynamo_orig_callable = eager
    model = SimpleNamespace(sample_actions=compiled)
    policy = SimpleNamespace(_sample_actions=compiled, _model=model)
    disable_compiled_sampling(policy)
    assert policy._sample_actions is eager
    assert policy._model.sample_actions is eager


def test_capture_preserves_batch_token_and_hidden_axes():
    layer = nn.Identity()
    value = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    with CaptureCalls(layer) as capture:
        output = layer(value)
    assert torch.equal(output, value)
    assert len(capture.calls) == 1
    assert capture.calls[0].shape == (1, 3, 8)


def test_projected_patch_changes_first_call_only():
    layer = nn.Identity()
    recipient = torch.zeros((1, 2, 3), dtype=torch.float32)
    donor = np.ones((1, 2, 3), dtype=np.float32)
    basis = np.zeros((6, 1), dtype=np.float32)
    basis[0, 0] = 1.0
    with ProjectedFirstCallPatch(layer, donor_activation=donor, basis=basis) as patch:
        first = layer(recipient)
        second = layer(recipient)
    assert first[0, 0, 0].item() == 1.0
    assert torch.count_nonzero(first).item() == 1
    assert torch.count_nonzero(second).item() == 0
    assert patch.receipt.calls_seen == 2
    assert patch.receipt.calls_patched == 1


def test_precomputed_delta_patch_changes_first_call_only():
    layer = nn.Identity()
    recipient = torch.zeros((1, 2, 3), dtype=torch.float32)
    delta = np.ones((1, 2, 3), dtype=np.float32)
    with FirstCallDeltaPatch(layer, delta) as patch:
        first = layer(recipient)
        second = layer(recipient)
    assert torch.equal(first, torch.ones_like(first))
    assert torch.equal(second, recipient)
    assert patch.receipt.calls_seen == 2
    assert patch.receipt.calls_patched == 1


def test_replay_calls_patch_replaces_each_matching_call():
    layer = nn.Identity()
    recipient = torch.zeros((1, 2, 3), dtype=torch.float32)
    source_calls = [
        np.ones((1, 2, 3), dtype=np.float32),
        np.full((1, 2, 3), 2.0, dtype=np.float32),
    ]
    with ReplayCallsPatch(layer, source_calls) as patch:
        first = layer(recipient)
        second = layer(recipient)
    assert torch.equal(first, torch.ones_like(first))
    assert torch.equal(second, torch.full_like(second, 2.0))
    assert patch.receipt.calls_seen == 2
    assert patch.receipt.calls_patched == 2
    assert patch.receipt.shape_mismatches == 0


def test_replay_calls_patch_rejects_unmatched_call_count():
    layer = nn.Identity()
    source_calls = [np.ones((1, 2, 3), dtype=np.float32)]
    with pytest.raises(RuntimeError, match="did not patch every matching call"):
        with ReplayCallsPatch(layer, source_calls):
            layer(torch.zeros((1, 2, 3), dtype=torch.float32))
            layer(torch.zeros((1, 2, 3), dtype=torch.float32))


def test_all_calls_delta_patch_applies_call_specific_deltas():
    layer = nn.Identity()
    recipient = torch.ones((1, 2, 3), dtype=torch.float32)
    deltas = [
        np.ones((1, 2, 3), dtype=np.float32),
        np.full((1, 2, 3), -0.5, dtype=np.float32),
    ]
    with AllCallsDeltaPatch(layer, deltas) as patch:
        first = layer(recipient)
        second = layer(recipient)
    assert torch.equal(first, torch.full_like(first, 2.0))
    assert torch.equal(second, torch.full_like(second, 0.5))
    assert patch.receipt.calls_patched == 2
