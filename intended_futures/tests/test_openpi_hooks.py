import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
from torch import nn

from intended_futures.openpi_hooks import CaptureCalls, ProjectedFirstCallPatch, disable_compiled_sampling


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
