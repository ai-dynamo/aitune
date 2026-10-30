# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT checkpoint metadata reuses supplied weights, while legacy checkpoints restore them."""

import pytest
import torch
from torch import nn

from aitune.torch.backend import (
    TorchAOBackend,
    TorchEagerBackend,
    TorchInductorJitBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.backend.backend import BackendState
from aitune.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask


@pytest.fixture(params=[TorchEagerBackend, TorchInductorJitBackend, TorchTensorRTJitBackend, TorchAOBackend])
def backend(request, mocker):
    # Only exercise serialization here; real GPU compilation is covered by backend lifecycle tests.
    mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_cuda_is_available")
    mocker.patch("aitune.torch.backend.torch_tensorrt_jit_backend.assert_torch_tensorrt")
    backend = request.param()
    backend._orig_module = make_model(1.0)
    backend._device = torch.device("cpu")
    return backend


def make_model(value: float) -> nn.Module:
    model = nn.Linear(4, 4)
    with torch.no_grad():
        model.weight.fill_(value)
        model.bias.fill_(value)
    model.register_buffer("scale", torch.tensor(value))
    return model.eval()


def test_new_checkpoint_preserves_supplied_weights(backend, mocker, tmp_path):
    state_dict = mocker.spy(backend._orig_module, "state_dict")
    state = backend.to_dict()
    state_dict.assert_not_called()
    assert backend.STATE_ORIG_MODULE not in state
    TorchSaveTask().save(tmp_path, state)
    state = TorchLoadTask().load(tmp_path)

    supplied = make_model(2.0)
    parameter = supplied.weight
    buffer = supplied.scale
    load_state_dict = mocker.spy(supplied, "load_state_dict")
    value = torch.ones(1, 4)
    expected = supplied(value)
    restored = type(backend).from_dict(supplied, state)

    load_state_dict.assert_not_called()
    assert restored.state == BackendState.CHECKPOINT_LOADED
    assert restored._orig_module is supplied
    assert supplied.weight is parameter
    assert supplied.scale is buffer
    torch.testing.assert_close(restored._orig_module(value), expected)
    torch.testing.assert_close(supplied.scale, torch.tensor(2.0))
    assert restored.key() == backend.key()


def test_legacy_checkpoint_restores_original_weights(backend, tmp_path):
    state = backend.to_dict()
    state[backend.STATE_ORIG_MODULE] = backend._orig_module.state_dict()
    TorchSaveTask().save(tmp_path, state)
    state = TorchLoadTask().load(tmp_path)

    supplied = make_model(2.0)
    value = torch.ones(1, 4)
    expected = backend._orig_module(value)
    assert not torch.equal(supplied(value), expected)
    restored = type(backend).from_dict(supplied, state)

    assert restored._orig_module is supplied
    torch.testing.assert_close(supplied(value), expected)
    torch.testing.assert_close(supplied.scale, backend._orig_module.scale)
