# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for tuned module."""

from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.config import AITuneConfig
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule


def get_tuned_module(check_graph=True, strict_mode=False):
    config = AITuneConfig()
    config.strict_mode = strict_mode
    forward_signature = ForwardSignature.from_callable(lambda x: x)
    graph1 = SampleMetadata.from_inputs(
        forward_signature.normalize((1,), {}).arguments,
        strict=strict_mode,
    )
    graph2 = SampleMetadata.from_inputs(
        forward_signature.normalize((torch.randn(2),), {}).arguments,
        strict=strict_mode,
    )
    backend1 = Mock()
    backend2 = Mock()
    backends = OrderedDict({graph1: backend1, graph2: backend2})

    module = TunedModule(
        backends=backends,
        check_graph=check_graph,
        config=config,
        module_name="test",
        forward_signature=forward_signature,
    )
    return backend1, backend2, module


def test_unique_backend():
    metadata = Mock(spec=SampleMetadata)
    backend = Mock(spec=Backend)
    backends = OrderedDict({metadata: backend})
    forward_signature = ForwardSignature.from_callable(lambda x: x)
    module = TunedModule(
        backends=backends,
        check_graph=False,
        module_name="test",
        forward_signature=forward_signature,
    )

    module(2)
    backend.infer.assert_called_with(2)


@pytest.mark.parametrize("check_graph", [True, False])
def test_multiple_dict_backends(check_graph):
    backend1, backend2, module = get_tuned_module(check_graph, strict_mode=True)

    module(1)
    backend1.infer.assert_called_with(1)
    x = torch.randn(8)
    module(x)
    backend2.infer.assert_called_with(x)

    with pytest.raises(RuntimeError):
        module("unseen")


@pytest.mark.parametrize("check_graph", [True, False])
def test_equivalent_call_layouts_use_same_backend(check_graph):
    def forward(x, y):
        return x + y

    config = AITuneConfig()
    config.strict_mode = True
    signature = ForwardSignature.from_callable(forward)
    forward_inputs = signature.normalize((1, 2), {})
    metadata = SampleMetadata.from_inputs(forward_inputs.arguments, strict=True)
    backend = Mock()
    module = TunedModule(
        OrderedDict({metadata: backend}),
        module_name="test",
        check_graph=check_graph,
        config=config,
        forward_signature=signature,
    )

    module(1, 2)
    module(1, y=2)
    module(x=1, y=2)
    module(y=2, x=1)

    assert backend.infer.call_count == 4
    backend.infer.assert_called_with(1, 2)


def test_deactivate():
    backend1, backend2, module = get_tuned_module(check_graph=True)

    module.deactivate()

    backend1.deactivate.assert_called()
    backend2.deactivate.assert_called()


def test_serialization():
    """Test serialization and deserialization of TunedModule."""
    kwargs = {}
    signature = ForwardSignature.from_callable(lambda x: x)
    graph1 = SampleMetadata.from_inputs(signature.normalize((1,), kwargs).arguments, strict=True)
    graph2 = SampleMetadata.from_inputs(signature.normalize((torch.randn(2),), kwargs).arguments, strict=True)
    backend1 = TorchInductorJitBackend()
    backend2 = TorchInductorJitBackend()
    backend1._orig_module = Mock(spec=torch.nn.Module)
    backend2._orig_module = Mock(spec=torch.nn.Module)
    backends = OrderedDict({graph1: backend1, graph2: backend2})

    module = TunedModule(
        backends=backends,
        check_graph=True,
        module_name="test",
        forward_signature=signature,
    )
    # Serialize
    state_dict = module.to_dict()

    # Deserialize
    new_module = TunedModule.from_dict(Mock(spec=torch.nn.Module), state_dict)
    assert new_module._check_graph == module._check_graph
    assert new_module._forward_signature == signature
    assert len(new_module._backends) == len(module._backends)
    assert all(k1 == k2 for k1, k2 in zip(new_module._backends.keys(), module._backends.keys(), strict=False))
