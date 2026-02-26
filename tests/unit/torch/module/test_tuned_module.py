# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for tuned module."""

from collections import OrderedDict
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torch_inductor_backend import TorchInductorBackend
from aitune.torch.config import AITuneConfig
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.tuned_module import TunedModule


def get_tuned_module(check_graph=True, strict_mode=False):
    config = AITuneConfig()
    config.strict_mode = strict_mode
    kwargs = {}
    graph1 = SampleMetadata.from_inputs((1,), kwargs, strict=strict_mode)
    graph2 = SampleMetadata.from_inputs((torch.randn(2),), kwargs, strict=strict_mode)
    backend1 = Mock()
    backend2 = Mock()
    backends = OrderedDict({graph1: backend1, graph2: backend2})

    module = TunedModule(backends=backends, check_graph=check_graph, config=config)
    return backend1, backend2, module


def test_unique_backend():
    metadata = Mock(spec=SampleMetadata)
    backend = Mock(spec=Backend)
    backends = OrderedDict({metadata: backend})
    module = TunedModule(backends=backends, check_graph=False)

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
        module(1, 2)


def test_deactivate():
    backend1, backend2, module = get_tuned_module(check_graph=True)

    module.deactivate()

    backend1.deactivate.assert_called()
    backend2.deactivate.assert_called()


def test_serialization():
    """Test serialization and deserialization of TunedModule."""
    kwargs = {}
    graph1 = SampleMetadata.from_inputs((1,), kwargs, strict=True)
    graph2 = SampleMetadata.from_inputs((torch.randn(2),), kwargs, strict=True)
    backend1 = TorchInductorBackend()
    backend2 = TorchInductorBackend()
    backend1._orig_module = Mock(spec=torch.nn.Module)
    backend2._orig_module = Mock(spec=torch.nn.Module)
    backends = OrderedDict({graph1: backend1, graph2: backend2})

    module = TunedModule(backends=backends, check_graph=True)
    # Serialize
    state_dict = module.to_dict()

    # Deserialize
    new_module = TunedModule.from_dict(Mock(spec=torch.nn.Module), state_dict)
    assert new_module._check_graph == module._check_graph
    assert len(new_module._backends) == len(module._backends)
    assert all(k1 == k2 for k1, k2 in zip(new_module._backends.keys(), module._backends.keys(), strict=False))
