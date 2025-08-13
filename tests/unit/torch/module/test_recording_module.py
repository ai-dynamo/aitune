# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Unit test for recording module."""

from unittest.mock import Mock

import pytest
import torch

from aitune.torch.config import AITuneConfig
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.tensor_spec import InfoLevel
from tests.toy_models.torch_models import ToyTorchModel


@pytest.fixture
def recording_module():
    config = AITuneConfig()
    config.max_num_samples_stored = 10
    config.min_num_samples = 2

    module = Mock(spec=torch.nn.Module)
    module.__call__ = lambda *args, **kwargs: args[0]
    return RecordingModule(module, "test-module", config)


def test_recording_module_call_same_sample(recording_module):
    recording_module(1)
    recording_module(1)

    assert recording_module.record_sample
    assert len(recording_module.graph_specs) == 1


def test_recording_module_call_different_tensors(recording_module):
    recording_module(torch.tensor(1))
    recording_module(torch.tensor(2))
    recording_module(torch.tensor([1, 1]))

    assert recording_module.record_sample
    assert len(recording_module.graph_specs) == 2


def test_recording_module_call_multiple_graphs(recording_module):
    recording_module(1, torch.tensor(42))  # first graph
    recording_module(2, torch.randn(2))  # second graph
    recording_module(2, torch.randn(3))  # second graph batch dimension
    recording_module(3, torch.randn(1, 1), torch.randn(2, 1))  # third graph
    recording_module(3, torch.randn(1, 3), torch.randn(2, 3))  # third graph batched dimensions

    assert recording_module.record_sample
    assert len(recording_module.graph_specs) == 3
    assert recording_module.graph_specs[0].input_spec.describe(InfoLevel.MEDIUM) == "((1, input__0[]), {})"
    assert recording_module.graph_specs[1].input_spec.describe(InfoLevel.MEDIUM) == "((2, input__0[dim0]), {})"
    assert (
        recording_module.graph_specs[2].input_spec.describe(InfoLevel.MEDIUM)
        == "((3, input__0[1, dim1], input__1[2, dim1]), {})"
    )
    assert recording_module.graph_specs[2].input_spec.tensor_specs[0].min_shape == [1, 1]
    assert recording_module.graph_specs[2].input_spec.tensor_specs[0].max_shape == [1, 3]
    assert recording_module.graph_specs[2].input_spec.tensor_specs[1].min_shape == [2, 1]
    assert recording_module.graph_specs[2].input_spec.tensor_specs[1].max_shape == [2, 3]


def test_recording_module_check_is_ready(recording_module):
    recording_module(1, torch.tensor(1))
    recording_module(2, torch.tensor(1))
    assert not recording_module.is_ready_for_optimization

    recording_module(1, torch.tensor(1))
    assert not recording_module.is_ready_for_optimization
    recording_module(2, torch.tensor(1))
    assert recording_module.is_ready_for_optimization


def test_samples_for_graph(recording_module):
    recording_module(1, torch.tensor(1))  # first graph
    recording_module(1, torch.tensor(1), a=5)  # second graph
    recording_module(2, torch.tensor(1))  # third graph
    recording_module(2, torch.tensor(1))  # third graph

    for graph_spec, num_expected_samples in zip(recording_module.graph_specs, [1, 1, 2], strict=False):
        samples = recording_module.samples_for_graph_spec(graph_spec)
        assert len(samples) == num_expected_samples

    samples = recording_module.samples_for_graph_spec(recording_module.graph_specs[1])
    args, kwargs = samples[0]  # take the only sample
    assert args == (1, torch.tensor(1))
    assert kwargs == {"a": 5}


def test_device():
    model = ToyTorchModel()
    recording = RecordingModule(model, "test-module")
    assert recording.device == torch.device("cpu")
