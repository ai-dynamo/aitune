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
from tests.toy_models.torch_models import ToyTorchModel


def recording_module(strict_mode):
    config = AITuneConfig()
    config.max_num_samples_stored = 10
    config.min_num_samples = 2
    config.strict_mode = strict_mode

    module = Mock(spec=torch.nn.Module)
    module.__call__ = lambda *args, **kwargs: args[0]
    return RecordingModule(module, "test-module", config)


@pytest.mark.parametrize("strict_mode", [True, False])
def test_recording_module_same_rank_tensors(strict_mode):
    rec_module = recording_module(strict_mode=strict_mode)
    rec_module(torch.tensor(2))
    rec_module(torch.tensor(2))

    assert len(rec_module.graph_specs) == 1


def test_recording_module_same_other_data():
    rec_module = recording_module(strict_mode=True)
    rec_module(1)
    rec_module(1)

    assert len(rec_module.graph_specs) == 1


@pytest.mark.parametrize("strict_mode", [True, False])
def test_recording_module_call_different_rank_tensors(strict_mode):
    rec_module = recording_module(strict_mode=strict_mode)
    rec_module(torch.tensor(2))
    rec_module(torch.tensor([1, 1]))

    assert len(rec_module.graph_specs) == 2


def test_recording_module_same_rank_tensors_different_other_data():
    rec_module = recording_module(strict_mode=True)
    rec_module(torch.tensor(2), "abc")
    rec_module(torch.tensor(2), "xyz")

    assert len(rec_module.graph_specs) == 2


def test_recording_module_call_multiple_graphs():
    rec_module = recording_module(strict_mode=False)
    rec_module(torch.tensor(42))  # first graph - dtype int
    rec_module(torch.randn(2))  # second graph - dtype fp32
    rec_module(torch.randn(3))  # second graph batch dimension
    rec_module(torch.randn(1, 1), torch.randn(2, 1))  # third graph
    rec_module(torch.randn(1, 3), torch.randn(2, 3))  # third graph batched dimensions

    assert rec_module.record_sample
    assert len(rec_module.graph_specs) == 3
    assert rec_module.graph_specs[0].input_spec.tensor_specs[0].shape == []
    assert rec_module.graph_specs[1].input_spec.tensor_specs[0].shape == ["dim0"]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].shape == [1, "dim1"]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].min_shape == [1, 1]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[0].max_shape == [1, 3]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[1].min_shape == [2, 1]
    assert rec_module.graph_specs[2].input_spec.tensor_specs[1].max_shape == [2, 3]


def test_recording_module_check_is_ready_non_strict():
    rec_module = recording_module(strict_mode=False)
    rec_module(torch.tensor(1))  # first graph, one sample
    assert not rec_module.is_ready_for_optimization

    rec_module(torch.randn(1, 1))  # second graph, one sample
    assert not rec_module.is_ready_for_optimization

    rec_module(torch.tensor(2))  # first graph, second sample
    rec_module(torch.randn(1, 1))  # second graph, second sample
    assert rec_module.is_ready_for_optimization


def test_samples_for_graph():
    rec_module = recording_module(strict_mode=True)
    rec_module(1, torch.tensor(1))  # first graph
    rec_module(1, torch.tensor(1), a=5)  # second graph
    rec_module(2, torch.tensor(1))  # third graph
    rec_module(2, torch.tensor(1))  # third graph

    for graph_spec, num_expected_samples in zip(rec_module.graph_specs, [1, 1, 2], strict=False):
        samples = rec_module.samples_for_graph_spec(graph_spec)
        assert len(samples) == num_expected_samples

    samples = rec_module.samples_for_graph_spec(rec_module.graph_specs[1])
    args, kwargs = samples[0]  # take the only sample
    assert args == (1, torch.tensor(1))
    assert kwargs == {"a": 5}


def test_device():
    model = ToyTorchModel()
    recording = RecordingModule(model, "test-module")
    assert recording.device == torch.device("cpu")
