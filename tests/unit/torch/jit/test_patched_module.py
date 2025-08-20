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
"""Test the patched module."""

import re
from unittest.mock import Mock, patch

import pytest
import torch

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import (
    PRINT_HIERARCHY_HEADER,
    PRINT_HIERARCHY_NO_MODULES_HEADER,
    GraphBreakException,
    PatchedModule,
)
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from tests.toy_models.torch_models import OUTPUT_SIZE, ToyComplexPipeline
from tests.utilities.helpers import requires_cuda


class TestSink:
    """Sink for capturing output from PatchedModule.print_hierarchy."""

    def __init__(self):
        self.output = []

    def write(self, text):
        self.output.append(text)


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_dry_run_success(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 0.0

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
        pipeline(x)

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=tuned.*(dry-run tuning success).*call_count=2", sink.output[1])


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_dry_run_failure(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 1.0

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
        pipeline(x)

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(tuning error).*call_count=2", sink.output[1])


@requires_cuda
@pytest.mark.parametrize("scenario", ["success", "correctness_error", "backend_build_error"])
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_success(mock_trt_backend_class, torch_device, scenario, mocker):
    config.dry_run = False
    config.detect_graph_breaks = False
    # prepare a mock backend
    mock_backend = Mock(name="MockTensorRTBackend")
    mock_backend.build.return_value = mock_backend
    mock_backend.name = "MockTensorRTBackend"

    mock_trt_backend_class.return_value = mock_backend

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    if scenario == "success":
        mock_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE)
    elif scenario == "correctness_error":
        mock_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE * 10)  # wrong output shapes returned
    elif scenario == "backend_build_error":
        mock_backend.build.side_effect = Exception("Backend build error")

    mocker.patch("copy.deepcopy", return_value=mock_backend)
    for x in inputs:
        pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    if scenario == "success":
        assert re.match(r".*ToyTorchModel.*state=tuned.*(MockTensorRTBackend).*call_count=2", sink.output[1])
    elif scenario == "correctness_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(tuning error).*call_count=2", sink.output[1])
    elif scenario == "backend_build_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(tuning error).*call_count=2", sink.output[1])


@requires_cuda
def test_jit_tuning_graph_break(torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = True

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)

    mocker.patch("aitune.torch.jit.patched_module.GraphBreakDetector.detect", side_effect=GraphBreakException)
    for x in inputs:
        pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(graph break).*call_count=2", sink.output[1])
    assert re.match(r".*Linear.*state=eager.*(graph break).*call_count=2", sink.output[2])


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_skip_module(mock_trt_backend_class, torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.skip_modules = ["ToyTorchModel"]
    # prepare a mock backend
    mock_backend = Mock(name="MockTensorRTBackend")
    mock_backend.build.return_value = mock_backend
    mock_backend.name = "MockTensorRTBackend"

    mock_trt_backend_class.return_value = mock_backend

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    mocker.patch("copy.deepcopy", return_value=mock_backend)
    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    for x in inputs:
        pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=skipped.*call_count=2", sink.output[1])
    assert re.match(r".*Linear.*state=detached.*call_count=2", sink.output[2])


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_no_modules(mock_trt_backend_class, torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = False

    # the following class has not parameters and should be skipped
    class IdentityModule(torch.nn.Module):
        def forward(self, x):
            return x

    # prepare a mock backend
    mock_backend = Mock(name="MockTensorRTBackend")
    mock_backend.build.return_value = mock_backend
    mock_backend.name = "MockTensorRTBackend"

    mock_trt_backend_class.return_value = mock_backend

    with prepare_for_jit_tuning():
        model = IdentityModule()
        model = model.to(torch_device)

    mocker.patch("copy.deepcopy", return_value=mock_backend)
    for _ in range(2):
        model(torch.randn(1, 10))

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_NO_MODULES_HEADER in sink.output[0]
