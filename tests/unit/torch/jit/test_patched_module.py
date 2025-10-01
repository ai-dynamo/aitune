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

import inspect
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
from tests.utilities.helpers import TestSink, requires_cuda


class TestType:
    """Just a class to mimic unsupported type."""


def mock_trt_backend(mock_trt_backend_class, mocker, infer_value):
    """Create a mock TensorRT backend for testing.

    Returns:
        Mock: A configured mock backend with build method and name attribute.
    """
    mock_backend = Mock()
    mock_backend.name = "MockTensorRTBackend"
    mock_backend.build.return_value = mock_backend
    if isinstance(infer_value, torch.Tensor):
        mock_backend.infer.return_value = infer_value
    else:
        mock_backend.infer.side_effect = infer_value

    mocker.patch("copy.deepcopy", return_value=mock_backend)
    mock_trt_backend_class.return_value = mock_backend
    return mock_backend


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_dry_run_success(mock_trt_backend, torch_device):
    config.dry_run = True
    config.inspect_mode = False
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
    config.inspect_mode = False
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
    config.inspect_mode = False
    config.detect_graph_breaks = False

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    if scenario == "success":
        infer_value = torch.randn(1, OUTPUT_SIZE)
    elif scenario == "correctness_error":
        infer_value = torch.randn(1, OUTPUT_SIZE * 10)  # wrong output shapes returned
    elif scenario == "backend_build_error":
        infer_value = Exception("Backend build error")

    mock_backend = mock_trt_backend(mock_trt_backend_class, mocker, infer_value)

    for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
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
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_with_module_hooks(mock_trt_backend_class, torch_device, mocker):
    config.dry_run = False
    config.inspect_mode = False
    config.detect_graph_breaks = False

    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)  # only to make num params > 0

        def forward(self, x):
            return x

    with prepare_for_jit_tuning():
        pipeline = TestNet().to(torch_device)

    hooks_history = []

    def pre_hook(module, input):  # noqa: A002
        hooks_history.append("pre_hook")
        return input

    def hook(module, input, output):  # noqa: A002
        hooks_history.append("forward_hook")
        return output

    pipeline.register_forward_hook(hook)
    pipeline.register_forward_pre_hook(pre_hook)

    mock_backend = mock_trt_backend(mock_trt_backend_class, mocker, infer_value=torch.randn(1))
    pipeline(torch.randn(1))
    assert hooks_history == ["pre_hook", "forward_hook"]
    hooks_history.clear()
    pipeline(torch.randn(2))
    assert hooks_history == ["pre_hook", "forward_hook"]

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*TestNet.*state=tuned.*(MockTensorRTBackend).*call_count=2", sink.output[1])


@requires_cuda
@pytest.mark.parametrize("unsupported_type_on_input", [True, False])
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_unsupported_type(mock_trt_backend_class, unsupported_type_on_input, torch_device, mocker):
    """Test that the patched module handles unsupported types.

    To mimic unsupported type, we use TestType object.

    The test model contains linear layer so that even if the parent layer cannot be tuned due to unsupported type,
    the child layer should be tuned.

    """
    config.dry_run = False
    config.inspect_mode = False
    config.detect_graph_breaks = False

    mock_trt_backend(mock_trt_backend_class, mocker, torch.randn(1, 10))

    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, unsupported_type=None, return_unsupported_type=False):
            y = self.linear(x)
            return y, TestType() if return_unsupported_type else y

    with prepare_for_jit_tuning():
        pipeline = TestNet().to(torch_device)

    # make two different inputs to detect batch axis
    x1 = torch.randn(1, 10).to(torch_device)
    x2 = torch.randn(2, 10).to(torch_device)

    if unsupported_type_on_input:
        pipeline(x1, TestType(), return_unsupported_type=False)
        pipeline(x2, TestType(), return_unsupported_type=False)
    else:
        pipeline(x1, return_unsupported_type=True)
        pipeline(x2, return_unsupported_type=True)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*TestNet.*state=eager.*(Unsupported type:.*TestType).*call_count=1", sink.output[1])
    # expected call count differs due to scenarios:
    if unsupported_type_on_input:
        call_cnt = 2  # unsupported exception detected before calling TestNet, 2 calls
    else:
        call_cnt = 3  # unsupported exception detected after calling TestNet, first call repeated 2 times + second call
    assert re.match(r".*Linear.*state=tuned.*(MockTensorRTBackend).*call_count=" + str(call_cnt), sink.output[2])


@requires_cuda
def test_jit_tuning_graph_break(torch_device, mocker):
    config.dry_run = False
    config.inspect_mode = False
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
    config.inspect_mode = False
    config.detect_graph_breaks = False
    config.skip_modules = ["ToyTorchModel"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    for x in inputs:
        pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend_class.build.assert_not_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=skipped.*call_count=1", sink.output[1])
    assert re.match(r".*Linear.*state=detached.*call_count=1", sink.output[2])
    assert re.match(r".*Linear.*state=detached.*call_count=1", sink.output[3])


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_skip_child_module_if_parent_failed(mock_trt_backend_class, torch_device, mocker):
    config.dry_run = False
    config.inspect_mode = False
    config.detect_graph_breaks = False
    config.skip_modules = ["Linear"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    mock_backend = mock_trt_backend(mock_trt_backend_class, mocker, infer_value=Exception("Backend build error"))

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    for x in inputs:
        pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(tuning error).*call_count=2", sink.output[1])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[2])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[3])


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_jit_tuning_no_modules(mock_trt_backend_class, torch_device, mocker):
    config.dry_run = False
    config.inspect_mode = False
    config.detect_graph_breaks = False

    # the following class has not parameters and should be skipped
    class IdentityModule(torch.nn.Module):
        def forward(self, x):
            return x

    mock_backend = mock_trt_backend(mock_trt_backend_class, mocker, torch.randn(1, 10))

    with prepare_for_jit_tuning():
        model = IdentityModule()
        model = model.to(torch_device)

    for _ in range(2):
        model(torch.randn(1, 10))

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_NO_MODULES_HEADER in sink.output[0]


@requires_cuda
@patch("aitune.torch.jit.patched_module.TensorRTBackend")
def test_forward_method_should_have_same_signature(mock_trt_backend, torch_device):
    config.dry_run = False
    config.inspect_mode = False

    class TestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, y, z, pos="test"):
            """Test forward method."""
            return x, y, z, pos

    with prepare_for_jit_tuning():
        model = TestNet().to(torch_device)

    # in init state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}
    model(1, 2, 3)
    # in recording state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}
    # in tuned state
    model(1, 2, 3)  # we are in recording state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}


def test_each_module_has_unique_cache_dir():
    config.dry_run = False
    config.inspect_mode = False

    with prepare_for_jit_tuning():
        # 10 identical modules - same parameters, same name
        modules = [torch.nn.Linear(10, 10) for _ in range(10)]

    for module in modules:
        # call each module to capture it by patched module
        module(torch.randn(1, 10))

    assert len(PatchedModule.heads) == 10

    cache_dirs = {m._create_graph_cache_dir("test") for m in PatchedModule.heads}
    assert len(cache_dirs) == 10
