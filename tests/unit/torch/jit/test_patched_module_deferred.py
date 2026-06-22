# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patched module."""

import copy
import inspect
import json
import re
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patched_module import (
    PRINT_HIERARCHY_HEADER,
    PRINT_HIERARCHY_NO_MODULES_HEADER,
    GraphBreakException,
    ModuleState,
    PatchedModule,
)
from aitune.torch.jit.patcher import Patcher, prepare_for_jit_tuning
from aitune.torch.tune_data.reporting import _active_report, report_tune_run_end
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from tests.toy_models.torch_models import OUTPUT_SIZE, ToyComplexPipeline
from tests.utilities.helpers import TestSink, requires_cuda


class CustomType:
    """Custom type for testing."""


@pytest.fixture(autouse=True)
def reset_jit_config():
    """Reset JIT config to defaults after each test so changes are scoped to the test."""
    yield
    config.reset_to_defaults()


@pytest.fixture
def mock_trt_backend():
    """Create a mock TensorRT backend for testing."""
    mock_backend = Mock()
    mock_backend.name = "MockTensorRTBackend"
    mock_backend.build.return_value = mock_backend
    mock_backend.key.return_value = "MockTensorRTBackend"
    strategy = FirstWinsStrategy(backends=[mock_backend])
    strategy.enable_performance_validation(False)
    config.strategy = strategy
    # in case strategy does a deepcopy, return self
    mock_backend.__deepcopy__ = lambda _: mock_backend
    return mock_backend


def test_set_original_forward_restores_unobserved_submodules():
    """Backend-boundary cleanup restores patched modules outside the observed call tree."""

    class ModelWithUnusedChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(2, 2)
            self.unused = torch.nn.Identity()

        def forward(self, x):
            return self.used(x)

    def is_wrapt_forward(forward):
        return type(forward).__name__ == "FunctionWrapper"

    config.mode = JITMode.TUNE_DEFERRED
    with prepare_for_jit_tuning():
        model = ModelWithUnusedChild()

    model(torch.randn(1, 2))

    assert len(PatchedModule.heads) == 1
    assert is_wrapt_forward(model.unused.forward)
    assert not any(child.__wrapped__ is model.unused for child in PatchedModule.heads[0]._children)

    with pytest.raises(NotImplementedError, match="object proxy must define __deepcopy__"):
        copy.deepcopy(model)

    PatchedModule.heads[0]._set_original_forward_for_hierarchy()

    assert not is_wrapt_forward(model.unused.forward)
    copy.deepcopy(model)


@requires_cuda
def test_jit_dry_run_success(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 0.0
    config.mode = JITMode.TUNE_DEFERRED

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with torch.inference_mode():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=tuned.*(dry-run tuning success).*call_count=3", sink.output[1])


@requires_cuda
def test_jit_deferred_tuning_not_triggered_automatically(mock_trt_backend, torch_device):
    """Test that deferred tuning does not trigger automatically during forward passes."""
    config.dry_run = False
    config.mode = JITMode.TUNE_DEFERRED

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with torch.inference_mode():
        # Run multiple forward passes
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    # Verify that module is still in recording state (not tuned)
    assert len(PatchedModule.heads) == 1
    assert PatchedModule.heads[0]._state == ModuleState.RECORDING
    mock_trt_backend.build.assert_not_called()

    # Now explicitly trigger tuning
    mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE)
    for head in PatchedModule.heads:
        head.try_tune()

    # Verify that module is now tuned
    assert PatchedModule.heads[0]._state == ModuleState.TUNED
    mock_trt_backend.build.assert_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=tuned.*(MockTensorRTBackend).*call_count=3", sink.output[1])


@requires_cuda
def test_jit_dry_run_failure(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 1.0
    config.mode = JITMode.TUNE_DEFERRED

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with torch.inference_mode():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=3", sink.output[1])


@requires_cuda
@pytest.mark.parametrize("scenario", ["success", "correctness_error", "backend_build_error"])
def test_jit_tuning_success(mock_trt_backend, torch_device, scenario):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    if scenario == "success":
        mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE)
    elif scenario == "correctness_error":
        mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE * 10)  # wrong output shapes returned
    elif scenario == "backend_build_error":
        mock_trt_backend.infer.side_effect = Exception("Backend build error")

    with torch.inference_mode():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    if scenario == "success":
        assert re.match(r".*ToyTorchModel.*state=tuned.*(MockTensorRTBackend).*call_count=3", sink.output[1])
    elif scenario == "correctness_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=3", sink.output[1])
    elif scenario == "backend_build_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=3", sink.output[1])


@requires_cuda
def test_jit_tuning_with_module_hooks(mock_trt_backend, torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED

    mock_trt_backend.infer.return_value = torch.randn(OUTPUT_SIZE)

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

    with torch.inference_mode():
        pipeline(torch.randn(1))
        assert hooks_history == ["pre_hook", "forward_hook"]
        hooks_history.clear()
        pipeline(torch.randn(2))
        assert hooks_history == ["pre_hook", "forward_hook"]

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*TestNet.*state=tuned.*(MockTensorRTBackend).*call_count=2", sink.output[1])


@requires_cuda
def test_jit_tuning_graph_break(mock_trt_backend, torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = True
    config.mode = JITMode.TUNE_DEFERRED
    config.max_depth_level = 2

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)

    mocker.patch("aitune.torch.jit.patched_module.GraphBreakDetector.detect", side_effect=GraphBreakException)
    with torch.inference_mode():
        for x in inputs:
            pipeline(x)

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(graph break).*call_count=3", sink.output[1])
    assert re.match(r".*Linear.*state=eager.*(graph break).*call_count=3", sink.output[2])


@requires_cuda
def test_jit_tuning_skip_module(mock_trt_backend, torch_device, mocker):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED
    config.max_depth_level = 2

    config.skip_modules = ["ToyTorchModel"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    with torch.inference_mode():
        for x in inputs:
            pipeline(x)

    # Explicitly trigger deferred tuning (even though module is skipped)
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=skipped.*call_count=1", sink.output[1])
    assert re.match(r".*Linear.*state=detached.*call_count=1", sink.output[2])
    assert re.match(r".*Linear.*state=detached.*call_count=1", sink.output[3])


@requires_cuda
def test_jit_tuning_skip_child_module_if_parent_failed(mock_trt_backend, torch_device):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED
    config.max_depth_level = 2

    config.skip_modules = ["Linear"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)

    with torch.inference_mode():
        for x in inputs:
            pipeline(x)

    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=3", sink.output[1])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[2])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[3])


@requires_cuda
def test_jit_tuning_no_modules(mock_trt_backend, torch_device):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED

    # the following class has not parameters and should be skipped
    class IdentityModule(torch.nn.Module):
        def forward(self, x):
            return x

    with prepare_for_jit_tuning():
        model = IdentityModule()
        model = model.to(torch_device)

    for _ in range(2):
        model(torch.randn(1, 10))

    # Explicitly trigger deferred tuning (no heads to tune)
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_NO_MODULES_HEADER in sink.output[0]


@requires_cuda
def test_forward_method_should_have_same_signature(mock_trt_backend, torch_device):
    config.dry_run = False
    config.mode = JITMode.TUNE_DEFERRED

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
    model(1, 2, 3)  # we are in recording state
    # Explicitly trigger deferred tuning
    for head in PatchedModule.heads:
        head.try_tune()
    # in tuned state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}


def test_each_module_has_unique_cache_dir():
    config.dry_run = False
    config.mode = JITMode.TUNE_DEFERRED

    with prepare_for_jit_tuning():
        # 10 identical modules - same parameters, same name
        modules = [torch.nn.Linear(10, 10) for _ in range(10)]

    for module in modules:
        # call each module to capture it by patched module
        module(torch.randn(1, 10))

    assert len(PatchedModule.heads) == 10

    cache_dirs = {m._create_graph_cache_dir("test") for m in PatchedModule.heads}
    assert len(cache_dirs) == 10


@requires_cuda
def test_jit_tuning_skip_module_when_not_match_min_parameters(mock_trt_backend, torch_device):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_DEFERRED
    config.min_parameters = 1e10
    config.skip_modules = ["ToyTorchModel"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    with torch.inference_mode():
        for x in inputs:
            pipeline(x)

    # Explicitly trigger deferred tuning (no heads to tune due to min_parameters)
    for head in PatchedModule.heads:
        head.try_tune()

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_not_called()


def test_jit_deferred_tune_before_forward_pass_does_not_crash():
    """Regression: Patcher.tune_deferred() must not raise on modules in INIT state.

    Modules created inside prepare_for_jit_tuning() land in _patched_modules without a _wrapper
    (set only on the first forward call). With min_samples > 1 and batch_axis_required=True,
    _should_be_tuned() previously crashed with AttributeError when accessing self._wrapper before
    any forward pass had occurred.
    """
    config.mode = JITMode.TUNE_DEFERRED
    config.min_samples = 2
    config.batch_axis_required = True

    with prepare_for_jit_tuning():
        torch.nn.Linear(10, 5)  # creates a PatchedModule in INIT state — _wrapper never set

    # Must not raise AttributeError: 'PatchedModule' object has no attribute '_wrapper'
    Patcher.tune_deferred()

    assert len(PatchedModule.heads) == 0


def test_jit_deferred_tuning_records_inspection_details_before_tuning(mocker, tmp_path):
    """Deferred tuning should snapshot every intercepted module before try_tune mutates state."""

    class ModelWithUnusedChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 4)
            self.unused = torch.nn.Linear(4, 4).to(dtype=torch.float16)

        def forward(self, x):
            return self.used(x)

    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.cache_dir = tmp_path / "cache"
    mock_config.tuning_data_output_path = report_path

    config.mode = JITMode.TUNE_DEFERRED
    config.max_depth_level = 2

    with prepare_for_jit_tuning():
        model = ModelWithUnusedChild()

    model(torch.randn(2, 4))

    def assert_inspection_details_collected(_module):
        report = _active_report.get()
        assert report is not None
        assert len(report.inspection_details) == 3

    mocker.patch.object(PatchedModule, "try_tune", autospec=True, side_effect=assert_inspection_details_collected)

    Patcher.tune_deferred()
    report_tune_run_end()

    report = json.loads(report_path.read_text())
    details = report["inspection_details"]
    assert len(details) == 3

    class_names = [detail["module_class"] for detail in details]
    assert class_names.count("torch.nn.modules.linear.Linear") == 2

    root = next(detail for detail in details if detail["module_class"].endswith("ModelWithUnusedChild"))
    assert isinstance(root["module_id"], int)
    assert root["parent_module_id"] is None
    assert set(root["dtypes"]) == {"torch.float16", "torch.float32"}

    unused_linear = next(
        detail
        for detail in details
        if detail["module_class"] == "torch.nn.modules.linear.Linear" and detail["state"] == "init"
    )
    assert unused_linear["call_count"] == 0
    assert unused_linear["module_name"] is None
    assert isinstance(unused_linear["module_id"], int)
    assert unused_linear["dtypes"] == ["torch.float16"]
    assert unused_linear["graphs"] == []

    observed_linear = next(
        detail
        for detail in details
        if detail["module_class"] == "torch.nn.modules.linear.Linear" and detail["state"] == "recording"
    )
    assert observed_linear["module_name"].startswith("ModelWithUnusedChild")
    assert observed_linear["module_name"].endswith("Linear")
    assert observed_linear["call_count"] == 1
    assert isinstance(observed_linear["module_id"], int)
    assert observed_linear["parent_module_id"] == root["module_id"]
    assert observed_linear["dtypes"] == ["torch.float32"]
    assert observed_linear["graphs"][0]["input_spec"]["tensor_data"][0]["shape"] == [2, 4]
    assert observed_linear["graphs"][0]["input_spec"]["tensor_data"][0]["dtype"] == "torch.float32"
    assert observed_linear["graphs"][0]["output_spec"]["tensor_data"][0]["shape"] == [2, 4]
    assert observed_linear["graphs"][0]["output_spec"]["tensor_data"][0]["dtype"] == "torch.float32"
