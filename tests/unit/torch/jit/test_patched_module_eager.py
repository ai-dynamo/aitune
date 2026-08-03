# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patched module."""

import errno
import inspect
import json
import re
from unittest.mock import Mock

import pytest
import torch

from aitune.torch.backend import Backend
from aitune.torch.backend.backend import DummyBackend
from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patched_module import (
    PRINT_HIERARCHY_HEADER,
    PRINT_HIERARCHY_NO_MODULES_HEADER,
    GraphBreakException,
    PatchedModule,
)
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.tune_data.reporting import report_tune_run_end
from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from aitune.torch.tune_strategy.tune_strategy import DummyTuneStrategy, TuneStrategy
from aitune.utils.disk_space import DiskSpaceError
from tests.toy_models.torch_models import OUTPUT_SIZE, ToyComplexPipeline
from tests.utilities.helpers import TestSink, requires_cuda


class CustomType:
    """Custom type for testing."""


class _SimpleJitModule(torch.nn.Module):
    """Simple JIT module for testing."""

    def forward(self, x):
        return x


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
    mock_backend.describe.return_value = "MockTensorRTBackend"
    mock_backend.key.return_value = "MockTensorRTBackend"

    mock_backend.build.return_value = mock_backend

    strategy = FirstWinsStrategy(backends=[mock_backend])
    strategy.enable_performance_validation(False)
    config.strategy = strategy
    # in case strategy does a deepcopy, return self
    mock_backend.__deepcopy__ = lambda _: mock_backend
    return mock_backend


@requires_cuda
def test_jit_dry_run_success(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 0.0
    config.mode = JITMode.TUNE_EAGER

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with torch.no_grad():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=tuned.*(dry-run tuning success).*call_count=1", sink.output[1])


@requires_cuda
def test_jit_dry_run_failure(mock_trt_backend, torch_device):
    config.dry_run = True
    config.dry_run_failure_probability = 1.0
    config.mode = JITMode.TUNE_EAGER

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with torch.no_grad():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    assert len(PatchedModule.heads) == 1
    mock_trt_backend.build.assert_not_called()

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=1", sink.output[1])


def test_jit_eager_records_inspection_details_for_tuned_module_subtree(mocker, tmp_path):
    """Eager tuning should snapshot the attempted module and its observed children."""

    class ParentWithChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.child(x)

    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.cache_dir = tmp_path / "cache"
    mock_config.tuning_data_output_path = report_path

    config.mode = JITMode.TUNE_EAGER
    config.device = torch.device("cpu")
    config.dry_run = True
    config.dry_run_failure_probability = 0.0
    config.max_depth_level = 2
    config.strategy = DummyTuneStrategy()

    with prepare_for_jit_tuning():
        model = ParentWithChild()

    model(torch.randn(2, 4))
    report_tune_run_end()

    report = json.loads(report_path.read_text())
    assert len(report["inspection_details"]) == 2
    module = report["modules"][0]
    parent = next(detail for detail in report["inspection_details"] if detail["module_id"] == module["module_id"])
    child = next(detail for detail in report["inspection_details"] if detail["parent_module_id"] == parent["module_id"])

    assert parent["module_class"].endswith("ParentWithChild")
    assert parent["state"] == "recording"
    assert parent["dtypes"] == ["torch.float32"]
    assert parent["child_module_ids"] == [child["module_id"]]
    assert parent["graphs"][0]["input_spec"]["tensor_data"][0]["dtype"] == "torch.float32"
    assert child["module_class"] == "torch.nn.modules.linear.Linear"
    assert child["state"] == "recording"
    assert child["dtypes"] == ["torch.float32"]
    assert child["graphs"][0]["input_spec"]["tensor_data"][0]["dtype"] == "torch.float32"
    assert module["module_name"] == "ParentWithChild"


def test_jit_eager_accumulates_inspection_details_for_multiple_top_modules(mocker, tmp_path):
    """Eager tuning should keep inspection details from every top-level module."""

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.cache_dir = tmp_path / "cache"
    mock_config.tuning_data_output_path = report_path

    config.mode = JITMode.TUNE_EAGER
    config.device = torch.device("cpu")
    config.dry_run = True
    config.dry_run_failure_probability = 0.0
    config.max_depth_level = 1
    config.strategy = DummyTuneStrategy()

    with prepare_for_jit_tuning():
        encoder = Encoder()
        decoder = Decoder()

    encoder(torch.randn(2, 4))
    decoder(torch.randn(2, 4))
    report_tune_run_end()

    report = json.loads(report_path.read_text())
    assert [module["module_name"] for module in report["modules"]] == ["Encoder", "Decoder"]
    assert len(report["inspection_details"]) == 2
    assert {detail["module_id"] for detail in report["inspection_details"]} == {
        module["module_id"] for module in report["modules"]
    }
    assert {detail["module_name"] for detail in report["inspection_details"]} == {"Encoder", "Decoder"}


def test_jit_eager_records_inspection_once_from_top_module_before_child_fallback(mocker, tmp_path):
    """Fallback tuning should not refresh inspection details from child attempts."""

    class ParentWithChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.child(x)

    class ParentFailsStrategy(TuneStrategy):
        def _describe_parts(self) -> list[str]:
            return ["Parent fails, child succeeds."]

        def to_json_dict(self) -> dict:
            return {}

        def _tune(self, module, name, graph_spec, data, device, cache_dir):
            if module.__class__.__name__ == "ParentWithChild":
                raise RuntimeError("parent tune failed")
            return DummyBackend()

    report_path = tmp_path / "report.json"
    mock_config = mocker.patch("aitune.torch.tune_data.reporting.config")
    mock_config.cache_dir = tmp_path / "cache"
    mock_config.tuning_data_output_path = report_path

    config.mode = JITMode.TUNE_EAGER
    config.device = torch.device("cpu")
    config.detect_graph_breaks = False
    config.max_depth_level = 2
    config.strategy = ParentFailsStrategy()

    with prepare_for_jit_tuning():
        model = ParentWithChild()

    model(torch.randn(2, 4))
    report_tune_run_end()

    report = json.loads(report_path.read_text())
    assert len(report["inspection_details"]) == 2
    parent = next(detail for detail in report["inspection_details"] if detail["parent_module_id"] is None)
    child = next(detail for detail in report["inspection_details"] if detail["parent_module_id"] == parent["module_id"])

    assert parent["module_class"].endswith("ParentWithChild")
    assert child["module_class"] == "torch.nn.modules.linear.Linear"
    assert child["allowed_to_tune"] is False


@requires_cuda
@pytest.mark.parametrize("scenario", ["success", "correctness_error", "backend_build_error"])
def test_jit_tuning_success(mock_trt_backend, torch_device, scenario):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_EAGER

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    if scenario == "success":
        mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE)
    elif scenario == "correctness_error":
        mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE * 10)  # wrong output shapes returned
    elif scenario == "backend_build_error":
        mock_trt_backend.infer.side_effect = Exception("Backend build error")

    with torch.no_grad():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    if scenario == "success":
        assert re.match(r".*ToyTorchModel.*state=tuned.*(MockTensorRTBackend).*call_count=1", sink.output[1])
    elif scenario == "correctness_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=1", sink.output[1])
    elif scenario == "backend_build_error":
        assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=1", sink.output[1])


@requires_cuda
def test_jit_tuning_with_module_hooks(mock_trt_backend, torch_device):
    config.min_samples = 2
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_EAGER

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

    with torch.no_grad():
        pipeline(torch.randn(1))
        assert hooks_history == ["pre_hook", "forward_hook"]
        hooks_history.clear()
        pipeline(torch.randn(2))
        assert hooks_history == ["pre_hook", "forward_hook"]

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*TestNet.*state=tuned.*(MockTensorRTBackend).*call_count=2", sink.output[1])


@requires_cuda
def test_jit_tuning_graph_break(mock_trt_backend, torch_device, mocker):
    config.max_depth_level = 2
    config.dry_run = False
    config.detect_graph_breaks = True
    config.mode = JITMode.TUNE_EAGER

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)

    mocker.patch("aitune.torch.jit.patched_module.GraphBreakDetector.detect", side_effect=GraphBreakException)
    with torch.no_grad():
        for x in inputs:
            pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(graph break).*call_count=1", sink.output[1]), sink.output[1]
    assert re.match(r".*Linear.*state=eager.*(graph break).*call_count=1", sink.output[2]), sink.output[2]


@requires_cuda
def test_jit_tuning_skip_module(mock_trt_backend, torch_device, mocker):
    config.max_depth_level = 2
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_EAGER

    config.skip_modules = ["ToyTorchModel"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    with torch.no_grad():
        for x in inputs:
            pipeline(x)

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
    config.max_depth_level = 2
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_EAGER

    config.skip_modules = ["Linear"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)

    with torch.no_grad():
        for x in inputs:
            pipeline(x)

    assert len(PatchedModule.heads) == 1

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_called()
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*ToyTorchModel.*state=eager.*(no better tuned version).*call_count=1", sink.output[1])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[2])
    assert re.match(r".*Linear.*state=skipped.*call_count=1", sink.output[3])


@requires_cuda
def test_jit_tune_raises_disk_space_error_on_enospc(mock_trt_backend, torch_device, mocker):
    """ENOSPC during a cache-write step must halt the JIT tune, not fall back to eager.

    Before the fix, the broad ``except Exception`` in :meth:`PatchedModule.tune`
    caught ``OSError(ENOSPC)`` and marked the module ``state=eager`` with
    ``no better tuned version`` — giving no indication that the cache disk was full.
    """
    config.dry_run = False
    config.inspect_mode = False
    config.detect_graph_breaks = False

    mock_trt_backend.infer.return_value = torch.randn(1, OUTPUT_SIZE)

    # Force the cache-write path inside the tune try-block to hit ENOSPC.
    mocker.patch.object(
        PatchedModule,
        "_create_graph_cache_dir",
        side_effect=OSError(errno.ENOSPC, "No space left on device"),
    )

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    with pytest.raises(DiskSpaceError), torch.no_grad():
        for x in pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device):
            pipeline(x)


@requires_cuda
def test_jit_tuning_no_modules(mock_trt_backend, torch_device):
    config.dry_run = False
    config.detect_graph_breaks = False
    config.mode = JITMode.TUNE_EAGER

    # the following class has not parameters and should be skipped
    class IdentityModule(torch.nn.Module):
        def forward(self, x):
            return x

    with prepare_for_jit_tuning():
        model = IdentityModule()
        model = model.to(torch_device)

    for _ in range(2):
        model(torch.randn(1, 10))

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_not_called()
    assert PRINT_HIERARCHY_NO_MODULES_HEADER in sink.output[0]


# ── capture_outputs hook preservation ────────────────────────────────────────


def test_jit_external_hook_preserved_in_forward_hooks_after_restore_proxy_cycle():
    """A forward hook registered after wrapping must survive _restore_original_forward/_proxy_forward."""
    inner = _SimpleJitModule()
    patched = PatchedModule(inner)

    hook_calls = []
    inner.register_forward_hook(lambda mod, inp, out: hook_calls.append(out))

    assert len(inner._forward_hooks) == 1

    patched._restore_original_forward()
    assert len(inner._forward_hooks) == 0  # cleared during restore — expected

    patched._proxy_forward()
    assert len(inner._forward_hooks) == 1  # must be restored


def test_jit_external_hook_fires_exactly_once_after_restore_proxy_cycle():
    """After the cycle the hook fires exactly once per forward call — no double-firing."""
    inner = _SimpleJitModule()
    patched = PatchedModule(inner)

    hook_calls = []
    inner.register_forward_hook(lambda mod, inp, out: hook_calls.append(out))

    patched._restore_original_forward()
    patched._proxy_forward()

    inner(1)
    assert len(hook_calls) == 1


def test_jit_external_pre_hook_preserved_after_restore_proxy_cycle():
    """A forward_pre_hook registered after wrapping must also survive the cycle."""
    inner = _SimpleJitModule()
    patched = PatchedModule(inner)

    pre_calls = []
    inner.register_forward_pre_hook(lambda mod, inp: pre_calls.append(inp))

    assert len(inner._forward_pre_hooks) == 1

    patched._restore_original_forward()
    assert len(inner._forward_pre_hooks) == 0

    patched._proxy_forward()
    assert len(inner._forward_pre_hooks) == 1


def test_jit_first_proxy_forward_without_prior_restore_does_not_crash():
    """_proxy_forward before any _restore_original_forward must not raise and uses init-time hooks."""
    inner = _SimpleJitModule()
    patched = PatchedModule(inner)

    patched._proxy_forward()  # must not raise
    assert inner._forward_hooks == patched._current_forward_hooks


def test_jit_tuning_backend_added_hooks():
    """Test that backend added hooks are preserved after a proxy forward."""
    hook_calls = []

    class TestStrategyWhichAddsHooks(TuneStrategy):
        def _tune(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs,
        ):
            # Imitate a backend which adds hooks to the module.
            module.register_forward_pre_hook(lambda mod, inp: hook_calls.append("backend pre hook"))
            module.register_forward_hook(lambda mod, inp, out: hook_calls.append("backend hook"))
            backend = Mock(spec=Backend)
            backend.name = "TestBackend"
            backend.describe.return_value = "TestBackend"
            return backend

        def describe(self):
            return "Test strategy which adds hooks."

        def _describe_parts(self):
            return ["Test strategy which adds hooks."]

        def to_json_dict(self):
            return {"type": "test_strategy_which_adds_hooks"}

    config.strategy = TestStrategyWhichAddsHooks()
    config.min_parameters = -1

    with prepare_for_jit_tuning():
        module = _SimpleJitModule()

    module.register_forward_pre_hook(lambda mod, inp: hook_calls.append("module pre hook"))
    module.register_forward_hook(lambda mod, inp, out: hook_calls.append("module hook"))

    module(1)

    hook_calls.clear()
    module(1)
    assert hook_calls == ["backend pre hook", "module pre hook", "backend hook", "module hook"]


@requires_cuda
def test_forward_method_should_have_same_signature(mock_trt_backend, torch_device):
    config.dry_run = False
    config.mode = JITMode.TUNE_EAGER

    class _SimpleJitModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, y, z, pos="test"):
            """Test forward method."""
            return x, y, z, pos

    with prepare_for_jit_tuning():
        model = _SimpleJitModule().to(torch_device)

    # in init state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}
    model(1, 2, 3)
    # in recording state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}
    # in tuned state
    model(1, 2, 3)  # we are in recording state
    assert set(inspect.signature(model.forward).parameters.keys()) == {"x", "y", "z", "pos"}


def test_get_fully_qualified_name():
    """_get_fully_qualified_name returns dot-separated path; index only for duplicates."""
    PatchedModule.fq_name_counter.clear()

    root = PatchedModule(torch.nn.Linear(1, 1))
    root._parent = None
    root._fq_name = "Linear"
    assert root._get_fully_qualified_name() == "Linear"

    parent = PatchedModule(torch.nn.Linear(1, 1))
    parent._parent = None
    parent._fq_name = "Model"
    child = PatchedModule(torch.nn.Linear(1, 1))
    child._parent = parent
    child._fq_name = "Linear"
    assert child._get_fully_qualified_name() == "Model.Linear"

    # 3-level hierarchy: grandparent → parent → child
    PatchedModule.fq_name_counter.clear()
    grandparent = PatchedModule(torch.nn.Linear(1, 1))
    grandparent._parent = None
    grandparent._fq_name = "GrandParent"
    mid = PatchedModule(torch.nn.Linear(1, 1))
    mid._parent = grandparent
    mid._fq_name = "GrandParent.Mid"
    grandchild = PatchedModule(torch.nn.Linear(1, 1))
    grandchild._parent = mid
    assert grandchild._get_fully_qualified_name() == "GrandParent.Mid.Linear"

    # Duplicate path: second "Linear" at root gets index 1
    PatchedModule.fq_name_counter.clear()
    first = PatchedModule(torch.nn.Linear(1, 1))
    first._parent = None
    first._fq_name = "Linear"
    second = PatchedModule(torch.nn.Linear(1, 1))
    second._parent = None
    second._fq_name = "Linear"
    assert first._get_fully_qualified_name() == "Linear"
    assert second._get_fully_qualified_name() == "Linear.1"


def test_each_module_has_unique_cache_dir():
    config.min_samples = 2
    config.dry_run = False
    config.mode = JITMode.TUNE_EAGER

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
    config.mode = JITMode.TUNE_EAGER
    config.min_parameters = 1e10
    config.skip_modules = ["ToyTorchModel"]

    with prepare_for_jit_tuning():
        pipeline = ToyComplexPipeline().to(torch_device)

    inputs = pipeline.inputs(batch_sizes=[1, 2, 4], device=torch_device)
    with torch.no_grad():
        for x in inputs:
            pipeline(x)

    assert len(PatchedModule.heads) == 0

    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)

    mock_trt_backend.build.assert_not_called()
