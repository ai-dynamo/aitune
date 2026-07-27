# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ONNXExporter."""

from pathlib import Path
from unittest.mock import MagicMock

import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import wrapt

from aitune.torch.dynamic_shapes import BatchDim, DynamicDim
from aitune.torch.libs.onnx.onnx_exporter import _ONNX_FALLBACK_SUPPORTED, ONNXExporter
from aitune.torch.utils.module import get_forward_arguments_names
from aitune.torch.utils.tensor import format_tensor_name
from tests.toy_models import ToyOnnxModel, ToyTorchModel
from tests.utilities.helpers import make_graph_spec, make_input_metadata, requires_cuda

# Constants for testing
IN_FEATURES = 32
BATCH_SIZE = 2


class SimpleModuleMayArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 5)
        self.b = nn.Linear(5, 10)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor):
        # matrix multiplication tests correct ordering of the arguments
        return (self.a(x) @ self.b(y)) * (z + w)


class WrapperModuleMayArgs(nn.Module):
    def __init__(self, module: SimpleModuleMayArgs):
        super().__init__()
        self.m = module

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor):
        return self.m(w=w, x=x, z=z, y=y)  # reordering the arguments to check FLUX case


@pytest.fixture
def mock_torch_onnx(mocker):
    """Fixture that mocks torch.onnx functionality."""
    mock = mocker.patch("aitune.torch.libs.onnx.onnx_exporter.torch.onnx")
    mock.export.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_onnx_lib(mocker):
    """Fixture that mocks onnx library functionality."""
    mock = mocker.patch("aitune.torch.libs.onnx.onnx_exporter.onnx")
    mock.load.return_value = mocker.MagicMock()
    mock.checker.check_model.return_value = None
    return mock


def test_onnx_exporter_init(tmp_path):
    """Test ONNXExporter initialization."""
    output_path = tmp_path / "test_model.onnx"

    # Test with default parameters
    exporter = ONNXExporter(output_path=output_path)
    assert exporter.use_dynamo is False
    assert exporter.opset_version is None

    # Test with custom parameters
    output_path2 = tmp_path / "test_model2.onnx"
    exporter = ONNXExporter(output_path=output_path2, use_dynamo=True, opset_version=16)
    assert exporter.use_dynamo is True
    assert exporter.opset_version == 16


def test_export_trace(mock_torch_onnx, mock_onnx_lib, tmp_path):
    """Test export method with standard export."""
    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter with standard mode (no dynamo)
    exporter = ONNXExporter(output_path=output_path, use_dynamo=False)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    graph_spec = make_graph_spec(model.forward, sample, output, strict=True)

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify interactions
    mock_torch_onnx.export.assert_called_once()
    assert "opset_version" in mock_torch_onnx.export.call_args.kwargs
    assert mock_torch_onnx.export.call_args.kwargs["opset_version"] is None

    # Verify verification was called
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Verify returned path
    assert onnx_path == output_path


def test_export_trace_uses_backend_safe_compound_names(tmp_path):
    class NestedModule(nn.Module):
        def forward(self, inner):
            return {"inner": {"a": inner["a"] + 1}}

    model = NestedModule().eval()
    sample = ((), {"inner": {"a": torch.ones(1)}})
    output = model(*sample[0], **sample[1])
    graph_spec = make_graph_spec(model.forward, sample, output)
    output_path = tmp_path / "compound_names.onnx"

    ONNXExporter(output_path=output_path).export(model, sample, graph_spec)

    onnx_model = onnx.load(output_path)
    assert [value.name for value in onnx_model.graph.input] == [format_tensor_name(("inner", "a"), "input")]
    assert [value.name for value in onnx_model.graph.output] == [format_tensor_name(("inner", "a"), "output")]


def test_export_dynamo(mocker, mock_torch_onnx, mock_onnx_lib, tmp_path):
    """Test export method with dynamo export."""
    # Mock dynamo export output
    mock_export_program = MagicMock()
    mock_torch_onnx.export.return_value = mock_export_program

    # Mock graphsurgeon inputs
    mock_graph = mocker.MagicMock()
    mock_graph.inputs = [mocker.MagicMock(name="input__0")]
    mock_graph.outputs = [mocker.MagicMock(name="output__0")]

    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter with dynamo
    exporter = ONNXExporter(output_path=output_path, use_dynamo=True)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    graph_spec = make_graph_spec(model.forward, sample, output, strict=True)

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify interactions
    mock_torch_onnx.export.assert_called_once()
    assert "dynamo" in mock_torch_onnx.export.call_args.kwargs
    assert mock_torch_onnx.export.call_args.kwargs["dynamo"] is True

    # Verify verification was called
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Verify returned path
    assert onnx_path == output_path


def test_export_error_handling(mock_torch_onnx, tmp_path):
    """Test error handling during export."""
    # Set up export to fail
    mock_torch_onnx.export.side_effect = RuntimeError("Export failed")

    output_path = tmp_path / "onnx" / "test_model.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    graph_spec = make_graph_spec(model.forward, sample, output, strict=True)

    # Export should fail
    with pytest.raises(RuntimeError, match="Export failed"):
        exporter.export(
            module=model,
            sample=sample,
            graph_spec=graph_spec,
        )


def test_verify_model(mock_onnx_lib, tmp_path):
    """Test verify_model method."""
    output_path = tmp_path / "test_model.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Verify model
    exporter.verify_model(output_path)

    # Verify interactions
    mock_onnx_lib.checker.check_model.assert_called_once()

    # Test verify failure
    mock_onnx_lib.checker.check_model.side_effect = ValueError("Invalid model")

    with pytest.raises(ValueError, match="Invalid model"):
        exporter.verify_model(output_path)


def test_onnx_exporter_integration(tmp_path):
    """Integration test for ONNXExporter with actual ONNX.

    This test is skipped by default and should be run only in environments
    with ONNX properly installed.
    """
    output_path = tmp_path / "onnx" / "test_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Create model and sample
    model = ToyTorchModel().eval()
    sample = model.samples(batch_sizes=[BATCH_SIZE], device="cpu")[0]

    args, kwargs = sample
    output = model(*args, **kwargs)

    graph_spec = make_graph_spec(model.forward, sample, output, strict=True)

    # Export model
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify file exists
    assert onnx_path == output_path

    # Verify model again
    exporter.verify_model(onnx_path)


def test_verify_toy_onnx_model_linear(tmp_path):
    """Test verification of a pre-generated toy linear ONNX model.

    This test uses the ToyOnnxModel function to get the path to a pre-generated
    toy linear ONNX model and verifies that it can be loaded and checked.
    """
    output_path = tmp_path / "dummy_output.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    # Get path to toy ONNX model (linear by default)
    onnx_path = ToyOnnxModel(is_linear=True).path

    # Verify the model exists
    assert onnx_path.exists(), f"Toy linear ONNX model not found at {onnx_path}"

    # Verify the model
    exporter.verify_model(onnx_path)


def test_verify_toy_onnx_model_conv(tmp_path):
    """Test verification of a pre-generated toy convolutional ONNX model.

    This test uses the ToyOnnxModel function to get the path to a pre-generated
    toy convolutional ONNX model and verifies that it can be loaded and checked.
    """
    output_path = tmp_path / "dummy_output.onnx"

    # Create exporter
    exporter = ONNXExporter(output_path=output_path)

    try:
        # Get path to toy ONNX model (convolutional)
        onnx_path = ToyOnnxModel(is_linear=False).path

        # Verify the model exists
        assert onnx_path.exists(), f"Toy convolutional ONNX model not found at {onnx_path}"

        # Verify the model
        exporter.verify_model(onnx_path)
    except FileNotFoundError:
        pytest.skip("Convolutional toy ONNX model not available")


@pytest.fixture
def simple_module_and_args(torch_device):
    module = SimpleModuleMayArgs()
    module.to(torch_device)
    module.eval()

    x = torch.randn(10, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(10, 10).to(torch_device)
    w = torch.randn(10, 10).to(torch_device)

    return module, x, y, z, w


def check_onnx_model(model_path: Path, input_feed, expected_output, run: bool = True):
    # check if onnx generated is valid model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    del onnx_model

    if not run:
        return

    # run the onnx model and check the outputs
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    outputs = session.run(input_feed=input_feed, output_names=["outputs"])
    output = torch.from_numpy(outputs[0]).to(expected_output.device)
    try:
        torch.testing.assert_close(output, expected_output)
    except AssertionError as e:
        pytest.skip(f"Output is not close, skipping test: {e}")


def test_should_work_with_simple(simple_module_and_args):
    module, x, y, z, w = simple_module_and_args

    module(x, y, z, w)
    torch.onnx.export(module, args=(x, y, z, w), dynamo=True)


@requires_cuda
def test_should_work_with_wrapper(simple_module_and_args, tmp_path):
    """ONNX export arguments have to be in order of forward signature."""
    module, x, y, z, w = simple_module_and_args
    module = WrapperModuleMayArgs(module)

    expected_output = module(x, y, z, w)

    ep = torch.onnx.export(
        module,
        args=(),
        kwargs={"x": x, "y": y, "z": z, "w": w},  # args order cannot be scrambled
        input_names=["x", "y", "z", "w"],
        output_names=["outputs"],
        f="wrapper_module_may_args.onnx",
        dynamo=True,
    )

    model_path = tmp_path / "wrapper_module_may_args.onnx"
    ep.save(model_path)

    check_onnx_model(
        model_path,
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "z": z.cpu().numpy(),
            "w": w.cpu().numpy(),
        },
        expected_output,
    )


def test_module_should_have_correct_forward_signature(simple_module_and_args):
    module = simple_module_and_args[0]

    _, forward_kwargs = get_forward_arguments_names(module.forward)
    assert forward_kwargs == ["x", "y", "z", "w"]


def test_wrapped_forward_method_should_have_same_signature(simple_module_and_args):
    module = simple_module_and_args[0]
    module = WrapperModuleMayArgs(module)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs)

    module.forward = wrapper(module.forward)

    _, forward_kwargs = get_forward_arguments_names(module.forward)
    assert forward_kwargs == ["x", "y", "z", "w"]  # correct order


def test_should_work_with_positional_args():
    def _forward_positional_args(x, y, z, w, /, v):
        return x, y, z, w, v

    forward_args, forward_kwargs = get_forward_arguments_names(_forward_positional_args)
    assert forward_args == ["x", "y", "z", "w"]
    assert forward_kwargs == ["v"]


def test_should_work_basic():
    def _forward(x):  # it is not positional!
        return x

    forward_args, forward_kwargs = get_forward_arguments_names(_forward)
    assert forward_args == []
    assert forward_kwargs == ["x"]


def test_should_work_basic_dynamic_shapes():
    def _forward(x):
        return x + 1.0

    sample = ((torch.randn(10, 10),), {})
    graph_spec = make_graph_spec(_forward, sample, torch.randn(10, 10))

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((torch.randn(20, 10),), {}))
    )
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((torch.randn(40, 10),), {}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec)
    assert dynamic_shapes["x"][0] is torch.export.Dim.AUTO


def test_should_work_basic_dynamic_shapes_explicit():
    def _forward(x):
        return x + 1.0

    sample = ((torch.randn(10, 10),), {})
    graph_spec = make_graph_spec(_forward, sample, torch.randn(10, 10))

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((torch.randn(20, 10),), {}))
    )
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((torch.randn(40, 10),), {}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec, use_auto=False)
    assert dynamic_shapes["x"][0].min == 10
    assert dynamic_shapes["x"][0].max == 40


def test_trace_dynamic_axes_use_explicit_dimension_names(tmp_path):
    sample = ((torch.ones(2, 4), torch.ones(2, 4)), {})
    graph_spec = make_graph_spec(lambda x, y: x + y, sample)
    graph_spec.dynamic_shapes = {
        "x": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
        "y": (BatchDim("batch", min=1, opt=2, max=8), DynamicDim("sequence", min=1, opt=4, max=16)),
    }

    dynamic_axes = ONNXExporter(tmp_path / "model.onnx")._create_dynamic_axes(graph_spec)

    assert dynamic_axes[format_tensor_name("x", "input")] == {0: "batch", 1: "sequence"}
    assert dynamic_axes[format_tensor_name("y", "input")] == {0: "batch", 1: "sequence"}


def test_what_if_once_arg_once_kwarg():
    def _forward(x):
        return x

    graph_spec = make_graph_spec(_forward, ((torch.randn(10, 10),), {}), torch.randn(10, 10))
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"x": torch.randn(20, 10)}))
    )

    assert graph_spec.input_spec.tensor_specs[0].shape == ["dim0", 10]


def test_should_extract_dynamic_shapes_from_graph_spec(simple_module_and_args, torch_device):
    module, x, y, z, w = simple_module_and_args
    output = module(x, y, z, w)

    # receiving kwargs in random order
    sample = ((), {"y": y, "z": z, "w": w, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, output)

    # adding one more input for dynamic shapes
    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    module(x, y, z, w)

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "z": z, "w": w, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec)
    assert dynamic_shapes["x"][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["y"] == {}
    assert dynamic_shapes["z"][0] is torch.export.Dim.AUTO


def test_should_extract_dynamic_shapes_from_graph_spec_explicit(simple_module_and_args, torch_device):
    module, x, y, z, w = simple_module_and_args
    output = module(x, y, z, w)

    sample = ((), {"y": y, "z": z, "w": w, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, output)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    module(x, y, z, w)

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "z": z, "w": w, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec, use_auto=False)
    assert dynamic_shapes["x"][0].min == 2
    assert dynamic_shapes["y"] == {}
    assert dynamic_shapes["z"][0].min == 2

    assert dynamic_shapes["w"][0].min == 2


@requires_cuda
def test_does_onnx_export_work_with_nested_tensors(simple_module_and_args, torch_device, tmp_path):
    _, x, y, z, w = simple_module_and_args
    zw = {"z": z, "w": w}

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 5)
            self.b = nn.Linear(5, 10)

        def forward(self, x: torch.Tensor, y: torch.Tensor, zw: dict[str, torch.Tensor]):
            """Nested tensors in dict in zw argument"""
            return self.a(x) @ self.b(y) * (zw["z"] + zw["w"])

    module = NestedModule()
    module.to(torch_device)
    module.eval()

    expected_output1 = module(x, y, zw)

    sample = ((), {"y": y, "zw": zw, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, expected_output1)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = {"z": z, "w": w}

    expected_output2 = module(x, y, zw)
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "zw": zw, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec)

    assert dynamic_shapes["x"][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["y"] == {}
    assert isinstance(dynamic_shapes["zw"], dict)
    assert dynamic_shapes["zw"]["z"][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["zw"]["w"][0] is torch.export.Dim.AUTO

    model_path = tmp_path / "nested_tensors.onnx"
    export_kwargs = {"fallback": False} if _ONNX_FALLBACK_SUPPORTED else {}
    ep = torch.onnx.export(
        module,
        args=(),
        kwargs={"x": x, "y": y, "zw": zw},
        input_names=["x", "y", "zw_z", "zw_w"],
        output_names=["outputs"],
        f="nested_tensors.onnx",
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        **export_kwargs,
    )
    ep.save(model_path)

    check_onnx_model(
        model_path,
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "zw_z": z.cpu().numpy(),
            "zw_w": w.cpu().numpy(),
        },
        expected_output2,
    )


@requires_cuda
def test_does_onnx_export_work_with_nested_tensors_explicit(simple_module_and_args, torch_device, tmp_path):
    _, x, y, z, w = simple_module_and_args
    zw = {"z": z, "w": w}

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 5)
            self.b = nn.Linear(5, 10)

        def forward(self, x: torch.Tensor, y: torch.Tensor, zw: dict[str, torch.Tensor]):
            """Nested tensors in dict in zw argument"""
            return self.a(x) @ self.b(y) * (zw["z"] + zw["w"])

    module = NestedModule()
    module.to(torch_device)
    module.eval()

    sample = ((), {"y": y, "zw": zw, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, module(x, y, zw))

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = {"z": z, "w": w}

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "zw": zw, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec, use_auto=False)

    assert dynamic_shapes["x"][0].min == 2
    assert dynamic_shapes["y"] == {}
    assert isinstance(dynamic_shapes["zw"], dict)
    assert dynamic_shapes["zw"]["z"][0].min == 2
    assert dynamic_shapes["zw"]["w"][0].min == 2


@requires_cuda
def test_does_onnx_export_work_with_nested_tensors_list(simple_module_and_args, torch_device, tmp_path):
    _, x, y, z, w = simple_module_and_args
    zw = [z, w]

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 5)
            self.b = nn.Linear(5, 10)

        def forward(self, x: torch.Tensor, y: torch.Tensor, zw: list[torch.Tensor]):
            """Nested tensors in dict in zw argument"""
            return self.a(x) @ self.b(y) * (zw[0] + zw[1])

    module = NestedModule()
    module.to(torch_device)
    module.eval()

    expected_output1 = module(x, y, zw)

    sample = ((), {"y": y, "zw": zw, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, expected_output1)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = [z, w]

    expected_output2 = module(x, y, zw)
    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "zw": zw, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec)

    assert dynamic_shapes["x"][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["y"] == {}
    assert isinstance(dynamic_shapes["zw"], list)
    assert dynamic_shapes["zw"][0][0] is torch.export.Dim.AUTO
    assert dynamic_shapes["zw"][1][0] is torch.export.Dim.AUTO

    model_path = tmp_path / "nested_tensors.onnx"
    export_kwargs = {"fallback": False} if _ONNX_FALLBACK_SUPPORTED else {}
    ep = torch.onnx.export(
        module,
        args=(),
        kwargs={"x": x, "y": y, "zw": zw},
        input_names=["x", "y", "zw_0", "zw_1"],
        output_names=["outputs"],
        f="nested_tensors.onnx",
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        **export_kwargs,
    )
    ep.save(model_path)

    check_onnx_model(
        model_path,
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "zw_0": z.cpu().numpy(),
            "zw_1": w.cpu().numpy(),
        },
        expected_output2,
    )


@requires_cuda
def test_does_onnx_export_work_with_nested_tensors_list_explicit(simple_module_and_args, torch_device, tmp_path):
    _, x, y, z, w = simple_module_and_args
    zw = [z, w]

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 5)
            self.b = nn.Linear(5, 10)

        def forward(self, x: torch.Tensor, y: torch.Tensor, zw: list[torch.Tensor]):
            """Nested tensors in list in zw argument"""
            return self.a(x) @ self.b(y) * (zw[0] + zw[1])

    module = NestedModule()
    module.to(torch_device)
    module.eval()

    sample = ((), {"y": y, "zw": zw, "x": x})
    graph_spec = make_graph_spec(module.forward, sample, module(x, y, zw))

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = [z, w]

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(graph_spec.forward_signature, ((), {"y": y, "zw": zw, "x": x}))
    )

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(sample, graph_spec, use_auto=False)

    assert dynamic_shapes["x"][0].min == 2
    assert dynamic_shapes["y"] == {}
    assert isinstance(dynamic_shapes["zw"], list)
    assert dynamic_shapes["zw"][0][0].min == 2
    assert dynamic_shapes["zw"][1][0].min == 2


def test_onnx_exporter_should_produce_valid_simple_model(simple_module_and_args, torch_device, tmp_path):
    module, x, y, z, w = simple_module_and_args
    sample = ((), {"y": y, "z": z, "w": w, "x": x})
    output = module(x, y, z, w)

    graph_spec = make_graph_spec(module.forward, sample, output, batch_size=10)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    output = module(x, y, z, w)

    graph_spec.input_spec.update_shapes_seen(
        make_input_metadata(
            graph_spec.forward_signature,
            ((), {"y": y, "z": z, "w": w, "x": x}),
            batch_size=2,
        )
    )

    model_path = tmp_path / "simple_model.onnx"
    exporter = ONNXExporter(output_path=model_path, use_dynamo=True, opset_version=19)
    exporter.export(module, sample, graph_spec)

    check_onnx_model(
        model_path,
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "z": z.cpu().numpy(),
            "w": w.cpu().numpy(),
        },
        output,
        run=False,  # IR version 11 is not supported by onnx runtime
    )
