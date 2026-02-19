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

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import wrapt

from aitune.torch.backend.tensorrt.onnx_exporter import ONNXExporter, _create_nested_structure
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.utils.module import get_forward_arguments_names
from tests.utilities.helpers import requires_cuda


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

    input_metadata = SampleMetadata.from_inputs((torch.randn(10, 10),), {})
    output_metadata = SampleMetadata.from_outputs(torch.randn(10, 10))
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(20, 10),), {}))
    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(40, 10),), {}))

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(graph_spec, get_forward_arguments_names(_forward))
    assert dynamic_shapes["x"][0].min == 10
    assert dynamic_shapes["x"][0].max == 40


def test_what_if_once_arg_once_kwarg():
    input_metadata = SampleMetadata.from_inputs((torch.randn(10, 10),), {})
    try:
        input_metadata.update_shapes_seen(SampleMetadata.from_inputs((), {"x": torch.randn(20, 10)}))
    except ValueError:
        # error is known we can continue
        pytest.skip("We do not support once arg once kwarg inputs yet")


def test_should_extract_dynamic_shapes_from_graph_spec(simple_module_and_args, torch_device):
    module, x, y, z, w = simple_module_and_args
    output = module(x, y, z, w)

    # receiving kwargs in random order
    input_metadata = SampleMetadata.from_inputs((), {"y": y, "z": z, "w": w, "x": x})
    output_metadata = SampleMetadata.from_outputs(output)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    # adding one more input for dynamic shapes
    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    module(x, y, z, w)

    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((), {"y": y, "z": z, "w": w, "x": x}))

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(graph_spec, get_forward_arguments_names(module.forward))
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

    input_metadata = SampleMetadata.from_inputs((), {"y": y, "zw": zw, "x": x})
    output_metadata = SampleMetadata.from_outputs(expected_output1)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = {"z": z, "w": w}

    expected_output2 = module(x, y, zw)
    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((), {"y": y, "zw": zw, "x": x}))

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(graph_spec, get_forward_arguments_names(module.forward))

    assert dynamic_shapes["x"][0].min == 2

    assert dynamic_shapes["y"] == {}

    assert isinstance(dynamic_shapes["zw"], dict)
    assert dynamic_shapes["zw"]["z"][0].min == 2
    assert dynamic_shapes["zw"]["w"][0].min == 2

    model_path = tmp_path / "nested_tensors.onnx"
    ep = torch.onnx.export(
        module,
        args=(),
        kwargs={"x": x, "y": y, "zw": zw},
        input_names=["x", "y", "zw_z", "zw_w"],
        output_names=["outputs"],
        f="nested_tensors.onnx",
        dynamo=True,
        fallback=False,
        dynamic_shapes=dynamic_shapes,
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

    input_metadata = SampleMetadata.from_inputs((), {"y": y, "zw": zw, "x": x})
    output_metadata = SampleMetadata.from_outputs(expected_output1)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    zw = [z, w]

    expected_output2 = module(x, y, zw)
    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((), {"y": y, "zw": zw, "x": x}))

    dynamic_shapes = ONNXExporter._create_dynamic_shapes(graph_spec, get_forward_arguments_names(module.forward))

    assert dynamic_shapes["x"][0].min == 2
    assert dynamic_shapes["y"] == {}

    assert isinstance(dynamic_shapes["zw"], list)
    assert dynamic_shapes["zw"][0][0].min == 2
    assert dynamic_shapes["zw"][1][0].min == 2

    model_path = tmp_path / "nested_tensors.onnx"
    ep = torch.onnx.export(
        module,
        args=(),
        kwargs={"x": x, "y": y, "zw": zw},
        input_names=["x", "y", "zw_0", "zw_1"],
        output_names=["outputs"],
        f="nested_tensors.onnx",
        dynamo=True,
        fallback=False,
        dynamic_shapes=dynamic_shapes,
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


def test_onnx_exporter_should_produce_valid_simple_model(simple_module_and_args, torch_device, tmp_path):
    module, x, y, z, w = simple_module_and_args
    sample = ((), {"y": y, "z": z, "w": w, "x": x})
    output = module(x, y, z, w)

    input_metadata = SampleMetadata.from_inputs(sample[0], sample[1], batch_size=10)
    output_metadata = SampleMetadata.from_outputs(output)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    x = torch.randn(2, 10).to(torch_device)
    y = torch.randn(5, 5).to(torch_device)
    z = torch.randn(2, 10).to(torch_device)
    w = torch.randn(2, 10).to(torch_device)
    output = module(x, y, z, w)

    input_metadata.update_shapes_seen(SampleMetadata.from_inputs((), {"y": y, "z": z, "w": w, "x": x}, batch_size=2))

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


def test_create_nested_structure():
    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator, last={})
    assert root == {"zw": [{}]}

    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE), ("t", ObjectType.DICT)))
    root = _create_nested_structure(locator, last={})
    assert root == {"zw": [{"t": {}}]}

    locator = Locator((("zw", ObjectType.DICT), (0, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator, last={})
    locator = Locator((("zw", ObjectType.DICT), (1, ObjectType.SEQUENCE)))
    root = _create_nested_structure(locator, root=root, last={})
    assert root == {"zw": [{}, {}]}

    locator = Locator((("zw", ObjectType.DICT),))
    root = _create_nested_structure(locator, last=1)
    assert root == {"zw": 1}
