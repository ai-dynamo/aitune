# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original ONNX tensor names survive the ordinary torch recording workflow."""

import pickle
from collections import OrderedDict

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.backend.tensorrt.modelopt_calibration import prepare_calibration_data
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.module.recording_module import RecordingModule
from aitune.torch.module.tensor_spec import TensorSpec


@pytest.fixture
def named_source(tmp_path):
    path = tmp_path / "source.onnx"
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Sub", ["input.1", "kwargs"], ["output.1"])],
            "subtract",
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", 3]) for name in ["input.1", "kwargs"]],
            [helper.make_tensor_value_info("output.1", TensorProto.FLOAT, ["N", 3])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.save(model, path)
    source = OnnxModule(path)
    yield source
    source.deactivate()


def _sample(layout, batch):
    x, y = torch.full((batch, 3), 5.0), torch.ones(batch, 3)
    if layout == "positional":
        return (x, y), {}
    if layout == "named":
        return (), {"kwargs": y, "input.1": x}  # Deliberately reverse ONNX graph order.
    return (x,), {"kwargs": y}


def _record(source, layout, tmp_path):
    recorder = RecordingModule(source, "onnx", cache_dir_resolver=lambda: tmp_path / "onnx")
    with global_context:
        for batch in [1, 4]:
            global_context.set(BATCH_SIZE_KEY, batch)
            args, kwargs = _sample(layout, batch)
            torch.testing.assert_close(recorder(*args, **kwargs)["output.1"], torch.full((batch, 3), 4.0))
    return recorder, recorder.graph_specs[0]


@pytest.mark.parametrize("layout", ["positional", "named", "mixed"])
def test_recording_uses_torch_shapes_and_preserves_original_names(named_source, layout, tmp_path):
    recorder, graph = _record(named_source, layout, tmp_path)

    class Subtract(torch.nn.Module):
        def forward(self, x, y):
            return {"output.1": x - y}

    eager = RecordingModule(Subtract(), "torch", cache_dir_resolver=lambda: tmp_path / "torch")
    with global_context:
        for batch in [1, 4]:
            global_context.set(BATCH_SIZE_KEY, batch)
            eager(torch.full((batch, 3), 5.0), torch.ones(batch, 3))
    for observed, expected in zip(
        graph.input_spec.tensor_specs, eager.graph_specs[0].input_spec.tensor_specs, strict=True
    ):
        assert observed.shape == expected.shape
        assert observed.min_shape == expected.min_shape == [1, 3]
        assert observed.max_shape == expected.max_shape == [4, 3]
        assert observed.get_batch_axis_multipliers() == expected.get_batch_axis_multipliers() == {0: 1}
    for metadata in [graph.input_spec, graph.post_input_spec]:
        assert {spec.name for _, spec in metadata.tensor_data} == {"input.1", "kwargs"}
    assert graph.output_spec.tensor_specs[0].name == "output.1"
    restored = GraphSpec.from_dict(pickle.loads(pickle.dumps(graph.to_dict())))
    args, kwargs = restored.make_batch(*_sample(layout, 1), 3)
    normalized = restored.forward_signature.normalize(args, kwargs)
    assert all(
        locator.get_value(normalized.arguments).shape == (3, 3) for locator, _ in restored.input_spec.tensor_data
    )
    assert restored.get_effective_input_shapes(*restored.input_spec.tensor_data[0]) == ([1, 3], [4, 3], [4, 3])
    calibration = prepare_calibration_data([_sample(layout, 1)], restored)
    assert set(calibration) == {"input.1", "kwargs"}
    assert calibration["input.1"].sum() == 15
    assert calibration["kwargs"].sum() == 3
    assert len(recorder.graph_specs) == 1


@pytest.mark.parametrize("layout", ["positional", "named", "mixed"])
def test_onnx_backend_skips_export_and_roundtrips_names(named_source, layout, tmp_path, monkeypatch):
    from aitune.torch.backend.onnx_runtime_backend import ONNXRuntimeBackend
    from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
    from aitune.torch.module.tuned_module import TunedModule

    recorder, graph = _record(named_source, layout, tmp_path)
    monkeypatch.setattr(ONNXRuntimeBackend, "_get_execution_providers", lambda self: ["CPUExecutionProvider"])

    def unexpected_export(*args, **kwargs):
        pytest.fail("An existing ONNX model must not be exported")

    monkeypatch.setattr(ONNXExporter, "export", unexpected_export)
    backend = ONNXRuntimeBackend()
    backend._device = torch.device("cpu")
    restored = None
    try:
        backend._build(named_source, graph, recorder.samples_for_graph_spec(graph), tmp_path)
        assert backend._onnx_model_artifact.path == named_source.path
        assert named_source._session is None
        state = pickle.loads(pickle.dumps(backend.to_dict()))
        restored = ONNXRuntimeBackend.from_dict(None, state)
        restored.activate()
        tuned = TunedModule(OrderedDict({graph.input_spec: restored}), "onnx", graph.forward_signature)
        for batch in [1, 3]:
            args, kwargs = _sample(layout, batch)
            torch.testing.assert_close(tuned(*args, **kwargs)["output.1"], torch.full((batch, 3), 4.0))
        assert "input_names" not in graph.forward_signature.to_dict()
    finally:
        backend._deactivate()
        if restored is not None:
            restored._deactivate()


def test_tensor_names_do_not_affect_graph_identity():
    original = TensorSpec.from_tensor(torch.ones(2, 3), batch_size=2)
    named = TensorSpec.from_dict(original.to_dict())
    named.name = "input.1"
    assert named == original
    assert hash(named) == hash(original)
    assert pickle.loads(pickle.dumps(named)).name == "input.1"
    assert TensorSpec.from_dict(named.to_dict()).name == "input.1"
    restored = TensorSpec.__new__(TensorSpec)
    restored.__setstate__((
        None,
        {key: value for key, value in original.to_dict().items() if key not in {"name", "type"}},
    ))
    assert restored.name is None
    assert restored == original


@pytest.mark.parametrize("layout", ["positional", "named", "mixed"])
@pytest.mark.parametrize("profile_mode", ["single", "samples", "explicit", "user"])
def test_tensorrt_profiles_and_bindings_use_original_names(named_source, layout, profile_mode, tmp_path):
    from types import SimpleNamespace

    from aitune.torch.backend.tensorrt.tensorrt_backend import ProfileMode, TensorRTBackend
    from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
    from aitune.torch.dynamic_shapes import BatchDim

    _, graph = _record(named_source, layout, tmp_path)
    backend = TensorRTBackend()
    backend._graph_spec = graph
    expected = ([1, 3], [4, 3], [4, 3])
    if profile_mode == "samples":
        backend._config.profiles = ProfileMode.SAMPLES_USED
        expected = ([2, 3], [2, 3], [2, 3])
    elif profile_mode == "explicit":
        graph.dynamic_shapes = {
            loc.path: (BatchDim("batch", min=1, opt=2, max=8), 3) for loc, _ in graph.input_spec.tensor_data
        }
        expected = ([1, 3], [2, 3], [8, 3])
    elif profile_mode == "user":
        profile = TensorRTProfile()
        for locator, _ in graph.input_spec.tensor_data:
            profile.add_input_shape(locator.path, (1, 3), (2, 3), (8, 3))
        backend._config.profiles = [profile]
        expected = ([1, 3], [2, 3], [8, 3])
    profiles = backend.get_profiles(graph, [_sample(layout, 2)])
    assert len(profiles) == 1
    assert set(profiles[0]) == {"input.1", "kwargs"}
    for shapes in profiles[0].values():
        assert tuple(list(shape) for shape in shapes) == expected
    backend._engine_info = SimpleNamespace(input_names=["input.1", "kwargs"])
    inputs = backend._prepare_inputs(*_sample(layout, 2))
    torch.testing.assert_close(inputs["input.1"], torch.full((2, 3), 5.0))
    torch.testing.assert_close(inputs["kwargs"], torch.ones(2, 3))
    backend._output_object = {"output.1": None}
    backend._output_allocator = SimpleNamespace(outputs={"output.1": inputs["input.1"] - inputs["kwargs"]})
    torch.testing.assert_close(backend._prepare_outputs_for_return()["output.1"], torch.full((2, 3), 4.0))


def test_torch_export_uses_the_same_tensor_name_resolution(tmp_path):
    from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter

    class Double(torch.nn.Module):
        def forward(self, x):
            return x * 2

    source = Double()
    recorder = RecordingModule(source, "torch", cache_dir_resolver=lambda: tmp_path / "torch")
    with global_context:
        for batch in [1, 4]:
            global_context.set(BATCH_SIZE_KEY, batch)
            recorder(torch.ones(batch, 3))
    graph = recorder.graph_specs[0]
    graph.input_spec.tensor_specs[0].name = "original.input"
    graph.output_spec.tensor_specs[0].name = "original.output"
    exporter = ONNXExporter(tmp_path / "export.onnx", use_dynamo=False)
    assert exporter._create_dynamic_axes(graph) == {"original.input": [0], "original.output": [0]}
    exported = onnx.load(exporter.export(source, ((torch.ones(2, 3),), {}), graph))
    assert [value.name for value in exported.graph.input] == ["original.input"]
    assert [value.name for value in exported.graph.output] == ["original.output"]
