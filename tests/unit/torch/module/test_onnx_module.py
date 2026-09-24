# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a real ONNX graph through recording and throughput tuning."""

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from aitune.torch import MaxThroughputStrategy, Module, tune
from aitune.torch.backend import ONNXRuntimeBackend
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule, onnx_checkpoint_placeholder
from aitune.torch.task.profiling import ProfilingConfig


@pytest.fixture
def onnx_add_path(tmp_path):
    path = tmp_path / "add.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])],
        "double",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)
    return path


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_onnx_module_simple_inference(onnx_add_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source = OnnxModule(onnx_add_path)

    x = torch.randn(3, 4, device=device).T  # Exercise non-contiguous input lifetime.

    torch.testing.assert_close(source(x)["y"], x * 2)
    torch.testing.assert_close(source(x=x)["y"], x * 2)


def test_onnx_module_record_and_tune(onnx_add_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda:0"

    source = OnnxModule(onnx_add_path)

    x = torch.randn(3, 4, device=device).T  # Exercise non-contiguous input lifetime.

    strategy = MaxThroughputStrategy([ONNXRuntimeBackend()], profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)

    module = Module(source, "onnx-double", strategy=strategy)
    try:
        dataset = DynamicShapeDataset([{"x": row} for row in x])
        tune(module, dataset, batch_sizes=[1, 2], device=device, ignore_failing_modules=False)

        assert len(module.graph_specs) == 1
        assert module.module.backends
        torch.testing.assert_close(module(x=x[:2])["y"], x[:2] * 2)

    finally:
        module.deactivate()
        source.deactivate()


@pytest.mark.parametrize("backend_type", ["ONNXRuntimeBackend", "TensorRTBackend"])
@pytest.mark.parametrize("named", [False, True])
def test_onnx_module_backend(onnx_add_path, named, backend_type):
    from aitune.torch import OneBackendStrategy
    from aitune.torch import backend as backends

    backend_cls = getattr(backends, backend_type)
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source = OnnxModule(onnx_add_path)
    x = torch.randn(4, 3, device="cuda")
    torch.testing.assert_close(source(x)["y"], x * 2)
    strategy = OneBackendStrategy(backend_cls(), profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    module = Module(source, "onnx-runtime-double", strategy=strategy)
    dataset = DynamicShapeDataset([{"x": row} if named else (row,) for row in x])
    try:
        tune(module, dataset, batch_sizes=[1, 2], device="cuda", ignore_failing_modules=False)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, backend_cls)
        if backend_type == "ONNXRuntimeBackend":
            assert backend._onnx_model_artifact.path == onnx_add_path
        assert source._session is None
        for batch in [1, 2]:
            actual = module(x=x[:batch]) if named else module(x[:batch])
            torch.testing.assert_close(actual["y"], x[:batch] * 2)
        restored = backend_cls.from_dict(None, backend.to_dict())
        try:
            restored.activate()
            actual = restored.infer(x=x[:2]) if named else restored.infer(x[:2])
            torch.testing.assert_close(actual["y"], x[:2] * 2)
        finally:
            restored.deactivate()
    finally:
        module.deactivate()
        source.deactivate()


def test_onnx_module_external_weights_checkpoint(tmp_path):
    from aitune.torch import OneBackendStrategy, load, save
    from aitune.torch.backend import ONNXRuntimeBackend

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    path = tmp_path / "weighted.onnx"
    weight = onnx.numpy_helper.from_array(torch.full((3,), 2.0).numpy(), name="scale")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["y"])],
        "scale",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
        initializer=[weight],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    source = OnnxModule(path)
    strategy = OneBackendStrategy(ONNXRuntimeBackend(), profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    module = Module(source, "onnx-external", strategy=strategy)
    x = torch.randn(4, 3, device="cuda")
    try:
        tune(module, DynamicShapeDataset([{"x": row} for row in x]), batch_sizes=[1, 2], device="cuda")
        checkpoint = tmp_path / "model.ait"
        save(module, checkpoint)
        module.deactivate()
        path.unlink()
        (tmp_path / "weights.bin").unlink()
        module = load(onnx_checkpoint_placeholder(), checkpoint)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, ONNXRuntimeBackend)
        torch.testing.assert_close(module(x=x[:2])["y"], x[:2] * 2)
    finally:
        # Loading deploys the backend; release its session directly for test cleanup.
        for backend in module.module.backends.values():
            backend._deactivate()
        source.deactivate()


def test_onnx_module_rejects_torch_quantization(onnx_add_path, tmp_path):
    from aitune.exceptions import AITuneUserInputError
    from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
    from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = OnnxModule(onnx_add_path)
    source(torch.ones(1, 3, device="cuda"))
    backend = TensorRTBackend(TensorRTBackendConfig(quantization_config=TorchQuantizationConfig()))
    backend._device = torch.device("cuda")
    with pytest.raises(AITuneUserInputError, match="Torch quantization requires a Torch module"):
        backend._build(source, None, None, tmp_path)
    assert source._session is None


@pytest.mark.parametrize("precision", ["fp16", "int8"])
def test_onnx_module_tensorrt_quantization(tmp_path, precision):
    from aitune.torch import OneBackendStrategy
    from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig
    from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCastConfig
    from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = tmp_path / "source.onnx"
    weight = onnx.numpy_helper.from_array(torch.full((16, 16, 1, 1), 0.5).numpy(), name="weight")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "weight"], ["y"], kernel_shape=[1, 1])],
        "convolution",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 16, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 16, 4, 4])],
        initializer=[weight],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    original_graph = path.read_bytes()
    original_weights = (tmp_path / "weights.bin").read_bytes()
    source = OnnxModule(path)
    config = ONNXAutoCastConfig() if precision == "fp16" else ONNXQuantizationConfig("int8", calibration_method="max")
    candidate = TensorRTBackend(TensorRTBackendConfig(quantization_config=config))
    copied = candidate._export_onnx(source, None, None, tmp_path / "copy")

    source = OnnxModule(copied)
    x = torch.ones(4, 16, 4, 4, device="cuda")
    torch.testing.assert_close(source(x)["y"], x * 8)
    strategy = OneBackendStrategy(candidate, profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    module = Module(source, "onnx-quantized", strategy=strategy)
    try:
        tune(
            module,
            DynamicShapeDataset([{"x": row} for row in x]),
            batch_sizes=[1, 2],
            device="cuda",
            ignore_failing_modules=False,
        )
        (backend,) = module.module.backends.values()
        assert isinstance(backend, TensorRTBackend)
        if precision == "int8":
            quantized = onnx.load(backend._engine_artifact.root / "onnx_ptq" / "model.onnx")
            assert any(node.op_type == "QuantizeLinear" for node in quantized.graph.node)
        for batch in [1, 2]:
            torch.testing.assert_close(module(x=x[:batch])["y"], x[:batch] * 8, rtol=1e-2, atol=1e-2)
        assert copied.read_bytes() == original_graph
        assert (copied.parent / "weights.bin").read_bytes() == original_weights
    finally:
        module.deactivate()
        source.deactivate()
