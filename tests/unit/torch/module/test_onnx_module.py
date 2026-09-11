# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a real ONNX graph through recording and throughput tuning."""

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from aitune.torch import MaxThroughputStrategy, Module, tune
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_onnx_module_record_and_tune(onnx_add_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source = OnnxModule(onnx_add_path)

    x = torch.randn(3, 4, device=device).T  # Exercise non-contiguous input lifetime.

    strategy = MaxThroughputStrategy([TorchEagerBackend()], profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
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


@pytest.mark.parametrize("named", [False, True])
def test_onnx_module_onnx_runtime_backend(onnx_add_path, named):
    from aitune.torch import OneBackendStrategy
    from aitune.torch.backend import ONNXRuntimeBackend

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    source = OnnxModule(onnx_add_path)
    x = torch.randn(4, 3, device="cuda")
    torch.testing.assert_close(source(x)["y"], x * 2)
    strategy = OneBackendStrategy(ONNXRuntimeBackend(), profiling_config=ProfilingConfig(batch_sizes=[1, 2]))
    strategy.enable_performance_validation(False)
    module = Module(source, "onnx-runtime-double", strategy=strategy)
    dataset = DynamicShapeDataset([{"x": row} if named else (row,) for row in x])
    try:
        tune(module, dataset, batch_sizes=[1, 2], device="cuda", ignore_failing_modules=False)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, ONNXRuntimeBackend)
        assert backend._onnx_model_artifact.path == onnx_add_path
        assert source._session is None
        actual = module(x=x[:2]) if named else module(x[:2])
        torch.testing.assert_close(actual["y"], x[:2] * 2)
        restored = ONNXRuntimeBackend.from_dict(None, backend.to_dict())
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
        module = load(module, checkpoint)
        (backend,) = module.module.backends.values()
        assert isinstance(backend, ONNXRuntimeBackend)
        torch.testing.assert_close(module(x=x[:2])["y"], x[:2] * 2)
    finally:
        # Loading deploys the backend; release its session directly for test cleanup.
        for backend in module.module.backends.values():
            backend._deactivate()
        source.deactivate()
