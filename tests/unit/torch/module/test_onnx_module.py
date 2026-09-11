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
