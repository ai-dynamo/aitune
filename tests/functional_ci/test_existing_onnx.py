# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune an existing ONNX file, with inline or external weights, through `OnnxModule`."""

from collections.abc import Callable
from pathlib import Path

import onnx
import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import Backend, ONNXRuntimeBackend, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.module.onnx_module import OnnxModule

from .lifecycle import TENSORRT_FP32, selected_backend
from .models import make_input, make_mlp

ONNX_BACKENDS: dict[str, Callable[[], Backend]] = {
    "tensorrt": lambda: TensorRTBackend(TensorRTBackendConfig(enable_tf32=False, use_cuda_graphs=False)),
    "onnxruntime-cuda": ONNXRuntimeBackend,
}


def _export_mlp(device: torch.device, directory: Path, external_data: bool) -> Path:
    """Export the toy MLP with input `x`, output `y`, and a dynamic batch axis."""
    path = directory / "mlp.onnx"
    torch.onnx.export(
        make_mlp(device),
        (make_input(device),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        dynamo=False,
    )
    if not external_data:
        return path
    external = directory / "external" / "mlp.onnx"
    external.parent.mkdir()
    onnx.save_model(
        onnx.load(str(path)), str(external), save_as_external_data=True, location="weights.bin", size_threshold=0
    )
    assert (external.parent / "weights.bin").is_file()
    return external


@pytest.mark.parametrize("external_data", [False, True], ids=["inline", "external-data"])
@pytest.mark.parametrize("backend_id", ONNX_BACKENDS)
def test_existing_onnx(backend_id: str, external_data: bool, device: torch.device, tmp_path):
    """Named ONNX inputs and outputs survive tuning and match the eager model across batch sizes."""
    backend = ONNX_BACKENDS[backend_id]()
    strategy = ait.OneBackendStrategy(backend).enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    tuned = ait.Module(OnnxModule(_export_mlp(device, tmp_path, external_data)), "onnx-mlp", strategy=strategy)
    samples = [{"x": row} for row in make_input(device, batch_size=4).unbind(0)]
    ait.tune(tuned, samples, batch_sizes=[1, 4], device=device)

    assert type(selected_backend(tuned)) is type(backend)
    reference = make_mlp(device)
    with torch.no_grad():
        for batch_size in (1, 3, 4):
            value = make_input(device, batch_size=batch_size, seed=20 + batch_size)
            outputs = tuned(x=value)
            assert set(outputs) == {"y"}
            torch.testing.assert_close(outputs["y"], reference(value), rtol=TENSORRT_FP32.rtol, atol=TENSORRT_FP32.atol)
