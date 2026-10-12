# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run exported deployment artifacts through their native runtimes, without AITune."""

from collections.abc import Callable
from pathlib import Path

import torch

from aitune.records import DeploymentArtifact

NativeRunner = Callable[[Path, list[torch.Tensor]], list[torch.Tensor]]


def _run_pt2(path: Path, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run an AOT Inductor package."""
    runner = torch._inductor.aoti_load_package(str(path), device_index=inputs[0].device.index or 0)
    result = runner(*inputs)
    return list(result) if isinstance(result, (list, tuple)) else [result]


def _run_tensorrt_plan(path: Path, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Deserialize a TensorRT plan and execute it on torch-allocated buffers."""
    import tensorrt as trt

    # TensorRT needs the logger and the runtime to outlive the engine and its execution context.
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None, f"cannot deserialize {path}"
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    for name, tensor in zip(input_names, inputs, strict=True):
        context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.contiguous().data_ptr())
    outputs = []
    for name in names:
        if name in input_names:
            continue
        dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[engine.get_tensor_dtype(name)]
        output = torch.empty(tuple(context.get_tensor_shape(name)), dtype=dtype, device=inputs[0].device)
        context.set_tensor_address(name, output.data_ptr())
        outputs.append(output)
    stream = torch.cuda.current_stream(inputs[0].device)
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def _run_onnx(path: Path, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run an ONNX model in a plain ONNX Runtime CUDA session."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    feed = {node.name: tensor.cpu().numpy() for node, tensor in zip(session.get_inputs(), inputs, strict=True)}
    return [torch.from_numpy(output).to(inputs[0].device) for output in session.run(None, feed)]


NATIVE_RUNNERS: dict[str, tuple[str, NativeRunner]] = {
    "pt2": ("aotinductor", _run_pt2),
    "tensorrt_plan": ("tensorrt", _run_tensorrt_plan),
    "onnx": ("onnxruntime", _run_onnx),
}


def run_exported_artifact(
    artifact: DeploymentArtifact, export_dir: Path, inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Copy the artifact files to `export_dir` and run the copy with the native runtime for its format."""
    runtime_name, runner = NATIVE_RUNNERS[artifact.model.format]
    assert artifact.runtime.name == runtime_name
    exported = artifact.model.export_files(export_dir / artifact.model.path.name)
    assert exported.is_file()
    with torch.no_grad():
        return runner(exported, inputs)
