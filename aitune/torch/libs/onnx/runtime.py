# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ONNX Runtime execution with Torch inputs.

OnnxModule and ONNXRuntimeBackend use this path so input binding, stream synchronization, and output
collection agree for source artifacts and tuned sessions. Inputs are matched to actual ONNX names;
structured Python output reconstruction and recorded input mutations belong to the caller.
"""

import ctypes
from logging import getLogger

import onnxruntime
import torch
from onnx import TensorProto

from aitune.torch.libs.cuda.memory import memcpy_to_torch

logger = getLogger(__name__)

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, int] = {
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.float16: TensorProto.FLOAT16,
    torch.float32: TensorProto.FLOAT,
    torch.float64: TensorProto.DOUBLE,
    torch.int8: TensorProto.INT8,
    torch.int16: TensorProto.INT16,
    torch.int32: TensorProto.INT32,
    torch.int64: TensorProto.INT64,
    torch.uint8: TensorProto.UINT8,
    torch.bool: TensorProto.BOOL,
}


def run_onnx(
    session: onnxruntime.InferenceSession, inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Bind tensors directly and return outputs on the requested device.

    Contiguous input buffers stay alive through output collection.
    CUDA execution first waits for current-stream Torch writes because ORT uses its own stream.
    CPU results use NumPy when supported; CPU BF16 and CUDA results are copied into Torch storage.
    Returned names are the session's graph outputs, including any internal mutation results.
    """
    # Keep contiguous buffers alive until outputs have been collected.
    inputs = {name: value.contiguous() for name, value in inputs.items()}
    # ONNX Runtime reads the bound buffers on its own stream, so finish the Torch writes first.
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    binding = session.io_binding()

    _bind_inputs(binding, inputs)
    _bind_outputs(binding, session, device)

    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    return _collect_outputs(binding, session, device)


def _bind_inputs(io_binding: onnxruntime.IOBinding, inputs: dict[str, torch.Tensor]) -> None:
    """Bind contiguous CPU or CUDA tensors directly by their memory pointers."""
    for name, value in inputs.items():
        logger.debug("Binding input %s: device=%s shape=%s dtype=%s", name, value.device, value.shape, value.dtype)
        element_type = _TORCH_DTYPE_TO_ONNX.get(value.dtype)
        if element_type is None:
            raise ValueError(f"Unsupported tensor dtype for ONNX Runtime IOBinding: {value.dtype}")
        io_binding.bind_input(
            name=name,
            device_type=value.device.type,
            device_id=value.device.index or 0,
            element_type=element_type,
            shape=list(value.shape),
            buffer_ptr=value.data_ptr(),
        )


def _bind_outputs(
    io_binding: onnxruntime.IOBinding, session: onnxruntime.InferenceSession, device: torch.device
) -> None:
    """Bind all outputs to the requested device without preallocating buffers.

    Tensors are retrieved after inference via ``_collect_outputs``.
    """
    for output in session.get_outputs():
        io_binding.bind_output(output.name, device.type, device.index or 0)


def _collect_outputs(
    io_binding: onnxruntime.IOBinding, session: onnxruntime.InferenceSession, device: torch.device
) -> dict[str, torch.Tensor]:
    """Collect ORT outputs in Torch tensors on the requested device."""
    outputs = session.get_outputs()
    if device.type == "cpu":
        if any(node.type == "tensor(bfloat16)" for node in outputs):
            return {
                node.name: _cpu_output_to_torch(value)
                for node, value in zip(outputs, io_binding.get_outputs(), strict=True)
            }
        return {
            node.name: torch.from_numpy(value)
            for node, value in zip(outputs, io_binding.copy_outputs_to_cpu(), strict=True)
        }
    return {
        node.name: memcpy_to_torch(ort_val.data_ptr(), list(ort_val.shape()), ort_val.data_type(), device)
        for node, ort_val in zip(outputs, io_binding.get_outputs(), strict=True)
    }


def _cpu_output_to_torch(value: onnxruntime.OrtValue) -> torch.Tensor:
    if value.data_type() != "tensor(bfloat16)":
        # A NumPy view may alias a bound input, so copy it while the input is still alive.
        return torch.from_numpy(value.numpy().copy())
    # ORT cannot convert BF16 to NumPy, so copy the bits directly from the output buffer.
    result = torch.empty(value.shape(), dtype=torch.bfloat16, device="cpu")
    if result.nbytes:
        ctypes.memmove(result.data_ptr(), value.data_ptr(), result.nbytes)
    return result


def prepare_onnx_inputs(session: onnxruntime.InferenceSession, args: tuple, kwargs: dict) -> dict[str, torch.Tensor]:
    """Match positional and named inputs to the original ONNX graph names."""
    names = [node.name for node in session.get_inputs()]
    if len(args) > len(names) or set(names[: len(args)]) & kwargs.keys():
        raise TypeError("Too many positional inputs or duplicate ONNX input names")
    inputs = dict(zip(names, args, strict=False)) | kwargs
    if inputs.keys() != set(names):
        raise TypeError(f"Expected ONNX inputs {names}, got {list(inputs)}")
    return inputs
