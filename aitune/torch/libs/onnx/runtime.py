# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ONNX Runtime execution with Torch inputs."""

from logging import getLogger

import numpy as np
import onnxruntime
import torch

from aitune.torch.libs.cuda.memory import memcpy_to_torch

logger = getLogger(__name__)

_TORCH_DTYPE_TO_NUMPY: dict[torch.dtype, type] = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def run_onnx(
    session: onnxruntime.InferenceSession, inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Bind tensors directly and return outputs on the requested device."""
    # Keep contiguous buffers alive until inference completes.
    inputs = {name: value.contiguous() for name, value in inputs.items()}
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
        np_dtype = _TORCH_DTYPE_TO_NUMPY.get(value.dtype)
        if np_dtype is None:
            raise ValueError(f"Unsupported tensor dtype for ONNX Runtime IOBinding: {value.dtype}")
        io_binding.bind_input(
            name=name,
            device_type=value.device.type,
            device_id=value.device.index or 0,
            element_type=np_dtype,
            shape=list(value.shape),
            buffer_ptr=value.data_ptr(),
        )


def _bind_outputs(
    io_binding: onnxruntime.IOBinding, session: onnxruntime.InferenceSession, device: torch.device
) -> None:
    """Tell ORT to allocate all outputs on the requested device.

    ORT owns the output buffers; shapes are resolved at inference time.
    Tensors are retrieved after inference via ``_collect_outputs``.
    """
    for output in session.get_outputs():
        io_binding.bind_output(output.name, device.type, device.index or 0)


def _collect_outputs(
    io_binding: onnxruntime.IOBinding, session: onnxruntime.InferenceSession, device: torch.device
) -> dict[str, torch.Tensor]:
    """Collect CPU outputs as arrays or copy CUDA outputs directly to Torch."""
    if device.type == "cpu":
        return {
            node.name: torch.from_numpy(value)
            for node, value in zip(session.get_outputs(), io_binding.copy_outputs_to_cpu(), strict=True)
        }
    return {
        node.name: memcpy_to_torch(ort_val.data_ptr(), list(ort_val.shape()), ort_val.data_type(), device)
        for node, ort_val in zip(session.get_outputs(), io_binding.get_outputs(), strict=True)
    }


def prepare_onnx_inputs(session: onnxruntime.InferenceSession, args: tuple, kwargs: dict) -> dict[str, torch.Tensor]:
    """Match positional and named inputs to the original ONNX graph names."""
    names = [node.name for node in session.get_inputs()]
    if len(args) > len(names) or set(names[: len(args)]) & kwargs.keys():
        raise TypeError("Too many positional inputs or duplicate ONNX input names")
    inputs = dict(zip(names, args, strict=False)) | kwargs
    if inputs.keys() != set(names):
        raise TypeError(f"Expected ONNX inputs {names}, got {list(inputs)}")
    return inputs
