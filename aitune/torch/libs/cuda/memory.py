# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA memory utilities for GPU-to-GPU transfers."""

import ctypes

import torch

from aitune.torch.libs.cuda.cudart import cudart

_ORT_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int8)": torch.int8,
    "tensor(int16)": torch.int16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(uint8)": torch.uint8,
    "tensor(bool)": torch.bool,
}


def memcpy_to_torch(ptr: int, shape: list[int], ort_dtype: str, device: torch.device) -> torch.Tensor:
    """Copy a CUDA device buffer into a new torch tensor via D2D memcpy (no CPU round-trip).

    Args:
        ptr: Raw CUDA pointer to the source buffer (e.g. ``OrtValue.data_ptr()``).
        shape: Shape of the buffer.
        ort_dtype: ORT data type string (e.g. ``"tensor(float)"``).
        device: Target CUDA device for the output tensor.

    Returns:
        A new CUDA torch tensor containing a copy of the source buffer.

    Raises:
        KeyError: If ``ort_dtype`` is not a recognised ORT type string.
        RuntimeError: If the CUDA memcpy fails.
    """
    torch_dtype = _ORT_DTYPE_TO_TORCH[ort_dtype]
    dst = torch.empty(shape, dtype=torch_dtype, device=device)
    status = cudart().cudaMemcpy(
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(dst.nbytes),
        ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
    )
    if status != 0:
        raise RuntimeError(f"cudaMemcpy D2D failed with status {status}")
    return dst
