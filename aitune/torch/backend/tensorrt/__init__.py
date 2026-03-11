# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT backend package for AITune.

This package provides functionality to convert PyTorch modules to optimized
TensorRT engines for accelerated inference. It includes components for ONNX
export, TensorRT engine building, and runtime inference.
"""

from aitune.torch.backend.tensorrt.onnx_autocast import ONNXAutoCastConfig
from aitune.torch.backend.tensorrt.onnx_quantization import ONNXQuantizationConfig
from aitune.torch.backend.tensorrt.tensorrt_backend import ProfileMode, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.tensorrt.tensorrt_profile import TensorRTProfile
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig

__all__ = [
    "TensorRTBackend",
    "TensorRTBackendConfig",
    "TensorRTProfile",
    "ProfileMode",
    "ONNXAutoCastConfig",
    "ONNXQuantizationConfig",
    "TorchQuantizationConfig",
]
