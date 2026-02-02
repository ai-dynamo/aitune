# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
