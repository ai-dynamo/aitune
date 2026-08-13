# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch backend module."""

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.onnx_runtime_backend import (
    ONNXExecutionProvider,
    ONNXRuntimeBackend,
    ONNXRuntimeBackendConfig,
)
from aitune.torch.backend.tensorrt import (
    ONNXAutoCastConfig,
    ONNXQuantizationConfig,
    TensorRTBackend,
    TensorRTBackendConfig,
    TensorRTProfile,
    TorchQuantizationConfig,
)
from aitune.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from aitune.torch.backend.torch_inductor_aot_backend import TorchInductorAotBackend, TorchInductorAotBackendConfig
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.backend.torch_tensorrt_aot_backend import (
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
)
from aitune.torch.backend.torch_tensorrt_jit_backend import (
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from aitune.torch.checkpoint.artifact import ArtifactPath

__all__ = [
    "ArtifactPath",
    "Backend",
    "ONNXAutoCastConfig",
    "ONNXExecutionProvider",
    "ONNXQuantizationConfig",
    "ONNXRuntimeBackend",
    "ONNXRuntimeBackendConfig",
    "TensorRTBackend",
    "TensorRTBackendConfig",
    "TensorRTProfile",
    "TorchInductorAotBackend",
    "TorchInductorAotBackendConfig",
    "TorchInductorJitBackend",
    "TorchInductorJitBackendConfig",
    "TorchAOBackend",
    "TorchAOBackendConfig",
    "TorchEagerBackend",
    "TorchEagerBackendConfig",
    "TorchTensorRTAotBackend",
    "TorchTensorRTAotBackendConfig",
    "TorchTensorRTConfig",
    "TorchTensorRTJitBackend",
    "TorchTensorRTJitBackendConfig",
    "TorchQuantizationConfig",
]
