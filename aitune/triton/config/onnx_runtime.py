# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated ONNX Runtime configuration for Triton."""

from typing import Literal

from tritonclient.grpc import model_config_pb2

from aitune.records import ONNXExecutionProvider
from aitune.triton.config.common import _BaseModelConfig


class ONNXRuntimeModelConfig(_BaseModelConfig):
    """Triton configuration specialized for ONNX Runtime artifacts."""

    platform: Literal["onnxruntime_onnx"] = "onnxruntime_onnx"
    execution_provider: ONNXExecutionProvider

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Preserve ONNX Runtime TensorRT provider selection."""
        config = super().to_protobuf()
        if self.execution_provider is ONNXExecutionProvider.TENSORRT:
            accelerator = config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "tensorrt"
        return config
