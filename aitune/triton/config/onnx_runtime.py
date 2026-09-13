# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated ONNX Runtime configuration for Triton."""

from typing import Any, Literal

from google.protobuf import json_format
from tritonclient.grpc import model_config_pb2

from aitune.records import DeploymentArtifact
from aitune.triton.config.common import _BaseModelConfig
from aitune.triton.config.options import ExecutionAccelerator


class ONNXRuntimeModelConfig(_BaseModelConfig):
    """Triton configuration specialized for ONNX Runtime artifacts."""

    platform: Literal["onnxruntime_onnx"] = "onnxruntime_onnx"
    execution_provider: Literal["cuda", "tensorrt"]
    gpu_execution_accelerators: tuple[ExecutionAccelerator, ...] = ()
    cpu_execution_accelerators: tuple[ExecutionAccelerator, ...] = ()

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Preserve ONNX Runtime TensorRT provider selection."""
        config = super().to_protobuf()
        for kind, accelerators in (
            ("gpu_execution_accelerator", self.gpu_execution_accelerators),
            ("cpu_execution_accelerator", self.cpu_execution_accelerators),
        ):
            target = getattr(config.optimization.execution_accelerators, kind)
            for accelerator in accelerators:
                if accelerator.name in {item.name for item in target}:
                    raise ValueError(f"Duplicate execution accelerator: {accelerator.name}")
                json_format.ParseDict(accelerator.model_dump(mode="json"), target.add())
        gpu = config.optimization.execution_accelerators.gpu_execution_accelerator
        if self.execution_provider == "tensorrt" and not any(a.name == "tensorrt" for a in gpu):
            accelerator = config.optimization.execution_accelerators.gpu_execution_accelerator.add()
            accelerator.name = "tensorrt"
        return config

    @classmethod
    def _artifact_options(cls, artifact: DeploymentArtifact) -> dict[str, Any]:
        """Read the execution provider selected for ONNX Runtime."""
        return {"execution_provider": artifact.runtime.options["execution_provider"]}
