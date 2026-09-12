# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated TensorRT plan configuration for Triton."""

from typing import Any, Literal

from pydantic import Field
from tritonclient.grpc import model_config_pb2

from aitune.records import DeploymentArtifact
from aitune.triton.config.common import _BaseModelConfig


class TensorRTModelConfig(_BaseModelConfig):
    """Triton configuration specialized for serialized TensorRT plans."""

    platform: Literal["tensorrt_plan"] = "tensorrt_plan"
    optimization_profile_count: int = Field(gt=0, strict=True)
    cuda_graphs: bool = False

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Add TensorRT optimization-profile selection to the common config."""
        config = super().to_protobuf()
        if self.cuda_graphs:
            config.optimization.cuda.graphs = True
        if self.optimization_profile_count > 1:
            group = config.instance_group.add(kind=model_config_pb2.ModelInstanceGroup.KIND_GPU)
            group.profile.extend(str(index) for index in range(self.optimization_profile_count))
        return config

    @classmethod
    def _artifact_options(cls, artifact: DeploymentArtifact) -> dict[str, Any]:
        """Read plan profile metadata and TensorRT runtime settings."""
        return {
            "optimization_profile_count": artifact.model.metadata.get("optimization_profile_count"),
            "cuda_graphs": artifact.runtime.options.get("use_cuda_graphs", False),
        }
