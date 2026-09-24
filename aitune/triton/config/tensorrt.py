# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated TensorRT plan configuration for Triton."""

from typing import Any, Literal

from pydantic import NonNegativeInt, StrictInt, field_validator
from tritonclient.grpc import model_config_pb2

from aitune.records import DeploymentArtifact
from aitune.triton.config.common import BaseModelConfig


class TensorRTModelConfig(BaseModelConfig):
    """Triton configuration specialized for serialized TensorRT plans."""

    platform: Literal["tensorrt_plan"] = "tensorrt_plan"
    optimization_profile_indices: tuple[StrictInt, ...]
    cuda_graphs: bool = False
    eager_batching: bool | None = None
    gather_kernel_buffer_threshold: NonNegativeInt | None = None

    @field_validator("optimization_profile_indices")
    @classmethod
    def _validate_profiles(cls, indices: tuple[int, ...]) -> tuple[int, ...]:
        """Accept a non-empty set of distinct non-negative profile indices."""
        if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
            raise ValueError("TensorRT optimization profiles must be non-empty, unique, and non-negative")
        return indices

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Add TensorRT optimization-profile selection to the common config."""
        config = super().to_protobuf()
        if self.cuda_graphs:
            config.optimization.cuda.graphs = True
        if self.eager_batching is not None:
            config.optimization.eager_batching = self.eager_batching
        if self.gather_kernel_buffer_threshold is not None:
            config.optimization.gather_kernel_buffer_threshold = self.gather_kernel_buffer_threshold
        profiles = tuple(str(index) for index in self.optimization_profile_indices)
        if not config.instance_group and profiles != ("0",):
            config.instance_group.add(kind=model_config_pb2.ModelInstanceGroup.KIND_GPU)
        for group in config.instance_group:
            if group.kind not in (
                model_config_pb2.ModelInstanceGroup.KIND_GPU,
                model_config_pb2.ModelInstanceGroup.KIND_AUTO,
            ):
                raise ValueError("TensorRT instances require KIND_GPU or KIND_AUTO")
            if group.profile and not set(group.profile).issubset(profiles):
                raise ValueError("Instance profile must be included in optimization_profile_indices")
            if not group.profile:
                group.profile.extend(profiles)
        return config

    @classmethod
    def _artifact_options(cls, artifact: DeploymentArtifact) -> dict[str, Any]:
        """Read plan profile metadata and TensorRT runtime settings."""
        count = artifact.model.metadata.get("optimization_profile_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("TensorRT model metadata requires a positive integer optimization_profile_count")
        return {
            "optimization_profile_indices": tuple(range(count)),
            "cuda_graphs": artifact.runtime.options.get("use_cuda_graphs", False),
        }
