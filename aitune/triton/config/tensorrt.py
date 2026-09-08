# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated TensorRT plan configuration for Triton."""

from typing import Literal

from pydantic import field_validator
from tritonclient.grpc import model_config_pb2

from aitune.triton.config.common import _BaseModelConfig


class TensorRTModelConfig(_BaseModelConfig):
    """Triton configuration specialized for serialized TensorRT plans."""

    platform: Literal["tensorrt_plan"] = "tensorrt_plan"
    optimization_profile_indices: tuple[int, ...]
    cuda_graphs: bool = False

    @field_validator("optimization_profile_indices")
    @classmethod
    def _validate_profiles(cls, indices: tuple[int, ...]) -> tuple[int, ...]:
        """Require every TensorRT optimization profile in index order."""
        if indices != tuple(range(len(indices))):
            raise ValueError("TensorRT optimization profiles must contain every index in order")
        return indices

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Add TensorRT optimization-profile selection to the common config."""
        config = super().to_protobuf()
        if self.cuda_graphs:
            config.optimization.cuda.graphs = True
        if len(self.optimization_profile_indices) > 1:
            group = config.instance_group.add(kind=model_config_pb2.ModelInstanceGroup.KIND_GPU)
            group.profile.extend(str(index) for index in self.optimization_profile_indices)
        return config
