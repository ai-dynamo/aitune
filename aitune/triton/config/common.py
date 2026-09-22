# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fields for backend-specific Triton model configurations."""

from enum import Enum
from typing import Any, Literal

from google.protobuf import text_format
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tritonclient.grpc import model_config_pb2

from aitune.records import BoundedTensorSpec, DeploymentArtifact, DType


class TritonDataType(str, Enum):
    """Tensor element types supported by AITune artifacts and Triton."""

    BOOL = "TYPE_BOOL"
    UINT8 = "TYPE_UINT8"
    INT8 = "TYPE_INT8"
    INT16 = "TYPE_INT16"
    INT32 = "TYPE_INT32"
    INT64 = "TYPE_INT64"
    FLOAT16 = "TYPE_FP16"
    FLOAT32 = "TYPE_FP32"
    FLOAT64 = "TYPE_FP64"


_DTYPES = {
    DType.BOOL: TritonDataType.BOOL,
    DType.UINT8: TritonDataType.UINT8,
    DType.INT8: TritonDataType.INT8,
    DType.INT16: TritonDataType.INT16,
    DType.INT32: TritonDataType.INT32,
    DType.INT64: TritonDataType.INT64,
    DType.FLOAT16: TritonDataType.FLOAT16,
    DType.FLOAT32: TritonDataType.FLOAT32,
    DType.FLOAT64: TritonDataType.FLOAT64,
}


class TritonTensorConfig(BaseModel):
    """One input or output in a Triton model configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    data_type: TritonDataType
    dims: tuple[int, ...]
    reshape: tuple[int, ...] | None = None

    @field_validator("dims")
    @classmethod
    def _validate_dims(cls, dims: tuple[int, ...]) -> tuple[int, ...]:
        """Allow concrete positive dimensions and Triton's dynamic marker."""
        if not dims or any(dimension != -1 and dimension < 1 for dimension in dims):
            raise ValueError("Triton tensor dimensions must be non-empty and contain positive values or -1")
        return dims


class _BaseModelConfig(BaseModel):
    """Internal fields and validation shared by supported Triton backends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    platform: Literal["tensorrt_plan", "onnxruntime_onnx", "torch_aoti"]
    max_batch_size: int = Field(ge=0)
    inputs: tuple[TritonTensorConfig, ...] = Field(min_length=1)
    outputs: tuple[TritonTensorConfig, ...] = Field(min_length=1)
    dynamic_batching: bool = False

    @classmethod
    def from_artifact(
        cls, artifact: DeploymentArtifact, *, name: str, max_batch_size: int, dynamic_batching: bool = False
    ) -> "_BaseModelConfig":
        """Combine the tensor interface with runtime-specific artifact settings."""
        batched = max_batch_size > 0
        return cls(
            name=name,
            max_batch_size=max_batch_size,
            inputs=tuple(tensor_config(tensor, batched=batched) for tensor in artifact.inputs),
            outputs=tuple(tensor_config(tensor, batched=batched) for tensor in artifact.outputs),
            dynamic_batching=dynamic_batching,
            **cls._artifact_options(artifact),
        )

    @model_validator(mode="after")
    def _validate_batching(self) -> "_BaseModelConfig":
        """Require a batched model contract before enabling the scheduler."""
        if self.dynamic_batching and self.max_batch_size == 0:
            raise ValueError("dynamic_batching requires a positive max_batch_size")
        return self

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Build Triton's protobuf representation."""
        config = model_config_pb2.ModelConfig(
            name=self.name,
            platform=self.platform,
            max_batch_size=self.max_batch_size,
        )
        for field_name, tensors in (("input", self.inputs), ("output", self.outputs)):
            target = getattr(config, field_name)
            for tensor in tensors:
                target_tensor = target.add(
                    name=tensor.name,
                    data_type=model_config_pb2.DataType.Value(tensor.data_type.value),
                    dims=tensor.dims,
                )
                if tensor.reshape is not None:
                    target_tensor.reshape.shape.extend(tensor.reshape)
                    target_tensor.reshape.SetInParent()
        if self.dynamic_batching:
            config.dynamic_batching.SetInParent()
        return config

    def to_pbtxt(self) -> str:
        """Render and schema-validate a Triton ``config.pbtxt`` document."""
        content = text_format.MessageToString(self.to_protobuf())
        if self.max_batch_size == 0:
            content += "max_batch_size: 0\n"
        text_format.Parse(content, model_config_pb2.ModelConfig())
        return content

    @classmethod
    def _artifact_options(cls, artifact: DeploymentArtifact) -> dict[str, Any]:
        """Read settings owned by the specialized Triton config."""
        raise NotImplementedError


def tensor_config(spec: BoundedTensorSpec, *, batched: bool) -> TritonTensorConfig:
    """Translate an artifact tensor spec into Triton's tensor representation."""
    if batched and spec.batch_axis != 0:
        raise ValueError(f"Triton batching requires batch_axis=0 for tensor {spec.name!r}")
    bounds = zip(spec.min_shape, spec.max_shape, strict=True)
    dimensions = tuple(minimum if minimum == maximum else -1 for minimum, maximum in bounds)
    if batched:
        dimensions = dimensions[1:]
    reshape = None
    if not dimensions:
        dimensions = (1,)
        reshape = ()
    return TritonTensorConfig(name=spec.name, data_type=_DTYPES[spec.dtype], dims=dimensions, reshape=reshape)
