# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fields for backend-specific Triton model configurations."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from google.protobuf import json_format, text_format
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tritonclient.grpc import model_config_pb2

from aitune.records import BoundedTensorSpec, DeploymentArtifact, DType
from aitune.triton.config.options import DynamicBatcher, InstanceGroup, ModelWarmup, SequenceBatcher


class TritonDataType(str, Enum):
    """Tensor element types supported by AITune artifacts and Triton."""

    BOOL = "TYPE_BOOL"
    UINT8 = "TYPE_UINT8"
    UINT16 = "TYPE_UINT16"
    UINT32 = "TYPE_UINT32"
    UINT64 = "TYPE_UINT64"
    BFLOAT16 = "TYPE_BF16"
    STRING = "TYPE_STRING"
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
    is_shape_tensor: bool = False
    is_non_linear_format_io: bool = False
    optional: bool = False
    format: Literal["FORMAT_NONE", "FORMAT_NHWC", "FORMAT_NCHW"] = "FORMAT_NONE"
    allow_ragged_batch: bool = False
    label_filename: str | None = None

    @field_validator("reshape")
    @classmethod
    def _validate_reshape(cls, dims: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """Allow scalar reshapes and one inferred dimension."""
        if dims is not None and (any(d < -1 for d in dims) or dims.count(-1) > 1):
            raise ValueError("reshape allows non-negative dimensions and at most one -1")
        return dims

    def to_config_dict(self, *, output: bool) -> dict[str, Any]:
        """Serialize only settings appropriate to an input or output tensor."""
        if output and (self.optional or self.allow_ragged_batch or self.format != "FORMAT_NONE"):
            raise ValueError("optional, format, and allow_ragged_batch are input-only settings")
        if not output and self.label_filename is not None:
            raise ValueError("label_filename is an output-only setting")
        data = self.model_dump(mode="json", exclude_none=True)
        if self.reshape is not None:
            data["reshape"] = {"shape": list(self.reshape)}
        excluded = ("optional", "format", "allow_ragged_batch") if output else ("label_filename",)
        for key in excluded:
            data.pop(key, None)
        return data

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
    dynamic_batching: bool | DynamicBatcher = False
    sequence_batching: SequenceBatcher | None = None
    instance_groups: tuple[InstanceGroup, ...] = ()
    parameters: dict[str, str] = Field(default_factory=dict)
    response_cache: bool | None = None
    decoupled: bool | None = None
    warmup: tuple[ModelWarmup, ...] = ()
    default_model_filename: str | None = None
    version_policy: dict[str, Any] | None = None
    optimization: dict[str, Any] | None = None
    batch_input: tuple[dict[str, Any], ...] = ()
    batch_output: tuple[dict[str, Any], ...] = ()
    metric_tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("default_model_filename")
    @classmethod
    def _validate_filename(cls, filename: str | None) -> str | None:
        """Keep the entry file inside its version directory."""
        if filename is not None and (
            not filename or filename in {".", ".."} or Path(filename).name != filename or "\\" in filename
        ):
            raise ValueError("default_model_filename must be a single filename")
        return filename

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
        if self.dynamic_batching and self.sequence_batching is not None:
            raise ValueError("dynamic_batching and sequence_batching are mutually exclusive")
        if isinstance(self.dynamic_batching, DynamicBatcher):
            if any(size > self.max_batch_size for size in self.dynamic_batching.preferred_batch_size):
                raise ValueError("preferred_batch_size cannot exceed max_batch_size")
        if self.platform != "tensorrt_plan" and any(group.profile for group in self.instance_groups):
            raise ValueError("Instance profiles are only supported by TensorRT")
        if len({sample.name for sample in self.warmup}) != len(self.warmup):
            raise ValueError("Warmup names must be unique")
        if any(sample.batch_size > max(1, self.max_batch_size) for sample in self.warmup):
            raise ValueError("Warmup batch_size exceeds the model batch limit")
        self.to_protobuf()
        return self

    def to_protobuf(self) -> model_config_pb2.ModelConfig:
        """Build Triton's protobuf representation."""
        data = self._common_config_dict()
        try:
            return json_format.ParseDict(data, model_config_pb2.ModelConfig())
        except (json_format.ParseError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Triton model configuration: {error}") from error

    def _common_config_dict(self) -> dict[str, Any]:
        """Translate public options to protobuf field names without dropping settings."""
        data = {
            "name": self.name,
            "platform": self.platform,
            "max_batch_size": self.max_batch_size,
            "input": [tensor.to_config_dict(output=False) for tensor in self.inputs],
            "output": [tensor.to_config_dict(output=True) for tensor in self.outputs],
            "instance_group": [group.model_dump(mode="json", exclude_none=True) for group in self.instance_groups],
            "parameters": {key: {"string_value": value} for key, value in self.parameters.items()},
            "model_warmup": [sample.model_dump(mode="json", exclude_none=True) for sample in self.warmup],
            "batch_input": list(self.batch_input),
            "batch_output": list(self.batch_output),
            "metric_tags": self.metric_tags,
        }
        data.update(self._optional_config_dict())
        return data

    def _optional_config_dict(self) -> dict[str, Any]:
        """Preserve explicit false values and scheduler message presence."""
        data: dict[str, Any] = {}
        if self.dynamic_batching:
            data["dynamic_batching"] = (
                self.dynamic_batching.model_dump(mode="json", exclude_none=True)
                if isinstance(self.dynamic_batching, DynamicBatcher)
                else {}
            )
        if self.sequence_batching is not None:
            data["sequence_batching"] = self.sequence_batching.model_dump(mode="json", exclude_none=True)
        if self.response_cache is not None:
            data["response_cache"] = {"enable": self.response_cache}
        if self.decoupled is not None:
            data["model_transaction_policy"] = {"decoupled": self.decoupled}
        for name in ("default_model_filename", "version_policy", "optimization"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

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
    return TritonTensorConfig(name=spec.name, data_type=_DTYPES[spec.dtype], dims=dimensions)
