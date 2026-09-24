# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable Triton scheduling, placement, and warmup settings."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator


class _Options(BaseModel):
    """Reject misspelled settings and retain an immutable configuration interface."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class QueuePolicy(_Options):
    """Request timeout and queue capacity policy; times are in microseconds."""

    timeout_action: Literal["REJECT", "DELAY"] = "REJECT"
    default_timeout_microseconds: NonNegativeInt = 0
    allow_timeout_override: bool = False
    max_queue_size: NonNegativeInt = 0


class DynamicBatcher(_Options):
    """Dynamic batching with preferred sizes, priorities, and queue policies."""

    preferred_batch_size: tuple[PositiveInt, ...] = ()
    max_queue_delay_microseconds: NonNegativeInt = 0
    preserve_ordering: bool = False
    priority_levels: NonNegativeInt = 0
    default_priority_level: NonNegativeInt = 0
    default_queue_policy: QueuePolicy | None = None
    priority_queue_policy: dict[int, QueuePolicy] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_priorities(self) -> "DynamicBatcher":
        """Require configured priority levels and defaults to agree."""
        if self.priority_levels:
            if not 1 <= self.default_priority_level <= self.priority_levels:
                raise ValueError("default_priority_level must be within 1..priority_levels")
        elif self.default_priority_level or self.priority_queue_policy:
            raise ValueError("Priority settings require priority_levels")
        if any(not 1 <= level <= self.priority_levels for level in self.priority_queue_policy):
            raise ValueError("Queue policy priority must be within 1..priority_levels")
        return self


class SequenceBatcher(_Options):
    """Sequence scheduling using Triton protobuf fields for controls and state.

    ``direct`` and ``oldest`` select mutually exclusive strategies. Nested mappings
    use the ModelSequenceBatching schema and are checked when the model config is built.
    """

    direct: dict[str, Any] | None = None
    oldest: dict[str, Any] | None = None
    max_sequence_idle_microseconds: NonNegativeInt | None = None
    control_input: tuple[dict[str, Any], ...] = ()
    state: tuple[dict[str, Any], ...] = ()
    iterative_sequence: bool = False

    @model_validator(mode="after")
    def _validate_strategy(self) -> "SequenceBatcher":
        """Allow at most one sequence scheduler strategy."""
        if self.direct is not None and self.oldest is not None:
            raise ValueError("Sequence batching accepts either direct or oldest, not both")
        return self


class InstanceGroup(_Options):
    """Model instance count, device placement, profiles, and rate limiting."""

    name: str | None = None
    kind: Literal["KIND_AUTO", "KIND_CPU", "KIND_GPU", "KIND_MODEL"] = "KIND_AUTO"
    count: PositiveInt = 1
    gpus: tuple[NonNegativeInt, ...] = ()
    profile: tuple[str, ...] = ()
    passive: bool = False
    host_policy: str | None = None
    rate_limiter: dict[str, Any] | None = None
    secondary_devices: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def _validate_devices(self) -> "InstanceGroup":
        """Reject GPU identifiers on CPU or model-managed instances."""
        if self.gpus and self.kind not in {"KIND_AUTO", "KIND_GPU"}:
            raise ValueError("gpus requires KIND_AUTO or KIND_GPU")
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("GPU identifiers must be unique")
        return self


class ModelWarmup(_Options):
    """Named warmup sample with protobuf-shaped input data specifications.

    Each input specifies ``data_type``, ``dims``, and one of ``zero_data``,
    ``random_data``, or ``input_data_file``. Files live under the model's warmup directory.
    """

    name: str = Field(min_length=1)
    batch_size: PositiveInt = 1
    inputs: dict[str, dict[str, Any]] = Field(min_length=1)
    count: NonNegativeInt = 0


class ExecutionAccelerator(_Options):
    """ONNX Runtime execution accelerator and backend-specific string parameters."""

    name: str = Field(min_length=1)
    parameters: dict[str, str] = Field(default_factory=dict)
