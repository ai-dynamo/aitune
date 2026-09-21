# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal schema building blocks for Model Analyzer configurations."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt


class _ConfigModel(BaseModel):
    """Strict immutable base for the generated Model Analyzer schema subset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_yaml(self) -> str:
        """Render this validated configuration as YAML."""
        return yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False)


class _LoadParameters(_ConfigModel):
    """Explicit client load values used by a manual search."""

    batch_sizes: tuple[PositiveInt, ...] = Field(min_length=1)
    concurrency: tuple[PositiveInt, ...] = Field(min_length=1)


class _DynamicBatchingParameters(_ConfigModel):
    """Dynamic batcher settings varied by a manual search."""

    max_queue_delay_microseconds: tuple[NonNegativeInt, ...] = Field(min_length=1)


class _InstanceGroupParameters(_ConfigModel):
    """GPU model instance counts varied by a manual search."""

    kind: Literal["KIND_GPU"] = "KIND_GPU"
    count: tuple[PositiveInt, ...] = Field(min_length=1)
    profile: tuple[str, ...] | None = None


class _ModelConfigParameters(_ConfigModel):
    """Triton model configuration values varied by a manual search."""

    max_batch_size: tuple[PositiveInt, ...] | None = None
    instance_group: tuple[_InstanceGroupParameters, ...]
    dynamic_batching: _DynamicBatchingParameters | None = None


class _ManualModelProfile(_ConfigModel):
    """Explicit search space for one published Triton model."""

    parameters: _LoadParameters
    model_config_parameters: _ModelConfigParameters


class _QuickModelAnalyzerConfig(_ConfigModel):
    """Bounded configuration for Model Analyzer's fast search mode."""

    model_repository: Path
    profile_models: tuple[str, ...] = Field(min_length=1)
    checkpoint_directory: Path
    output_model_repository_path: Path
    override_output_model_repository: Literal[False] = False
    export_path: Path
    perf_analyzer_flags: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    run_config_search_mode: Literal["quick"] = "quick"
    run_config_search_min_instance_count: int = Field(default=1, ge=1)
    run_config_search_max_instance_count: int = Field(default=5, ge=1)
    run_config_search_min_model_batch_size: int | None = Field(default=None, ge=1)
    run_config_search_max_model_batch_size: int | None = Field(default=None, ge=1)


class _ManualModelAnalyzerConfig(_ConfigModel):
    """Exhaustive bounded Model Analyzer configuration."""

    model_repository: Path
    profile_models: dict[str, _ManualModelProfile] = Field(min_length=1, max_length=1)
    checkpoint_directory: Path
    output_model_repository_path: Path
    override_output_model_repository: Literal[False] = False
    export_path: Path
    perf_analyzer_flags: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    run_config_search_mode: Literal["brute"] = "brute"
    run_config_search_disable: Literal[True] = True
