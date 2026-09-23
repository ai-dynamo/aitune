# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded Triton Model Analyzer configurations."""

import json
from base64 import b64encode
from math import prod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from tritonclient.grpc import model_config_pb2

from aitune.records import DeploymentArtifact, DType

_CONFIG_FILE_NAME = "config.yaml"
_INPUT_DATA_FILE_NAME = "input-data.json"
_DEFAULT_MAX_INSTANCE_COUNT = 3


class _ModelAnalyzerConfig(BaseModel):
    """Validated configuration for Model Analyzer's quick search mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_repository: Path
    profile_models: tuple[str, ...] = Field(min_length=1)
    checkpoint_directory: Path
    output_model_repository_path: Path
    override_output_model_repository: Literal[False] = False
    export_path: Path
    perf_analyzer_flags: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    latency_budget: int | None = Field(default=None, ge=1)
    run_config_search_mode: Literal["quick"] = "quick"
    run_config_search_min_instance_count: int = Field(default=1, ge=1)
    run_config_search_max_instance_count: int = Field(default=3, ge=1)
    run_config_search_min_model_batch_size: int | None = Field(default=None, ge=1)
    run_config_search_max_model_batch_size: int | None = Field(default=None, ge=1)
    run_config_search_max_concurrency: int = Field(ge=1)

    def to_yaml(self) -> str:
        """Render this validated configuration as YAML."""
        return yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False)


def write_model_analyzer_config(
    artifact: DeploymentArtifact,
    *,
    config: model_config_pb2.ModelConfig,
    model_directory: Path,
    destination: Path,
    latency_budget_ms: int | None,
    output_directory: Path,
    input_data_path: Path,
) -> None:
    """Write the Model Analyzer config into the new model directory."""
    perf_flags, input_data = _profiling_inputs(artifact, config)
    if input_data is not None:
        perf_flags["input-data"] = (str(input_data_path.resolve()),)

    analyzer_config = _quick_config(config, model_directory, destination, perf_flags, latency_budget_ms)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / _CONFIG_FILE_NAME).write_text(analyzer_config.to_yaml())
    if input_data is not None:
        (output_directory / _INPUT_DATA_FILE_NAME).write_text(json.dumps(input_data, indent=2) + "\n")


def _profiling_inputs(
    artifact: DeploymentArtifact, config: model_config_pb2.ModelConfig
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any] | None]:
    """Select concrete input shapes and representative values from the artifact."""
    shapes = {tensor.name: tensor.min_shape for tensor in artifact.inputs}
    flags = []
    for name, shape in shapes.items():
        dimensions = shape[1:] if config.max_batch_size else shape
        flags.append(f"{name}:{','.join(str(dimension) for dimension in dimensions)}")
    input_data = _input_data(artifact, shapes, batched=config.max_batch_size > 0)
    return {"shape": tuple(flags)}, input_data


def _input_data(
    artifact: DeploymentArtifact, shapes: dict[str, tuple[int, ...]], *, batched: bool
) -> dict[str, Any] | None:
    """Create one Perf Analyzer request from representative backend inputs."""
    if not artifact.sample_inputs:
        return None

    dtypes = {tensor.name: tensor.dtype for tensor in artifact.inputs}
    tensors: dict[str, Any] = {}
    for sample in artifact.sample_inputs:
        target_shape = tuple(shapes[sample.name])
        source_shape = sample.shape
        source_values = sample.values
        if batched:
            if not source_shape or source_shape[0] < 1:
                raise ValueError(f"Representative input {sample.name!r} has no batch values")
            source_shape = source_shape[1:]
            source_values = source_values[: prod(source_shape)]
            target_shape = target_shape[1:]

        target_size = prod(target_shape)
        if not source_values:
            raise ValueError(f"Representative input {sample.name!r} has no values")
        repeats = (target_size + len(source_values) - 1) // len(source_values)
        values = (source_values * repeats)[:target_size]
        content: Any = list(values)
        if dtypes[sample.name] is DType.FLOAT16:
            # Perf Analyzer rejects numeric JSON for FP16; send little-endian half values as binary.
            content = {"b64": b64encode(np.asarray(values, dtype="<f2").tobytes()).decode("ascii")}
        tensors[sample.name] = {
            "content": content,
            "shape": list(target_shape),
        }
    return {"data": [tensors]}


def _quick_config(
    config: model_config_pb2.ModelConfig,
    model_directory: Path,
    destination: Path,
    perf_flags: dict[str, tuple[str, ...]],
    latency_budget_ms: int | None,
) -> _ModelAnalyzerConfig:
    """Build a quick search within the published model's batch limit."""
    batched = config.max_batch_size > 0
    return _ModelAnalyzerConfig(
        model_repository=model_directory.parent.resolve(),
        perf_analyzer_flags=perf_flags,
        latency_budget=latency_budget_ms,
        profile_models=(config.name,),
        checkpoint_directory=(destination / "checkpoints").resolve(),
        output_model_repository_path=(destination / "model-repository").resolve(),
        export_path=(destination / "results").resolve(),
        run_config_search_max_instance_count=_DEFAULT_MAX_INSTANCE_COUNT,
        run_config_search_min_model_batch_size=1 if batched else None,
        run_config_search_max_model_batch_size=config.max_batch_size if batched else None,
        run_config_search_max_concurrency=2 * max(config.max_batch_size, 1),
    )
