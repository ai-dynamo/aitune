# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded Triton Model Analyzer configuration files."""

import json
from math import prod
from pathlib import Path
from typing import Any

from tritonclient.grpc import model_config_pb2

from aitune.records import DeploymentArtifact
from aitune.triton.model_analyzer._schema import (
    _DynamicBatchingParameters,
    _InstanceGroupParameters,
    _LoadParameters,
    _ManualModelAnalyzerConfig,
    _ManualModelProfile,
    _ModelConfigParameters,
    _QuickModelAnalyzerConfig,
)

_FAST_CONFIG_FILE_NAME = "fast.yaml"
_MANUAL_CONFIG_FILE_NAME = "manual.yaml"
_INPUT_DATA_FILE_NAME = "input-data.json"
_DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32)
_DEFAULT_MAX_INSTANCE_COUNT = 5
_DEFAULT_QUEUE_DELAYS_MICROSECONDS = (0, 100, 500)


def write_model_analyzer_configs(
    artifact: DeploymentArtifact,
    *,
    config: model_config_pb2.ModelConfig,
    model_directory: Path,
    destination: Path,
    staging: Path,
    input_data_path: Path,
) -> None:
    """Write Model Analyzer configs into an in-progress model publication."""
    fast, manual, input_data = _configs(
        artifact,
        model_directory,
        destination,
        config=config,
        input_data_path=input_data_path,
    )
    staging.mkdir(parents=True, exist_ok=True)
    (staging / _FAST_CONFIG_FILE_NAME).write_text(fast.to_yaml())
    (staging / _MANUAL_CONFIG_FILE_NAME).write_text(manual.to_yaml())
    if input_data is not None:
        (staging / _INPUT_DATA_FILE_NAME).write_text(json.dumps(input_data, indent=2) + "\n")


def _bounded_values(min_batch: int, max_batch: int) -> tuple[int, ...]:
    """Return powers of two within the bounds, including both endpoints."""
    values = [min_batch]
    batch_size = 1
    while batch_size < max_batch:
        if batch_size > min_batch:
            values.append(batch_size)
        batch_size *= 2
    if min_batch < max_batch:
        values.append(max_batch)
    return tuple(values)


def _profiling_inputs(
    artifact: DeploymentArtifact, config: model_config_pb2.ModelConfig
) -> tuple[dict[str, tuple[str, ...]], int, int, dict[str, Any] | None]:
    """Select concrete input shapes and compatible batch bounds from the artifact."""
    shapes = {tensor.name: tensor.min_shape for tensor in artifact.inputs}
    flags = []
    for name, shape in shapes.items():
        dimensions = shape[1:] if config.max_batch_size else shape
        flags.append(f"{name}:{','.join(str(dimension) for dimension in dimensions)}")
    input_data = _input_data(artifact, shapes, batched=config.max_batch_size > 0)
    return {"shape": tuple(flags)}, 1, config.max_batch_size, input_data


def _input_data(
    artifact: DeploymentArtifact, shapes: dict[str, tuple[int, ...]], *, batched: bool
) -> dict[str, Any] | None:
    """Create one Perf Analyzer request from representative backend inputs."""
    if not artifact.sample_inputs:
        return None

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
        tensors[sample.name] = {
            "content": list((source_values * repeats)[:target_size]),
            "shape": list(target_shape),
        }
    return {"data": [tensors]}


def _optimization_profiles(config: model_config_pb2.ModelConfig) -> tuple[str, ...] | None:
    """Return TensorRT profiles enabled by the published model configuration."""
    if config.platform != "tensorrt_plan":
        return None
    profiles = tuple(dict.fromkeys(profile for group in config.instance_group for profile in group.profile))
    return profiles or None


def _configs(
    artifact: DeploymentArtifact,
    model_directory: Path,
    destination: Path,
    *,
    config: model_config_pb2.ModelConfig,
    input_data_path: Path,
) -> tuple[
    _QuickModelAnalyzerConfig | _ManualModelAnalyzerConfig,
    _ManualModelAnalyzerConfig,
    dict[str, Any] | None,
]:
    """Build fast and exhaustive configurations from a published model."""
    perf_flags, minimum_batch, maximum_batch, input_data = _profiling_inputs(artifact, config)
    if input_data is not None:
        perf_flags["input-data"] = (str(input_data_path.resolve()),)

    repository = model_directory.parent.resolve()
    model_name = config.name
    batch_sizes = _bounded_values(minimum_batch, maximum_batch) if maximum_batch > 0 else (1,)
    batch_range = {
        "run_config_search_min_model_batch_size": minimum_batch,
        "run_config_search_max_model_batch_size": maximum_batch,
    }
    if config.max_batch_size == 0:
        batch_range = {}

    profiles = _optimization_profiles(config)

    def manual_config(
        label: str,
        *,
        selected_batch_sizes: tuple[int, ...],
        concurrency: tuple[int, ...],
        instance_counts: tuple[int, ...],
        queue_delays: tuple[int, ...],
    ) -> _ManualModelAnalyzerConfig:
        model_parameters = _ModelConfigParameters(
            max_batch_size=selected_batch_sizes if config.max_batch_size > 0 else None,
            instance_group=(_InstanceGroupParameters(count=instance_counts, profile=profiles),),
            dynamic_batching=(
                _DynamicBatchingParameters(max_queue_delay_microseconds=queue_delays)
                if config.max_batch_size > 0
                else None
            ),
        )
        return _ManualModelAnalyzerConfig(
            model_repository=repository,
            perf_analyzer_flags=perf_flags,
            profile_models={
                model_name: _ManualModelProfile(
                    parameters=_LoadParameters(batch_sizes=selected_batch_sizes, concurrency=concurrency),
                    model_config_parameters=model_parameters,
                )
            },
            checkpoint_directory=(destination / f"{label}-checkpoints").resolve(),
            output_model_repository_path=(destination / f"{label}-model-repository").resolve(),
            export_path=(destination / f"{label}-results").resolve(),
        )

    manual = manual_config(
        "manual",
        selected_batch_sizes=batch_sizes,
        concurrency=_DEFAULT_CONCURRENCY,
        instance_counts=tuple(range(1, _DEFAULT_MAX_INSTANCE_COUNT + 1)),
        queue_delays=_DEFAULT_QUEUE_DELAYS_MICROSECONDS,
    )
    if profiles is not None:
        # Quick search replaces instance_group and can discard the profiles enabled
        # by the published model. Keep every profile available in a smaller explicit
        # sweep without using profile shape bounds to constrain the search.
        fast_batch_sizes = tuple(dict.fromkeys((1, maximum_batch))) if maximum_batch > 0 else (1,)
        fast = manual_config(
            "fast",
            selected_batch_sizes=fast_batch_sizes,
            concurrency=(1, 8, 32),
            instance_counts=(1, 2),
            queue_delays=(0,),
        )
    else:
        fast = _QuickModelAnalyzerConfig(
            model_repository=repository,
            perf_analyzer_flags=perf_flags,
            profile_models=(model_name,),
            checkpoint_directory=(destination / "fast-checkpoints").resolve(),
            output_model_repository_path=(destination / "fast-model-repository").resolve(),
            export_path=(destination / "fast-results").resolve(),
            run_config_search_max_instance_count=_DEFAULT_MAX_INSTANCE_COUNT,
            **batch_range,
        )
    return fast, manual, input_data
