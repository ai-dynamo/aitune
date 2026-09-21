# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded Triton Model Analyzer configurations."""

import json
import os
import shutil
import tempfile
from math import prod
from pathlib import Path
from typing import Any, Literal

import yaml
from google.protobuf import text_format
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator
from tritonclient.grpc import model_config_pb2

from aitune.exceptions import AITuneError, AITuneUserInputError
from aitune.records import DeploymentArtifact

_CONFIG_FILE_NAME = "config.pbtxt"
_FAST_CONFIG_FILE_NAME = "fast.yaml"
_MANUAL_CONFIG_FILE_NAME = "manual.yaml"
_INPUT_DATA_FILE_NAME = "input-data.json"
_DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32)
_DEFAULT_QUEUE_DELAYS_MICROSECONDS = (0, 100, 500)


class ModelAnalyzerConfigError(AITuneError):
    """Raised when Model Analyzer configuration cannot be generated."""


class _ConfigModel(BaseModel):
    """Strict immutable base for the generated Model Analyzer schema subset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_yaml(self) -> str:
        """Render this validated configuration as YAML."""
        return yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False)


class QuickModelAnalyzerConfig(_ConfigModel):
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

    @model_validator(mode="after")
    def _validate_ranges(self) -> "QuickModelAnalyzerConfig":
        """Require complete, ordered search ranges."""
        if self.run_config_search_min_instance_count > self.run_config_search_max_instance_count:
            raise ValueError("minimum instance count cannot exceed maximum instance count")
        batch_range = (
            self.run_config_search_min_model_batch_size,
            self.run_config_search_max_model_batch_size,
        )
        if (batch_range[0] is None) != (batch_range[1] is None):
            raise ValueError("minimum and maximum model batch size must be provided together")
        minimum_batch_size, maximum_batch_size = batch_range
        if (
            minimum_batch_size is not None
            and maximum_batch_size is not None
            and minimum_batch_size > maximum_batch_size
        ):
            raise ValueError("minimum model batch size cannot exceed maximum model batch size")
        return self


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


class ManualModelAnalyzerConfig(_ConfigModel):
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


def _platform(artifact: DeploymentArtifact) -> str:
    """Return the Triton platform expected for the model format and runtime."""
    if (artifact.model.format, artifact.runtime.name) == ("tensorrt_plan", "tensorrt"):
        return "tensorrt_plan"
    if (artifact.model.format, artifact.runtime.name) == ("onnx", "onnxruntime"):
        return "onnxruntime_onnx"
    if (artifact.model.format, artifact.runtime.name) == ("pt2", "aotinductor"):
        return "torch_aoti"
    raise ModelAnalyzerConfigError(
        f"Model Analyzer does not support {artifact.model.format!r} with runtime {artifact.runtime.name!r}"
    )


def _read_published_config(model_directory: Path) -> model_config_pb2.ModelConfig:
    """Load and protobuf-validate a published Triton model config."""
    config_path = model_directory / _CONFIG_FILE_NAME
    if not model_directory.is_dir() or not config_path.is_file():
        raise AITuneUserInputError(f"Expected a published Triton model directory, got {model_directory}")
    try:
        return text_format.Parse(config_path.read_text(), model_config_pb2.ModelConfig())
    except (OSError, text_format.ParseError) as error:
        raise ModelAnalyzerConfigError(f"Cannot read Triton model configuration at {config_path}: {error}") from error


def _validate_model(artifact: DeploymentArtifact, model_directory: Path, config: model_config_pb2.ModelConfig) -> None:
    """Ensure the artifact and published model describe the same deployment."""
    if config.name != model_directory.name:
        raise ModelAnalyzerConfigError(
            f"Triton config names model {config.name!r}, but its directory is {model_directory.name!r}"
        )
    expected_platform = _platform(artifact)
    if config.platform != expected_platform:
        raise ModelAnalyzerConfigError(
            f"{artifact.model.format!r} requires Triton platform {expected_platform!r}, got {config.platform!r}"
        )
    if tuple(tensor.name for tensor in config.input) != artifact.input_names:
        raise ModelAnalyzerConfigError("Triton model inputs do not match the artifact")
    if tuple(tensor.name for tensor in config.output) != artifact.output_names:
        raise ModelAnalyzerConfigError("Triton model outputs do not match the artifact")

    supported_batch_size = artifact.max_batch_size
    if config.max_batch_size > 0 and (supported_batch_size is None or config.max_batch_size > supported_batch_size):
        raise ModelAnalyzerConfigError(
            f"Triton max_batch_size {config.max_batch_size} exceeds the artifact's tuned batch bounds"
        )


def _bounded_values(maximum: int, minimum: int = 1) -> tuple[int, ...]:
    """Return powers of two within the bounds, including both endpoints."""
    if not 1 <= minimum <= maximum:
        raise ModelAnalyzerConfigError(f"Invalid batch bounds: minimum={minimum}, maximum={maximum}")
    values = [minimum]
    value = 1
    while value < maximum:
        if value > minimum:
            values.append(value)
        value *= 2
    if values[-1] != maximum:
        values.append(maximum)
    return tuple(values)


def _profiling_inputs(
    artifact: DeploymentArtifact, config: model_config_pb2.ModelConfig
) -> tuple[dict[str, tuple[str, ...]], int, int, dict[str, Any] | None]:
    """Select concrete input shapes and compatible batch bounds from the artifact."""
    minimum_batch, maximum_batch = 1, config.max_batch_size
    shapes = {tensor.name: tensor.min_shape for tensor in artifact.inputs}
    if (artifact.model.format, artifact.runtime.name) == ("tensorrt_plan", "tensorrt"):
        profiles = _profile_indices(config)
        profile_data = artifact.model.metadata.get("optimization_profiles")
        if not profile_data or len(profile_data) != artifact.model.metadata.get("optimization_profile_count"):
            raise ModelAnalyzerConfigError("TensorRT model metadata requires optimization_profiles matching its count")
        for index in profiles:
            if not index.isdecimal() or int(index) >= len(profile_data):
                raise ModelAnalyzerConfigError(f"Invalid TensorRT profile index: {index!r}")
        for index in profiles:
            profile = profile_data[int(index)]
            if not config.max_batch_size:
                break
            minimum_batch = max(bounds["min_shape"][0] for bounds in profile.values())
            maximum_batch = min(config.max_batch_size, *(bounds["max_shape"][0] for bounds in profile.values()))
            if minimum_batch <= maximum_batch:
                break
        else:
            raise ModelAnalyzerConfigError("No enabled TensorRT profile supports a common batch size")
        shapes = {name: bounds["opt_shape"] for name, bounds in profile.items()}
    flags = []
    for name, shape in shapes.items():
        dimensions = shape[1:] if config.max_batch_size else shape
        flags.append(f"{name}:{','.join(str(dimension) for dimension in dimensions)}")
    input_data = _input_data(artifact, shapes, batched=config.max_batch_size > 0)
    return {"shape": tuple(flags)}, minimum_batch, maximum_batch, input_data


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
                raise ModelAnalyzerConfigError(f"Representative input {sample.name!r} has no batch values")
            source_shape = source_shape[1:]
            source_values = source_values[: prod(source_shape)]
            target_shape = target_shape[1:]

        target_size = prod(target_shape)
        if not source_values:
            raise ModelAnalyzerConfigError(f"Representative input {sample.name!r} has no values")
        repeats = (target_size + len(source_values) - 1) // len(source_values)
        tensors[sample.name] = {
            "content": list((source_values * repeats)[:target_size]),
            "shape": list(target_shape),
        }
    return {"data": [tensors]}


def _profile_indices(config: model_config_pb2.ModelConfig) -> tuple[str, ...]:
    """Return enabled TensorRT profiles, including Triton's implicit profile zero."""
    profiles = (index for group in config.instance_group for index in (tuple(group.profile) or ("0",)))
    return tuple(dict.fromkeys(profiles)) or ("0",)


def _configs(
    artifact: DeploymentArtifact,
    model_directory: Path,
    destination: Path,
    *,
    config: model_config_pb2.ModelConfig,
    input_data_path: Path | None,
    max_instance_count: int,
    queue_delay_microseconds: tuple[int, ...],
) -> tuple[
    QuickModelAnalyzerConfig | ManualModelAnalyzerConfig,
    ManualModelAnalyzerConfig,
    dict[str, Any] | None,
]:
    """Build fast and exhaustive configurations from a published model."""
    _validate_model(artifact, model_directory, config)
    perf_flags, minimum_batch, maximum_batch, input_data = _profiling_inputs(artifact, config)
    if input_data is not None:
        input_data_path = destination / _INPUT_DATA_FILE_NAME if input_data_path is None else input_data_path
        perf_flags["input-data"] = (str(input_data_path.resolve()),)

    repository = model_directory.parent.resolve()
    model_name = config.name
    batch_sizes = _bounded_values(maximum_batch, minimum_batch) if maximum_batch > 0 else (1,)
    batch_range = {
        "run_config_search_min_model_batch_size": minimum_batch,
        "run_config_search_max_model_batch_size": maximum_batch,
    }
    if config.max_batch_size == 0:
        batch_range = {}

    profiles: tuple[str, ...] | None = None
    if config.platform == "tensorrt_plan":
        profiles = _profile_indices(config)
        if profiles == ("0",) and not config.instance_group:
            profiles = None

    def manual_config(
        label: str,
        *,
        selected_batch_sizes: tuple[int, ...],
        concurrency: tuple[int, ...],
        instance_counts: tuple[int, ...],
        queue_delays: tuple[int, ...],
    ) -> ManualModelAnalyzerConfig:
        model_parameters = _ModelConfigParameters(
            max_batch_size=selected_batch_sizes if config.max_batch_size > 0 else None,
            instance_group=(_InstanceGroupParameters(count=instance_counts, profile=profiles),),
            dynamic_batching=(
                _DynamicBatchingParameters(max_queue_delay_microseconds=queue_delays)
                if config.max_batch_size > 0
                else None
            ),
        )
        return ManualModelAnalyzerConfig(
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
        instance_counts=tuple(range(1, max_instance_count + 1)),
        queue_delays=queue_delay_microseconds,
    )
    if minimum_batch > 1 or profiles not in (None, ("0",)):
        # Model Analyzer's quick generator replaces instance_group and would
        # discard TensorRT profile selection. Its default client batch of one also
        # cannot serve profiles whose minimum batch is larger. Use a bounded sweep.
        fast_batch_sizes = tuple(dict.fromkeys((minimum_batch, maximum_batch))) if config.max_batch_size > 0 else (1,)
        fast = manual_config(
            "fast",
            selected_batch_sizes=fast_batch_sizes,
            concurrency=(1, 8, 32),
            instance_counts=tuple(range(1, min(max_instance_count, 2) + 1)),
            queue_delays=(0,),
        )
    else:
        fast = QuickModelAnalyzerConfig(
            model_repository=repository,
            perf_analyzer_flags=perf_flags,
            profile_models=(model_name,),
            checkpoint_directory=(destination / "fast-checkpoints").resolve(),
            output_model_repository_path=(destination / "fast-model-repository").resolve(),
            export_path=(destination / "fast-results").resolve(),
            run_config_search_max_instance_count=max_instance_count,
            **batch_range,
        )
    return fast, manual, input_data


def generate_model_analyzer_configs(
    artifact: DeploymentArtifact,
    /,
    *,
    model_path: str | os.PathLike[str],
    path: str | os.PathLike[str],
    max_instance_count: int = 5,
    queue_delay_microseconds: tuple[int, ...] = _DEFAULT_QUEUE_DELAYS_MICROSECONDS,
) -> Path:
    """Generate fast and exhaustive Model Analyzer YAML for a published model.

    ``fast.yaml`` uses Model Analyzer's quick search where possible. TensorRT
    profile selections beyond profile zero or minimum batches above one require
    a reduced brute sweep to preserve those constraints.
    ``manual.yaml`` enumerates the complete recommended bounded search space.
    Publication already writes defaults; use this function only to customize them.
    Concrete input shapes use TensorRT profile optima or other artifacts' minimum
    shapes. Input values use Perf Analyzer's synthetic-data defaults.

    Args:
        artifact: Tuned artifact used to create the published model.
        model_path: Model directory returned by :func:`aitune.triton.publish`.
        path: New directory in which to write ``fast.yaml`` and ``manual.yaml``.
        max_instance_count: Largest GPU model instance count to explore.
        queue_delay_microseconds: Dynamic batching delays for the manual search.

    Returns:
        Directory containing the two generated configuration files.
    """
    if not isinstance(max_instance_count, int) or isinstance(max_instance_count, bool) or max_instance_count < 1:
        raise AITuneUserInputError(f"max_instance_count must be a positive integer, got {max_instance_count!r}")
    if not queue_delay_microseconds or any(
        not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 for delay in queue_delay_microseconds
    ):
        raise AITuneUserInputError("queue_delay_microseconds must contain non-negative integers")

    model_directory = Path(model_path)
    destination = Path(path)
    if destination.exists():
        raise ModelAnalyzerConfigError(f"{destination} already exists; configuration generation never replaces files")

    config = _read_published_config(model_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".aitune-{destination.name}-", dir=destination.parent))
    try:
        _write_model_analyzer_configs(
            artifact,
            config=config,
            model_directory=model_directory,
            destination=destination,
            staging=staging,
            max_instance_count=max_instance_count,
            queue_delay_microseconds=queue_delay_microseconds,
        )
        staging.rename(destination)
    except Exception as error:
        raise ModelAnalyzerConfigError(f"Failed to generate Model Analyzer configurations: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _write_model_analyzer_configs(
    artifact: DeploymentArtifact,
    *,
    config: model_config_pb2.ModelConfig,
    model_directory: Path,
    destination: Path,
    staging: Path,
    input_data_path: Path | None = None,
    max_instance_count: int = 5,
    queue_delay_microseconds: tuple[int, ...] = _DEFAULT_QUEUE_DELAYS_MICROSECONDS,
) -> None:
    """Write staged configs with paths pointing to their final deployment location."""
    fast, manual, input_data = _configs(
        artifact,
        model_directory,
        destination,
        config=config,
        input_data_path=input_data_path,
        max_instance_count=max_instance_count,
        queue_delay_microseconds=queue_delay_microseconds,
    )
    staging.mkdir(parents=True, exist_ok=True)
    (staging / _FAST_CONFIG_FILE_NAME).write_text(fast.to_yaml())
    (staging / _MANUAL_CONFIG_FILE_NAME).write_text(manual.to_yaml())
    if input_data is not None:
        (staging / _INPUT_DATA_FILE_NAME).write_text(json.dumps(input_data, indent=2) + "\n")
