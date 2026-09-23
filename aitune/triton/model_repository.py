# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from tuned AITune artifacts."""

import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from aitune.exceptions import AITunePublicationError, AITuneUserInputError
from aitune.records import DeploymentArtifact
from aitune.triton.config import (
    BaseModelConfig,
    ONNXRuntimeModelConfig,
    TensorRTModelConfig,
    TorchAOTIModelConfig,
)
from aitune.triton.model_analyzer import write_model_analyzer_config

logger = logging.getLogger(__name__)

PublicationError = AITunePublicationError

_CONFIG_FILE_NAME = "config.pbtxt"
_MODEL_CONFIGS: dict[str, type[BaseModelConfig]] = {
    "tensorrt": TensorRTModelConfig,
    "onnxruntime": ONNXRuntimeModelConfig,
    "aotinductor": TorchAOTIModelConfig,
}


def publish(
    model: DeploymentArtifact | str | os.PathLike[str],
    /,
    *,
    path: str | os.PathLike[str],
    model_name: str | None = None,
    model_version: int = 1,
    dynamic_batching: bool | None = None,
    max_batch_size: int | None = None,
    latency_budget_ms: int | None = None,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None = None,
    additional_files: Sequence[str | os.PathLike[str]] = (),
    resources: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Publish a tuned artifact or an existing model file to a Triton repository.

    The operation never replaces an existing model. Files are written directly
    into the new model directory, so a failure can leave an incomplete directory.
    For artifacts, a Model Analyzer config is generated under
    ``model_analyzer/config.yaml`` with input shapes derived from the artifact.
    Existing model files require an explicit backend-specific ``config`` and
    are copied without compilation or tuning. Callers must keep generation
    separate from updates to a live Triton repository.

    Supported model format/runtime pairs are ``onnx``/``onnxruntime``,
    ``tensorrt_plan``/``tensorrt``, and ``pt2``/``aotinductor``. ONNX provider
    selection comes from ``runtime.options["execution_provider"]``. TensorRT uses
    ``model.metadata["optimization_profile_count"]`` and the optional runtime
    setting ``use_cuda_graphs``. PT2 uses ``model.metadata["structured_call"]``.

    Args:
        model: Deployment artifact or path to an existing model file.
        path: Triton model repository root.
        model_name: New model directory name for an artifact; file publication uses ``config.name``.
        model_version: Positive Triton model version.
        dynamic_batching: Let Triton combine independent client requests. Enabled by default.
        max_batch_size: Optional implicit batch limit within the artifact's tuned bounds.
            It can be set without enabling the dynamic batcher.
        latency_budget_ms: Optional p99 latency limit for Model Analyzer, in milliseconds.
        config: Backend-specific configuration required for an existing file.
        additional_files: ONNX external-data paths relative to the source file's directory.
        resources: Model-relative auxiliary destination paths mapped to local files.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If publication arguments are invalid.
        AITunePublicationError: If the artifact cannot be represented or publication fails.
    """
    if not isinstance(model, DeploymentArtifact):
        return _publish_existing_request(
            model,
            path=path,
            model_name=model_name,
            model_version=model_version,
            dynamic_batching=dynamic_batching,
            max_batch_size=max_batch_size,
            latency_budget_ms=latency_budget_ms,
            config=config,
            additional_files=additional_files,
            resources=resources,
        )
    if config is not None or additional_files or resources:
        raise AITuneUserInputError("Artifact publication derives config and additional files from the artifact")
    artifact = model
    model_name = _validate_target(model_name, model_version)
    if latency_budget_ms is not None and (
        not isinstance(latency_budget_ms, int) or isinstance(latency_budget_ms, bool) or latency_budget_ms < 1
    ):
        raise AITuneUserInputError(f"latency_budget_ms must be a positive integer, got {latency_budget_ms!r}")

    if dynamic_batching is None:
        dynamic_batching = True

    file_name, multi_file = _artifact_layout(artifact)
    _validate_artifact_files(artifact, multi_file=multi_file)
    config = _model_config(
        artifact,
        model_name=model_name,
        dynamic_batching=dynamic_batching,
        max_batch_size=max_batch_size,
    )

    try:
        repository = Path(path)
        model_directory = repository / model_name
        if model_directory.exists():
            raise AITunePublicationError(f"{model_directory} already exists; Triton publication never replaces a model")

        repository.mkdir(parents=True, exist_ok=True)
        model_directory.mkdir()
        version_directory = model_directory / str(model_version)
        version_directory.mkdir(parents=True)
        (model_directory / _CONFIG_FILE_NAME).write_text(config.to_pbtxt())
        destination = version_directory / file_name
        if artifact.model.additional_files:
            destination = destination / file_name
        artifact.model.export_files(destination)
        repository_path = repository.resolve()
        write_model_analyzer_config(
            artifact,
            config=config.to_protobuf(),
            model_directory=model_directory,
            destination=repository_path.parent / f"{repository_path.name}-model-analyzer" / model_name,
            latency_budget_ms=latency_budget_ms,
            output_directory=model_directory / "model_analyzer",
            input_data_path=model_directory / "model_analyzer" / "input-data.json",
        )
    except Exception as error:
        if isinstance(error, AITunePublicationError):
            raise
        raise AITunePublicationError(
            f"Failed to publish Triton model {model_name!r}: {error}. "
            f"An incomplete model directory may remain at {Path(path) / model_name}"
        ) from error

    logger.info("Published Triton model to %s", model_directory)
    return model_directory


def _publish_existing_request(
    model: str | os.PathLike[str],
    *,
    path: str | os.PathLike[str],
    model_name: str | None,
    model_version: int,
    dynamic_batching: bool | None,
    max_batch_size: int | None,
    latency_budget_ms: int | None,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None,
    additional_files: Sequence[str | os.PathLike[str]],
    resources: Mapping[str, str | os.PathLike[str]] | None,
) -> Path:
    """Require file publication settings to come from the supplied config."""
    if not isinstance(model, str | os.PathLike):
        raise AITuneUserInputError("model must be an artifact or a model file path")
    if config is None:
        raise AITuneUserInputError("config is required when publishing a model file")
    if (
        model_name is not None
        or dynamic_batching is not None
        or max_batch_size is not None
        or latency_budget_ms is not None
    ):
        raise AITuneUserInputError("For a model file, specify deployment settings in config")
    return _publish_existing_file(
        model,
        path=path,
        config=config,
        model_version=model_version,
        additional_files=additional_files,
        resources=resources,
    )


def _artifact_layout(artifact: DeploymentArtifact) -> tuple[str, bool]:
    """Return Triton's file name and multi-file behavior."""
    format_runtime = (artifact.model.format, artifact.runtime.name)
    if format_runtime == ("tensorrt_plan", "tensorrt"):
        return "model.plan", False
    if format_runtime == ("onnx", "onnxruntime"):
        return "model.onnx", True
    if format_runtime == ("pt2", "aotinductor"):
        return "model.pt2", False
    raise AITunePublicationError(
        f"Unsupported Triton model format/runtime pair: {format_runtime!r}. "
        "Supported pairs are tensorrt_plan/tensorrt, onnx/onnxruntime, and pt2/aotinductor"
    )


def _validate_artifact_files(artifact: DeploymentArtifact, *, multi_file: bool) -> None:
    """Reject additional files when Triton's format has no defined layout."""
    if artifact.model.additional_files and not multi_file:
        raise AITunePublicationError(
            f"{artifact.model.format!r} is a single-file Triton format, but the artifact has additional files"
        )


def _batch_size(artifact: DeploymentArtifact, dynamic_batching: bool, requested: int | None) -> int:
    """Resolve and validate the Triton deployment batch limit."""
    if requested is None and not dynamic_batching:
        return 0
    if requested is not None and (not isinstance(requested, int) or isinstance(requested, bool) or requested < 1):
        raise AITuneUserInputError(f"max_batch_size must be a positive integer, got {requested!r}")

    supported = artifact.max_batch_size
    required = 2 if dynamic_batching else 1
    if supported is None or supported < required:
        raise AITunePublicationError(
            "Cannot publish a batched model: every input and output must have a batch axis starting at 1 "
            f"and support a batch size of at least {required}"
        )
    if any(tensor.batch_axis != 0 for tensor in artifact.inputs + artifact.outputs):
        raise AITunePublicationError("Triton implicit batching requires batch_axis=0 for every input and output")
    if requested is not None and requested > supported:
        raise AITunePublicationError(
            f"max_batch_size {requested} exceeds the artifact's bounded batch maximum of {supported}"
        )
    batch_size = supported if requested is None else requested
    if dynamic_batching and batch_size < 2:
        raise AITuneUserInputError("dynamic_batching requires max_batch_size of at least 2")
    return batch_size


def _model_config(
    artifact: DeploymentArtifact,
    *,
    model_name: str,
    dynamic_batching: bool,
    max_batch_size: int | None,
) -> BaseModelConfig:
    """Build a validated Triton configuration from an artifact."""
    if not artifact.inputs or not artifact.outputs:
        raise AITunePublicationError("Triton publication requires at least one input and one output")
    batch_size = _batch_size(artifact, dynamic_batching, max_batch_size)
    config_type = _MODEL_CONFIGS.get(artifact.runtime.name)
    if config_type is None:
        raise AITunePublicationError(f"Unsupported Triton runtime: {artifact.runtime.name!r}")
    try:
        return config_type.from_artifact(
            artifact,
            name=model_name,
            max_batch_size=batch_size,
            dynamic_batching=dynamic_batching,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AITunePublicationError(f"Invalid Triton configuration for {artifact.runtime.name!r}: {error}") from error


def _publish_existing_file(
    model_file: str | os.PathLike[str],
    *,
    path: str | os.PathLike[str],
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig,
    model_version: int,
    additional_files: Sequence[str | os.PathLike[str]],
    resources: Mapping[str, str | os.PathLike[str]] | None,
) -> Path:
    """Copy a supplied model using its explicit, validated Triton configuration."""
    layouts = {
        TensorRTModelConfig: "model.plan",
        ONNXRuntimeModelConfig: "model.onnx",
        TorchAOTIModelConfig: "model.pt2",
    }
    if type(config) not in layouts:
        raise AITunePublicationError(
            "config must be a TensorRTModelConfig, ONNXRuntimeModelConfig, or TorchAOTIModelConfig"
        )
    _validate_target(config.name, model_version)
    source = Path(model_file)
    relative_paths = _additional_file_paths(source, additional_files)
    if relative_paths and not isinstance(config, ONNXRuntimeModelConfig):
        raise AITunePublicationError("Only ONNX models support additional files")
    file_name = config.default_model_filename or layouts[type(config)]
    if Path(file_name) in relative_paths:
        raise AITuneUserInputError(f"An additional file cannot overwrite {file_name}")
    resource_files = _resource_files(config, resources or {})
    repository = Path(path)
    model_directory = repository / config.name
    if model_directory.exists():
        raise AITunePublicationError(f"{model_directory} already exists; Triton publication never replaces a model")

    try:
        repository.mkdir(parents=True, exist_ok=True)
        model_directory.mkdir()
        version_directory = model_directory / str(model_version)
        version_directory.mkdir()
        (model_directory / _CONFIG_FILE_NAME).write_text(config.to_pbtxt())
        destination = version_directory / file_name
        if relative_paths:
            destination = destination / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        for relative in relative_paths:
            target = destination.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.parent / relative, target)
        for relative, resource_source in resource_files.items():
            target = model_directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resource_source, target)
    except Exception as error:
        raise AITunePublicationError(
            f"Failed to publish Triton model {config.name!r}: {error}. "
            f"An incomplete model directory may remain at {model_directory}"
        ) from error
    logger.info("Published Triton model to %s", model_directory)
    return model_directory


def _validate_target(model_name: str | None, model_version: int) -> str:
    """Return a validated repository model name before writing files."""
    if not isinstance(model_name, str):
        raise AITuneUserInputError(f"Invalid Triton model name: {model_name!r}")
    if (
        not model_name
        or model_name != model_name.strip()
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        raise AITuneUserInputError(f"Invalid Triton model name: {model_name!r}")
    if not isinstance(model_version, int) or isinstance(model_version, bool) or model_version < 1:
        raise AITuneUserInputError(f"model_version must be a positive integer, got {model_version!r}")
    return model_name


def _additional_file_paths(source: Path, additional_files: Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
    """Validate the relative file layout of an externally supplied model."""
    paths = tuple(Path(relative) for relative in additional_files)
    for relative in paths:
        if relative == Path() or relative.is_absolute() or ".." in relative.parts:
            raise AITuneUserInputError(f"Additional file path must stay inside the model directory: {relative}")
        if relative == Path(source.name):
            raise AITuneUserInputError("The model entry file cannot also be an additional file")
    if len(paths) != len(set(paths)):
        raise AITuneUserInputError("Additional file paths must be unique")
    return paths


def _resource_files(
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig,
    resources: Mapping[str, str | os.PathLike[str]],
) -> dict[Path, Path]:
    """Validate auxiliary file destinations and require files referenced by config."""
    files = {Path(relative): Path(source) for relative, source in resources.items()}
    for relative in files:
        if (
            relative == Path()
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] == _CONFIG_FILE_NAME
            or relative.parts[0].isdigit()
        ):
            raise AITuneUserInputError(f"Invalid model resource destination: {relative}")
    protobuf = config.to_protobuf()
    required = [Path(output.label_filename) for output in protobuf.output if output.label_filename]
    required.extend(
        Path("warmup") / value.input_data_file
        for sample in protobuf.model_warmup
        for value in sample.inputs.values()
        if value.input_data_file
    )
    required.extend(
        Path("initial_state") / initial.data_file
        for state in protobuf.sequence_batching.state
        for initial in state.initial_state
        if initial.data_file
    )
    missing = set(required) - files.keys()
    if missing:
        raise AITuneUserInputError(f"Missing model resources: {sorted(str(path) for path in missing)}")
    return files
