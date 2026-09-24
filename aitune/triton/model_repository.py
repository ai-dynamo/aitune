# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from artifacts and existing model files."""

import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from aitune.exceptions import AITunePublicationError, AITuneUserInputError
from aitune.records import DeploymentArtifact
from aitune.triton.config import BaseModelConfig, ONNXRuntimeModelConfig, TensorRTModelConfig, TorchAOTIModelConfig
from aitune.triton.config.common import tensor_config
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
    latency_budget_ms: int | None = None,
    latency_percentile: int = 95,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None = None,
    additional_files: Sequence[str | os.PathLike[str]] = (),
    resources: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Publish a tuned artifact or an existing model file to a Triton repository.

    The operation never replaces an existing model. Files are written directly
    into the new model directory, so a failure can leave an incomplete directory.
    For artifacts, a Model Analyzer config is generated under
    ``model_analyzer/config.yaml`` with input shapes derived from the artifact.
    Without an explicit config, the Triton batch limit and scheduler are derived
    from the artifact's tensor bounds and runtime call structure.
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
        model_name: New model directory name for an artifact; defaults to ``config.name`` when config is supplied.
        model_version: Positive Triton model version.
        latency_budget_ms: Optional latency limit for Model Analyzer, in milliseconds.
        latency_percentile: Latency percentile used for stabilization and the optional budget (90, 95, or 99).
        config: Complete backend-specific configuration. For an artifact, it is derived when omitted and must match
            the artifact's executable interface when supplied. An existing model file requires it.
        additional_files: ONNX external-data paths relative to the source file's directory.
        resources: Model-relative auxiliary destination paths mapped to local files.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If publication arguments are invalid.
        AITunePublicationError: If the artifact cannot be represented or publication fails.
    """
    if isinstance(model, DeploymentArtifact):
        if config is not None and type(config) not in _MODEL_CONFIGS.values():
            raise AITunePublicationError(
                "config must be a TensorRTModelConfig, ONNXRuntimeModelConfig, or TorchAOTIModelConfig"
            )
        if additional_files or resources:
            raise AITuneUserInputError("Artifact publication derives additional files from the artifact")
        return _publish_deployment_artifact(
            model,
            path=path,
            model_name=model_name,
            model_version=model_version,
            latency_budget_ms=latency_budget_ms,
            latency_percentile=latency_percentile,
            config=config,
        )
    if not isinstance(model, str | os.PathLike):
        raise AITuneUserInputError("model must be an artifact or a model file path")
    if config is None:
        raise AITuneUserInputError("config is required when publishing a model file")
    if latency_budget_ms is not None or latency_percentile != 95:
        raise AITuneUserInputError("Model Analyzer options require a deployment artifact")
    if model_name is not None:
        raise AITuneUserInputError("For a model file, specify deployment settings in config")
    return _publish_existing_model(
        model,
        path=path,
        model_version=model_version,
        config=config,
        additional_files=additional_files,
        resources=resources,
    )


def _publish_deployment_artifact(
    artifact: DeploymentArtifact,
    *,
    path: str | os.PathLike[str],
    model_name: str | None,
    model_version: int,
    latency_budget_ms: int | None,
    latency_percentile: int,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None,
) -> Path:
    """Publish an artifact and generate its Model Analyzer configuration."""
    model_name = _validate_target(
        model_name if model_name is not None else config.name if config else None, model_version
    )
    if latency_budget_ms is not None and (
        not isinstance(latency_budget_ms, int) or isinstance(latency_budget_ms, bool) or latency_budget_ms < 1
    ):
        raise AITuneUserInputError(f"latency_budget_ms must be a positive integer, got {latency_budget_ms!r}")
    if (
        not isinstance(latency_percentile, int)
        or isinstance(latency_percentile, bool)
        or latency_percentile not in (90, 95, 99)
    ):
        raise AITuneUserInputError(f"latency_percentile must be one of 90, 95, 99, got {latency_percentile!r}")

    file_name, multi_file = _artifact_layout(artifact)
    _validate_artifact_files(artifact, multi_file=multi_file)
    config = _model_config(artifact, model_name=model_name, config=config, file_name=file_name)
    _validate_version_policy(config, model_version)
    _resource_files(config, {})
    try:
        config_text = config.to_pbtxt()
    except (TypeError, ValueError) as error:
        raise AITunePublicationError(f"Invalid Triton configuration for {model_name!r}: {error}") from error

    try:
        repository = Path(path)
        model_directory, version_directory = _create_model_directories(repository, model_name, model_version)
        (model_directory / _CONFIG_FILE_NAME).write_text(config_text)
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
            latency_percentile=latency_percentile,
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


def _create_model_directories(repository: Path, model_name: str, model_version: int) -> tuple[Path, Path]:
    """Create a new Triton model entry without replacing an existing one."""
    model_directory = repository / model_name
    if model_directory.exists():
        raise AITunePublicationError(f"{model_directory} already exists; Triton publication never replaces a model")
    repository.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir()
    version_directory = model_directory / str(model_version)
    version_directory.mkdir()
    return model_directory, version_directory


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


def _model_config(
    artifact: DeploymentArtifact,
    *,
    model_name: str,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None,
    file_name: str,
) -> BaseModelConfig:
    """Derive or validate the Triton configuration for an artifact."""
    if not artifact.inputs or not artifact.outputs:
        raise AITunePublicationError("Triton publication requires at least one input and one output")
    config_type = _MODEL_CONFIGS.get(artifact.runtime.name)
    if config_type is None:
        raise AITunePublicationError(f"Unsupported Triton runtime: {artifact.runtime.name!r}")
    try:
        generated = config_type.from_artifact(artifact, name=model_name)
    except (KeyError, TypeError, ValueError) as error:
        raise AITunePublicationError(f"Invalid Triton configuration for {artifact.runtime.name!r}: {error}") from error
    if config is None:
        return generated
    _validate_artifact_config(config, generated, artifact, file_name=file_name)
    return config


def _validate_artifact_config(
    config: BaseModelConfig, generated: BaseModelConfig, artifact: DeploymentArtifact, *, file_name: str
) -> None:
    """Require a supplied config to describe the artifact's executable contract."""
    if type(config) is not type(generated) or config.name != generated.name:
        raise AITunePublicationError("Supplied Triton config does not match the artifact runtime or model name")
    batched = config.max_batch_size > 0
    expected_inputs = tuple(tensor_config(tensor, batched=batched) for tensor in artifact.inputs)
    expected_outputs = tuple(tensor_config(tensor, batched=batched) for tensor in artifact.outputs)
    if config.inputs != expected_inputs or config.outputs != expected_outputs:
        raise AITunePublicationError("Supplied Triton config tensor interface does not match the artifact")
    if config.default_model_filename not in (None, file_name):
        raise AITunePublicationError("Supplied Triton config default_model_filename does not match the artifact layout")
    if (
        isinstance(config, TensorRTModelConfig)
        and isinstance(generated, TensorRTModelConfig)
        and not set(config.optimization_profile_indices).issubset(generated.optimization_profile_indices)
    ):
        raise AITunePublicationError("Supplied Triton config optimization profiles do not match the artifact")
    if (
        isinstance(config, TorchAOTIModelConfig)
        and isinstance(generated, TorchAOTIModelConfig)
        and config.structured_call != generated.structured_call
    ):
        raise AITunePublicationError("Supplied Triton config structured_call does not match the artifact")


def _publish_existing_model(
    model: str | os.PathLike[str],
    *,
    path: str | os.PathLike[str],
    model_version: int,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig,
    additional_files: Sequence[str | os.PathLike[str]],
    resources: Mapping[str, str | os.PathLike[str]] | None,
) -> Path:
    """Publish an existing file using its explicit Triton configuration."""
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
    _validate_version_policy(config, model_version)
    source = Path(model)
    relative_paths = _additional_file_paths(source, additional_files)
    if relative_paths and not isinstance(config, ONNXRuntimeModelConfig):
        raise AITunePublicationError("Only ONNX models support additional files")
    file_name = config.default_model_filename or layouts[type(config)]
    if Path(file_name) in relative_paths:
        raise AITuneUserInputError(f"An additional file cannot overwrite {file_name}")
    resource_files = _resource_files(config, resources or {})
    repository = Path(path)
    model_directory = repository / config.name
    try:
        model_directory, version_directory = _create_model_directories(repository, config.name, model_version)
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
        if isinstance(error, AITunePublicationError):
            raise
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


def _validate_version_policy(config: BaseModelConfig, model_version: int) -> None:
    """Require a specific version policy to serve the version being published."""
    if config.version_policy is not None and "specific" in config.version_policy:
        allowed_versions = config.version_policy["specific"].get("versions", ())
        if model_version not in allowed_versions:
            raise AITuneUserInputError(
                f"model_version {model_version} is excluded by config.version_policy.specific: {allowed_versions}"
            )


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
    config: BaseModelConfig,
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
