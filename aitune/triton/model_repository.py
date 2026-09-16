# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from artifacts and existing model files."""

import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from aitune.exceptions import AITuneError, AITuneUserInputError
from aitune.records import DeploymentArtifact
from aitune.triton.config import (
    ONNXRuntimeModelConfig,
    TensorRTModelConfig,
    TorchAOTIModelConfig,
)
from aitune.triton.config.common import _BaseModelConfig
from aitune.triton.model_analyzer import _write_model_analyzer_configs

logger = logging.getLogger(__name__)

__all__ = [
    "ONNXRuntimeModelConfig",
    "PublicationError",
    "TensorRTModelConfig",
    "TorchAOTIModelConfig",
    "publish",
]

_CONFIG_FILE_NAME = "config.pbtxt"
_MODEL_CONFIGS: dict[str, type[_BaseModelConfig]] = {
    "tensorrt": TensorRTModelConfig,
    "onnxruntime": ONNXRuntimeModelConfig,
    "aotinductor": TorchAOTIModelConfig,
}


class PublicationError(AITuneError):
    """Raised when an artifact cannot be published as a Triton model."""


def _artifact_layout(artifact: DeploymentArtifact) -> tuple[str, bool]:
    """Return Triton's file name and multi-file behavior."""
    format_runtime = (artifact.model.format, artifact.runtime.name)
    if format_runtime == ("tensorrt_plan", "tensorrt"):
        return "model.plan", False
    if format_runtime == ("onnx", "onnxruntime"):
        return "model.onnx", True
    if format_runtime == ("pt2", "aotinductor"):
        return "model.pt2", False
    raise PublicationError(
        f"Unsupported Triton model format/runtime pair: {format_runtime!r}. "
        "Supported pairs are tensorrt_plan/tensorrt, onnx/onnxruntime, and pt2/aotinductor"
    )


def _validate_artifact_files(artifact: DeploymentArtifact, *, multi_file: bool) -> None:
    """Reject additional files when Triton's format has no defined layout."""
    if artifact.model.additional_files and not multi_file:
        raise PublicationError(
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
        raise PublicationError(
            "Cannot publish a batched model: every input and output must have a batch axis starting at 1 "
            f"and support a batch size of at least {required}"
        )
    if any(tensor.batch_axis != 0 for tensor in artifact.inputs + artifact.outputs):
        raise PublicationError("Triton implicit batching requires batch_axis=0 for every input and output")
    if requested is not None and requested > supported:
        raise PublicationError(
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
) -> _BaseModelConfig:
    """Build a validated Triton configuration from an artifact."""
    if not artifact.inputs or not artifact.outputs:
        raise PublicationError("Triton publication requires at least one input and one output")
    batch_size = _batch_size(artifact, dynamic_batching, max_batch_size)
    config_type = _MODEL_CONFIGS.get(artifact.runtime.name)
    if config_type is None:
        raise PublicationError(f"Unsupported Triton runtime: {artifact.runtime.name!r}")
    try:
        return config_type.from_artifact(
            artifact, name=model_name, max_batch_size=batch_size, dynamic_batching=dynamic_batching
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PublicationError(f"Invalid Triton configuration for {artifact.runtime.name!r}: {error}") from error


def publish(
    model: DeploymentArtifact | str | os.PathLike[str],
    /,
    *,
    path: str | os.PathLike[str],
    model_name: str | None = None,
    model_version: int = 1,
    dynamic_batching: bool = False,
    max_batch_size: int | None = None,
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig | None = None,
    additional_files: Sequence[str | os.PathLike[str]] = (),
    resources: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Publish a tuned artifact or an existing model file to a Triton repository.

    For artifacts, supply ``model_name`` and optional batching settings. For an
    existing file, supply a specialized ``config`` containing the model name,
    tensor interface, and batching settings. The configuration type selects the
    backend; file contents are not inspected, compiled, or tuned.

    Args:
        model: Tuned artifact or path to an existing ONNX, TensorRT, or PT2 file.
        path: Triton model repository root.
        model_name: Target model name for an artifact. For a file, use ``config.name``.
        model_version: Positive Triton model version.
        dynamic_batching: Enable request batching for an artifact.
        max_batch_size: Implicit batch limit within an artifact's tuned bounds.
        config: Explicit backend-specific configuration required for a file.
        additional_files: ONNX external-data paths relative to the source file's directory.
            Artifact additional files are taken from the artifact record instead.
        resources: Model-relative destination paths mapped to local files, for example
            {"warmup/sample.bin": "samples/input.bin"} or {"labels.txt": "labels.txt"}.

    Returns:
        Path to the new model directory. Existing models are never replaced.

    Raises:
        AITuneUserInputError: If arguments are invalid or mix file and artifact options.
        PublicationError: If the model cannot be represented or publication fails.
    """
    if isinstance(model, DeploymentArtifact):
        if config is not None or additional_files or resources:
            raise AITuneUserInputError("Artifact publication derives config and additional files from the artifact")
        if model_name is None:
            raise AITuneUserInputError("model_name is required when publishing an artifact")
        return _publish_artifact(
            model,
            path=path,
            model_name=model_name,
            model_version=model_version,
            dynamic_batching=dynamic_batching,
            max_batch_size=max_batch_size,
        )
    if not isinstance(model, str | os.PathLike):
        raise AITuneUserInputError("model must be an artifact or a model file path")
    if config is None:
        raise AITuneUserInputError("config is required when publishing a model file")
    if model_name is not None or dynamic_batching or max_batch_size is not None:
        raise AITuneUserInputError("For a model file, specify the model name and batching settings in config")
    return _publish_file(
        model,
        path=path,
        config=config,
        model_version=model_version,
        additional_files=additional_files,
        resources=resources,
    )


def _publish_artifact(
    artifact: DeploymentArtifact,
    /,
    *,
    path: str | os.PathLike[str],
    model_name: str,
    model_version: int = 1,
    dynamic_batching: bool = False,
    max_batch_size: int | None = None,
) -> Path:
    """Publish one tuned artifact into a new Triton model repository entry.

    The operation never replaces an existing model. Files are copied and staged
    before the completed model directory is moved into the repository. Model Analyzer
    configs are generated automatically under ``model_analyzer/fast.yaml`` and
    ``model_analyzer/manual.yaml``, with input shapes derived from the artifact.

    Supported model format/runtime pairs are ``onnx``/``onnxruntime``,
    ``tensorrt_plan``/``tensorrt``, and ``pt2``/``aotinductor``. ONNX provider
    selection comes from ``runtime.options["execution_provider"]``. TensorRT uses
    ``model.metadata["optimization_profile_count"]`` and the optional runtime
    setting ``use_cuda_graphs``. Analyzer shapes come from
    ``model.metadata["optimization_profiles"]``. PT2 uses
    ``model.metadata["structured_call"]``.

    Args:
        artifact: Deployment record containing model files, tensor specs, and runtime settings.
        path: Triton model repository root.
        model_name: New model directory name.
        model_version: Positive Triton model version.
        dynamic_batching: Let Triton combine independent client requests.
        max_batch_size: Optional implicit batch limit within the artifact's tuned bounds.
            It can be set without enabling the dynamic batcher.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If publication arguments are invalid.
        PublicationError: If the artifact cannot be represented or publication fails.
    """
    _validate_target(model_name, model_version)

    file_name, multi_file = _artifact_layout(artifact)
    _validate_artifact_files(artifact, multi_file=multi_file)
    config = _model_config(
        artifact,
        model_name=model_name,
        dynamic_batching=dynamic_batching,
        max_batch_size=max_batch_size,
    )

    return _publish_model(
        path=path,
        config=config,
        model_version=model_version,
        file_name=file_name,
        nested=bool(artifact.model.additional_files),
        export_files=artifact.model.export_files,
        artifact=artifact,
    )


def _validate_target(model_name: str, model_version: int) -> None:
    """Validate repository directory components before writing files."""
    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name != model_name.strip()
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        raise AITuneUserInputError(f"Invalid Triton model name: {model_name!r}")
    if not isinstance(model_version, int) or isinstance(model_version, bool) or model_version < 1:
        raise AITuneUserInputError(f"model_version must be a positive integer, got {model_version!r}")


def _publish_model(
    *,
    path: str | os.PathLike[str],
    config: _BaseModelConfig,
    model_version: int,
    file_name: str,
    nested: bool,
    export_files: Callable[[Path], object],
    resources: Mapping[Path, Path] | None = None,
    artifact: DeploymentArtifact | None = None,
) -> Path:
    """Stage a model and its configuration before publishing the directory."""
    model_name = config.name
    repository = Path(path)
    model_directory = repository / model_name
    if model_directory.exists():
        raise PublicationError(f"{model_directory} already exists; Triton publication never replaces a model")

    repository.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".aitune-{model_name}-", dir=repository))
    try:
        staged_model = staging / model_name
        version_directory = staged_model / str(model_version)
        version_directory.mkdir(parents=True)
        (staged_model / _CONFIG_FILE_NAME).write_text(config.to_pbtxt())
        destination = version_directory / file_name
        if nested:
            destination = destination / file_name
        export_files(destination)
        for relative, source in (resources or {}).items():
            target = staged_model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if artifact is not None:
            _write_model_analyzer_configs(
                artifact,
                config=config.to_protobuf(),
                model_directory=model_directory,
                destination=repository.resolve().parent / f"{repository.resolve().name}-model-analyzer" / model_name,
                staging=staged_model / "model_analyzer",
            )
        staged_model.rename(model_directory)
    except Exception as error:
        raise PublicationError(f"Failed to publish Triton model {model_name!r}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Published Triton model to %s", model_directory)
    return model_directory


def _publish_file(
    model_file: str | os.PathLike[str],
    /,
    *,
    path: str | os.PathLike[str],
    config: TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig,
    model_version: int = 1,
    additional_files: Sequence[str | os.PathLike[str]] = (),
    resources: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Publish an existing model file using an explicit Triton configuration.

    No tuning, compilation, or model inspection is performed. The caller supplies
    the executable tensor interface and supported batching in ``config``. Its type
    selects the backend and destination filename, regardless of the source suffix.
    Existing model directories are never replaced.

    Args:
        model_file: Existing ONNX graph, TensorRT plan, or AOTInductor PT2 package.
        path: Triton model repository root.
        config: Backend-specific configuration, including the target model name.
            For implicit batching, tensor dimensions omit the leading batch axis.
        model_version: Positive Triton model version.
        additional_files: ONNX external-data paths relative to ``model_file.parent``.
            Relative paths are preserved beside the published graph.
        resources: Auxiliary files mapped by destination relative to the model directory.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If target arguments or additional file paths are invalid.
        PublicationError: If the configuration is unsupported or copying fails.
    """
    layouts = {
        TensorRTModelConfig: "model.plan",
        ONNXRuntimeModelConfig: "model.onnx",
        TorchAOTIModelConfig: "model.pt2",
    }
    if type(config) not in layouts:
        raise PublicationError("config must be a TensorRTModelConfig, ONNXRuntimeModelConfig, or TorchAOTIModelConfig")
    _validate_target(config.name, model_version)
    source = Path(model_file)
    relative_paths = _additional_file_paths(source, additional_files)
    if relative_paths and not isinstance(config, ONNXRuntimeModelConfig):
        raise PublicationError("Only ONNX models support additional files")
    file_name = config.default_model_filename or layouts[type(config)]
    resource_files = _resource_files(config, resources or {})
    if Path(file_name) in relative_paths:
        raise AITuneUserInputError(f"An additional file cannot overwrite {file_name}")

    def copy_files(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        for relative in relative_paths:
            target = destination.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.parent / relative, target)

    return _publish_model(
        path=path,
        config=config,
        model_version=model_version,
        file_name=file_name,
        nested=bool(relative_paths),
        export_files=copy_files,
        resources=resource_files,
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
