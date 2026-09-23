# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from tuned AITune artifacts."""

import logging
import os
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

_CONFIG_FILE_NAME = "config.pbtxt"
_MODEL_CONFIGS: dict[str, type[BaseModelConfig]] = {
    "tensorrt": TensorRTModelConfig,
    "onnxruntime": ONNXRuntimeModelConfig,
    "aotinductor": TorchAOTIModelConfig,
}


def publish(
    artifact: DeploymentArtifact,
    /,
    *,
    path: str | os.PathLike[str],
    model_name: str,
    model_version: int = 1,
    dynamic_batching: bool = True,
    max_batch_size: int | None = None,
    latency_budget_ms: int | None = None,
) -> Path:
    """Publish one tuned artifact into a new Triton model repository entry.

    The operation never replaces an existing model. Files are written directly
    into the new model directory, so a failure can leave an incomplete directory.
    A Model Analyzer config is generated under ``model_analyzer/config.yaml``,
    with input shapes derived from the artifact. Callers must keep generation
    separate from updates to a live Triton repository.

    Supported model format/runtime pairs are ``onnx``/``onnxruntime``,
    ``tensorrt_plan``/``tensorrt``, and ``pt2``/``aotinductor``. ONNX provider
    selection comes from ``runtime.options["execution_provider"]``. TensorRT uses
    ``model.metadata["optimization_profile_count"]`` and the optional runtime
    setting ``use_cuda_graphs``. PT2 uses ``model.metadata["structured_call"]``.

    Args:
        artifact: Deployment record containing model files, tensor specs, and runtime settings.
        path: Triton model repository root.
        model_name: New model directory name.
        model_version: Positive Triton model version.
        dynamic_batching: Let Triton combine independent client requests. Enabled by default.
        max_batch_size: Optional implicit batch limit within the artifact's tuned bounds.
            It can be set without enabling the dynamic batcher.
        latency_budget_ms: Optional p99 latency limit for Model Analyzer, in milliseconds.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If publication arguments are invalid.
        AITunePublicationError: If the artifact cannot be represented or publication fails.
    """
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
    if latency_budget_ms is not None and (
        not isinstance(latency_budget_ms, int) or isinstance(latency_budget_ms, bool) or latency_budget_ms < 1
    ):
        raise AITuneUserInputError(f"latency_budget_ms must be a positive integer, got {latency_budget_ms!r}")

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
