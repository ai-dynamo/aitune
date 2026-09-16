# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from tuned AITune artifacts."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from aitune.exceptions import AITuneError, AITuneUserInputError
from aitune.records import DeploymentArtifact
from aitune.triton.config import (
    ONNXRuntimeModelConfig,
    TensorRTModelConfig,
    TorchAOTIModelConfig,
)
from aitune.triton.config.common import _BaseModelConfig

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
    if not dynamic_batching:
        if requested is not None:
            raise AITuneUserInputError("max_batch_size requires dynamic_batching=True")
        return 0
    if requested is not None and (not isinstance(requested, int) or isinstance(requested, bool) or requested < 2):
        raise AITuneUserInputError(f"max_batch_size must be an integer of at least 2, got {requested!r}")

    supported = artifact.max_batch_size
    if supported is None or supported < 2:
        raise PublicationError(
            "Cannot enable dynamic batching: every input and output must have a batch axis starting at 1 "
            "and support a batch size of at least 2"
        )
    if any(tensor.batch_axis != 0 for tensor in artifact.inputs + artifact.outputs):
        raise PublicationError("Triton dynamic batching requires batch_axis=0 for every input and output")
    if requested is not None and requested > supported:
        raise PublicationError(
            f"max_batch_size {requested} exceeds the artifact's bounded batch maximum of {supported}"
        )
    return supported if requested is None else requested


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
        return config_type.from_artifact(artifact, name=model_name, max_batch_size=batch_size)
    except (KeyError, TypeError, ValueError) as error:
        raise PublicationError(f"Invalid Triton configuration for {artifact.runtime.name!r}: {error}") from error


def publish(
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
    before the completed model directory is moved into the repository.

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
        dynamic_batching: Let Triton combine independent client requests.
        max_batch_size: Optional cap within the artifact's tuned batch bounds.

    Returns:
        Path to the generated model directory.

    Raises:
        AITuneUserInputError: If publication arguments are invalid.
        PublicationError: If the artifact cannot be represented or publication fails.
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

    file_name, multi_file = _artifact_layout(artifact)
    _validate_artifact_files(artifact, multi_file=multi_file)
    config = _model_config(
        artifact,
        model_name=model_name,
        dynamic_batching=dynamic_batching,
        max_batch_size=max_batch_size,
    )

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
        if artifact.model.additional_files:
            destination = destination / file_name
        artifact.model.export_files(destination)
        staged_model.rename(model_directory)
    except Exception as error:
        raise PublicationError(f"Failed to publish Triton model {model_name!r}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Published Triton model to %s", model_directory)
    return model_directory
