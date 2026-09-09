# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Triton model repositories from tuned AITune artifacts."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import ValidationError

from aitune.exceptions import AITuneError, AITuneUserInputError
from aitune.records import Artifact, ONNXArtifact, PT2Artifact, TensorRTPlanArtifact
from aitune.triton.config import (
    ONNXRuntimeModelConfig,
    TensorRTModelConfig,
    TorchAOTIModelConfig,
    tensor_config,
)
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


class PublicationError(AITuneError):
    """Raised when an artifact cannot be published as a Triton model."""


def _artifact_layout(artifact: Artifact) -> tuple[str, bool]:
    """Return Triton's file name and multi-file behavior."""
    if isinstance(artifact, TensorRTPlanArtifact):
        return "model.plan", False
    if isinstance(artifact, ONNXArtifact):
        return "model.onnx", True
    if isinstance(artifact, PT2Artifact):
        return "model.pt2", False
    supported = "ONNXArtifact, PT2Artifact, and TensorRTPlanArtifact"
    raise PublicationError(f"Triton publication supports {supported}, got {type(artifact).__name__}")


def _validate_artifact_files(artifact: Artifact, *, multi_file: bool) -> None:
    """Reject companion files when Triton's format has no defined layout."""
    if artifact.companions and not multi_file:
        raise PublicationError(
            f"{type(artifact).__name__} is a single-file Triton format, but the artifact has companion files"
        )


def _batch_size(artifact: Artifact, dynamic_batching: bool, requested: int | None) -> int:
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
    artifact: Artifact,
    *,
    model_name: str,
    dynamic_batching: bool,
    max_batch_size: int | None,
) -> TensorRTModelConfig | ONNXRuntimeModelConfig | TorchAOTIModelConfig:
    """Build a validated Triton configuration from an artifact."""
    if not artifact.inputs or not artifact.outputs:
        raise PublicationError("Triton publication requires at least one input and one output")
    batch_size = _batch_size(artifact, dynamic_batching, max_batch_size)
    batched = batch_size > 0
    try:
        inputs = tuple(tensor_config(tensor, batched=batched) for tensor in artifact.inputs)
        outputs = tuple(tensor_config(tensor, batched=batched) for tensor in artifact.outputs)
        if isinstance(artifact, TensorRTPlanArtifact):
            return TensorRTModelConfig(
                name=model_name,
                max_batch_size=batch_size,
                inputs=inputs,
                outputs=outputs,
                dynamic_batching=dynamic_batching,
                optimization_profile_indices=tuple(range(artifact.optimization_profile_count)),
                cuda_graphs=artifact.use_cuda_graphs,
            )
        if isinstance(artifact, ONNXArtifact):
            return ONNXRuntimeModelConfig(
                name=model_name,
                max_batch_size=batch_size,
                inputs=inputs,
                outputs=outputs,
                dynamic_batching=dynamic_batching,
                execution_provider=artifact.execution_provider,
            )
        if isinstance(artifact, PT2Artifact):
            return TorchAOTIModelConfig(
                name=model_name,
                max_batch_size=batch_size,
                inputs=inputs,
                outputs=outputs,
                dynamic_batching=dynamic_batching,
                structured_call=artifact.structured_call,
            )
    except ValidationError as error:
        raise PublicationError(f"Invalid Triton configuration for {type(artifact).__name__}: {error}") from error
    raise AssertionError(f"Unhandled Triton artifact: {type(artifact).__name__}")


def publish(
    artifact: Artifact,
    /,
    *,
    path: str | os.PathLike[str],
    model_name: str,
    model_version: int = 1,
    dynamic_batching: bool = False,
    max_batch_size: int | None = None,
) -> Path:
    """Publish one tuned artifact into a new Triton model repository entry.

    The operation never replaces an existing model. Files are verified and staged
    before the completed model directory is moved into the repository. Model Analyzer
    configs are generated automatically under ``model_analyzer/fast.yaml`` and
    ``model_analyzer/manual.yaml``, with input shapes derived from the artifact.

    Args:
        artifact: Tuned artifact to publish.
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
        if artifact.companions:
            destination = destination / file_name
        artifact.export_file(destination)
        _write_model_analyzer_configs(
            artifact,
            config=config.to_protobuf(),
            model_directory=model_directory,
            destination=repository.parent / f"{repository.name}-model-analyzer" / model_name,
            staging=staged_model / "model_analyzer",
        )
        staged_model.rename(model_directory)
    except Exception as error:
        raise PublicationError(f"Failed to publish Triton model {model_name!r}: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Published Triton model to %s", model_directory)
    return model_directory
