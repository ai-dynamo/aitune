# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable references to tuned files that publishers can consume."""

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aitune.records.shapes import BoundedTensorSpec


@dataclass(frozen=True, kw_only=True)
class Artifact:
    """Reference a finalized tuning result and its tensor interface.

    An artifact exists only after tuning has established concrete tensor bounds.
    The record neither owns nor serializes its cache file. The AITune Package
    design will define source provenance and persistence.

    Args:
        inputs: Ordered tuned input specifications.
        outputs: Ordered tuned output specifications.
        path: Main tuned file in the AITune cache that the runtime opens.
        additional_files: Relative paths to files required by the main file, such
            as separate weights, resolved relative to ``path.parent``.
    """

    inputs: tuple[BoundedTensorSpec, ...]
    outputs: tuple[BoundedTensorSpec, ...]
    path: Path
    additional_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        """Validate invariants that the field types cannot express."""
        for label, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(tensor.name for tensor in tensors)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique, got {names}")

        for relative_path in self.additional_files:
            if relative_path == Path() or relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Additional file path must stay inside the artifact directory, got {relative_path}")
        if len(self.additional_files) != len(set(self.additional_files)):
            raise ValueError(f"Additional file paths must be unique, got {self.additional_files}")
        if Path(self.path.name) in self.additional_files:
            raise ValueError("An additional file cannot also be the artifact's main file")

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the complete artifact file set, with the main file first."""
        return (
            self.path,
            *(self.path.parent / relative_path for relative_path in self.additional_files),
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return input names in executable order."""
        return tuple(tensor.name for tensor in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return output names in executable order."""
        return tuple(tensor.name for tensor in self.outputs)

    @property
    def max_batch_size(self) -> int | None:
        """Return the largest batch size supported by every tensor, if known.

        Every input and output must have a known batch axis and support batch
        size one. Differing maximums are intersected by using the smallest one.
        """
        tensors = self.inputs + self.outputs
        if not tensors:
            return None

        maximums = []
        for tensor in tensors:
            if tensor.batch_axis is None or tensor.min_batch_size != 1:
                return None
            maximums.append(tensor.max_shape[tensor.batch_axis])
        return min(maximums)

    def export_files(self, path: str | os.PathLike[str]) -> Path:
        """Copy the main file and its additional files to the destination.

        Copies directly to the destination, overwriting existing files. A copy
        failure may leave an incomplete export. The caller is responsible for
        keeping sources unchanged during copying and for staging and publishing
        the complete artifact.

        Args:
            path: Destination main file. Additional files retain their relative
                paths beside it. Missing parent directories are created.

        Returns:
            The destination main file path.

        Raises:
            OSError: If directory creation or copying fails.
        """
        destination = Path(path)
        planned = [(self.path, destination)]
        planned.extend(
            (
                self.path.parent / relative_path,
                destination.parent / relative_path,
            )
            for relative_path in self.additional_files
        )
        destinations = tuple(target for _, target in planned)
        if len(destinations) != len(set(destinations)):
            raise ValueError("The exported main file would overwrite one of its additional files")

        for source, target in planned:
            if source.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return destination


@dataclass(frozen=True, kw_only=True)
class TensorRTPlanArtifact(Artifact):
    """A serialized TensorRT plan produced by tuning.

    Args:
        optimization_profile_count: Number of profiles stored in the plan.
        use_cuda_graphs: Whether the selected AITune runtime used CUDA graphs.
    """

    optimization_profile_count: int = 1
    use_cuda_graphs: bool = False

    def __post_init__(self) -> None:
        """Validate TensorRT-specific runtime metadata."""
        super().__post_init__()
        if self.optimization_profile_count < 1:
            raise ValueError("A TensorRT plan must contain at least one optimization profile")


class ONNXExecutionProvider(str, Enum):
    """ONNX Runtime execution provider selected during tuning."""

    CUDA = "cuda"
    TENSORRT = "tensorrt"


@dataclass(frozen=True, kw_only=True)
class ONNXArtifact(Artifact):
    """An ONNX model finalized as a tuning result for ONNX Runtime."""

    execution_provider: ONNXExecutionProvider = ONNXExecutionProvider.CUDA


@dataclass(frozen=True, kw_only=True)
class PT2Artifact(Artifact):
    """An AOTInductor package consumable by Triton's ``torch_aoti`` backend.

    Args:
        structured_call: Whether the package embeds structured inputs or outputs.
    """

    structured_call: bool
