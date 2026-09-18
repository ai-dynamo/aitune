# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable deployment descriptions shared by tuning frontends and runtime adapters."""

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aitune.records.shapes import BoundedTensorSpec


@dataclass(frozen=True, kw_only=True)
class ModelFiles:
    """Describe the files and format properties of one executable model.

    Args:
        format: Format identifier, such as ``onnx``, ``tensorrt_plan``, or ``pt2``.
            Custom formats do not require registration with AITune.
        path: Main model file opened by the runtime.
        additional_files: Required file paths relative to ``path.parent``.
        metadata: Format-specific properties, such as a plan's profile count.
            Producers and consumers of the format define these keys.
    """

    format: str
    path: Path
    additional_files: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the format identifier and relative file layout."""
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("ModelFiles.format must be a non-empty string")
        for relative_path in self.additional_files:
            if relative_path == Path() or relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Additional file path must stay inside the model directory, got {relative_path}")
        if len(self.additional_files) != len(set(self.additional_files)):
            raise ValueError(f"Additional file paths must be unique, got {self.additional_files}")
        if Path(self.path.name) in self.additional_files:
            raise ValueError("An additional file cannot also be the model's main file")

    @property
    def files(self) -> tuple[Path, ...]:
        """Return all source paths, with the main file first."""
        return (self.path, *(self.path.parent / relative_path for relative_path in self.additional_files))

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
            ValueError: If an export target would overwrite a model source file.
            OSError: If directory creation or copying fails.
        """
        destination = Path(path)
        planned = [(self.path, destination)]
        model_directory = self.path.parent.resolve()
        additional_sources = tuple(
            (self.path.parent / relative_path).resolve() for relative_path in self.additional_files
        )
        for source in additional_sources:
            if not source.is_relative_to(model_directory):
                raise ValueError(f"Additional file source must stay inside the model directory, got {source}")
        planned.extend(
            (
                source,
                destination.parent / relative_path,
            )
            for source, relative_path in zip(additional_sources, self.additional_files, strict=True)
        )
        resolved_sources = {source.resolve() for source, _ in planned}
        for source, target in planned:
            resolved_source = source.resolve()
            resolved_target = target.resolve()
            if resolved_target != resolved_source and resolved_target in resolved_sources:
                raise ValueError(f"Exporting to {target} would overwrite one of the model's source files")
        for source, target in planned:
            if source.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return destination


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    """Preserve runtime choices independently of a deployment platform.

    Args:
        name: Runtime identifier, such as ``onnxruntime`` or a user-defined name.
        options: Runtime settings selected during tuning. For ``onnxruntime``,
            ``execution_provider`` is ``cuda`` or ``tensorrt``. Custom runtimes
            define their own option names and values; adapters interpret them.
    """

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require a runtime identifier that adapters can dispatch on."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("RuntimeConfig.name must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class DeploymentArtifact:
    """Describe one tuned model for built-in or user-defined deployment adapters.

    This record composes model files, a tensor interface, and runtime settings.
    It neither loads the model nor defines deployment-platform configuration.

    Args:
        model: Model format, files, and format-specific metadata.
        inputs: Input tensor specifications in executable order.
        outputs: Output tensor specifications in executable order.
        runtime: Runtime identifier and options selected during tuning.
    """

    model: ModelFiles
    inputs: tuple[BoundedTensorSpec, ...]
    outputs: tuple[BoundedTensorSpec, ...]
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        """Require unique names within each side of the tensor interface."""
        for label, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(tensor.name for tensor in tensors)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique, got {names}")

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
