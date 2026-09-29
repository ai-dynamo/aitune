# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable deployment descriptions shared by tuning frontends and runtime adapters."""

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
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
            ValueError: If export targets collide with each other or with model source files.
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
        resolved_plan = tuple((source.resolve(), target.resolve()) for source, target in planned)
        resolved_sources = {source for source, _ in resolved_plan}
        resolved_targets = tuple(target for _, target in resolved_plan)
        if len(resolved_targets) != len(set(resolved_targets)):
            raise ValueError("Multiple model files would be exported to the same target")
        for resolved_source, resolved_target in resolved_plan:
            if resolved_target != resolved_source and resolved_target in resolved_sources:
                raise ValueError(f"Exporting to {resolved_target} would overwrite one of the model's source files")
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
class TensorSample:
    """Portable values for one representative input tensor."""

    name: str
    shape: tuple[int, ...]
    values: tuple[bool | int | float, ...]

    def __post_init__(self) -> None:
        """Require the flattened values to match the recorded shape."""
        if not self.name:
            raise ValueError("TensorSample.name must be a non-empty string")
        if any(dimension < 0 for dimension in self.shape) or prod(self.shape) != len(self.values):
            raise ValueError(f"TensorSample values do not match shape {self.shape}")

    def to_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe primitive values."""
        return {"name": self.name, "shape": self.shape, "values": self.values}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSample":
        """Restore representative tensor values from checkpoint state."""
        return cls(name=data["name"], shape=tuple(data["shape"]), values=tuple(data["values"]))


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
        sample_inputs: Representative input values in executable order, when retained.
    """

    model: ModelFiles
    inputs: tuple[BoundedTensorSpec, ...]
    outputs: tuple[BoundedTensorSpec, ...]
    runtime: RuntimeConfig
    sample_inputs: tuple[TensorSample, ...] = ()

    def __post_init__(self) -> None:
        """Require unique names within each side of the tensor interface."""
        for label, tensors in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(tensor.name for tensor in tensors)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} tensor names must be unique, got {names}")
        if self.sample_inputs and tuple(sample.name for sample in self.sample_inputs) != self.input_names:
            raise ValueError("Representative sample names must match artifact inputs in executable order")

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
