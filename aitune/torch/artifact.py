# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate Torch graph metadata into deployment-neutral artifact records."""

from collections.abc import Sequence
from typing import Literal

import torch

from aitune.records import BoundedTensorSpec, DType
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator
from aitune.torch.module.tensor_spec import TensorSpec

_TORCH_DTYPE_TO_RECORD = {
    torch.bool: DType.BOOL,
    torch.uint8: DType.UINT8,
    torch.int8: DType.INT8,
    torch.int16: DType.INT16,
    torch.int32: DType.INT32,
    torch.int64: DType.INT64,
    torch.float16: DType.FLOAT16,
    torch.float32: DType.FLOAT32,
    torch.float64: DType.FLOAT64,
}

TensorKind = Literal["input", "output"]


def _batch_axis(graph_spec: GraphSpec, locator: Locator, tensor_spec: TensorSpec, kind: TensorKind) -> int | None:
    """Return the single directly representable logical batch axis, if known."""
    definition = graph_spec.get_shape_definition(locator) if kind == "input" else None
    if definition is not None:
        axes = [axis for axis, dimension in enumerate(definition) if isinstance(dimension, BatchDim)]
    else:
        axes = [axis for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items() if multiplier == 1]
    return axes[0] if len(axes) == 1 else None


def bounded_tensor_specs(
    graph_spec: GraphSpec,
    kind: TensorKind,
    *,
    recorded_names: Sequence[str] | None = None,
    metadata_indices: Sequence[int] | None = None,
    artifact_names: Sequence[str] | None = None,
) -> tuple[BoundedTensorSpec, ...]:
    """Create ordered bounded tensor specs from a recorded Torch graph.

    Backends whose executable retains exporter names select tensors with
    ``recorded_names``. Backends with a positional ABI can instead select metadata by
    index and provide their finalized ``artifact_names``. Selection methods are
    mutually exclusive so an adapter cannot accidentally combine two orderings.
    """
    if recorded_names is not None and metadata_indices is not None:
        raise ValueError("Select artifact tensors by recorded name or metadata index, not both")

    metadata = graph_spec.input_spec if kind == "input" else graph_spec.output_spec
    tensor_data = metadata.tensor_data
    if recorded_names is not None:
        indices_by_name = {
            graph_spec.tensor_name(locator, tensor_spec, kind): index
            for index, (locator, tensor_spec) in enumerate(tensor_data)
        }
        try:
            selected_indices = tuple(indices_by_name[name] for name in recorded_names)
        except KeyError as error:
            raise ValueError(f"Artifact {kind} {error.args[0]!r} is missing from the recorded graph") from error
        default_names = tuple(recorded_names)
    else:
        selected_indices = tuple(range(len(tensor_data))) if metadata_indices is None else tuple(metadata_indices)
        try:
            default_names = tuple(graph_spec.tensor_name(*tensor_data[index], kind) for index in selected_indices)
        except IndexError as error:
            raise ValueError(f"Artifact {kind} metadata index is outside the recorded graph") from error

    names = default_names if artifact_names is None else tuple(artifact_names)
    if len(names) != len(selected_indices):
        raise ValueError(f"Artifact {kind} names and selected tensors must have the same length")

    result = []
    for name, index in zip(names, selected_indices, strict=True):
        locator, tensor_spec = tensor_data[index]
        try:
            dtype = _TORCH_DTYPE_TO_RECORD[tensor_spec.dtype]
        except KeyError as error:
            raise ValueError(f"Artifact {kind} {name!r} uses unsupported torch dtype {tensor_spec.dtype}") from error

        if kind == "input":
            min_shape, _, max_shape = graph_spec.get_effective_input_shapes(locator, tensor_spec)
        else:
            min_shape, max_shape = graph_spec.get_effective_output_shapes(tensor_spec)
        result.append(
            BoundedTensorSpec(
                name=name,
                dtype=dtype,
                min_shape=tuple(min_shape),
                max_shape=tuple(max_shape),
                batch_axis=_batch_axis(graph_spec, locator, tensor_spec, kind),
            )
        )
    return tuple(result)


__all__ = ["bounded_tensor_specs"]
