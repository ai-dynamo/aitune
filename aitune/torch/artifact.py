# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate Torch graph metadata into deployment-neutral artifact records."""

from collections.abc import Sequence
from typing import Literal, cast

import torch

from aitune.records import BoundedTensorSpec, DType, TensorSample
from aitune.torch.dynamic_shapes import BatchDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_store import Sample
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


def artifact_input_samples(
    graph_spec: GraphSpec,
    samples: Sequence[Sample],
    *,
    recorded_names: Sequence[str] | None = None,
    metadata_indices: Sequence[int] | None = None,
    artifact_names: Sequence[str] | None = None,
) -> tuple[tuple[TensorSample, ...], ...]:
    """Capture recorded calls as portable deployment input requests.

    Each request contains all input tensors in executable order. Only the first
    request for each combination of input shapes is retained, since later calls
    with the same shapes do not add profiling coverage.

    Args:
        graph_spec: Recorded graph metadata containing tensor names and locators.
        samples: Original module calls represented as ``(args, kwargs)`` pairs.
        recorded_names: Recorded graph input names in executable order.
        metadata_indices: Graph input positions in executable order.
        artifact_names: Final executable names for the selected inputs.

    Returns:
        Portable requests, each containing input samples in executable order.
    """
    requests = []
    seen_shapes = set()
    for sample in samples:
        request = _artifact_input_sample(
            graph_spec,
            sample,
            recorded_names=recorded_names,
            metadata_indices=metadata_indices,
            artifact_names=artifact_names,
        )
        shapes = tuple(tensor.shape for tensor in request)
        if shapes not in seen_shapes:
            requests.append(request)
            seen_shapes.add(shapes)
    return tuple(requests)


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
            default_names = tuple(
                graph_spec.tensor_name(tensor_data[index][0], tensor_data[index][1], kind) for index in selected_indices
            )
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


def _batch_axis(graph_spec: GraphSpec, locator: Locator, tensor_spec: TensorSpec, kind: TensorKind) -> int | None:
    """Return the single directly representable logical batch axis, if known."""
    definition = graph_spec.get_shape_definition(locator) if kind == "input" else None
    if definition is not None:
        axes = [axis for axis, dimension in enumerate(definition) if isinstance(dimension, BatchDim)]
    else:
        axes = [axis for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items() if multiplier == 1]
    return axes[0] if len(axes) == 1 else None


def _artifact_input_sample(
    graph_spec: GraphSpec,
    sample: Sample,
    *,
    recorded_names: Sequence[str] | None = None,
    metadata_indices: Sequence[int] | None = None,
    artifact_names: Sequence[str] | None = None,
) -> tuple[TensorSample, ...]:
    """Capture one recorded Torch call as portable deployment input values.

    A recorded sample follows the module's Python call signature, while an exported
    executable can expose a different input order or different names. This function
    uses ``GraphSpec`` locators to read the original tensors, reorders them to match
    the executable, and stores their shape and flattened values in ``TensorSample``
    records. The result contains no Torch objects, so it can be kept in a deployment
    artifact and later converted to input formats such as Perf Analyzer JSON.

    By default, the result uses the input order and original tensor names stored in
    ``GraphSpec``. A backend whose executable selects or reorders named inputs can
    provide ``recorded_names``. Positional formats such as PT2 instead provide
    ``metadata_indices`` and use ``artifact_names`` for their final public names.
    The two explicit selection methods are mutually exclusive.

    Args:
        graph_spec: Recorded graph metadata containing tensor names and locators.
        sample: Original module call represented as ``(args, kwargs)``.
        recorded_names: Recorded graph input names in executable order.
        metadata_indices: Graph input positions in executable order.
        artifact_names: Final executable names for the selected inputs. Recorded
            names are used when this is omitted.

    Returns:
        Portable input samples in executable order.

    Raises:
        ValueError: If selection is ambiguous, an input cannot be found, names and
            selected inputs have different lengths, or a selected value is not a
            tensor.
    """
    if recorded_names is not None and metadata_indices is not None:
        raise ValueError("Select artifact tensors by recorded name or metadata index, not both")

    tensor_data = graph_spec.input_spec.tensor_data
    if recorded_names is not None:
        # Graph metadata is stored in discovery order. Resolve names back to metadata
        # positions so the result follows the executable's requested order.
        indices_by_name = {
            graph_spec.tensor_name(locator, tensor_spec, "input"): index
            for index, (locator, tensor_spec) in enumerate(tensor_data)
        }
        try:
            selected_indices = tuple(indices_by_name[name] for name in recorded_names)
        except KeyError as error:
            raise ValueError(f"Artifact input {error.args[0]!r} is missing from the recorded graph") from error
        default_names = tuple(recorded_names)
    else:
        selected_indices = tuple(range(len(tensor_data))) if metadata_indices is None else tuple(metadata_indices)
        try:
            default_names = tuple(
                graph_spec.tensor_name(tensor_data[index][0], tensor_data[index][1], "input")
                for index in selected_indices
            )
        except IndexError as error:
            raise ValueError("Artifact input metadata index is outside the recorded graph") from error

    names = default_names if artifact_names is None else tuple(artifact_names)
    if len(names) != len(selected_indices):
        raise ValueError("Artifact input names and selected tensors must have the same length")

    args, kwargs = sample
    # Bind positional and keyword arguments to one mapping because GraphSpec locators
    # describe paths from normalized forward parameters, not directly from args/kwargs.
    normalized = graph_spec.forward_signature.normalize(args, kwargs)
    result = []
    for name, index in zip(names, selected_indices, strict=True):
        locator, _ = tensor_data[index]
        tensor = locator.get_value(normalized.arguments)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Recorded artifact input {name!r} is not a tensor")
        # Deployment records must not retain a device allocation or a Torch object.
        # Flattening keeps the original shape explicit while making values portable.
        values = cast(list[bool | int | float], tensor.detach().cpu().reshape(-1).tolist())
        result.append(TensorSample(name=name, shape=tuple(tensor.shape), values=tuple(values)))
    return tuple(result)
