# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for building ``torch.export.export(dynamic_shapes=...)`` and preparing sample inputs."""

import logging
from copy import deepcopy
from typing import Any

import torch

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.recording_module import Sample

# Setup logger
logger = logging.getLogger(__name__)


def print_dynamic_shapes(dynamic_shapes: dict) -> None:
    """Print the dynamic shapes."""
    logger.debug("Extracted dynamic shapes:")
    for key, value in dynamic_shapes.items():
        logger.debug("%s:", key)
        for idx, ds in (value or {}).items():
            if isinstance(ds, torch.export.dynamic_shapes._Dim):
                logger.debug("  %s: min:%d, max:%d, %s", idx, ds.min, ds.max, str(ds).split(".")[-1])
            else:
                logger.debug("  %s: %s", idx, ds)


def make_batch_dim(tensor_specs) -> Any:
    """Create a ``torch.export.Dim`` for batch axes, or ``None`` if the batch is static.

    Uses the actual axis values from ``TensorSpec.min_shape`` / ``max_shape`` directly.
    For a CFG-doubled UNet with ``batch_sizes=[1, 2]``, ``axis_0 ∈ [2, 4]``, so the
    Dim range is ``[2, 4]`` and the axis constraint is just ``batch_dim`` with no multiplier.

    Returns ``None`` when ``min_val == max_val``: all recordings had the same batch axis
    size, so the dimension is static and should not be marked dynamic.
    """
    if not any(ts.has_batch_axis() for ts in tensor_specs):
        return None
    min_val = float("inf")
    max_val = 0
    for ts in tensor_specs:
        for axis in ts.get_batch_axis_multipliers():
            min_val = min(min_val, ts.min_shape[axis])
            max_val = max(max_val, ts.max_shape[axis])
    if min_val == max_val:
        return None
    return torch.export.Dim("batch", min=int(min_val), max=int(max_val))


def axis_dims_for_tensor(
    tensor_spec,
    batch_dim: Any,
    *,
    tensor_name: str,
    use_auto: bool = True,
    dim_cache: dict | None = None,
) -> dict[int, Any]:
    """Return the ``{axis_index: Dim}`` mapping for a single tensor spec.

    ``batch*`` axes share the common ``batch_dim`` so torch.export treats them as
    a single symbolic value across all tensors.

    ``dim*`` axes use ``Dim.AUTO`` (when ``use_auto=True``, default) so torch.export
    infers all constraints automatically:

    - **Cross-tensor equality**: when two tensors' axes must be equal at runtime
      (e.g. batch on ``sample`` and ``encoder_hidden_states`` in UNet), ``Dim.AUTO``
      detects this during tracing and merges the symbols automatically. Explicit
      per-tensor ``Dim`` objects would instead enforce independence and raise a
      ``ConstraintViolationError``.
    - **Divisibility**: when a spatial axis must be a multiple of 8 (e.g. UNet
      latent spatial dims), ``Dim.AUTO`` infers ``axis = 8 * base_dim`` automatically.
      An explicit range like ``[32, 64]`` would include invalid values and raise a
      ``ConstraintViolationError``.

    0/1 hint-specialisation (which only affects ``Dim.AUTO``) is avoided upstream
    in ``prepare_export_sample`` by expanding any size-1 dynamic axis to 2.

    With ``use_auto=False``, ``dim*`` axes get explicit ``Dim(name, min, max)``
    instances instead of ``Dim.AUTO``. Required for downstream tooling that needs
    concrete ranges (e.g. ONNXRuntime quantization shape inference). When
    ``dim_cache`` is provided, axes with identical ``(min, max)`` ranges share the
    same ``Dim`` instance — preserving cross-tensor equality without relying on
    torch.export's auto-merge.
    """
    axis_dims: dict[int, Any] = {}
    if batch_dim is not None:
        for axis in tensor_spec.get_batch_axis_multipliers():
            axis_dims[axis] = batch_dim
    for i, entry in enumerate(tensor_spec.shape):
        if isinstance(entry, str) and entry.startswith("dim"):
            if tensor_spec.min_shape[i] != tensor_spec.max_shape[i]:
                if use_auto:
                    axis_dims[i] = torch.export.Dim.AUTO
                else:
                    key = (tensor_spec.min_shape[i], tensor_spec.max_shape[i])
                    if dim_cache is not None and key in dim_cache:
                        axis_dims[i] = dim_cache[key]
                    else:
                        dim = torch.export.Dim(f"{tensor_name}_dim_{i}", min=key[0], max=key[1])
                        if dim_cache is not None:
                            dim_cache[key] = dim
                        axis_dims[i] = dim
    return axis_dims


def build_dynamic_shapes(
    sample: Sample,
    graph_spec: GraphSpec,
    *,
    use_auto: bool = True,
) -> dict:
    """Build a dynamic_shapes dict for ``torch.export.export``, keyed by parameter name.

    Creates one ``torch.export.Dim`` per symbolic dimension class, using the ranges
    accumulated by ``update_shapes_seen`` across all recorded samples:

    - **Batch axes** (``"batch*"``): a single ``batch`` Dim covering the observed batch
      range. Axes with a multiplier > 1 use a derived expression (e.g. ``2 * batch_dim``).
    - **Dynamic axes** (``"dim*"``): ``Dim.AUTO`` per axis (when ``use_auto=True``,
      default), letting torch.export infer all constraints automatically — cross-tensor
      equality (e.g. batch shared across ``sample`` and ``encoder_hidden_states`` in
      UNet) and divisibility (e.g. spatial dims that must be multiples of 8). Axes
      whose min and max are identical are treated as static.

    ``Dim.AUTO`` 0/1 hint-specialisation is avoided by ``prepare_export_sample``,
    which expands any size-1 dynamic axis to 2 before the export call.

    Args:
        sample: The normalized args and kwargs passed to ``torch.export.export``.
        graph_spec: The recorded graph spec.
        use_auto: When ``True`` (default), non-batch dynamic axes use ``Dim.AUTO``.
            When ``False``, they get explicit ``Dim(name, min, max)`` instances shared
            across tensors with identical ``(min, max)`` ranges. ``False`` is required
            for downstream tooling that needs concrete ranges (e.g. ONNXRuntime
            quantization shape inference).

    Returns:
        A dict mapping each forward parameter name to its ``{axis: Dim}`` constraints,
        covering both positional args and keyword arguments. Tensors with no dynamic
        axes get an empty inner dict. Callers that pass the result to
        ``torch.export.export`` should convert the all-empty case to ``None`` (the
        signal that no dynamic shapes are needed).
    """
    input_spec = graph_spec.input_spec
    args, kwargs = sample
    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    batch_dim = make_batch_dim(input_spec.tensor_specs)
    dim_cache: dict | None = None if use_auto else {}

    result: dict[str, Any] = {}
    for locator, tensor_spec in input_spec.tensor_data:
        if tensor_spec.has_batch_axis() or tensor_spec.has_dynamic_axis():
            dims = axis_dims_for_tensor(
                tensor_spec,
                batch_dim,
                tensor_name=locator.display_path,
                use_auto=use_auto,
                dim_cache=dim_cache,
            )
        else:
            dims = {}
        _raise_on_locator_user_type(locator)
        result = _create_nested_structure(locator, root=result)
        locator.set_value(result, dims)

    # torch.export requires every forward parameter to be present, including non-tensors.
    for key, value in forward_inputs.arguments.items():
        if key not in result:
            result[key] = {} if isinstance(value, (dict, list, tuple)) else None

    return result


def prepare_export_sample(sample: Sample, graph_spec: GraphSpec) -> Sample:
    """Return a copy of ``sample`` with all dynamic axes having hint > 1.

    ``torch.export.export`` specialises any axis whose hint is 0 or 1 as a
    compile-time constant, breaking inference at other sizes.  Two cases:

    - **batch* axes**: delegate to ``input_spec.make_batch(batch_size=min(max_batch_size, 2))``
      which scales all batch tensors consistently. When ``max_batch_size == 1``, the batch
      dimension is static (``min == max``), so no expansion is needed.  ``make_batch``
      internally deepcopies the sample, so no separate copy is needed for this path.
    - **dim* axes** (JIT mode, where batch is mislabelled): for each axis that is
      truly dynamic (``min_shape < max_shape``) but has a current hint ≤ 1, double
      the tensor along that axis via ``torch.cat``.  A hint of 2 is sufficient to
      keep the dimension symbolic.
    """
    input_spec = graph_spec.input_spec

    if input_spec.has_batch_axis():
        max_batch_size = graph_spec.get_max_batch_size()
        batch_size = min(max_batch_size, 2)
        args, kwargs = graph_spec.make_batch(*sample, batch_size=batch_size)
    else:
        args, kwargs = deepcopy(sample)

    forward_inputs = graph_spec.forward_signature.normalize(args, kwargs)
    for locator, tensor_spec in input_spec.tensor_data:
        try:
            tensor = locator.get_value(forward_inputs.arguments)
        except (IndexError, KeyError):
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        modified = False
        for i, entry in enumerate(tensor_spec.shape):
            if (
                isinstance(entry, str)
                and entry.startswith("dim")
                and tensor_spec.min_shape[i] < tensor_spec.max_shape[i]
                and tensor_spec.max_shape[i] >= 2
                and tensor.shape[i] <= 1
            ):
                tensor = torch.cat([tensor, tensor], dim=i)
                modified = True
        if modified:
            forward_inputs.arguments = locator.set_value(forward_inputs.arguments, tensor)
    return forward_inputs.args, forward_inputs.kwargs


def _create_nested_structure(locator: Locator, root: Any = None, last: Any = None) -> Any:
    """As locator does not create parent objects, we need to create them manually.

    Args:
        locator: Locator object
        root: Root object in the path
        last: Last object in the path
    Returns:
        Root object in the path
    """
    parents = [[] if T == ObjectType.SEQUENCE else {} for _, T in locator.path_iter()]
    accessors = [a for a, _ in locator.path_iter()]

    if root is not None:
        parents[0] = root

    parent = parents[0]
    for accessor, current in zip(accessors, parents[1:] + [last], strict=True):
        if isinstance(parent, list) and len(parent) <= accessor:
            parent.extend([None] * (accessor - len(parent) + 1))

        if accessor in parent:
            current = parent[accessor]

        parent[accessor] = current
        parent = current

    return parents[0]


def _raise_on_locator_user_type(locator: Locator) -> None:
    """Raise an exception if the locator is a user type."""
    if any(T in [ObjectType.USER_TYPE, ObjectType.DATACLASS] for _, T in locator.path_iter()):
        raise ValueError("Dynamic shapes does not support user types or dataclasses.")
