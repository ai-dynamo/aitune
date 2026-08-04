# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contains SampleMetadata which represents metadata of a function inputs (args and kwargs) or outputs."""

import copy
import pickle
from collections.abc import Mapping
from typing import Any, Literal

import torch
from tabulate import tabulate

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.config import config
from aitune.torch.module.locator import Locator
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec
from aitune.utils.serialization import json_serialize


def _hash_value(value: Any) -> int:
    """Hash nested values, falling back safely for unsupported unhashable objects."""
    if isinstance(value, Mapping):
        return hash(frozenset((_hash_value(key), _hash_value(item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return hash(frozenset(_hash_value(item) for item in value))
    if isinstance(value, (list, tuple)):
        return hash(tuple(_hash_value(item) for item in value))

    try:
        return hash(value)
    except TypeError:
        # Equal metadata must always have equal hashes. A constant fallback preserves that
        # contract for arbitrary unhashable leaf objects; __eq__ resolves collisions.
        return 0


class SampleMetadata:
    """Metadata description of inputs (args and kwargs) or outputs of a function.

    SampleMetadata captures and tracks metadata about function inputs (args and kwargs) and outputs. It serves
    several important purposes:

    - Tensor Tracking: Automatically discovers and tracks all tensors in nested structures
      (tuples, lists, dicts, dataclasses, custom objects).
    - Shape Inference: Learns about dynamic dimensions and batch axes by observing multiple samples
      with different shapes via `update_shapes_seen()`.
    - Dynamic Batching: Supports scaling tensors to different batch sizes based on learned patterns
      using `make_batch()`.

    Operating Modes:
        The metadata can be created in two modes based on the `strict` flag:

        - Non-strict mode (`strict=False`, default): Only tensors are tracked. Primitives and
          other non-tensor data are ignored.
        - Strict mode (`strict=True`): All data types are tracked, including primitives.

    Equality Rules:
        For non-tensor data, metadata equality is strict - two samples are equal only if all parts
        are equal:
            sample(1) == sample(1)
            sample(1) != sample(2)
            sample(1, 1) != sample(1)

        For PyTorch tensors, equality is determined by rank (number of dimensions):
            # Same rank (equal metadata)
            sample(tensor(1)) == sample(tensor(1))
            sample(tensor(1)) == sample(tensor(2))
            sample(tensor(1, 1)) == sample(tensor(1, 7))
            sample(tensor(1, 4, 2)) == sample(tensor(2, 2, 4))

            # Different rank (different metadata)
            sample(tensor(1)) != sample(tensor(1, 1))
            sample(tensor(1, 1)) != sample(tensor(1, 1, 1))

        These rules ensure that samples differing in structure or tensor rank are treated as having
        different metadata, supporting the creation of distinct multi-graph representations.

    Key Concepts:
        - Dynamic Dimensions: Dimensions that vary across samples (e.g., sequence length in NLP).
          Marked as 'dim0', 'dim1', etc.
        - Batch Axes: Dimensions that scale proportionally with batch size. Marked as 'batch0',
          'batch1', etc. with associated multipliers.

    Example - Basic Usage:
        >>> inputs = {
        ...     "t": torch.randn(2, 3),
        ...     "values": (1, 2, 3, torch.randn(2, 2)),
        ...     "other": "abc",
        ... }
        >>> # Non-strict mode: only tensors tracked
        >>> SampleMetadata.from_inputs(inputs, strict=False)
        Tensors:
        ╒═══════════════╤═════════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
        │ Access Path   │ Semantic Path   │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
        ╞═══════════════╪═════════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
        │ t             │ t               │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
        ├───────────────┼─────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
        │ values[3]     │ ('values', 3)   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
        ╘═══════════════╧═════════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
        <BLANKLINE>
        >>> # Strict mode: all data tracked
        >>> SampleMetadata.from_inputs(inputs, strict=True)
        Tensors:
        ╒═══════════════╤═════════════════╤═════════╤═════════════╤═════════════╤═══════════════╕
        │ Access Path   │ Semantic Path   │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
        ╞═══════════════╪═════════════════╪═════════╪═════════════╪═════════════╪═══════════════╡
        │ t             │ t               │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
        ├───────────────┼─────────────────┼─────────┼─────────────┼─────────────┼───────────────┤
        │ values[3]     │ ('values', 3)   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
        ╘═══════════════╧═════════════════╧═════════╧═════════════╧═════════════╧═══════════════╛
        Other:
        ╒═══════════════╤═════════════════╤═════════╕
        │ Access Path   │ Semantic Path   │ Value   │
        ╞═══════════════╪═════════════════╪═════════╡
        │ values[0]     │ ('values', 0)   │ 1       │
        ├───────────────┼─────────────────┼─────────┤
        │ values[1]     │ ('values', 1)   │ 2       │
        ├───────────────┼─────────────────┼─────────┤
        │ values[2]     │ ('values', 2)   │ 3       │
        ├───────────────┼─────────────────┼─────────┤
        │ other         │ other           │ abc     │
        ╘═══════════════╧═════════════════╧═════════╛
        <BLANKLINE>

    Example - Tensor Rank Equality:
        >>> s1 = SampleMetadata.from_inputs({"x": torch.randn(1, 2, 3)})
        >>> s2 = SampleMetadata.from_inputs({"x": torch.randn(2, 2, 3)})
        >>> s1 == s2  # Same rank, different shape
        True

    Note:
    1. The __init__ method should not be used directly. instead:
        - for inputs - use SampleMetadata.from_inputs(bound_arguments)
        - for outputs - use SampleMetadata.from_outputs(output)

    2. In order to support Hugging Face integrations with kv cache the following behavior is changed:
        - the equality and hash operator ignore all tensors and return value corresponding to the llm_graph_type,
        which is either "prefill" or "decode". This is deduced based on the cache_position tensor. It cannot be
        done otherwise because even though kv cache is lazily initialized, it can be cached on subsequent generate
        calls i.e. it can always have tensors irrespective of the phase.


    See Also:
        - TensorSpec: Underlying representation of individual tensors
        - Locator: Navigation mechanism for nested structures
        - sample_metadata_walkthrough.ipynb: Comprehensive tutorial on using SampleMetadata
    """

    def __init__(
        self,
        tensor_data: tuple[tuple[Locator, TensorSpec], ...],
        other_data: tuple[tuple[Locator, Any], ...],
        strict: bool = False,
        llm_phase: Literal["prefill", "decode"] | None = None,
    ) -> None:
        """Create SampleMetadata from provided tensor data and other data.

        If strict is False, other data is empty i.e. everything except tensors is ignored.
        """
        self._tensor_data = tensor_data
        self._other_data = other_data
        self._strict = strict
        self._llm_phase = llm_phase

    def __str__(self) -> str:
        """Convert sample metadata to string."""
        return self.describe(InfoLevel.SHORT)

    def __repr__(self):
        """Return representation of metadata."""
        return self.describe(InfoLevel.FULL)

    def __eq__(self, __value: object) -> bool:
        """Equality operator."""
        if not isinstance(__value, type(self)):
            return False

        if self._llm_phase is None:
            return dict(self._tensor_data) == dict(__value._tensor_data) and dict(self._other_data) == dict(
                __value._other_data
            )
        else:
            # LLM integration: check only LLM phase
            return self._llm_phase == __value._llm_phase

    def __hash__(self) -> int:
        """Compute hash of sample metadata."""
        if self._llm_phase is None:
            return hash(frozenset(self._tensor_data)) ^ self._hash_other_data()
        else:
            # LLM integration: hash only LLM phase
            return hash(self._llm_phase)

    def _hash_other_data(self) -> int:
        """Compute an order-independent hash for non-tensor data."""
        return hash(frozenset((locator, _hash_value(value)) for locator, value in self._other_data))

    def detected_dynamic_axis(self) -> bool:
        """Check if dynamic axes are detected in the metadata."""
        return any(ts.has_dynamic_axis() or ts.has_batch_axis() for ts in self.tensor_specs)

    @property
    def llm_phase(self) -> Literal["prefill", "decode", ""]:
        """Get LLM graph type."""
        if self._llm_phase is None:
            return ""
        return self._llm_phase

    @property
    def other_data(self) -> tuple[tuple[Locator, Any], ...]:
        """Get list of other data."""
        return self._other_data

    @property
    def tensor_data(self) -> tuple[tuple[Locator, TensorSpec]]:
        """Get list of tensor data."""
        return self._tensor_data

    @property
    def tensor_specs(self) -> list[TensorSpec]:
        """Get list of tensor specs."""
        return [ts for _, ts in self._tensor_data]

    @staticmethod
    def from_inputs(inputs: dict[str, Any], strict: bool = False, batch_size: int | None = None) -> "SampleMetadata":
        """Create SampleMetadata from inputs keyed by forward parameter name.

        If strict is True, then other data is also included.
        """
        batch_size = batch_size or global_context.get(BATCH_SIZE_KEY, float("nan"))
        tensor_data, other_data, cache_position = [], [], None
        for locator, value in Locator.find_leaves(inputs):
            if torch.is_tensor(value):
                tensor_data.append((locator, TensorSpec.from_tensor(value, batch_size)))
            elif strict:
                other_data.append((locator, value))
            if config.enable_transformers_integration and locator.root_name == "cache_position":
                cache_position = value

        if cache_position is None:
            llm_phase = None
        else:
            llm_phase = "prefill" if cache_position[0].item() == 0 else "decode"

        return SampleMetadata(tuple(tensor_data), tuple(other_data), strict, llm_phase)

    @staticmethod
    def from_outputs(output: Any, strict: bool = False, batch_size: int | None = None) -> "SampleMetadata":
        """Create SampleMetadata from outputs.

        If strict is True, then other data is also included.
        """
        batch_size = batch_size or global_context.get(BATCH_SIZE_KEY, float("nan"))
        tensor_data, other_data = [], []
        for locator, value in Locator.find_leaves(output, is_output=True):
            if torch.is_tensor(value):
                tensor_data.append((locator, TensorSpec.from_tensor(value, batch_size)))
            elif strict:
                other_data.append((locator, value))
        return SampleMetadata(tuple(tensor_data), tuple(other_data), strict, llm_phase=None)

    @staticmethod
    def from_dict(data: dict) -> "SampleMetadata":
        """Create SampleMetadata from dictionary.

        Args:
            data: Dictionary containing serialized metadata

        Returns:
            A SampleMetadata instance
        """
        return SampleMetadata(
            pickle.loads(data["tensor_data"]),
            pickle.loads(data["other_data"]),
            data["strict"],
            data["llm_phase"],
        )

    def to_dict(self) -> dict:
        """Convert sample metadata to a serializable dictionary.

        Returns:
            A dictionary representation of the metadata that can be serialized.
        """
        return {
            "tensor_data": pickle.dumps(self._tensor_data),
            "other_data": pickle.dumps(self._other_data),
            "strict": self._strict,
            "llm_phase": self._llm_phase,
        }

    def to_json_dict(self) -> dict:
        """Convert sample metadata to a JSON-serializable dictionary.

        Returns:
            A dictionary representation safe for ``json.dumps``.
        """
        tensor_data = []
        for locator, ts in self._tensor_data:
            tensor_data.append({
                "access_path": locator.display_path,
                "semantic_path": locator.path,
                "shape": ts.shape,
                "min_shape": ts.min_shape,
                "max_shape": ts.max_shape,
                "dtype": ts.dtype,
            })
        other_data = [
            {
                "access_path": locator.display_path,
                "semantic_path": locator.path,
                "value": value,
            }
            for locator, value in self._other_data
        ]
        return json_serialize({
            "tensor_data": tensor_data,
            "other_data": other_data,
            "strict": self._strict,
            "llm_phase": self._llm_phase,
        })

    def describe(self, info_level: InfoLevel = InfoLevel.FULL) -> str:
        """Get information describing sample metadata."""
        tensors, tensor_header = [], []
        for locator, ts in self._tensor_data:
            access_path = locator.display_path
            if info_level == InfoLevel.SHORT:
                tensors.append(access_path)
            elif info_level == InfoLevel.MEDIUM:
                tensor_header = ["Access Path", "Semantic Path", "Shape"]
                tensors.append((access_path, str(locator.path), ts.shape))
            elif info_level == InfoLevel.FULL:
                tensor_header = ["Access Path", "Semantic Path", "Shape", "Min Shape", "Max Shape", "Dtype"]
                tensors.append((access_path, str(locator.path), ts.shape, ts.min_shape, ts.max_shape, ts.dtype))

        others, other_header = [], []
        for locator, value in self._other_data:
            access_path = locator.display_path
            if info_level == InfoLevel.SHORT:
                others.append(f"{access_path}={value}")
            else:
                other_header = ["Access Path", "Semantic Path", "Value"]
                others.append((access_path, str(locator.path), str(value)))

        if info_level == InfoLevel.SHORT:
            result = "Tensors: " + ", ".join(tensors)
            if others:
                result += " Others: " + ", ".join(others)
        else:
            tbl_fmt = "simple" if info_level == InfoLevel.MEDIUM else "fancy_grid"
            result = "Tensors:\n" + tabulate(tensors, headers=tensor_header, tablefmt=tbl_fmt) + "\n"
            if others:
                result += "Other:\n" + tabulate(others, headers=other_header, tablefmt=tbl_fmt) + "\n"

        return result

    def make_batch(self, inputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
        """Extrapolate all tensors in forward inputs to the specified batch size.

        Args:
            inputs: Inputs keyed by forward parameter name
            batch_size: Batch size

        Returns:
            Bound inputs with all tensors having the specified batch size
        """
        inputs = copy.deepcopy(inputs)
        for locator, tensor_spec in self._tensor_data:
            inputs = locator.set_value(inputs, batch_tensor(locator.get_value(inputs), tensor_spec, batch_size))

        return inputs

    def update_shapes_seen(self, other: "SampleMetadata"):
        """Update shapes seen from other SampleMetadata."""
        tensor_data = dict(self._tensor_data)
        other_tensor_data = dict(other._tensor_data)
        if tensor_data != other_tensor_data:
            raise ValueError(
                f"Cannot update shapes seen, because tensor data is different, {self.describe(InfoLevel.FULL)} != {other.describe(InfoLevel.FULL)}"
            )
        for locator, tensor_spec in tensor_data.items():
            tensor_spec.update_shapes_seen(other_tensor_data[locator])

    def update_max_batch_size(self, inputs: dict[str, Any], max_batch_size: int):
        """Update input spec with max batch size information."""
        inputs = self.make_batch(inputs, max_batch_size)
        max_batch_metadata = SampleMetadata.from_inputs(inputs, batch_size=max_batch_size, strict=True)
        self.update_shapes_seen(max_batch_metadata)

    def has_batch_axis(self) -> bool:
        """Check if metadata has batch axis."""
        return all(tensor_spec.has_batch_axis() for tensor_spec in self.tensor_specs)


def batch_tensor(tensor: torch.Tensor, tensor_spec: TensorSpec, batch_size: int) -> torch.Tensor:
    """Batch tensor so that instead of its own batch size it uses specified batch size.

    Args:
        tensor: Tensor to batch
        tensor_spec: Metadata to batch tensor according to
        batch_size: Batch size

    Returns:
        Tensor of specified batch size
    """
    if not tensor_spec.get_batch_axis_multipliers():
        return tensor  # no batch axes, return tensors unchanged

    for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items():
        effective_batch_size = multiplier * batch_size
        if tensor.shape[axis] > effective_batch_size:
            # Slice the tensor along this axis to match the effective batch size e.g. [:, :bs, :]
            slice_indices = [slice(None)] * len(tensor.shape)
            slice_indices[axis] = slice(effective_batch_size)
            tensor = tensor[tuple(slice_indices)]
        elif tensor.shape[axis] < effective_batch_size:
            # Slice the tensor along this axis to have only 1 element e.g. [:, :1, :]
            slice_indices = [slice(None)] * len(tensor.shape)
            slice_indices[axis] = slice(1)
            tensor = tensor[tuple(slice_indices)]
            # Repeat the tensor along this axis to match the effective batch size
            repeat_indices = [1] * len(tensor.shape)
            repeat_indices[axis] = effective_batch_size
            tensor = tensor.repeat(repeat_indices)
    return tensor
