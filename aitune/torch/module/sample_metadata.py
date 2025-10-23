# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains SampleMetadata which represents metadata of a function inputs (args and kwargs) or outputs."""

import copy
import pickle
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from tabulate import tabulate

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.module.locator import Locator
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec


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
        >>> args = 1, 2, 3, torch.randn(2, 2)
        >>> kwargs = {"t": torch.randn(2, 3), "other": "abc"}
        >>> # Non-strict mode: only tensors tracked
        >>> SampleMetadata.from_inputs(args, kwargs, strict=False)
        Tensors:
        ╒═══════════╤══════════╤═════════╤═════════════╤═════════════╤═══════════════╕
        │ Locator   │ Name     │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
        ╞═══════════╪══════════╪═════════╪═════════════╪═════════════╪═══════════════╡
        │ [3]       │ args_3   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
        ├───────────┼──────────┼─────────┼─────────────┼─────────────┼───────────────┤
        │ ['t']     │ kwargs_t │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
        ╘═══════════╧══════════╧═════════╧═════════════╧═════════════╧═══════════════╛
        <BLANKLINE>
        >>> # Strict mode: all data tracked
        >>> SampleMetadata.from_inputs(args, kwargs, strict=True)
        Tensors:
        ╒═══════════╤══════════╤═════════╤═════════════╤═════════════╤═══════════════╕
        │ Locator   │ Name     │ Shape   │ Min Shape   │ Max Shape   │ Dtype         │
        ╞═══════════╪══════════╪═════════╪═════════════╪═════════════╪═══════════════╡
        │ [3]       │ args_3   │ [2, 2]  │ [2, 2]      │ [2, 2]      │ torch.float32 │
        ├───────────┼──────────┼─────────┼─────────────┼─────────────┼───────────────┤
        │ ['t']     │ kwargs_t │ [2, 3]  │ [2, 3]      │ [2, 3]      │ torch.float32 │
        ╘═══════════╧══════════╧═════════╧═════════════╧═════════════╧═══════════════╛
        Other:
        ╒═══════════╤══════════════╤═════════╕
        │ Locator   │ Name         │ Value   │
        ╞═══════════╪══════════════╪═════════╡
        │ [0]       │ args_0       │ 1       │
        ├───────────┼──────────────┼─────────┤
        │ [1]       │ args_1       │ 2       │
        ├───────────┼──────────────┼─────────┤
        │ [2]       │ args_2       │ 3       │
        ├───────────┼──────────────┼─────────┤
        │ ['other'] │ kwargs_other │ abc     │
        ╘═══════════╧══════════════╧═════════╛
        <BLANKLINE>

    Example - Tensor Rank Equality:
        >>> s1 = SampleMetadata.from_inputs((torch.randn(1, 2, 3),), kwargs={})
        >>> s2 = SampleMetadata.from_inputs((torch.randn(2, 2, 3),), kwargs={})
        >>> s1 == s2  # Same rank, different shape
        True

    Note: the __init__ method should not be used directly. instead:

        - for inputs - use SampleMetadata.from_inputs(args, kwargs)
        - for outputs - use SampleMetadata.from_outputs(output)

    See Also:
        - TensorSpec: Underlying representation of individual tensors
        - Locator: Navigation mechanism for nested structures
        - sample_metadata_walkthrough.ipynb: Comprehensive tutorial on using SampleMetadata
    """

    def __init__(
        self,
        tensor_data: tuple[tuple[Locator, TensorSpec]],
        other_data: tuple[tuple[Locator, str, str]],
        strict: bool = False,
    ) -> None:
        """Create SampleMetadata from provided tensor data and other data.

        If strict is False, other data is empty i.e. everything except tensors is ignored.
        """
        self._tensor_data = tensor_data
        self._other_data = other_data
        self._strict = strict

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
        return self._tensor_data == __value._tensor_data and self._other_data == __value._other_data

    def __hash__(self) -> int:
        """Compute hash of sample metadata."""
        return hash(self._tensor_data) ^ hash(self._other_data)

    def detected_dynamic_axis(self) -> bool:
        """Check if dynamic axes are detected in the metadata."""
        return any(ts.has_dynamic_axis() or ts.has_batch_axis() for ts in self.tensor_specs)

    def get_names(self) -> Sequence[str]:
        """Get names of tensors."""
        return [ts.name for ts in self.tensor_specs]

    def get_names_mapping(self) -> tuple[list[str], dict[str, list[Any]]]:
        """Get tensor names.

        Returns:
            Tuple of list of tensor names for args and dictionary with key kwarg name, value - list of tensor names under this kwarg
            - for args it returns list of tensor names
            - for kwargs it returns dictionary with key kwarg name, value - list of tensor names under this kwarg
        """
        args_names: list[str] = []
        kwargs_names: dict[str, list[Any]] = defaultdict(list)  # make pytype happy
        for locator, tensor_spec in self._tensor_data:
            if tensor_spec.name.startswith("args"):
                args_names.append(tensor_spec.name)
            else:
                kwarg_name = locator._path[0][0]  # TODO:: better api for this?
                kwargs_names[kwarg_name].append(tensor_spec.name)

        return args_names, kwargs_names

    @property
    def other_data(self) -> tuple[tuple[Locator, str, str]]:
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
    def from_inputs(
        args: Any, kwargs: dict[str, Any], strict: bool = False, batch_size: int | None = None
    ) -> "SampleMetadata":
        """Create SampleMetadata from inputs: args and kwargs.

        If strict is True, then other data is also included.
        """
        batch_size = batch_size or global_context.get(BATCH_SIZE_KEY, float("nan"))
        tensor_data, other_data = [], []
        for prefix, data_source in zip(("args", "kwargs"), (args, kwargs), strict=False):
            for locator, value in Locator.find_leaves(data_source):
                name = prefix + sanitize_tensor_name(str(locator))
                if torch.is_tensor(value):
                    tensor_data.append((locator, TensorSpec.from_tensor(name, value, batch_size)))
                elif strict:
                    other_data.append((locator, name, str(value)))

        return SampleMetadata(tuple(tensor_data), tuple(other_data), strict)

    @staticmethod
    def from_outputs(output: Any, strict: bool = False, batch_size: int | None = None) -> "SampleMetadata":
        """Create SampleMetadata from outputs.

        If strict is True, then other data is also included.
        """
        batch_size = batch_size or global_context.get(BATCH_SIZE_KEY, float("nan"))
        tensor_data, other_data = [], []
        for locator, value in Locator.find_leaves(output):
            name = "outputs" + sanitize_tensor_name(str(locator))
            if torch.is_tensor(value):
                tensor_data.append((locator, TensorSpec.from_tensor(name, value, batch_size)))
            elif strict:
                other_data.append((locator, name, str(value)))
        return SampleMetadata(tuple(tensor_data), tuple(other_data), strict)

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
        }

    def describe(self, info_level: InfoLevel = InfoLevel.FULL) -> str:
        """Get information describing sample metadata."""
        tensors, tensor_header = [], []
        for locator, ts in self._tensor_data:
            if info_level == InfoLevel.SHORT:
                tensors.append(ts.name)
            elif info_level == InfoLevel.MEDIUM:
                tensor_header = ["Locator", "Name", "Shape"]
                tensors.append((str(locator), ts.name, ts.shape))
            elif info_level == InfoLevel.FULL:
                tensor_header = ["Locator", "Name", "Shape", "Min Shape", "Max Shape", "Dtype"]
                tensors.append((str(locator), ts.name, ts.shape, ts.min_shape, ts.max_shape, ts.dtype))

        others, other_header = [], []
        for locator, name, value in self._other_data:
            if info_level == InfoLevel.SHORT:
                others.append(f"{name}={value}")
            else:
                other_header = ["Locator", "Name", "Value"]
                others.append((str(locator), name, value))

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

    def make_batch(self, args: Any, kwargs: dict[str, Any], batch_size: int) -> tuple[Any, dict[str, Any]]:
        """Takes args and kwargs and extrapolates all tensors according to tensor specs to have specified batch size.

        Args:
            args: Args to make batch from
            kwargs: Kwargs to make batch from
            batch_size: Batch size

        Returns:
            args, kwargs with all tensors having specified batch size
        """
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        for locator, tensor_spec in self._tensor_data:
            if tensor_spec.name.startswith("args"):
                args = locator.set_value(args, batch_tensor(locator.get_value(args), tensor_spec, batch_size))
            else:
                kwargs = locator.set_value(kwargs, batch_tensor(locator.get_value(kwargs), tensor_spec, batch_size))

        return args, kwargs

    def update_shapes_seen(self, other: "SampleMetadata"):
        """Update shapes seen from other SampleMetadata."""
        if hash(self._tensor_data) != hash(other._tensor_data):
            raise ValueError(
                f"Cannot update shapes seen, because tensor data is different, {self.describe(InfoLevel.FULL)} != {other.describe(InfoLevel.FULL)}"
            )
        for i, tensor_spec in enumerate(self.tensor_specs):
            tensor_spec.update_shapes_seen(other.tensor_specs[i])

    def update_max_batch_size(self, sample: Any, max_batch_size: int):
        """Update input spec with max batch size information."""
        args, kwargs = sample
        args, kwargs = self.make_batch(args, kwargs, max_batch_size)
        max_batch_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=max_batch_size, strict=True)
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


def sanitize_tensor_name(name: str) -> str:
    """Sanitize tensor name to be used as a tensor name."""
    return name.translate(str.maketrans("[", "_", "]'"))
