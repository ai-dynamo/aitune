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
"""Contains SampleMetadata which represents metadata of a sample."""

import itertools
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Union

import numpy as np
import torch

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec

PYTHON_PRIMITIVE_TYPES = (int, float, bool, bytes, str, type(None))
# The following is type of metadata which consists of python primitives, mappings and sequences
# Note: any supported type have to have __hash__ and __eq__ implemented for hashing purposes
MetadataType = Union[int, float, bool, bytes, str, type(None), "TensorSpec", Mapping, Sequence]


class SampleMetadata:
    """Metadata description of a sample of data.

    The sample can consist of sequences, mappings, python primitives and tensors.
    If a given part of sample is different the metadata will be considered different e.g.
    sample(1) == sample(1)
    sample(1) != sample(2)
    sample(1, 1) != sample(1)

    The rule is a bit different for tensors:
    - tensors of the same rank are considered equal
    - tensors of the different rank are considered different

    Same tensors:
        sample(tensor(1)) == sample(tensor(1))
        sample(tensor(1)) == sample(tensor(2))
        sample(tensor(1, 1)) == sample(tensor(1, 7)) # 2nd axis is dynamic
        sample(tensor(1, 4, 2)) == sample(tensor(2, 2, 4)) # all axes are dynamic

    Different tensors:
        sample(tensor(1)) != sample(tensor(1, 1))
        sample(tensor(1, 1)) != sample(tensor(1, 1, 1))

    This two rules allow to differentiate between samples of data in order to create different multi-graphs.

    Example usage:
        >>> args = 1, 2, 3, torch.randn(2, 2)
        >>> kwargs = {"t": torch.randn(2, 3)}
        >>> SampleMetadata.from_sample((args, kwargs), prefix="test")
        ((1, 2, 3, test__0[2, 2]), {t: test__1[2, 3]})

        >>> SampleMetadata.from_sample((args, kwargs), names=["t1", "t2"])
        ((1, 2, 3, t1[2, 2]), {t: t2[2, 3]})

        >>> s1 = SampleMetadata.from_sample(torch.randn(1, 2, 3), prefix="t")
        >>> s2 = SampleMetadata.from_sample(torch.randn(2, 2, 3), prefix="t")
        >>> s1 == s2
        True
    """

    def __init__(self, metadata: MetadataType) -> None:
        """Create SampleMetadata from provided metadata."""
        self._metadata = metadata
        self._tensor_specs = []
        self._find_tensor_specs(metadata, self._tensor_specs)

    def __str__(self) -> str:
        """Convert sample metadata to string."""
        return self.describe(InfoLevel.SHORT)

    def __repr__(self):
        """Return representation of metadata."""
        return self.describe(InfoLevel.MEDIUM)

    def __eq__(self, __value: object) -> bool:
        """Compare sample metadata."""
        if not isinstance(__value, type(self)):
            return False
        return self._metadata == __value._metadata

    def __hash__(self) -> int:
        """Compute hash of sample metadata."""
        return self._hash(self._metadata, 0)

    def get_names_mapping(self) -> tuple[Sequence[Any], dict[str, Any]]:
        """Get mapping of metadata to names. Flatten the structure to collect exact mapping used for samples.

        Returns:
            Tuple of lists of names for arguments and keyword arguments
        """
        metadata = self._metadata
        if isinstance(metadata, (PYTHON_PRIMITIVE_TYPES, Mapping)):
            metadata = (metadata,)

        if isinstance(metadata, TensorSpec):
            args, kwargs = [metadata], {}
        elif isinstance(metadata[-1], Mapping):
            args, kwargs = metadata[:-1], metadata[-1]
        else:
            args, kwargs = metadata, {}

        args_mapping, kwargs_mapping = [], {}
        for arg in args:
            flattened = {}
            self._flatten_sample(arg, arg, flattened, include_constants=False)
            args_mapping.extend(list(flattened.keys()))

        for key, arg in kwargs.items():  # pytype: disable=attribute-error
            flattened = {}
            self._flatten_sample(arg, arg, flattened, include_constants=False)
            kwargs_mapping[key] = list(flattened.keys())

        return args_mapping, kwargs_mapping

    def get_names(self) -> Sequence[str]:
        """Get names of tensors in PyTreeMetadata."""
        return list(self.flatten_sample(self._metadata).keys())

    @property
    def tensor_specs(self) -> list[TensorSpec]:
        """Get list of tensor specs."""
        return self._tensor_specs

    @staticmethod
    def from_sample(
        sample: Any, names: Iterable[str] | None = None, prefix: str = "", batch_size: int | None = None
    ) -> "SampleMetadata":
        """Create SampleMetadata from sample.

        Args:
            sample: A sample from which SampleMetadata will be created
            names: Names of tensors in the sample
            prefix: A prefix for names of tensors in the sample. Used only if names are not provided.
            batch_size: Batch size to use for tensors. If not provided, it will be taken from global context or nan if not set.

        Returns:
            SampleMetadata created from sample
        """
        if names is None:
            if prefix == "":
                raise ValueError("Prefix must be provided if names are not provided")
            names = (f"{prefix}__{i}" for i in itertools.count(start=0, step=1))
        else:
            names = iter(names)
        metadata, _ = SampleMetadata._from_sample(sample, names, batch_size)
        return SampleMetadata(metadata)

    @staticmethod
    def from_dict(data: dict) -> "SampleMetadata":
        """Create SampleMetadata from dictionary.

        Args:
            data: Dictionary containing serialized metadata

        Returns:
            A SampleMetadata instance
        """
        return SampleMetadata(SampleMetadata._from_metadata(data["metadata"]))

    def to_dict(self) -> dict:
        """Convert sample metadata to a serializable dictionary.

        Returns:
            A dictionary representation of the metadata that can be serialized.
        """
        return {"metadata": self._serialize_metadata(self._metadata)}

    def describe(self, info_level: InfoLevel = InfoLevel.FULL) -> str:
        """Get information describing metadata."""
        return self._describe(self._metadata, info_level)

    def flatten_sample(self, sample: Any) -> dict[str, Any]:
        """Flatten sample according to sample metadata.

        Nested structure of sample is flattened to one level dict with keys corresponding to sample metadata.
        """
        flattened_sample = {}
        self._flatten_sample(sample, self._metadata, flattened_sample)
        return flattened_sample

    def unflatten_sample(self, sample: dict[str, Any], wrap_input: bool = False) -> Any:
        """Unflatten sample according to sample metadata.

        Reverse process of flatten, flat structure is reversed to original nested structure according to metadata.
        If wrap_input is True, then single tensor will be wrapped in tuple.
        """
        unflatten_sample = self._unflatten_sample(sample, self._metadata)
        if wrap_input and isinstance(self._metadata, (str, Mapping)):
            unflatten_sample = (unflatten_sample,)
        return unflatten_sample

    def make_batch(self, sample: Any, batch_size: int) -> Any:
        """Returns sample with all tensors having specified batch size.

        Args:
            sample: Sample to make batch from
            metadata: Metadata to make batch from
            batch_size: Batch size

        Returns:
            Sample with all tensors having specified batch size
        """
        return self._make_batch(sample, self._metadata, batch_size)

    def update_shapes_seen(self, other: "SampleMetadata"):
        """Update shapes seen from other SampleMetadata."""
        if hash(self) != hash(other):
            raise ValueError(f"SampleMetadata to update shapes seen must be the same, {self} != {other}")
        for i, tensor_spec in enumerate(self._tensor_specs):
            tensor_spec.update_shapes_seen(other.tensor_specs[i])

    def update_max_batch_size(self, sample: Any, max_batch_size: int):
        """Update input spec with max batch size information."""
        max_batch_sample = self.make_batch(sample, max_batch_size)
        max_batch_metadata = SampleMetadata.from_sample(max_batch_sample, prefix="input", batch_size=max_batch_size)
        self.update_shapes_seen(max_batch_metadata)

    @staticmethod
    def _from_metadata(metadata: Any) -> MetadataType:
        """Helper method to recursively deserialize metadata contents.

        Args:
            metadata: serialized metadata

        Returns:
            Deserialized metadata
        """
        if isinstance(metadata, dict) and metadata.get("type") == "TensorSpec":
            return TensorSpec.from_dict(metadata)
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return metadata
        elif isinstance(metadata, dict):
            return {key: SampleMetadata._from_metadata(value) for key, value in metadata.items()}
        elif isinstance(metadata, list):
            return [SampleMetadata._from_metadata(item) for item in metadata]
        elif isinstance(metadata, tuple):
            return tuple(SampleMetadata._from_metadata(item) for item in metadata)
        else:
            raise TypeError(f"Unsupported type: {type(metadata)}")

    @staticmethod
    def _from_sample(sample, names, batch_size: int | None = None) -> tuple[MetadataType, Iterable[str]]:
        """Create SampleMetadata from sample in a recursive manner.

        Args:
            sample: Sample to create SampleMetadata from
            names: iterator of names for tensors in the sample
            batch_size: Batch size to use for tensors. If not provided, it will be taken from global context or nan if not set.

        Returns:
            Tuple of metadata and names
        """
        if torch.is_tensor(sample) or isinstance(sample, np.ndarray):
            tensor_spec = TensorSpec.from_tensor(
                next(names),
                sample,
                batch_size
                or global_context.get(
                    BATCH_SIZE_KEY, float("nan")
                ),  # TBD global context should be set always, bs_multipliers won't work
            )
            return tensor_spec, names
        if isinstance(sample, PYTHON_PRIMITIVE_TYPES):
            return sample, names
        if isinstance(sample, Mapping):
            metadata = {}
            for key, item in sorted(sample.items()):
                submetadata, names = SampleMetadata._from_sample(item, names, batch_size)
                metadata[key] = submetadata
            return metadata, names
        if isinstance(sample, Sequence):
            metadata = []
            for item in sample:
                submetadata, names = SampleMetadata._from_sample(item, names, batch_size)
                metadata.append(submetadata)
            if isinstance(sample, list):
                return metadata, names
            return tuple(metadata), names
        raise TypeError(f"Unsupported type: {type(sample)}")

    def _flatten_sample(self, sample, metadata: MetadataType, flatten_sample: dict[str, Any], include_constants=False):
        """Flatten sample according to metadata in a recursive manner.

        Args:
            sample: Sample to flatten
            metadata: Metadata to flatten sample according to
            flatten_sample: Dictionary to store flattened sample
            include_constants: Whether to include constants in the flattened sample
        """
        if isinstance(metadata, TensorSpec):
            flatten_sample[metadata.name] = sample
        elif isinstance(sample, PYTHON_PRIMITIVE_TYPES):
            if include_constants:
                flatten_sample[f"const_{uuid.uuid4()}"] = sample
        elif isinstance(sample, Mapping):
            for key, item in sample.items():
                self._flatten_sample(item, metadata[key], flatten_sample, include_constants=include_constants)
        elif isinstance(sample, Sequence):
            i = 0
            for item in sample:
                self._flatten_sample(item, metadata[i], flatten_sample, include_constants=include_constants)
                i += 1
        else:
            raise TypeError(f"Unsupported type: {type(sample)}")

    def _unflatten_sample(self, sample: Any, metadata: MetadataType) -> MetadataType:
        """Unflatten sample according to metadata in a recursive manner.

        Args:
            sample: Sample to unflatten
            metadata: Metadata to unflatten sample according to

        Returns:
            Unflattened sample
        """
        if isinstance(metadata, TensorSpec):
            return sample[metadata.name]
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return metadata
        elif isinstance(metadata, Mapping):
            return {key: self._unflatten_sample(sample, item) for key, item in metadata.items()}
        elif isinstance(metadata, Sequence):
            inner = (self._unflatten_sample(sample, item) for item in metadata)
            if isinstance(metadata, list):
                return list(inner)
            elif isinstance(metadata, tuple):
                return tuple(inner)

    def _hash(self, metadata: MetadataType, hash_):
        """Helper method to recursively hash metadata contents."""
        if isinstance(metadata, TensorSpec) or isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return hash_ ^ hash(metadata)
        elif isinstance(metadata, Mapping):
            for key, value in metadata.items():
                hash_ ^= hash(key)
                hash_ ^= self._hash(value, hash_)
            return hash_
        elif isinstance(metadata, Sequence):
            for value in metadata:
                hash_ ^= self._hash(value, hash_)
            return hash_

    def _describe(self, metadata: MetadataType, info_level: InfoLevel) -> str:
        """Get information describing metadata in a recursive manner."""
        if isinstance(metadata, TensorSpec):
            return metadata.describe(info_level)
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return str(metadata)
        elif isinstance(metadata, Mapping):
            parts = []
            for k, v in metadata.items():
                parts.append(f"{k}: {self._describe(v, info_level)}")
            return "{" + ", ".join(parts) + "}"
        elif isinstance(metadata, list):
            parts = []
            for item in metadata:
                parts.append(self._describe(item, info_level))
            return "[" + ", ".join(parts) + "]"  # type: ignore[bad-return-type]
        elif isinstance(metadata, tuple):
            if len(metadata) == 1:  # make similar to python e.g. (1,)
                return f"({self._describe(metadata[0], info_level)},)"

            parts = []
            for item in metadata:
                parts.append(self._describe(item, info_level))
            return "(" + ", ".join(parts) + ")"  # type: ignore[bad-return-type]

    def _serialize_metadata(self, metadata: MetadataType) -> Any:
        """Helper method to recursively serialize metadata contents.

        Args:
            metadata: The metadata to serialize

        Returns:
            A serializable representation of the metadata
        """
        if isinstance(metadata, TensorSpec):
            return metadata.to_dict()
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return metadata
        elif isinstance(metadata, Mapping):
            return {key: self._serialize_metadata(value) for key, value in metadata.items()}
        elif isinstance(metadata, list):
            return [self._serialize_metadata(item) for item in metadata]
        elif isinstance(metadata, tuple):
            return tuple(self._serialize_metadata(item) for item in metadata)

    def _find_tensor_specs(self, metadata: MetadataType, tensor_specs: list[TensorSpec]):
        """Find all tensor specs in metadata."""
        if isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return
        if isinstance(metadata, TensorSpec):
            tensor_specs.append(metadata)
        elif isinstance(metadata, Mapping):
            for value in metadata.values():
                self._find_tensor_specs(value, tensor_specs)
        elif isinstance(metadata, Sequence):
            for item in metadata:
                self._find_tensor_specs(item, tensor_specs)

    def _make_batch(self, sample: Any, metadata: MetadataType, batch_size: int) -> Any:
        """Traverses through metadata and makes tensors to be of specified batch size.

        Args:
            sample: Sample to make batch from
            metadata: Metadata to make batch from
            batch_size: Batch size

        Returns:
            Sample with all tensors having
        """
        if isinstance(metadata, TensorSpec):
            return batch_tensor(sample, metadata, batch_size)
        elif isinstance(metadata, PYTHON_PRIMITIVE_TYPES):
            return sample
        elif isinstance(metadata, Mapping):
            return {key: self._make_batch(item, metadata[key], batch_size) for key, item in sample.items()}
        elif isinstance(metadata, Sequence):
            return tuple(self._make_batch(item, metadata[i], batch_size) for i, item in enumerate(sample))

    def has_batch_axis(self) -> bool:
        """Check if metadata has batch axis."""
        return all(tensor_spec.has_batch_axis() for tensor_spec in self._tensor_specs)


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
