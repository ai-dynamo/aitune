# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Python-side input changes that tensor backends must reproduce."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_metadata import SampleMetadata


class InputMutationKind(str, Enum):
    """How a recorded input leaf changed during forward."""

    COPY_TENSOR = "copy_tensor"
    REPLACE_TENSOR = "replace_tensor"
    REPLACE_CONSTANT = "replace_constant"


@dataclass(frozen=True)
class InputMutation:
    """One observable input change made by a module call."""

    locator: Locator
    kind: InputMutationKind
    output_index: int | None = None
    value: Any = None


@dataclass(frozen=True)
class ForwardContract:
    """Recorded output structure and input changes for one graph."""

    output_template: Any
    input_mutations: tuple[InputMutation, ...] = ()


@dataclass(frozen=True)
class _TensorMetadata:
    """Tensor properties that cannot be reproduced by copying values."""

    shape: tuple[int, ...]
    stride: tuple[int, ...] | None
    storage_offset: int | None
    layout: torch.layout
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _InputLeafSnapshot:
    """Pre-call identity and value information for one input leaf."""

    value: Any
    tensor_version: int | None
    tensor_metadata: _TensorMetadata | None = None
    tensor_value: torch.Tensor | None = None


@dataclass(frozen=True)
class ForwardContractCapture:
    """Compare normalized inputs before and after one module call."""

    _before: dict[Locator, _InputLeafSnapshot]

    @classmethod
    def before_call(
        cls,
        arguments: dict[str, Any],
        snapshot_arguments: dict[str, Any] | None = None,
    ) -> ForwardContractCapture:
        """Capture input leaves immediately before forward executes."""
        return cls({
            locator: _snapshot_leaf(locator, value, snapshot_arguments)
            for locator, value in Locator.find_leaves(arguments)
        })

    def after_call(
        self,
        arguments: dict[str, Any],
        output: Any,
        output_spec: SampleMetadata,
    ) -> ForwardContract:
        """Describe supported input leaf changes observed after forward."""
        output_tensors = output_spec.flatten_tensors(output)
        output_indices_by_identity = {id(tensor): index for index, tensor in enumerate(output_tensors)}
        next_output_index = len(output_tensors)
        mutations = []

        for locator, value in Locator.find_leaves(arguments):
            kind = _mutation_kind(self._before.get(locator), value)
            if kind is None:
                continue
            if kind == InputMutationKind.REPLACE_CONSTANT:
                mutations.append(InputMutation(locator, kind, value=deepcopy(value)))
                continue

            output_index = output_indices_by_identity.get(id(value))
            if output_index is None:
                output_index = next_output_index
                next_output_index += 1
            mutations.append(InputMutation(locator, kind, output_index=output_index))

        return ForwardContract(
            output_template=output_spec.without_tensors(output),
            input_mutations=tuple(mutations),
        )


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Read a tensor version counter when its execution mode provides one."""
    try:
        return tensor._version
    except Exception:
        return None


def _tensor_metadata(tensor: torch.Tensor) -> _TensorMetadata:
    """Read metadata used to distinguish copying from replacement."""
    try:
        stride = tuple(tensor.stride())
        storage_offset = tensor.storage_offset()
    except RuntimeError:
        stride = None
        storage_offset = None
    return _TensorMetadata(
        shape=tuple(tensor.shape),
        stride=stride,
        storage_offset=storage_offset,
        layout=tensor.layout,
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _snapshot_leaf(
    locator: Locator,
    value: Any,
    snapshot_arguments: dict[str, Any] | None,
) -> _InputLeafSnapshot:
    """Save the identity and mutation evidence for one input leaf."""
    if not isinstance(value, torch.Tensor):
        return _InputLeafSnapshot(value, None)

    version = _tensor_version(value)
    tensor_value = None
    if version is None:
        if snapshot_arguments is not None:
            candidate = locator.get_value(snapshot_arguments)
            tensor_value = candidate if isinstance(candidate, torch.Tensor) else None
        if tensor_value is None or tensor_value is value:
            with torch.no_grad():
                tensor_value = value.detach().clone()
    return _InputLeafSnapshot(
        value=value,
        tensor_version=version,
        tensor_metadata=_tensor_metadata(value),
        tensor_value=tensor_value,
    )


def _tensor_values_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensors exactly when a version counter is unavailable."""
    try:
        if left.dtype.is_floating_point or left.dtype.is_complex:
            return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
        return torch.equal(left, right)
    except Exception:
        return False


def _values_equal(left: Any, right: Any) -> bool:
    """Compare arbitrary non-tensor leaves without leaking comparison errors."""
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def _mutation_kind(previous: _InputLeafSnapshot | None, value: Any) -> InputMutationKind | None:
    """Classify a leaf change by the restoration behavior it requires."""
    if isinstance(value, torch.Tensor):
        if previous is None or not isinstance(previous.value, torch.Tensor) or previous.value is not value:
            return InputMutationKind.REPLACE_TENSOR
        if previous.tensor_metadata != _tensor_metadata(value):
            return InputMutationKind.REPLACE_TENSOR
        current_version = _tensor_version(value)
        if previous.tensor_version is not None and current_version is not None:
            return InputMutationKind.COPY_TENSOR if previous.tensor_version != current_version else None
        if previous.tensor_value is None or not _tensor_values_equal(previous.tensor_value, value):
            return InputMutationKind.COPY_TENSOR
        return None
    if previous is None or not _values_equal(previous.value, value):
        return InputMutationKind.REPLACE_CONSTANT
    return None
