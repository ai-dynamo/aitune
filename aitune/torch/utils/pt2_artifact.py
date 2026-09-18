# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary PT2 adapter for AOTInductor's pytree call contract.

TODO: Replace this module with the shared recorded export contract from
``static-cache/04-export-cleanup`` once that branch is merged. In particular,
the shared contract will own canonical tensor ordering and remove the need to
inspect live trees by object identity here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch
from torch.utils import _pytree

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator
from aitune.torch.module.sample_store import Sample
from aitune.torch.module.tensor_spec import TensorSpec


class PT2CallContractState(TypedDict):
    """Checkpoint representation of a temporary PT2 call contract.

    ``structured`` is true when an input argument or the output is a container instead of a tensor. For example, the
    call may pass tensors inside a dictionary or return tensors in a tuple.
    """

    input_order: tuple[int, ...]
    output_order: tuple[int, ...]
    structured: bool


@dataclass(frozen=True, slots=True)
class PT2CallContract:
    """Validated mapping from a PyTorch call tree to PT2's ordinal tensor ABI."""

    input_order: tuple[int, ...]
    output_order: tuple[int, ...]
    structured: bool

    @classmethod
    def capture(cls, graph_spec: GraphSpec, sample: Sample, output: Any) -> "PT2CallContract":
        """Capture tensor leaf ordering from the exact call used for export."""
        args, kwargs = sample
        normalized = graph_spec.forward_signature.normalize(args, kwargs)
        _validate_tensor_tree(args, "args")
        _validate_tensor_tree(kwargs, "kwargs")
        _validate_tensor_tree(output, "result")
        return cls(
            input_order=_metadata_order(
                (args, kwargs),
                graph_spec.input_spec.tensor_data,
                normalized.arguments,
                "input",
            ),
            output_order=_metadata_order(
                output,
                graph_spec.output_spec.tensor_data,
                output,
                "output",
            ),
            structured=any(not isinstance(value, torch.Tensor) for value in normalized.arguments.values())
            or not isinstance(output, torch.Tensor),
        )

    def to_dict(self) -> PT2CallContractState:
        """Return checkpoint-safe primitive values."""
        return {
            "input_order": self.input_order,
            "output_order": self.output_order,
            "structured": self.structured,
        }

    @classmethod
    def from_dict(cls, data: PT2CallContractState) -> "PT2CallContract":
        """Restore a call contract from checkpoint state."""
        return cls(
            input_order=tuple(data["input_order"]),
            output_order=tuple(data["output_order"]),
            structured=data["structured"],
        )


def _validate_tensor_tree(value: Any, path: str) -> None:
    """Require tensors under containers supported by Triton's PT2 runtime."""
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, tuple | list):
        for index, child in enumerate(value):
            _validate_tensor_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                not isinstance(key, str)
                or not key.isprintable()
                or any(character.isspace() or character in {'"', "[", "]"} for character in key)
            ):
                raise ValueError(f"PT2 deployment does not support dictionary key {key!r} at {path}")
            _validate_tensor_tree(child, f"{path}[{key!r}]")
        return
    raise ValueError(
        "PT2 deployment supports only tensor leaves under tuples, lists, and dictionaries; "
        f"got {type(value).__name__} at {path}"
    )


def _metadata_order(
    tree: Any,
    tensor_data: Sequence[tuple[Locator, TensorSpec]],
    metadata_root: Any,
    label: Literal["input", "output"],
) -> tuple[int, ...]:
    """Map PyTorch pytree tensor order to GraphSpec metadata indices."""
    leaves, _ = _pytree.tree_flatten(tree)
    if not leaves or any(not isinstance(leaf, torch.Tensor) for leaf in leaves):
        raise ValueError(f"PT2 deployment requires at least one tensor-only {label}")

    indices_by_identity: dict[int, list[int]] = {}
    for index, (locator, _) in enumerate(tensor_data):
        tensor = locator.get_value(metadata_root)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"GraphSpec {label} metadata no longer locates a tensor at {locator}")
        indices_by_identity.setdefault(id(tensor), []).append(index)

    order: list[int] = []
    for leaf in leaves:
        candidates = indices_by_identity.get(id(leaf))
        if not candidates:
            raise ValueError(f"PyTorch and GraphSpec disagree about the flattened {label} tensor order")
        order.append(candidates.pop(0))
    if len(order) != len(tensor_data) or any(candidate for candidate in indices_by_identity.values()):
        raise ValueError(f"PyTorch and GraphSpec disagree about the flattened {label} tensor count")
    return tuple(order)
