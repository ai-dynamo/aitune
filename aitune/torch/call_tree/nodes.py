# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call paths and nodes of the context-keyed call tree.

A node is identified by its calling context: the chain of call labels that led to it, outermost first. The same
module called from two places therefore gives two nodes (for example an LLM used for prefill and for generation).
Labels come from the adapter that installed the wrappers: object paths in AOT mode.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

# The chain of call labels from the outermost call to this one; a node's identity.
# E.g. ("root.forward", "root.vlm.forward", "root.vlm.lm_head.forward").
CallPath = tuple[str, ...]

PATH_SEPARATOR = " > "

ROOT_LABEL = "root"


def short_label(label: str) -> str:
    """Label without the leading ``root.`` of the adapter.

    Example:
        >>> short_label("root.vlm.lm_head.forward"), short_label("root.forward")
        ('vlm.lm_head.forward', 'forward')
    """
    return label.removeprefix(f"{ROOT_LABEL}.")


def format_call_path(path: CallPath) -> str:
    """The written form of a call path: its labels without ``root.``, outermost first, joined by `` > ``.

    Example:
        >>> format_call_path(("root.forward", "root.vlm.forward", "root.vlm.lm_head.forward"))
        'forward > vlm.forward > vlm.lm_head.forward'
    """
    return PATH_SEPARATOR.join(short_label(label) for label in path)


@dataclass(frozen=True, slots=True)
class Leaf:
    """A leaf of a ``CallSignature``: its name and qualified class name.

    A leaf is one argument or result of a call, or a value nested in one. Tensors are ``TensorLeaf``s, which add
    dtype and shape; any other object, such as a ``DynamicCache``, is a plain ``Leaf``. The name is the parameter
    name, or ``output`` for the result, followed by a suffix per container for nested values, such as ``x``,
    ``inputs['input_ids']`` or ``output['logits']``. It is relative to the call, not the node path.
    """

    name: str
    type: str


@dataclass(frozen=True, slots=True)
class TensorLeaf(Leaf):
    """A ``Leaf`` for a tensor, with its dtype and exact shape."""

    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass
class CallSignature:
    """The arguments and results of a group of a node's calls that passed exactly the same leaves, and those calls.

    A tensor is recorded as a ``TensorLeaf`` with its dtype and exact shape, any other object as a plain ``Leaf``
    with its type; primitives (numbers, strings, ``None``, enums, and torch's ``dtype``, ``device``, ``Size``, ...)
    are not recorded. Calls that differ in any leaf, in its type or, for a tensor, its dtype or shape, or in which
    arguments they pass, have separate signatures. Leaves keep the order of the walk. Named parameters are always in
    declaration order, but a dict, including ``**kwargs``, with the same entries inserted in another order gives a
    separate signature. Leaves store class names, not classes, so signatures stay picklable for rank
    synchronization.

    Attributes:
        inputs: The signature's argument leaves by ``Leaf.name``.
        outputs: The signature's result leaves by ``Leaf.name``, or None when the calls raised.
        call_indices: The indices (0-based) of the node's calls with exactly this signature. Indices count the node's
            calls in the order they started, across all iterations — the order ``CallTree.sequences`` records —
            not reset per iteration.

    Example:
        >>> import torch
        >>> from aitune.torch.call_tree.recorder import CallRecorder
        >>> class Cache: ...
        >>> class Layer(torch.nn.Module):
        ...     def forward(self, x, state=None):
        ...         return x
        >>> recorder, layer = CallRecorder(), Layer()
        >>> with recorder:
        ...     for args in (
        ...         ((torch.zeros(1, 4, 8),), {"state": Cache()}),
        ...         ((torch.zeros(1, 4, 8),), {}),
        ...         ((torch.zeros(1, 1, 8),), {}),
        ...         ((torch.zeros(1, 4, 8),), {}),  # same signature as call 1
        ...     ):
        ...         frame = recorder.begin_call("root.forward", layer, "forward", *args)
        ...         recorder.end_call(frame, torch.zeros(1, 4, 8))
        >>> node = recorder.result().nodes[("root.forward",)]
        >>> [sig.call_indices for sig in node.signatures]
        [[0], [1, 3], [2]]
        >>> node.calls_passing("state")
        [0]
        >>> node.shapes("x")
        [(1, 4, 8), (1, 1, 8)]
        >>> node.signatures[1].inputs["x"].shape
        (1, 4, 8)
        >>> entry = recorder.result().to_dict()["nodes"][0]
        >>> entry["signatures"][1]["inputs"]["x"]
        {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [1, 4, 8]}
    """

    inputs: dict[str, Leaf] = field(default_factory=dict)
    outputs: dict[str, Leaf] | None = None
    call_indices: list[int] = field(default_factory=list)


@dataclass
class CallNode:
    """One node of the call tree.

    ``bytes_in`` and ``bytes_out`` measure the tensor data passed into and returned from the node's calls. Only
    tensors count, each as ``numel() * element_size()``, and a tensor passed twice in one call counts once. Tensors
    inside lists, tuples, dicts and ``ModelOutput`` count too; ``UserDict``s, such as transformers'
    ``BatchFeature`` and ``BatchEncoding``, are walked into their values, and dataclass instances into their fields.
    Everything else counts as 0 bytes: numbers, strings, ``None``, and objects such as a ``DynamicCache`` or
    ``StaticCache``, including the tensors they hold. The ``signatures`` are recorded by the same walk and also list
    the non-tensor objects by type.

    Attributes:
        path: Labels of the calling chain, outermost first. Identifies the node and must not change after creation:
            ``path_str`` is cached from it.
        label: Adapter name of this call, for example ``root.vlm.model.visual.forward``.
        owner_class: Class name of the object the method belongs to.
        method: Name of the called method (``forward`` for module calls).
        module: Module the method belongs to, or None for methods of plain objects.
        calls: Number of calls over all recorded iterations.
        time_ms: Time in milliseconds over all recorded iterations.
        bytes_in: Tensor bytes in the arguments, summed over all recorded iterations. Bytes per call:
            ``bytes_in / calls``.
        bytes_out: Tensor bytes in the results, summed over all recorded iterations.
        signatures: The distinct ``CallSignature``s of the node's calls, in first-seen order; each records the
            start-order indices of the calls that had it, so ``sum(len(s.call_indices)) == calls`` and every index
            appears exactly once. A signature whose calls raised has ``outputs`` None.
        children: Call paths of the called nodes in first-call order.
        first_seen: Position of the node's first call among all nodes.

    Example:
        >>> node = CallNode(
        ...     path=("root.forward", "root.dit_forward"),
        ...     label="root.dit_forward",
        ...     owner_class="Vla",
        ...     method="dit_forward",
        ... )
        >>> node.path_str, node.parent_path
        ('forward > dit_forward', ('root.forward',))
    """

    path: CallPath
    label: str
    owner_class: str
    method: str
    module: nn.Module | None = None
    calls: int = 0
    time_ms: float = 0.0
    bytes_in: int = 0
    bytes_out: int = 0
    signatures: list[CallSignature] = field(default_factory=list)
    children: list[CallPath] = field(default_factory=list)
    first_seen: int = 0

    def calls_passing(self, name: str, side: Literal["inputs", "outputs"] = "inputs") -> list[int]:
        """The sorted indices of the calls that passed the argument ``name`` on ``side``.

        Calls that raised have no outputs, so they never count for ``side="outputs"``.
        """
        indices = set()
        for signature in self.signatures:
            leaves = signature.outputs if side == "outputs" else signature.inputs
            if leaves is not None and name in leaves:
                indices.update(signature.call_indices)
        return sorted(indices)

    def shapes(self, name: str, side: Literal["inputs", "outputs"] = "inputs") -> list[tuple[int, ...]]:
        """The distinct tensor shapes passed as the argument ``name`` on ``side``, in first-seen order.

        Signatures where ``name`` is missing or not a tensor are skipped.
        """
        seen = []
        for signature in self.signatures:
            leaves = signature.outputs if side == "outputs" else signature.inputs
            if leaves is None:
                continue
            leaf = leaves.get(name)
            if isinstance(leaf, TensorLeaf) and leaf.shape not in seen:
                seen.append(leaf.shape)
        return seen

    @property
    def parent_path(self) -> CallPath:
        """Call path of the calling node; empty for outermost calls."""
        return self.path[:-1]

    @functools.cached_property
    def path_str(self) -> str:
        """The call path in written form (``forward > dit_forward``): its call labels, outermost first."""
        return format_call_path(self.path)
