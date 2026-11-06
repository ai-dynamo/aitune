# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-keyed call tree recorded by the call recorder.

A node is identified by its calling context: the chain of call labels that led to it, outermost first. The same
module called from two places therefore gives two nodes (for example an LLM used for prefill and for generation).
Labels come from the adapter that installed the wrappers: object paths in AOT mode.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from aitune.torch.call_tree.sharing import SharedWeights

NodeKey = tuple[str, ...]

UnitSpec = str | nn.Module | Sequence[str | nn.Module]

PATH_SEPARATOR = " > "

ROOT_LABEL = "root"


def short_label(label: str) -> str:
    """Label without the leading ``root.`` of the adapter.

    Example:
        >>> short_label("root.vlm.lm_head.forward"), short_label("root.forward")
        ('vlm.lm_head.forward', 'forward')
    """
    return label.removeprefix(f"{ROOT_LABEL}.")


def key_path(key: NodeKey) -> str:
    """The written form of a node key: its labels without ``root.``, outermost first, joined by `` > ``.

    Example:
        >>> key_path(("root.forward", "root.vlm.forward", "root.vlm.lm_head.forward"))
        'forward > vlm.forward > vlm.lm_head.forward'
    """
    return PATH_SEPARATOR.join(short_label(label) for label in key)


@dataclass
class CallNode:
    """One node of the call tree.

    Attributes:
        key: Labels of the calling chain, outermost first. Identifies the node.
        label: Adapter name of this call, for example ``root.vlm.model.visual.forward``.
        owner_class: Class name of the object the method belongs to.
        method: Name of the called method (``forward`` for module calls).
        module: Module the method belongs to, or None for methods of plain objects.
        calls: Number of calls over all recorded iterations.
        time_ms: Time in milliseconds over all recorded iterations.
        children: Keys of the called nodes in first-call order.
        first_seen: Position of the node's first call among all nodes.

    Example:
        >>> node = CallNode(
        ...     key=("root.forward", "root.dit_forward"), label="root.dit_forward", owner_class="Vla", method="dit_forward"
        ... )
        >>> node.path, node.parent_key
        ('forward > dit_forward', ('root.forward',))
    """

    key: NodeKey
    label: str
    owner_class: str
    method: str
    module: nn.Module | None = None
    calls: int = 0
    time_ms: float = 0.0
    children: list[NodeKey] = field(default_factory=list)
    first_seen: int = 0

    @property
    def parent_key(self) -> NodeKey:
        """Key of the calling node; empty for outermost calls."""
        return self.key[:-1]

    @property
    def path(self) -> str:
        """The node key in written form (``forward > dit_forward``): its call labels, outermost first."""
        return key_path(self.key)


@dataclass
class CallTree:
    """Call tree of one recording.

    Attributes:
        nodes: All nodes by key.
        iterations: Number of recorded iterations.
        sequences: Per iteration, the keys of the entered nodes in call order.
        external_apply: For each module (by ``id``), the labels of the calls during which ``nn.Module._apply``
            (``.to()``, ``.cuda()``, ...) reached it. ``<outside any call>`` if no recorded call was running.
        weight_accesses: For each parameter or buffer (by ``id``), the recorded accesses to it outside an op that
            offloading preserves: ``(node key, "read" | "device", caller label)``.
        modules: Label prefix and module for every instrumented module, used for parameter names and sizes.

    Example:
        >>> outer = CallNode(
        ...     key=("root.forward",), label="root.forward", owner_class="Vla", method="forward", calls=2, time_ms=30.0
        ... )
        >>> dit = CallNode(
        ...     key=("root.forward", "root.dit.forward"),
        ...     label="root.dit.forward",
        ...     owner_class="Dit",
        ...     method="forward",
        ...     calls=10,
        ...     time_ms=25.0,
        ...     first_seen=1,
        ... )
        >>> outer.children.append(dit.key)
        >>> tree = CallTree(nodes={n.key: n for n in (outer, dit)}, iterations=2)
        >>> [n.path for n in tree.roots], tree.total_ms
        (['forward'], 30.0)
        >>> [n.path for n in tree.descendants(outer)]
        ['forward > dit.forward']
        >>> [(n.path, n.calls) for n in tree.find("dit")]
        [('forward > dit.forward', 10)]
    """

    nodes: dict[NodeKey, CallNode] = field(default_factory=dict)
    iterations: int = 0
    sequences: list[list[NodeKey]] = field(default_factory=list)
    external_apply: dict[int, set[str]] = field(default_factory=dict)
    weight_accesses: dict[int, set[tuple[NodeKey, str, str]]] = field(default_factory=dict)
    modules: list[tuple[str, nn.Module]] = field(default_factory=list)

    @property
    def roots(self) -> list[CallNode]:
        """Outermost nodes in first-call order."""
        return sorted((n for n in self.nodes.values() if len(n.key) == 1), key=lambda n: n.first_seen)

    @property
    def total_ms(self) -> float:
        """Time of all outermost calls over all recorded iterations."""
        return sum(n.time_ms for n in self.roots)

    def descendants(self, node: CallNode) -> list[CallNode]:
        """All nodes below ``node``."""
        out, todo = [], list(node.children)
        while todo:
            child = self.nodes[todo.pop()]
            out.append(child)
            todo.extend(child.children)
        return out

    def find(self, text: str) -> list[CallNode]:
        """Nodes whose label contains ``text``, in first-call order."""
        return sorted((n for n in self.nodes.values() if text in n.label), key=lambda n: n.first_seen)

    def shared_weights_of(self, unit: UnitSpec) -> "list[SharedWeights]":
        """The weights ``unit`` holds that something outside it still uses or moves.

        This is what tuning ``unit`` alone and offloading its weights would break: code outside it that reads one
        of its weights (a tied head, ``F.linear(x, unit.weight)`` in a caller's forward), code that looks up a
        weight's device, or code outside it that moves its modules with ``.to()`` or ``.cuda()``. Reads and device
        lookups come from the ops recorded by ``WeightAccessMode`` during the call-tree pass. Weights with the same
        readers, device lookups and movers form one group.

        Args:
            unit: A node path (``"forward > dit_forward"``, see ``CallNode.path``), a module (all its recorded
                ``forward`` calls), or a list of them as one unit.

        Returns:
            The groups, sorted by their weights; empty if the unit can be tuned and offloaded on its own.

        Raises:
            ValueError: If ``unit`` matches no recorded call.
        """
        from aitune.torch.call_tree.sharing import shared_weights_of

        return shared_weights_of(self, unit)
