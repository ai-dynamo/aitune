# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node keys and nodes of the context-keyed call tree.

A node is identified by its calling context: the chain of call labels that led to it, outermost first. The same
module called from two places therefore gives two nodes (for example an LLM used for prefill and for generation).
Labels come from the adapter that installed the wrappers: object paths in AOT mode.
"""

from dataclasses import dataclass, field

from torch import nn

NodeKey = tuple[str, ...]

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
