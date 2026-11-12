# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-keyed call tree recorded by the call recorder."""

from dataclasses import dataclass, field

from torch import nn

from aitune.torch.call_tree.nodes import CallNode, NodeKey
from aitune.torch.call_tree.weights.accesses import WeightAccess


@dataclass
class CallTree:
    """Call tree of one recording.

    Attributes:
        nodes: All nodes by key.
        iterations: Number of recorded iterations.
        sequences: Per iteration, the keys of the entered nodes in call order.
        weight_accesses: For each parameter or buffer (by weight name: its object path, e.g.
            ``vlm.lm_head.weight``; a tied weight appears under each of its names), the ``WeightAccess`` records of
            accesses to it: reads, device checks and moves. Names stay valid when ``.to()`` or casts replace the
            tensors.
        modules: Label prefix and module for every instrumented module, used for parameter names and sizes.

    Example:
        >>> outer = CallNode(
        ...     key=("root.forward",), label="root.forward", owner_class="Vla", method="forward", calls=2, time_ms=30.0
        ... )
        >>> dit_module = nn.Linear(2, 2)
        >>> dit = CallNode(
        ...     key=("root.forward", "root.dit.forward"),
        ...     label="root.dit.forward",
        ...     owner_class="Dit",
        ...     method="forward",
        ...     module=dit_module,
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
        >>> [n.path for n in tree.nodes_of(dit_module)]
        ['forward > dit.forward']
    """

    nodes: dict[NodeKey, CallNode] = field(default_factory=dict)
    iterations: int = 0
    sequences: list[list[NodeKey]] = field(default_factory=list)
    weight_accesses: dict[str, set[WeightAccess]] = field(default_factory=dict)
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

    def nodes_of(self, module: nn.Module, method: str = "forward") -> list[CallNode]:
        """Nodes recorded for calls of ``module.<method>``.

        A node is one chain of callers, so a module called from one place has one node, however often it ran there.
        A module called from several places (e.g. an LLM used by both ``prefill`` and ``generate``) has one node per
        place.

        Matching is by object identity (``node.module is module``), not by label, so a module that is reachable under
        several names (e.g. a tied embedding) is found under whichever name was recorded.

        Args:
            module: The module object that was recorded. After ``ait.wrap`` the attribute holds the wrapper instead,
                which matches nothing; pass a reference to the original module.
            method: Name of the recorded method.

        Returns:
            The matching nodes in first-call order; empty if the module was not recorded. Methods of objects that
            are not ``nn.Module`` cannot be looked up this way, because their nodes have no ``module``.
        """
        return sorted(
            (n for n in self.nodes.values() if n.module is module and n.method == method), key=lambda n: n.first_seen
        )

    def find(self, text: str) -> list[CallNode]:
        """Nodes whose label contains ``text``, in first-call order."""
        return sorted((n for n in self.nodes.values() if text in n.label), key=lambda n: n.first_seen)
