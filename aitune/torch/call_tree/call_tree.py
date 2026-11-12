# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-keyed call tree recorded by the call recorder."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from aitune.torch.call_tree.nodes import ROOT_LABEL, CallNode, NodeKey
from aitune.torch.call_tree.weights.accesses import WeightAccess

SCHEMA_VERSION = 1


def _object_path(label_prefix: str) -> str:
    """The object path of a module's label prefix: ``root`` gives ``""``, ``root.vlm.lm_head`` gives ``vlm.lm_head``."""
    return label_prefix.removeprefix(ROOT_LABEL).removeprefix(".")


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

    def to_dict(self) -> dict[str, Any]:
        """The call tree as plain JSON data, for tools that read it without torch or aitune.

        ``save`` writes this dict to a file. Node values are totals over ``iterations``: divide by ``iterations`` for
        per iteration, by ``calls`` for per call.

        Returns:
            A dict of str, int, float, list and dict values with these keys:

            - ``schema_version``: format version. Fields may be added under the same version, so readers should
              ignore keys they don't know; it changes when a field is removed, renamed or changes meaning.
            - ``iterations``: number of recorded iterations.
            - ``total_ms``: time of the outermost calls over all iterations (``CallTree.total_ms``).
            - ``nodes``: one dict per node, in first-call order, with ``call_path`` (``CallNode.path``, which
              identifies the node), ``owner_class``, ``calls``, ``time_ms``, ``bytes_in`` and ``bytes_out``.
            - ``modules``: for each instrumented module, by object path, its ``class`` (qualified name) and
              ``parameter_bytes``: ``numel() * element_size()`` summed over the parameters and buffers of the module
              and its submodules, also for ``meta`` tensors. Each tensor object counts once, so a weight tied by
              assigning the same ``Parameter`` to two submodules counts once; different tensor objects that share
              storage count separately, as in ``CallNode.bytes_in``.
            - ``weights``: for each weight name, its recorded accesses as ``{"by": call_path, "kind": ...}``,
              ``kind`` being ``read``, ``device`` or ``move``.

            A node's parent, object path and method follow from its ``call_path``::

                parent = call_path.rpartition(" > ")[0]  # "" for outermost nodes
                object_path, _, method = call_path.rpartition(" > ")[2].rpartition(".")

            The object path is ``ModuleInfo.object_path``: the attribute path from the inspected object to the
            object whose method was called. Inspecting ``model``, ``language_model.layers.0`` is
            ``model.language_model.layers[0]``, and ``""`` is ``model`` itself, also for its own methods such as
            ``dit_forward``. For a module node it is the key into ``modules``.

        Example:
            >>> node = CallNode(
            ...     key=("root.forward", "root.dit.forward"),
            ...     label="root.dit.forward",
            ...     owner_class="Dit",
            ...     method="forward",
            ...     first_seen=1,
            ... )
            >>> outer = CallNode(key=("root.forward",), label="root.forward", owner_class="Vla", method="forward")
            >>> tree = CallTree(nodes={n.key: n for n in (outer, node)}, iterations=2)
            >>> exported = tree.to_dict()["nodes"][1]
            >>> exported["call_path"], exported["owner_class"]
            ('forward > dit.forward', 'Dit')
            >>> exported["call_path"].rpartition(" > ")[2].rpartition(".")
            ('dit', '.', 'forward')
        """
        modules: dict[str, Any] = {}
        for label, module in self.modules:
            object_path = _object_path(label)
            if object_path not in modules:  # keep the first prefix that maps to the same object path
                modules[object_path] = {
                    "class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "parameter_bytes": sum(
                        t.numel() * t.element_size()
                        for t in {id(t): t for t in (*module.parameters(), *module.buffers())}.values()
                    ),
                }
        return {
            "schema_version": SCHEMA_VERSION,
            "iterations": self.iterations,
            "total_ms": self.total_ms,
            "nodes": [
                {
                    "call_path": node.path,
                    "owner_class": node.owner_class,
                    "calls": node.calls,
                    "time_ms": node.time_ms,
                    "bytes_in": node.bytes_in,
                    "bytes_out": node.bytes_out,
                }
                for node in sorted(self.nodes.values(), key=lambda n: n.first_seen)
            ],
            "modules": modules,
            "weights": {
                name: [
                    {"by": access.by, "kind": access.kind} for access in sorted(accesses, key=lambda a: (a.kind, a.by))
                ]
                for name, accesses in sorted(self.weight_accesses.items())
            },
        }

    def save(self, path: str | Path) -> None:
        """Write the call tree to ``path`` as JSON, in the format of ``to_dict``.

        In distributed runs call this on one rank: the tree is already synchronized across ranks.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
