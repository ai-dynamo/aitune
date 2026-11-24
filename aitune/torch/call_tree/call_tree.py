# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-keyed call tree recorded by the call recorder."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from aitune.torch.call_tree.nodes import ROOT_LABEL, CallNode, CallPath, Leaf, TensorLeaf, format_call_path
from aitune.torch.call_tree.weights.accesses import WeightAccess

SCHEMA_VERSION = 1


def _leaves_dict(leaves: dict[str, Leaf]) -> dict[str, dict]:
    """A signature's leaves for export, keyed by name; the name is the key only, not repeated in the entry."""
    entries: dict[str, dict[str, object]] = {}
    for name, leaf in leaves.items():
        entry: dict[str, object] = {"type": leaf.type}
        if isinstance(leaf, TensorLeaf):
            entry["dtype"] = str(leaf.dtype)
            entry["shape"] = list(leaf.shape)
        if leaf.source is not None:
            entry["source"] = {
                "call_path": format_call_path(leaf.source.path),
                "side": leaf.source.side,
                "name": leaf.source.name,
            }
        entries[name] = entry
    return entries


def _object_path(label_prefix: str) -> str:
    """The object path of a module's label prefix: ``root`` gives ``""``, ``root.vlm.lm_head`` gives ``vlm.lm_head``."""
    return label_prefix.removeprefix(ROOT_LABEL).removeprefix(".")


@dataclass
class CallTree:
    """Call tree of one recording.

    Attributes:
        nodes: All nodes by call path.
        iterations: Number of recorded iterations.
        sequences: Per iteration, the call paths of the entered nodes in the order the calls started.
        weight_accesses: For each parameter or buffer (by weight name: its object path, e.g.
            ``vlm.lm_head.weight``; a tied weight appears under each of its names), the ``WeightAccess`` records of
            accesses to it: reads, device checks and moves. Names stay valid when ``.to()`` or casts replace the
            tensors.
        modules: Label prefix and module for every instrumented module, used for parameter names and sizes.

    Example:
        >>> outer = CallNode(
        ...     path=("root.forward",), label="root.forward", owner_class="Vla", method="forward", calls=2, time_ms=30.0
        ... )
        >>> dit_module = nn.Linear(2, 2)
        >>> dit = CallNode(
        ...     path=("root.forward", "root.dit.forward"),
        ...     label="root.dit.forward",
        ...     owner_class="Dit",
        ...     method="forward",
        ...     module=dit_module,
        ...     calls=10,
        ...     time_ms=25.0,
        ...     first_seen=1,
        ... )
        >>> outer.children.append(dit.path)
        >>> tree = CallTree(nodes={n.path: n for n in (outer, dit)}, iterations=2)
        >>> [n.path_str for n in tree.roots], tree.total_ms
        (['forward'], 30.0)
        >>> [n.path_str for n in tree.descendants(outer)]
        ['forward > dit.forward']
        >>> [(n.path_str, n.calls) for n in tree.find("dit")]
        [('forward > dit.forward', 10)]
        >>> [n.path_str for n in tree.nodes_of(dit_module)]
        ['forward > dit.forward']
    """

    nodes: dict[CallPath, CallNode] = field(default_factory=dict)
    iterations: int = 0
    sequences: list[list[CallPath]] = field(default_factory=list)
    weight_accesses: dict[str, set[WeightAccess]] = field(default_factory=dict)
    modules: list[tuple[str, nn.Module]] = field(default_factory=list)

    @property
    def roots(self) -> list[CallNode]:
        """Outermost nodes in first-call order."""
        return sorted((n for n in self.nodes.values() if len(n.path) == 1), key=lambda n: n.first_seen)

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

    def iteration_starts(self) -> dict[CallPath, list[int]]:
        """For each node, how many of its calls were made before each iteration began.

        ``iteration_starts()[path][i]`` is the index of the node's first call in iteration ``i`` when the node is
        called there, so it maps the call indices in ``CallSignature.call_indices`` to iterations. It assumes that all
        calls are recorded inside iterations, as ``inspect`` does.

        Example:
            >>> outer = CallNode(path=("root.forward",), label="root.forward", owner_class="Vla", method="forward")
            >>> step = CallNode(path=("root.forward", "root.step"), label="root.step", owner_class="Vla", method="step")
            >>> tree = CallTree(
            ...     nodes={n.path: n for n in (outer, step)},
            ...     iterations=2,
            ...     sequences=[[outer.path, step.path, step.path], [outer.path, step.path]],
            ... )
            >>> tree.iteration_starts()[step.path]
            [0, 2]
        """
        seen: Counter[CallPath] = Counter()
        starts: dict[CallPath, list[int]] = {path: [] for path in self.nodes}
        for sequence in self.sequences:
            for path in self.nodes:
                starts[path].append(seen[path])
            seen.update(sequence)
        return starts

    def to_dict(self) -> dict[str, Any]:
        """The call tree as plain JSON data, for tools that read it without torch or aitune.

        Node values are totals over ``iterations``: divide by ``iterations`` for per iteration, by ``calls`` for
        per call.

        Returns:
            A dict of str, int, float, list and dict values with these keys:

            - ``schema_version``: format version. Fields may be added under the same version, so readers should
              ignore keys they don't know; it changes when a field is removed, renamed or changes meaning.
            - ``iterations``: number of recorded iterations.
            - ``total_ms``: time of the outermost calls over all iterations (``CallTree.total_ms``).
            - ``nodes``: one dict per node, in first-call order, with ``call_path`` (``CallNode.path_str``, which
              identifies the node), ``owner_class``, ``calls``, ``time_ms``, ``bytes_in``, ``bytes_out`` and
              ``signatures``: the node's distinct ``CallSignature``s in first-seen order, each with its
              ``call_indices`` (the indices of the calls that had it) and ``inputs``/``outputs`` dicts of leaves
              by path — ``outputs`` is null for a signature whose calls raised. Every leaf has ``type``; tensors
              also ``dtype`` and ``shape``. A leaf whose object a recorded call had before also has ``source``:
              ``{"call_path": ..., "side": ..., "name": ...}``, where it was last seen (``Leaf.source``).
              ``iteration_starts``: ``iteration_starts[i]`` is how many of the node's calls were made before
              iteration ``i`` began (``CallTree.iteration_starts``).
            - ``modules``: for each instrumented module, by object path, its ``class`` (qualified name),
              ``parameters`` and ``parameter_bytes``. ``parameters`` maps the local name of each of the module's own
              parameters and buffers — not its submodules', which have their own entries if instrumented — to its
              ``dtype`` and ``shape``, also for ``meta`` tensors. ``parameter_bytes`` is
              ``numel() * element_size()`` summed over the parameters and buffers of the module and its submodules,
              also for ``meta`` tensors. Each tensor object counts once, so a weight tied by assigning the same
              ``Parameter`` to two submodules counts once; different tensor objects that share storage count
              separately, as in ``CallNode.bytes_in``.
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
            ...     path=("root.forward", "root.dit.forward"),
            ...     label="root.dit.forward",
            ...     owner_class="Dit",
            ...     method="forward",
            ...     first_seen=1,
            ... )
            >>> outer = CallNode(path=("root.forward",), label="root.forward", owner_class="Vla", method="forward")
            >>> tree = CallTree(nodes={n.path: n for n in (outer, node)}, iterations=2)
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
                    "parameters": {
                        name: {"dtype": str(t.dtype), "shape": list(t.shape)}
                        for name, t in (
                            *module.named_parameters(recurse=False, remove_duplicate=False),
                            *module.named_buffers(recurse=False, remove_duplicate=False),
                        )
                    },
                    "parameter_bytes": sum(
                        t.numel() * t.element_size()
                        for t in {id(t): t for t in (*module.parameters(), *module.buffers())}.values()
                    ),
                }
        iteration_starts = self.iteration_starts()
        return {
            "schema_version": SCHEMA_VERSION,
            "iterations": self.iterations,
            "total_ms": self.total_ms,
            "nodes": [
                {
                    "call_path": node.path_str,
                    "owner_class": node.owner_class,
                    "calls": node.calls,
                    "time_ms": node.time_ms,
                    "bytes_in": node.bytes_in,
                    "bytes_out": node.bytes_out,
                    "signatures": [
                        {
                            "call_indices": list(sig.call_indices),
                            "inputs": _leaves_dict(sig.inputs),
                            "outputs": None if sig.outputs is None else _leaves_dict(sig.outputs),
                        }
                        for sig in node.signatures
                    ],
                    "iteration_starts": iteration_starts[node.path],
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
