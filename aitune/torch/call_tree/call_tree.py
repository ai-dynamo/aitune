# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context-keyed call tree recorded by the call recorder."""

from dataclasses import dataclass, field, replace
from typing import Any

from torch import nn

from aitune.torch.call_tree.nodes import (
    ROOT_LABEL,
    CallNode,
    CallPath,
    Leaf,
    TensorLeaf,
    format_call_path,
)
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
        weight_accesses: For each parameter or buffer (by weight name: its object path, e.g.
            ``vlm.lm_head.weight``; a tied weight appears under each of its names), the ``WeightAccess`` records of
            accesses to it: reads, device checks and moves. Names stay valid when ``.to()`` or casts replace the
            tensors.
        modules: Label prefix and module for every instrumented module, used for parameter names and sizes.

    Example:
        >>> outer = CallNode(
        ...     path=("root.forward",),
        ...     label="root.forward",
        ...     owner_class="Vla",
        ...     method="forward",
        ...     calls_per_sample=[1, 1],
        ...     total_time_ms=30.0,
        ... )
        >>> dit_module = nn.Linear(2, 2)
        >>> dit = CallNode(
        ...     path=("root.forward", "root.dit.forward"),
        ...     label="root.dit.forward",
        ...     owner_class="Dit",
        ...     method="forward",
        ...     module=dit_module,
        ...     calls_per_sample=[5, 5],
        ...     total_time_ms=25.0,
        ...     first_seen=1,
        ... )
        >>> outer.children.append(dit.path)
        >>> tree = CallTree(nodes={n.path: n for n in (outer, dit)})
        >>> [n.path_str for n in tree.roots], tree.total_time_ms
        (['forward'], 30.0)
        >>> [n.path_str for n in tree.descendants(outer)]
        ['forward > dit.forward']
        >>> [(n.path_str, n.total_calls) for n in tree.find("dit")]
        [('forward > dit.forward', 10)]
        >>> [n.path_str for n in tree.nodes_of(dit_module)]
        ['forward > dit.forward']
    """

    nodes: dict[CallPath, CallNode] = field(default_factory=dict)
    weight_accesses: dict[str, set[WeightAccess]] = field(default_factory=dict)
    modules: list[tuple[str, nn.Module]] = field(default_factory=list)

    @property
    def roots(self) -> list[CallNode]:
        """Outermost nodes in first-call order."""
        return sorted((n for n in self.nodes.values() if len(n.path) == 1), key=lambda n: n.first_seen)

    @property
    def total_time_ms(self) -> float:
        """Time of all outermost calls over all recorded samples; nodes with no timed calls do not count."""
        return sum(n.total_time_ms for n in self.roots if n.total_time_ms is not None)

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

    def copy(self) -> "CallTree":
        """A snapshot of the tree: mutable containers are copied, immutable values shared.

        Each node is replaced by a copy whose ``children`` and ``signatures`` are new —
        each signature with copied ``inputs``/``outputs`` dicts and ``call_indices`` list — so later writes into
        them (a timing pass's times, a ``synchronize_call_tree``'s values) don't reach this tree. Each node's
        ``calls_per_sample`` list, the per-weight ``WeightAccess`` sets and the ``modules`` list are copied too.
        Shared: the ``nn.Module``s (in ``modules`` and on the nodes), the signature leaves (``Leaf``/``TensorLeaf``
        are frozen dataclasses) and the ``WeightAccess`` records (``NamedTuple``s).

        Example:
            >>> node = CallNode(path=("m.forward",), label="m.forward", owner_class="M", method="forward")
            >>> tree = CallTree(nodes={("m.forward",): node})
            >>> copied = tree.copy()
            >>> copied.nodes[("m.forward",)].total_time_ms = 1.0
            >>> tree.nodes[("m.forward",)].total_time_ms is None
            True
        """
        return CallTree(
            nodes={
                path: replace(
                    node,
                    children=list(node.children),
                    calls_per_sample=list(node.calls_per_sample),
                    signatures=[
                        replace(
                            sig,
                            inputs=dict(sig.inputs),
                            outputs=None if sig.outputs is None else dict(sig.outputs),
                            call_indices=list(sig.call_indices),
                        )
                        for sig in node.signatures
                    ],
                )
                for path, node in self.nodes.items()
            },
            weight_accesses={name: set(records) for name, records in self.weight_accesses.items()},
            modules=list(self.modules),
        )

    def find(self, text: str) -> list[CallNode]:
        """Nodes whose label contains ``text``, in first-call order."""
        return sorted((n for n in self.nodes.values() if text in n.label), key=lambda n: n.first_seen)

    def to_dict(self) -> dict[str, Any]:
        """The call tree as plain JSON data, for tools that read it without torch or aitune.

        Node values are totals over all calls and samples: divide by ``total_calls`` for per call;
        ``calls_per_sample`` gives the per-sample counts.

        Returns:
            A dict of str, int, float, list and dict values with these keys:

            - ``schema_version``: format version. Fields may be added under the same version, so readers should
              ignore keys they don't know; it changes when a field is removed, renamed or changes meaning.
            - ``total_time_ms``: time of the outermost calls over all samples (``CallTree.total_time_ms``).
            - ``nodes``: one dict per node, in first-call order, with ``call_path`` (``CallNode.path_str``, which
              identifies the node), ``owner_class``, ``total_calls``, ``total_time_ms`` (the total measured time of the
              node's calls, null when never timed), ``total_bytes_in``, ``total_bytes_out`` and
              ``signatures``: the node's distinct ``CallSignature``s in first-seen order, each with its
              ``call_indices`` (the indices of the calls that had it) and ``inputs``/``outputs`` dicts of leaves
              by path — ``outputs`` is null for a signature whose calls raised. Every leaf has ``type``; tensors
              also ``dtype`` and ``shape``. A leaf whose object a recorded call had before also has ``source``:
              ``{"call_path": ..., "side": ..., "name": ...}``, where it was last seen (``Leaf.source``).
              ``calls_per_sample``: ``calls_per_sample[i]`` is how many of the node's calls sample ``i``
              made (``CallNode.calls_per_sample``).
            - ``modules``: for each instrumented module, by object path, its ``class`` (qualified name),
              ``parameters`` and ``parameter_bytes``. ``parameters`` maps the local name of each of the module's own
              parameters and buffers — not its submodules', which have their own entries if instrumented — to its
              ``dtype`` and ``shape``, also for ``meta`` tensors. ``parameter_bytes`` is
              ``numel() * element_size()`` summed over the parameters and buffers of the module and its submodules,
              also for ``meta`` tensors. Each tensor object counts once, so a weight tied by assigning the same
              ``Parameter`` to two submodules counts once; different tensor objects that share storage count
              separately, as in ``CallNode.total_bytes_in``.
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
            >>> tree = CallTree(nodes={n.path: n for n in (outer, node)})
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
        return {
            "schema_version": SCHEMA_VERSION,
            "total_time_ms": self.total_time_ms,
            "nodes": [
                {
                    "call_path": node.path_str,
                    "owner_class": node.owner_class,
                    "total_calls": node.total_calls,
                    "inside_compiled_part": node.inside_compiled_part,
                    "total_time_ms": node.total_time_ms,
                    "total_bytes_in": node.total_bytes_in,
                    "total_bytes_out": node.total_bytes_out,
                    "signatures": [
                        {
                            "call_indices": list(sig.call_indices),
                            "inputs": _leaves_dict(sig.inputs),
                            "outputs": None if sig.outputs is None else _leaves_dict(sig.outputs),
                        }
                        for sig in node.signatures
                    ],
                    "calls_per_sample": list(node.calls_per_sample),
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
