# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Units of the call tree: a module, a node path or a list of them, as the recorded calls and tensors they cover.

A unit is what tuning would wrap. ``resolve_unit`` finds the recorded calls it stands for, and ``owned_tensors`` the
parameters and buffers that offloading it would move.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from torch import nn

from aitune.torch.call_tree.nodes import CallNode

if TYPE_CHECKING:
    from aitune.torch.call_tree.call_tree import CallTree

UnitSpec = str | nn.Module | Sequence[str | nn.Module]


def resolve_unit(tree: CallTree, unit: UnitSpec) -> list[CallNode]:
    """The recorded nodes of ``unit``: a node path, a module, or a list of them.

    A module stands for all its recorded ``forward`` calls. A module without one (an exact ``nn.Sequential`` the
    recorder skips, or a ``forward`` that takes a callable argument) stands for the outermost recorded calls of the
    module and its submodules. A module that is part of the recorded model but never ran resolves to no nodes, so
    every access to its weights counts as outside.
    """
    items = unit_items(unit)
    if not items:
        raise ValueError("The unit names no calls")
    by_path = {n.path: n for n in tree.nodes.values()}
    nodes: list[CallNode] = []
    for item in items:
        if isinstance(item, nn.Module):
            matches = tree.nodes_of(item)
            if not matches:
                matches = _outermost_inside(tree, item)
            if not matches:
                recorded = {id(m) for _, root in tree.modules for m in root.modules()}
                if id(item) in recorded:
                    # Part of the recorded model but never called: no nodes, so every access is outside it.
                    continue
                raise ValueError(f"Module {type(item).__name__} is not part of the recorded model")
        elif isinstance(item, str):
            if item not in by_path:
                last = item.rsplit(" > ", 1)[-1]
                similar = [n.path for n in tree.nodes.values() if n.path.endswith(last)][:5]
                raise ValueError(f"No recorded call at {item!r}; calls ending the same: {similar}")
            matches = [by_path[item]]
        else:
            raise TypeError(f"Expected a node path or a module, got {type(item).__name__}")
        nodes.extend(n for n in matches if n not in nodes)
    return sorted(nodes, key=lambda n: n.first_seen)


def unit_items(unit: UnitSpec) -> list:
    """The unit's members as a list: a single path or module becomes a one-item list."""
    return [unit] if isinstance(unit, (str, nn.Module)) else list(unit)


def _outermost_inside(tree: CallTree, module: nn.Module) -> list[CallNode]:
    """Recorded calls of ``module`` or its submodules whose callers are outside it, in first-call order."""
    inside = {id(m) for m in module.modules()}
    keys = {k for k, n in tree.nodes.items() if n.module is not None and id(n.module) in inside}
    return sorted(
        (tree.nodes[k] for k in keys if not any(k[:i] in keys for i in range(1, len(k)))), key=lambda n: n.first_seen
    )


def owned_tensors(tree: CallTree, unit: UnitSpec, nodes: list[CallNode]) -> set[int]:
    """The ids of the parameters and buffers ``unit`` owns: what offloading it would move.

    Offloading a module moves all of its parameters and buffers, so a module owns all of them; a path owns what it
    would wrap. ``nodes`` are the unit's nodes from ``resolve_unit``.
    """
    items = unit_items(unit)
    owned = {id(p) for item in items if isinstance(item, nn.Module) for p in (*item.parameters(), *item.buffers())}
    for item in items:
        if isinstance(item, str):
            owned |= set().union(*(_node_tensors(tree, n) for n in nodes if n.path == item))
    return owned


def _node_tensors(tree: CallTree, node: CallNode) -> set[int]:
    """The ids of the parameters and buffers a node would wrap.

    A ``forward`` node owns all of its module's parameters and buffers, because offloading moves all of them, even
    the ones this ``forward`` did not run. Other methods own their module's direct tensors, which they may use
    without a module call (``F.linear(x, self.weight)``), plus the union of what they call.
    """
    direct = (
        {id(t) for t in (*node.module.parameters(recurse=False), *node.module.buffers(recurse=False))}
        if node.module is not None
        else set()
    )
    if node.method == "forward" and node.module is not None:
        return {id(t) for t in (*node.module.parameters(), *node.module.buffers())}
    out = set(direct)
    for key in node.children:
        out |= _node_tensors(tree, tree.nodes[key])
    return out
