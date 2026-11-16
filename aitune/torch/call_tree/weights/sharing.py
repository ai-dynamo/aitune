# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared weights: weights that tuning a unit would offload while something else still needs them.

The analysis works on the weights a unit *owns* (what tuning would offload: the module it wraps), not only on the
weights it uses. A weight — a parameter or a buffer — is shared if code outside the unit reads it or looks up its
device (recorded by ``WeightAccessMode`` while the call tree records), or if code outside the unit moves the unit's
modules (``nn.Module._apply``: ``.to()``, ``.cuda()``, recorded by the recorder's ``_apply`` probe). Reported as one
``SharedWeight`` entry per weight, sorted by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

import torch

from aitune.torch.call_tree.callers import UNRECORDED_PREFIX
from aitune.torch.call_tree.nodes import CallNode
from aitune.torch.call_tree.units import UnitSpec, owned_tensors, resolve_unit
from aitune.torch.call_tree.weights.accesses import WeightAccess, named_weights

if TYPE_CHECKING:
    from aitune.torch.call_tree.call_tree import CallTree

logger = getLogger(__name__)


@dataclass
class SharedWeight:
    """A parameter or buffer a unit holds that code outside the unit still reads, looks up the device of, or moves.

    Attributes:
        name: The weight's name from the root, e.g. ``llm.embed.weight``; a tied weight is named by its first name.
        aliases: Its other names, sorted (a tied ``lm_head.weight``); empty if it has one name.
        bytes: ``numel * element_size``. Views of one storage are each counted in full.
        accesses: The accesses from outside the unit, sorted by ``(by, kind)``: reads, device lookups and moves.
            Unrecorded code is reported by its qualified name (``SentenceTransformer.encode``).
    """

    name: str
    aliases: list[str] = field(default_factory=list)
    bytes: int = 0
    accesses: list[WeightAccess] = field(default_factory=list)


def shared_weights_of(tree: CallTree, unit: UnitSpec) -> list[SharedWeight]:
    """The weights ``unit`` holds that something outside it still uses or moves.

    This is what tuning ``unit`` alone and offloading its weights would break: code outside it that reads one of its
    weights (a tied head, ``F.linear(x, unit.weight)`` in a caller's forward), code that looks up a weight's device,
    or code outside it that moves its modules with ``.to()`` or ``.cuda()``. Reads and device lookups come from the
    ops recorded by ``WeightAccessMode`` during the call-tree pass; moves are recorded by the recorder's
    ``nn.Module._apply`` probe.

    Args:
        tree: The call tree from ``ait.inspect(..., call_tree=True)``.
        unit: A node path (``"forward > dit_forward"``, see ``CallNode.path``), a module (all its recorded
            ``forward`` calls), or a list of them as one unit.

    Returns:
        One entry per weight, sorted by name; empty if the unit can be tuned and offloaded on its own.

    Raises:
        ValueError: If ``unit`` is empty, names a path with no recorded call, or is a module that is not part of
            the recorded model.
        TypeError: If ``unit`` is neither a node path nor a module, nor a list of them.
    """
    return [weight for weight, _ in shared_weights_with_ids(tree, unit)]


def shared_weights_with_ids(tree: CallTree, unit: UnitSpec) -> list[tuple[SharedWeight, int]]:
    """Like ``shared_weights_of``, paired with the id of each weight's live tensor for keeping it on the device.

    Raises:
        ValueError: If ``unit`` is empty, names a path with no recorded call, or is a module that is not part of
            the recorded model.
        TypeError: If ``unit`` is neither a node path nor a module, nor a list of them.
    """
    nodes = resolve_unit(tree, unit)
    owned = owned_tensors(tree, unit, nodes)
    name_to_tensor, id_to_names = _weight_index(tree)
    missing = set(tree.weight_accesses) - name_to_tensor.keys()
    if missing:
        logger.debug("Recorded weight accesses for names no longer present: %s", sorted(missing))
    accesses = _accesses(tree, nodes, owned, name_to_tensor)

    tensors = {id(t): t for _, m in tree.modules for t in (*m.parameters(), *m.buffers())}
    out = []
    for p in owned:
        if p not in accesses:
            continue
        name, *aliases = id_to_names.get(p, [f"<param {p}>"])
        tensor = tensors.get(p)
        out.append((
            SharedWeight(
                name=name,
                aliases=sorted(aliases),
                bytes=tensor.numel() * tensor.element_size() if tensor is not None else 0,
                accesses=sorted(accesses[p], key=lambda a: (a.by, a.kind)),
            ),
            p,
        ))
    return sorted(out, key=lambda item: item[0].name)


def _weight_index(tree: CallTree) -> tuple[dict[str, torch.Tensor], dict[int, list[str]]]:
    """The live weights of the tree's registered modules: by name, and each tensor id to its names, first name first.

    Built at query time, so the tensors are the current ones even if ``.to()`` or a cast replaced them since the
    recording. First-name order is parameters then buffers per module, in ``tree.modules`` order.
    """
    name_to_tensor: dict[str, torch.Tensor] = {}
    id_to_names: dict[int, list[str]] = {}
    for prefix, module in tree.modules:
        for name, tensor in named_weights(prefix, module):
            name_to_tensor[name] = tensor
            names = id_to_names.setdefault(id(tensor), [])
            if name not in names:
                names.append(name)
    return name_to_tensor, id_to_names


def _accesses(
    tree: CallTree, nodes: list[CallNode], owned: set[int], name_to_tensor: dict[str, torch.Tensor]
) -> dict[int, set[WeightAccess]]:
    """For each owned weight, the accesses from outside the unit.

    An access inside the unit — ``by`` is a unit node's path or below one — does not count; an access from an
    ancestor (a caller of the unit, e.g. its forward doing ``F.linear(x, unit.weight)``) is outside and counts.
    Unrecorded and outside-any-call markers start with ``<`` so they never match a unit path.
    """
    unit_paths = {n.path for n in nodes}
    out: dict[int, set[WeightAccess]] = {}
    for weight_name, records in tree.weight_accesses.items():
        tensor = name_to_tensor.get(weight_name)
        if tensor is None:
            continue  # name no longer present (the weight was removed or renamed after recording)
        tensor_id = id(tensor)
        if tensor_id not in owned:
            continue
        for access in records:
            if any(access.by == p or access.by.startswith(p + " > ") for p in unit_paths):
                continue  # inside the unit
            out.setdefault(tensor_id, set()).add(WeightAccess(_code_name(access.by), access.kind))
    for tensor_id, paths in _opaque_reads(tree, nodes, owned).items():
        for path in paths:
            out.setdefault(tensor_id, set()).add(WeightAccess(path, "read"))
    return out


def _opaque_reads(tree: CallTree, nodes: list[CallNode], owned: set[int]) -> dict[int, set[str]]:
    """Owned tensors held directly by an outside module whose ``forward`` ran, as reads of that call.

    The call may pass them to code torch functions do not see, such as a C++ or CUDA extension (TransformerEngine,
    apex), even when it touches its other tensors with visible ops. Counting them keeps those tensors;
    over-reporting only keeps more memory. A call of the unit's own module that leads into the unit (its
    ``forward`` around a method unit) is not a separate holder.
    """
    unit_keys = {n.key for n in nodes}
    out: dict[int, set[str]] = {}
    for node in tree.nodes.values():
        if (
            node.module is None
            or node.method != "forward"
            or any(node.key[: len(k)] == k for k in unit_keys)
            or any(u.key[: len(node.key)] == node.key and u.module is node.module for u in nodes)
        ):
            continue
        direct = {id(t) for t in (*node.module.parameters(recurse=False), *node.module.buffers(recurse=False))}
        for tensor_id in direct & owned:
            out.setdefault(tensor_id, set()).add(node.path)
    return out


def _code_name(by: str) -> str:
    """An access's ``by`` as reported: unrecorded code by its qualified name, recorded paths unchanged."""
    if by.startswith(UNRECORDED_PREFIX):
        return by[len(UNRECORDED_PREFIX) : -1]
    return by
