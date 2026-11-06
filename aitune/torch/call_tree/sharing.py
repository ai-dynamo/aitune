# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared weights: weights that tuning a unit would offload while something else still needs them.

The analysis works on the weights a unit *owns* (what tuning would offload: the module it wraps), not only on the
weights it uses. A weight is shared if code outside the unit reads it or looks up its device (recorded by
``WeightAccessMode`` while the call tree records), or if code outside the unit moves the unit's modules
(``nn.Module._apply``: ``.to()``, ``.cuda()``). Weights with the same readers, device lookups and movers form one
``SharedWeights`` group.
"""

from dataclasses import dataclass, field

from torch import nn

from aitune.torch.call_tree.call_tree import CallNode, CallTree, NodeKey, UnitSpec, key_path, short_label

UNRECORDED_PREFIX = "<unrecorded: "


@dataclass
class SharedWeights:
    """A group of weights that tuning their holder would offload while something else still needs them.

    Attributes:
        params: The weights: the path of a module with several parameters, all in the group, else parameter names.
        bytes: Memory of the weights, counted once.
        held_by: The unit that holds the weights: a module's object path or a node path.
        read_by: Paths of module calls outside the unit that read the weights.
        device_by: Paths of calls outside the unit that looked up a weight's device (``.device``, ``is_cuda``);
            a ``meta`` weight reports the wrong device, so these must stay as well.
        moved_by: Code outside the unit that moves them with ``.to()`` or ``.cuda()``: a call path, or the qualified
            name of code that is not recorded (``SentenceTransformer.encode``).
    """

    params: list[str] = field(default_factory=list)
    bytes: int = 0
    held_by: list[str] = field(default_factory=list)
    read_by: list[str] = field(default_factory=list)
    device_by: list[str] = field(default_factory=list)
    moved_by: list[str] = field(default_factory=list)


def shared_weights_of(tree: CallTree, unit: UnitSpec) -> list[SharedWeights]:
    """The groups of weights ``unit`` holds that something outside it still uses or moves; see ``CallTree``."""
    return [group for group, _ in shared_weight_groups(tree, unit)]


def shared_weight_groups(tree: CallTree, unit: UnitSpec) -> list[tuple[SharedWeights, set[int]]]:
    """Like ``shared_weights_of``, with the ids of each group's parameters, for keeping them on the device.

    Raises:
        ValueError: If ``unit`` matches no recorded call.
        TypeError: If ``unit`` is neither a node path nor a module, nor a list of them.
    """
    nodes = resolve_unit(tree, unit)
    name = unit_name(tree, unit)
    items = _items(unit)
    # Offloading a module moves all of its parameters and buffers, so a module owns all of them; a path owns what
    # it would wrap.
    owned = {id(p) for item in items if isinstance(item, nn.Module) for p in (*item.parameters(), *item.buffers())}
    for item in items:
        if isinstance(item, str):
            owned |= set().union(*(owned_tensors(tree, n) for n in nodes if n.path == item))
    subtree = [m for n in nodes for m in (n, *tree.descendants(n))]
    modules = {id(m): m for item in items if isinstance(item, nn.Module) for m in item.modules()}
    modules.update({id(m): m for n in subtree if n.module is not None for m in n.module.modules()})
    readers, devices = _accesses(tree, nodes, owned)
    movers = _movers(tree, subtree, modules, owned)

    groups: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], set[int]] = {}
    for p in owned:
        read = tuple(sorted(readers.get(p, ())))
        device = tuple(sorted(devices.get(p, ())))
        moved = tuple(sorted(movers.get(p, ())))
        if read or device or moved:
            groups.setdefault((read, device, moved), set()).add(p)
    sizes = {id(t): t.numel() * t.element_size() for _, m in tree.modules for t in (*m.parameters(), *m.buffers())}
    paths = _ParamPaths(tree)
    out = [
        (
            SharedWeights(
                params=paths.describe(params),
                bytes=sum(sizes.get(p, 0) for p in params),
                held_by=[name],
                read_by=list(read),
                device_by=list(device),
                moved_by=list(moved),
            ),
            params,
        )
        for (read, device, moved), params in groups.items()
    ]
    return sorted(out, key=lambda item: item[0].params)


def resolve_unit(tree: CallTree, unit: UnitSpec) -> list[CallNode]:
    """The recorded nodes of ``unit``: a node path, a module, or a list of them.

    A module stands for all its recorded ``forward`` calls. A module without one (an exact ``nn.Sequential`` the
    recorder skips, or a ``forward`` that takes a callable argument) stands for the outermost recorded calls of the
    module and its submodules.
    """
    items = _items(unit)
    if not items:
        raise ValueError("The unit names no calls")
    by_path = {n.path: n for n in tree.nodes.values()}
    nodes: list[CallNode] = []
    for item in items:
        if isinstance(item, nn.Module):
            matches = [n for n in tree.nodes.values() if n.module is item and n.method == "forward"]
            if not matches:
                matches = _outermost_inside(tree, item)
            if not matches:
                raise ValueError(f"Module {type(item).__name__} has no recorded calls")
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


def _items(unit: UnitSpec) -> list:
    return [unit] if isinstance(unit, (str, nn.Module)) else list(unit)


def _outermost_inside(tree: CallTree, module: nn.Module) -> list[CallNode]:
    """Recorded calls of ``module`` or its submodules whose callers are outside it, in first-call order."""
    inside = {id(m) for m in module.modules()}
    keys = {k for k, n in tree.nodes.items() if n.module is not None and id(n.module) in inside}
    return sorted(
        (tree.nodes[k] for k in keys if not any(k[:i] in keys for i in range(1, len(k)))), key=lambda n: n.first_seen
    )


def unit_name(tree: CallTree, unit: UnitSpec) -> str:
    """How ``held_by`` names a unit: a module's object path (``root`` for the inspected module), or the node path."""
    items = _items(unit)
    module_paths = {id(m): short_label(prefix) for prefix, m in tree.modules}
    return " + ".join(module_paths.get(id(i), type(i).__name__) if isinstance(i, nn.Module) else str(i) for i in items)


def owned_tensors(tree: CallTree, node: CallNode) -> set[int]:
    """Return the ids of the parameters and buffers of the module(s) a unit would wrap.

    A ``forward`` node owns its module's tensors, except a ``forward`` that delegates to one other module: it owns
    the delegate's tensors plus its own direct ones (a container that runs one of several children owns only the
    part it runs). Other methods own their module's direct tensors, which they may use without a module call
    (``F.linear(x, self.weight)``), plus the union of what they call.
    """
    direct = (
        {id(t) for t in (*node.module.parameters(recurse=False), *node.module.buffers(recurse=False))}
        if node.module is not None
        else set()
    )
    if node.method == "forward" and node.module is not None:
        forward_kids = [tree.nodes[k] for k in node.children if tree.nodes[k].method == "forward"]
        if len(forward_kids) == 1 and forward_kids[0].module is not node.module:
            return direct | owned_tensors(tree, forward_kids[0])
        return {id(t) for t in (*node.module.parameters(), *node.module.buffers())}
    out = set(direct)
    for key in node.children:
        out |= owned_tensors(tree, tree.nodes[key])
    return out


def _accesses(
    tree: CallTree, nodes: list[CallNode], owned: set[int]
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """For each owned weight, the calls outside the unit that read it or looked up its device (recorded ops).

    An access inside the unit — some unit node key prefixes the access's node key — does not count; an access from
    an ancestor (a caller of the unit, e.g. its forward doing ``F.linear(x, unit.weight)``) is outside and counts.
    """
    unit_keys = {n.key for n in nodes}
    readers: dict[int, set[str]] = {}
    devices: dict[int, set[str]] = {}
    for tensor_id, records in tree.weight_accesses.items():
        if tensor_id not in owned:
            continue
        for node_key, kind, caller in records:
            if any(node_key[: len(k)] == k for k in unit_keys):
                continue  # inside the unit
            name = key_path(node_key) if node_key else _code_name(caller)
            if kind == "device":
                devices.setdefault(tensor_id, set()).add(name)
            else:
                readers.setdefault(tensor_id, set()).add(name)
    for tensor_id, paths in _opaque_reads(tree, unit_keys, owned).items():
        readers.setdefault(tensor_id, set()).update(paths)
    return readers, devices


def _opaque_reads(tree: CallTree, unit_keys: set[NodeKey], owned: set[int]) -> dict[int, set[str]]:
    """Owned tensors held by an outside module whose recorded call showed no op on any of its own tensors.

    Such a call most likely passed them to code torch functions do not see, such as a C++ or CUDA extension
    (TransformerEngine, apex). Treating the call as a read of everything it holds keeps those tensors; a call with
    recorded ops on its tensors still counts only what it was seen to read.
    """
    seen: dict[NodeKey, set[int]] = {}
    for tensor_id, records in tree.weight_accesses.items():
        for node_key, _, _ in records:
            seen.setdefault(node_key, set()).add(tensor_id)
    out: dict[int, set[str]] = {}
    for node in tree.nodes.values():
        if node.module is None or node.method != "forward" or any(node.key[: len(k)] == k for k in unit_keys):
            continue
        direct = {id(t) for t in (*node.module.parameters(recurse=False), *node.module.buffers(recurse=False))}
        if direct & owned and not direct & seen.get(node.key, set()):
            for tensor_id in direct & owned:
                out.setdefault(tensor_id, set()).add(node.path)
    return out


def _movers(
    tree: CallTree, subtree: list[CallNode], modules: dict[int, nn.Module], owned: set[int]
) -> dict[int, set[str]]:
    """For each owned weight, the code outside the unit that moved the unit's modules."""
    movers: dict[int, set[str]] = {}
    labels = {n.label for n in subtree}
    for module_id, module in modules.items():
        for caller in tree.external_apply.get(module_id, set()) - labels:
            for param in module.parameters(recurse=False):
                if id(param) in owned:
                    movers.setdefault(id(param), set()).add(_code_name(caller))
    return movers


def _code_name(label: str) -> str:
    """A recorded call label without ``root.``, or the qualified name of unrecorded code."""
    if label.startswith(UNRECORDED_PREFIX):
        return label[len(UNRECORDED_PREFIX) : -1]
    return short_label(label)


class _ParamPaths:
    """Write a set of parameters as module paths where a whole module is in the set, else as parameter names."""

    def __init__(self, tree: CallTree):
        self.names = _param_names(tree)
        self.modules = sorted(
            ((short_label(prefix), {id(p) for p in m.parameters()}) for prefix, m in tree.modules if "." in prefix),
            key=lambda item: item[0].count("."),
        )

    def describe(self, params: set[int]) -> list[str]:
        left, out = set(params), []
        for path, module_params in self.modules:
            if len(module_params) > 1 and module_params <= left:
                out.append(path)
                left -= module_params
        out += [self.names.get(p, f"<param {p}>") for p in left]
        return sorted(out)


def _param_names(tree: CallTree) -> dict[int, str]:
    names: dict[int, str] = {}
    for prefix, module in tree.modules:
        base = short_label(prefix) if "." in prefix else ""
        for name, tensor in (*module.named_parameters(), *module.named_buffers()):
            names.setdefault(id(tensor), f"{base}.{name}" if base else name)
    return names
