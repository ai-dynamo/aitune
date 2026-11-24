# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structure pass's call recorder.

The recorder builds a context-keyed call tree. It installs no hooks on modules: each inspection mode (an adapter)
wraps the methods it finds and runs them through the active recorder's ``call``::

    recorder = CallRecorder.active()
    if recorder is None:
        return original(*args, **kwargs)
    return recorder.call(original, args, kwargs, label=label, obj=obj, method=name)

The only hook the recorder owns is a process-wide probe on ``nn.Module._apply`` (``.to()``, ``.cuda()``, ...), active
inside a ``with`` block, which records no times. A second pass with ``TimingRecorder`` times the recorded
calls without that probe or the weight-access mode.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import weakref
from collections import UserDict
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallNode, CallTree
from aitune.torch.call_tree.callers import unrecorded_caller
from aitune.torch.call_tree.methods import is_compiled_call
from aitune.torch.call_tree.nodes import CallPath, CallSignature, Leaf, TensorLeaf, _Source
from aitune.torch.call_tree.recording.call_recorder import NO_RESULT, CallRecorder
from aitune.torch.call_tree.weights.accesses import WeightAccess, WeightAccessLog, WeightAccessMode, storage_key
from aitune.torch.module.forward_signature import ForwardSignature

# A raw signature key: the input leaves, and the output leaves or None when the call raised.
_SignatureKey = tuple[tuple[Leaf, ...], tuple[Leaf, ...] | None]


class StructureRecorder(CallRecorder):
    """Record a context-keyed call tree from begin_call/end_call callbacks: the structure pass.

    Inside a ``with`` block it installs the ``nn.Module._apply`` probe and the weight-access mode and records
    every wrapped call until the block ends; it records no times — ``TimingRecorder`` times them in a second
    pass, never inside this one. Everything restores on exit, also when the recorded code raised; repeated
    ``with`` blocks keep accumulating.

    Example:
        >>> recorder = StructureRecorder()
        >>> with recorder:
        ...     with recorder.record_sample():
        ...         outer = recorder.begin_call("root.forward", object(), "forward", ())
        ...         inner = recorder.begin_call("root.head.forward", object(), "forward", ())
        ...         recorder.end_call(inner)
        ...         recorder.end_call(outer)
        >>> sorted(recorder.result().nodes)
        [('root.forward',), ('root.forward', 'root.head.forward')]
    """

    def __init__(self) -> None:
        """Create an empty recorder."""
        self._nodes: dict[CallPath, CallNode] = {}
        # id of the called function -> (the function, kept so its id isn't reused; its ForwardSignature or None)
        self._declared_signatures: dict[int, tuple[Any, ForwardSignature | None]] = {}
        # per node: raw signature key -> its CallSignature, the same object as in the node's ``signatures``
        self._signatures: dict[CallPath, dict[_SignatureKey, CallSignature]] = {}
        self._stack: list[_Frame] = []
        # what the outermost calls of the current sample returned; inside a call, its frame's ``visible``
        self._top_visible = _SourceIndex()
        self._sequences: list[list[CallPath]] = []
        self._current_sequence: list[CallPath] | None = None
        self._apply_depth = 0
        self._original_apply: Callable[..., Any] | None = None
        self._weight_access_log = WeightAccessLog()
        self._weight_access_mode: WeightAccessMode | None = None
        self._modules: dict[int, tuple[str, nn.Module]] = {}

    def register_module(self, label_prefix: str, module: nn.Module) -> None:
        """Register an instrumented module for parameter names and sizes.

        Args:
            label_prefix: The adapter's label for the module without the method, e.g. ``root.vlm.lm_head``.
            module: The module.
        """
        self._modules.setdefault(id(module), (label_prefix, module))

    def begin_call(
        self, label: str, obj: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ) -> _Frame | None:
        """Record the start of a call.

        Args:
            label: The adapter's label of the call.
            obj: The object the method belongs to.
            method: Method name.
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.

        Returns:
            A frame to pass to ``end_call``, or None if this recorder is not the active one.
        """
        if CallRecorder.active() is not self:
            return None
        parent = self._stack[-1].compiled if self._stack else False
        parent_path = self._stack[-1].node.path if self._stack else ()
        path = (*parent_path, label)
        node = self._nodes.get(path)
        if node is None:
            node = CallNode(
                path=path,
                label=label,
                owner_class=type(obj).__name__,
                method=method,
                module=obj if isinstance(obj, nn.Module) else None,
                first_seen=len(self._nodes),
            )
            self._nodes[path] = node
            if self._stack:
                self._stack[-1].node.children.append(path)
        # a node stays timeable if any call of it ran outside a compiled part
        node.inside_compiled_part = parent if node.calls == 0 else node.inside_compiled_part and parent
        node.calls += 1
        with torch._C.DisableTorchFunction():
            values, input_bytes = _leaves_and_bytes(self._input_leaves(obj, method, args, kwargs))
        node.total_bytes_in += input_bytes
        if self._current_sequence is not None:
            self._current_sequence.append(path)
        input_key, visible = self._link_inputs(path, values)
        frame = _Frame(
            node=node,
            input_key=input_key,
            call_index=node.calls - 1,
            visible=visible,
            compiled=parent or is_compiled_call(obj, method),
        )
        self._stack.append(frame)
        return frame

    def end_call(self, frame: _Frame | None, result: Any = NO_RESULT) -> None:
        """Record the end of a call started with ``begin_call``.

        Args:
            frame: The frame ``begin_call`` returned.
            result: The call's return value, to count its tensor bytes; omit it when the call raised.
        """
        if frame is None:
            return
        if self._stack and self._stack[-1] is frame:
            self._stack.pop()
        if result is NO_RESULT:
            output_key = None
        else:
            with torch._C.DisableTorchFunction():
                values, output_bytes = _leaves_and_bytes(list(_named_leaves(result, "output")))
            frame.node.total_bytes_out += output_bytes
            output_key = self._link_outputs(frame.node.path, values, frame.visible)
        signatures = self._signatures.setdefault(frame.node.path, {})
        key = (frame.input_key, output_key)
        signature = signatures.get(key)
        if signature is None:
            signature = signatures[key] = _signature(key)
            frame.node.signatures.append(signature)
        signature.call_indices.append(frame.call_index)

    @contextlib.contextmanager
    def record_sample(self) -> Generator[None]:
        """Record one sample (one call of the inference function) inside the ``with`` block.

        An exception inside the block drops the sample it was recording: its sequence is not appended.
        """
        self._current_sequence = []
        self._top_visible = _SourceIndex()  # a new top level: nothing links across samples
        try:
            yield
        except BaseException:
            self._current_sequence = None
            raise
        self._sequences.append(self._current_sequence)
        self._current_sequence = None

    def result(self) -> CallTree:
        """Return the recorded call tree.

        The returned tree is a snapshot — a ``CallTree.copy()`` of the recorder's state — so later recording
        does not change it.
        """
        return CallTree(
            nodes=self._nodes,
            sequences=self._sequences,
            weight_accesses=self._weight_access_log.snapshot(),
            modules=list(self._modules.values()),
        ).copy()

    def _start(self) -> None:
        """Install the ``nn.Module._apply`` probe and the weight-access mode."""
        self._original_apply = nn.Module._apply
        try:
            nn.Module._apply = self._make_apply_probe(self._original_apply)
            self._weight_access_log.start(self._modules.values())
            weight_access_mode = WeightAccessMode(
                self._weight_access_log, lambda: self._stack[-1].node.path_str if self._stack else ""
            )
            weight_access_mode.__enter__()
            self._weight_access_mode = weight_access_mode
        except BaseException:
            self._stop(completed=False)
            raise

    def _stop(self, completed: bool) -> None:
        """End the structure pass and restore ``nn.Module._apply``, also when the recorded code raised."""
        self._stack.clear()
        self._current_sequence = None
        if self._weight_access_mode is not None:
            self._weight_access_mode.__exit__(None, None, None)
            self._weight_access_mode = None
        if self._original_apply is not None:
            nn.Module._apply = self._original_apply
            self._original_apply = None

    def _input_leaves(
        self, obj: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None
    ) -> list[tuple[str, Any]]:
        """The call's arguments bound to parameter names, walked into ``(name, leaf)`` pairs."""
        bound = getattr(obj, method, None)
        function = getattr(bound, "__func__", bound)  # per instance when the instance replaced the method
        entry = self._declared_signatures.get(id(function))
        if entry is None:
            try:
                declared = ForwardSignature.from_callable(bound)
            except (TypeError, ValueError):
                declared = None
            entry = self._declared_signatures[id(function)] = (function, declared)
        declared = entry[1]
        arguments = {"args": args, "kwargs": kwargs or {}}
        if declared is not None:
            with contextlib.suppress(TypeError, ValueError):
                arguments = dict(declared.normalize(args, kwargs or {}).arguments)
        seen: set[int] = set()  # container ids seen in this call's input walk
        return [leaf for parameter, value in arguments.items() for leaf in _named_leaves(value, parameter, seen)]

    def _link_inputs(self, path: CallPath, values: list[tuple[Any, Leaf]]) -> tuple[tuple[Leaf, ...], _SourceIndex]:
        """Source a call's arguments from what its caller sees; return its input signature key and what it sees.

        An argument visible in the caller gets ``source`` set to where it was last seen: one of the caller's
        arguments or what one of the caller's earlier child calls returned. The call's arguments then become visible
        inside it.
        """
        caller = self._visible_in_caller()
        own = _SourceIndex()
        leaves = []
        for obj, leaf in values:
            source = caller.source_of(obj)
            if source is not None:
                leaf = replace(leaf, source=source)
            own.add(obj, _Source(path, "inputs", leaf.name))
            leaves.append(leaf)
        return tuple(leaves), own

    def _link_outputs(self, path: CallPath, values: list[tuple[Any, Leaf]], own: _SourceIndex) -> tuple[Leaf, ...]:
        """Source a call's results from what it saw (``own``), make them visible in the caller; return the output key.

        A result visible inside the call gets ``source`` set to where it was: one of the call's arguments or what one
        of its child calls returned.
        """
        caller = self._visible_in_caller()
        leaves = []
        for obj, leaf in values:
            source = own.source_of(obj)
            if source is not None:
                leaf = replace(leaf, source=source)
            caller.add(obj, _Source(path, "outputs", leaf.name))
            leaves.append(leaf)
        return tuple(leaves)

    def _visible_in_caller(self) -> _SourceIndex:
        """What the caller of a starting or ending call sees: the innermost open call's ``visible`` or the top level."""
        return self._stack[-1].visible if self._stack else self._top_visible

    def _make_apply_probe(self, original_apply: Callable) -> Callable:
        """Build the replacement for ``nn.Module._apply`` that ``_start`` installs.

        ``nn.Module._apply`` is the method that ``.to()``, ``.cuda()``, ``.half()`` and similar calls use to convert a
        module's parameters and buffers. The replacement runs ``original_apply`` unchanged. While the recorder is
        recording, it first records a ``"move"`` access on every tracked weight of the module and, unless
        ``recurse=False``, of all its submodules: the innermost recorded call's path, or the code found by
        ``unrecorded_caller`` if no recorded call is running. Only the outermost ``_apply`` of a conversion records,
        because ``_apply`` calls itself on each child. Recording goes through the storage map before
        ``original_apply`` runs, so a tied alias outside the moved subtree is moved as well, and the storage reads
        themselves do not count as accesses.

        Args:
            original_apply: The ``nn.Module._apply`` to call; restored when the ``with`` block ends.

        Returns:
            A function with the signature of ``nn.Module._apply``.
        """
        recorder = self

        def _apply(module: nn.Module, fn: Callable, *args: Any, **kwargs: Any) -> Any:
            # ``nn.Module._apply`` recurses through child ``_apply`` calls; only the outermost one records.
            if CallRecorder.active() is recorder and recorder._apply_depth == 0:
                caller = recorder._stack[-1].node.path_str if recorder._stack else unrecorded_caller()
                recurse = kwargs.get("recurse", args[0] if args else True)
                with torch._C.DisableTorchFunction():
                    for submodule in module.modules() if recurse else [module]:
                        for weight in (*submodule.parameters(recurse=False), *submodule.buffers(recurse=False)):
                            key = storage_key(weight)
                            if recorder._weight_access_log.lookup(weight, key):
                                recorder._weight_access_log.record(key, WeightAccess(caller, "move"))
            recorder._apply_depth += 1
            try:
                if recorder._weight_access_mode is not None:
                    with recorder._weight_access_mode.paused():
                        return original_apply(module, fn, *args, **kwargs)
                return original_apply(module, fn, *args, **kwargs)
            finally:
                recorder._apply_depth -= 1
                if (
                    recorder._apply_depth == 0
                    and CallRecorder.active() is recorder
                    and recorder._weight_access_mode is not None
                ):
                    # A move or cast gives the tensors new storage; keep tracking them under it.
                    recorder._weight_access_log.retrack()

        return _apply


def _named_leaves(value: Any, root: str, seen: set[int] | None = None) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, leaf)`` for every leaf inside ``value``, with ``name`` built on ``root``.

    Leaves are tensors and every other value a container does not cover. ``seen`` holds the ids of the containers
    already walked — the caller shares one set across a call's arguments — so a container reached again, whether
    by a cycle or shared as another argument, is skipped: each container is walked once, under its first name.

    The walk covers the containers pytree registers (lists, tuples, dicts, namedtuples, ``ModelOutput``, ...) and
    two kinds that pytree does not flatten:

    - ``UserDict``s, such as ``transformers.BatchFeature``, via their ``.data``: it is a plain dict, so the walk
      runs no code of the mapping's class.
    - Dataclass instances, via the values of their fields (``dataclasses.fields``), read with ``getattr``, which
      runs no code of the class unless a property shadows the field; a field without a value, such as an unset
      slot, is skipped. Dataclasses that pytree flattens itself, such as ``ModelOutput``, are walked by pytree.

    Everything else is a leaf and not looked into. The suffixes follow pytree's ``keystr`` style: ``x[0]`` for
    sequences, ``x['key']`` for mappings, ``x.field`` for namedtuples and dataclass fields. pytree's
    ``tree_flatten_with_path`` isn't used because transformers registers ``ModelOutput`` without a
    ``flatten_with_keys_fn``, so it raises for model outputs.

    Example:
        >>> t = torch.zeros(2)
        >>> [(name, type(leaf).__name__) for name, leaf in _named_leaves({"ids": t, "extra": [t, "x"]}, "inputs")]
        [("inputs['ids']", 'Tensor'), ("inputs['extra'][0]", 'Tensor'), ("inputs['extra'][1]", 'str')]
    """
    seen = set() if seen is None else seen
    stack = [(root, value)]
    while stack:
        name, obj = stack.pop()
        # Only containers are walked once; a tensor or other leaf reached under two names is yielded under both.
        if torch.is_tensor(obj):
            yield name, obj
            continue
        children = _children(obj)
        if children is None:
            yield name, obj  # a non-tensor leaf
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        # Pushed reversed, so the stack pops the children in their own order.
        stack.extend((name + suffix, child) for suffix, child in reversed(children))


def _children(obj: Any) -> list[tuple[str, Any]] | None:
    """The direct children of ``obj`` as ``(suffix, child)`` pairs, or None when ``obj`` is a leaf.

    ``suffix`` is appended to the container's name to name the child: the child at ``[0]`` of ``x`` is ``x[0]``. The
    suffixes depend on the kind of container:

    - ``UserDict``: its ``.data``, a plain dict, under an empty suffix, so its entries are named like a dict's.
    - A container pytree registers with keys (lists, tuples, dicts, namedtuples): pytree's keys, such as ``[0]``,
      ``['key']`` or ``.field``.
    - A container pytree registers without keys, such as transformers' ``ModelOutput``: see ``_suffixed_children``.
    - A dataclass that pytree does not register: its fields, as ``.field`` (see ``_field_values``).

    Anything else is a leaf.

    Example:
        >>> [suffix for suffix, _ in _children([1, 2])], [suffix for suffix, _ in _children({"a": 1})]
        (['[0]', '[1]'], ["['a']"])
        >>> _children(object()) is None
        True
    """
    if isinstance(obj, UserDict):
        return [("", obj.data)]
    node_def = torch.utils._pytree.SUPPORTED_NODES.get(torch.utils._pytree._get_node_type(obj))
    if node_def is None:
        return _field_values(obj) if dataclasses.is_dataclass(obj) and not isinstance(obj, type) else None
    if node_def.flatten_with_keys_fn is not None:
        return [(str(key), child) for key, child in node_def.flatten_with_keys_fn(obj)[0]]
    return _suffixed_children(*node_def.flatten_fn(obj))


def _suffixed_children(flat: list[Any], context: Any) -> list[tuple[str, Any]]:
    """The ``(suffix, child)`` pairs of a container that pytree registers without keys.

    ``flat`` and ``context`` are what pytree's ``flatten_fn`` returns for the container: its children, and what
    pytree needs to rebuild it. When ``context`` is a list of distinct strings, one per child, those strings name the
    children like a dict's entries. That is how transformers registers ``ModelOutput``, with its field names, so its
    children get suffixes such as ``['logits']``. Any other context gives positional suffixes: ``[0]``, ``[1]``, ...

    Example:
        >>> _suffixed_children([1, 2], ["logits", "hidden_states"])
        [("['logits']", 1), ("['hidden_states']", 2)]
        >>> _suffixed_children([1, 2], None)
        [('[0]', 1), ('[1]', 2)]
    """
    if (
        isinstance(context, (list, tuple))
        and len(context) == len(flat)
        and all(isinstance(k, str) for k in context)
        and len(set(context)) == len(context)
    ):
        return [(f"[{key!r}]", child) for key, child in zip(context, flat, strict=True)]
    return [(f"[{i}]", child) for i, child in enumerate(flat)]


def _field_values(instance: Any) -> list[tuple[str, Any]]:
    """The ``(.field, value)`` pairs of a dataclass instance's fields, skipping fields that have no value."""
    values = []
    for field in dataclasses.fields(instance):
        with contextlib.suppress(AttributeError):
            values.append((f".{field.name}", getattr(instance, field.name)))
    return values


# Types left out of signatures entirely: Python's scalars, strings and enums, and torch's ``dtype``, ``device``,
# ``Size``, ``layout`` and ``memory_format``. Their values are immutable, compared by value and hold no tensor data,
# so their type alone says nothing about a call. Recording their values instead is open in #184.
_PRIMITIVES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    enum.Enum,
    torch.dtype,
    torch.device,
    torch.Size,
    torch.layout,
    torch.memory_format,
)


def _leaves_and_bytes(leaves: list[tuple[str, Any]]) -> tuple[list[tuple[Any, Leaf]], int]:
    """The walked leaves as ``(object, Leaf)`` pairs, and their tensor byte total, in one pass.

    There is a named ``Leaf`` for each tensor and non-primitive object, in walk order. ``_link_inputs`` and
    ``_link_outputs`` set their sources and build the raw signature key from them. Each tensor object counts its
    bytes once. Callers run it with torch functions disabled so the metadata reads don't go through the
    weight-access mode.
    """
    values: list[tuple[Any, Leaf]] = []
    seen: set[int] = set()
    total = 0
    for name, obj in leaves:
        if torch.is_tensor(obj):
            values.append((obj, TensorLeaf(name, _class_name(type(obj)), obj.dtype, tuple(obj.shape))))
            if id(obj) not in seen:
                seen.add(id(obj))
                total += obj.numel() * obj.element_size()
        elif not isinstance(obj, _PRIMITIVES):
            values.append((obj, Leaf(name, _class_name(type(obj)))))
    return values, total


class _SourceIndex:
    """An index of the values visible inside one call, or at the top level of a sample: where each was last seen.

    Values are matched by identity. An entry is keyed by the object's ``id`` and holds a weak reference to it: the
    reference confirms that the id still belongs to the same object, because CPython gives a freed object's id to
    new objects, and it keeps recording from holding any value alive. Values that don't support weak references
    are not tracked.

    Example:
        >>> index = _SourceIndex()
        >>> x = torch.ones(2)
        >>> index.add(x, _Source(("root.forward",), "inputs", "x"))
        >>> index.source_of(x)
        _Source(path=('root.forward',), side='inputs', name='x')
        >>> index.source_of(torch.ones(2)) is None  # an equal tensor, but another object
        True
        >>> del x
        >>> index.source_of(torch.ones(2)) is None  # even if the new tensor got the freed one's id
        True
        >>> size = (2, 3)  # a tuple doesn't support weak references
        >>> index.add(size, _Source(("root.forward",), "inputs", "size"))
        >>> index.source_of(size) is None
        True
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        """Start with no visible values."""
        self._entries: dict[int, tuple[weakref.ref[Any], _Source]] = {}

    def add(self, obj: Any, source: _Source) -> None:
        """Make ``obj`` visible here as last seen at ``source``, replacing an earlier entry for the same object."""
        if type(obj).__weakrefoffset__:
            self._entries[id(obj)] = (weakref.ref(obj), source)

    def source_of(self, obj: Any) -> _Source | None:
        """Where ``obj`` was last seen here, or None if it wasn't."""
        entry = self._entries.get(id(obj))
        if entry is None or entry[0]() is not obj:
            return None
        return entry[1]


@functools.cache
def _class_name(cls: type) -> str:
    """A class's signature type string: builtin ``__qualname__``, else ``module.QualName``."""
    return cls.__qualname__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"


def _signature(key: _SignatureKey) -> CallSignature:
    """A new ``CallSignature`` for one raw signature key, with no calls yet: ``outputs`` None when the calls raised."""
    input_key, output_key = key
    return CallSignature(
        inputs={leaf.name: leaf for leaf in input_key},
        outputs=None if output_key is None else {leaf.name: leaf for leaf in output_key},
        call_indices=[],
    )


@dataclass
class _Frame:
    """An open call: its node, the raw key of its inputs, its call index and what it sees.

    ``call_index`` is the call's position among the node's calls in start order (the same order
    ``CallTree.sequences`` records). ``visible`` holds the values visible inside the call: its own
    arguments and what its child calls returned. ``compiled`` says whether calls inside this call run compiled.
    """

    node: CallNode
    input_key: tuple[Leaf, ...]
    call_index: int
    visible: _SourceIndex
    compiled: bool
