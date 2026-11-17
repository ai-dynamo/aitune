# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mode-independent call recorder.

The recorder builds a context-keyed call tree. It installs no hooks on modules: each inspection mode (an adapter)
wraps the methods it finds and calls ``begin_call`` and ``end_call`` around the original call::

    frame = recorder.begin_call(label, obj, name, args, kwargs)
    try:
        result = original(*args, **kwargs)
    except BaseException:
        recorder.end_call(frame)
        raise
    recorder.end_call(frame, result)
    return result

The only hook the recorder owns is a process-wide probe on ``nn.Module._apply`` (``.to()``, ``.cuda()``, ...), active
inside a ``with recorder:`` block. Timing uses CUDA events per call (no per-call synchronization) when CUDA is
available and ``time.perf_counter`` otherwise; ``flush()`` resolves the pending events once per iteration.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import time
from collections import UserDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Self

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallNode, CallTree
from aitune.torch.call_tree.callers import unrecorded_caller
from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container
from aitune.torch.call_tree.nodes import CallPath, CallSignature, Leaf, TensorLeaf
from aitune.torch.call_tree.weights.accesses import WeightAccess, WeightAccessLog, WeightAccessMode, storage_key
from aitune.torch.module.forward_signature import ForwardSignature

_NO_RESULT = object()

# A raw signature key: the input leaves, and the output leaves or None when the call raised.
_SignatureKey = tuple[tuple[Leaf, ...], tuple[Leaf, ...] | None]


class CallRecorder:
    """Record a context-keyed call tree from begin_call/end_call callbacks.

    Example:
        >>> recorder = CallRecorder()
        >>> with recorder:
        ...     recorder.begin_iteration()
        ...     outer = recorder.begin_call("root.forward", object(), "forward", ())
        ...     inner = recorder.begin_call("root.head.forward", object(), "forward", ())
        ...     recorder.end_call(inner)
        ...     recorder.end_call(outer)
        ...     recorder.end_iteration()
        >>> sorted(recorder.result().nodes)
        [('root.forward',), ('root.forward', 'root.head.forward')]
    """

    # The recorder whose ``nn.Module._apply`` probe is installed; the probe is process-wide, so only one may record.
    _recording_now: CallRecorder | None = None

    def __init__(self) -> None:
        """Create an empty recorder."""
        self._nodes: dict[CallPath, CallNode] = {}
        # id of the called function -> (the function, kept so its id isn't reused; its ForwardSignature or None)
        self._declared_signatures: dict[int, tuple[Any, ForwardSignature | None]] = {}
        # per node: raw signature key -> its CallSignature, the same object as in the node's ``signatures``
        self._signatures: dict[CallPath, dict[_SignatureKey, CallSignature]] = {}
        self._stack: list[CallNode] = []
        self._events: list[tuple[CallNode, Any, Any]] = []
        self._sequences: list[list[CallPath]] = []
        self._current_sequence: list[CallPath] | None = None
        self._apply_depth = 0
        self._original_apply: Callable[..., Any] | None = None
        self._weight_access_log = WeightAccessLog()
        self._weight_access_mode: WeightAccessMode | None = None
        self._modules: dict[int, tuple[str, nn.Module]] = {}
        self._active = False
        self._use_cuda = torch.cuda.is_available()

    def __enter__(self) -> Self:
        """Start recording: install the ``nn.Module._apply`` probe; calls are recorded until ``__exit__``."""
        if CallRecorder._recording_now is not None:
            raise RuntimeError("Another CallRecorder is already recording; the nn.Module._apply probe is process-wide.")
        self._original_apply = nn.Module._apply
        CallRecorder._recording_now = self
        try:
            nn.Module._apply = self._make_apply_probe(self._original_apply)
            self._weight_access_log.start(self._modules.values())
            weight_access_mode = WeightAccessMode(
                self._weight_access_log, lambda: self._stack[-1].path_str if self._stack else ""
            )
            weight_access_mode.__enter__()
            self._weight_access_mode = weight_access_mode
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self._active = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop recording and restore ``nn.Module._apply``, also when the recorded code raised."""
        self._active = False
        self._stack.clear()
        self._current_sequence = None
        if self._weight_access_mode is not None:
            self._weight_access_mode.__exit__(None, None, None)
            self._weight_access_mode = None
        if self._original_apply is not None:
            nn.Module._apply = self._original_apply
            self._original_apply = None
        CallRecorder._recording_now = None

    @property
    def active(self) -> bool:
        """Whether the recorder is inside its ``with`` block."""
        return self._active

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
            kwargs: Keyword arguments of the call; a callable among them makes the call transparent as well.

        Returns:
            A frame to pass to ``end_call``, or None if the call is not recorded (not recording, exact torch container,
            or a helper that runs a callable argument).
        """
        if not self._active or is_transparent_container(obj) or is_transparent_call(args, kwargs):
            return None
        parent_path = self._stack[-1].path if self._stack else ()
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
                self._stack[-1].children.append(path)
        node.calls += 1
        with torch._C.DisableTorchFunction():
            input_key, input_bytes = _key_and_bytes(self._input_leaves(obj, method, args, kwargs))
        node.bytes_in += input_bytes
        if self._current_sequence is not None:
            self._current_sequence.append(path)
        self._stack.append(node)
        start, device = self._start_timer()
        return _Frame(node=node, start=start, device=device, input_key=input_key, call_index=node.calls - 1)

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

    def end_call(self, frame: _Frame | None, result: Any = _NO_RESULT) -> None:
        """Record the end of a call started with ``begin_call``.

        Args:
            frame: The frame ``begin_call`` returned.
            result: The call's return value, to count its tensor bytes; omit it when the call raised.
        """
        if frame is None:
            return
        self._stop_timer(frame)
        if result is _NO_RESULT:
            output_key = None
        else:
            with torch._C.DisableTorchFunction():
                outputs = list(_named_leaves(result, "output"))
                output_key, output_bytes = _key_and_bytes(outputs)
            frame.node.bytes_out += output_bytes
        signatures = self._signatures.setdefault(frame.node.path, {})
        key = (frame.input_key, output_key)
        signature = signatures.get(key)
        if signature is None:
            signature = signatures[key] = _signature(key)
            frame.node.signatures.append(signature)
        signature.call_indices.append(frame.call_index)
        if self._stack and self._stack[-1] is frame.node:
            self._stack.pop()

    def begin_iteration(self) -> None:
        """Mark the start of one iteration (for example one call of the inference function)."""
        self._current_sequence = []

    def end_iteration(self) -> None:
        """Mark the end of the current iteration."""
        if self._current_sequence is not None:
            self._sequences.append(self._current_sequence)
        self._current_sequence = None

    def result(self) -> CallTree:
        """Return the recorded call tree, with times resolved to milliseconds.

        Idempotent: resolved event times accumulate on the nodes, so calling ``result`` again returns the
        same times, and recording more calls afterwards keeps accumulating. The returned tree is a snapshot: its nodes,
        their ``children`` and ``signatures`` are copies, so later recording does not change it.
        """
        self.flush()
        nodes = {
            path: replace(
                node,
                children=list(node.children),
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
            for path, node in self._nodes.items()
        }
        return CallTree(
            nodes=nodes,
            iterations=len(self._sequences),
            sequences=list(self._sequences),
            weight_accesses=self._weight_access_log.snapshot(),
            modules=list(self._modules.values()),
        )

    def flush(self) -> None:
        """Resolve pending CUDA event pairs into the accumulated totals so they are not held until the end.

        Adapters call it once per iteration; pending events stay bounded by one iteration. Each end event is
        synchronized on its own, so pairs recorded on any device resolve without a device-wide synchronization.
        """
        if not self._events:
            return
        if self._use_cuda:
            for _, _, end in self._events:
                end.synchronize()
        for node, start, end in self._events:
            node.time_ms += start.elapsed_time(end)
        self._events.clear()

    def _start_timer(self) -> tuple[Any, int | None]:
        """Start timing a call: record a CUDA event, or read ``time.perf_counter`` without CUDA.

        Returns:
            The start of the timer and the CUDA device the start event was recorded on (``None`` without CUDA).
        """
        if self._use_cuda:
            device = torch.cuda.current_device()
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event, device
        return time.perf_counter(), None

    def _stop_timer(self, frame: _Frame) -> None:
        """Stop timing a call: queue the CUDA event pair for ``flush``, or add the CPU time to the node's total."""
        if self._use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(frame.device))
            self._events.append((frame.node, frame.start, end))
        else:
            frame.node.time_ms += (time.perf_counter() - frame.start) * 1000.0

    def _make_apply_probe(self, original_apply: Callable) -> Callable:
        """Build the replacement for ``nn.Module._apply`` that ``__enter__`` installs.

        ``nn.Module._apply`` is the method that ``.to()``, ``.cuda()``, ``.half()`` and similar calls use to convert a
        module's parameters and buffers. The replacement runs ``original_apply`` unchanged. While the recorder is
        recording, it first records a ``"move"`` access on every tracked weight of the module and, unless
        ``recurse=False``, of all its submodules: the innermost recorded call's path, or the code found by
        ``unrecorded_caller`` if no recorded call is running. Only the outermost ``_apply`` of a conversion records,
        because ``_apply`` calls itself on each child. Recording goes through the storage map before
        ``original_apply`` runs, so a tied alias outside the moved subtree is moved as well, and the storage reads
        themselves do not count as accesses.

        Args:
            original_apply: The ``nn.Module._apply`` to call; restored by ``__exit__``.

        Returns:
            A function with the signature of ``nn.Module._apply``.
        """
        recorder = self

        def _apply(module: nn.Module, fn: Callable, *args: Any, **kwargs: Any) -> Any:
            # ``nn.Module._apply`` recurses through child ``_apply`` calls; only the outermost one records.
            if recorder._active and recorder._apply_depth == 0:
                caller = recorder._stack[-1].path_str if recorder._stack else unrecorded_caller()
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
                if recorder._apply_depth == 0 and recorder._active and recorder._weight_access_mode is not None:
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

    Everything else is a leaf and not looked into. The steps follow pytree's ``keystr`` style: ``x[0]`` for
    sequences, ``x['key']`` for mappings, ``x.field`` for namedtuples and dataclass fields. pytree's
    ``tree_flatten_with_path`` isn't used because transformers registers ``ModelOutput`` without a
    ``flatten_with_keys_fn``, so it raises for model outputs.
    """
    seen = set() if seen is None else seen
    stack = [(root, value)]
    while stack:
        name, obj = stack.pop()
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
        stack.extend((name + step, child) for step, child in reversed(children))


def _children(obj: Any) -> list[tuple[str, Any]] | None:
    """A container's children as ``(step, child)`` pairs for ``_named_leaves``, or None if ``obj`` is a leaf."""
    if isinstance(obj, UserDict):
        return [("", obj.data)]
    node_def = torch.utils._pytree.SUPPORTED_NODES.get(torch.utils._pytree._get_node_type(obj))
    if node_def is None:
        return _field_values(obj) if dataclasses.is_dataclass(obj) and not isinstance(obj, type) else None
    if node_def.flatten_with_keys_fn is not None:
        return [(str(key), child) for key, child in node_def.flatten_with_keys_fn(obj)[0]]
    flat, context = node_def.flatten_fn(obj)
    # transformers registers ModelOutput without keys: its context is the field names. Other
    # registrations may use non-string or duplicate contexts; those children get positional names instead.
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


def _key_and_bytes(leaves: list[tuple[str, Any]]) -> tuple[tuple[Leaf, ...], int]:
    """The raw signature key of the walked leaves and their tensor byte total in one pass.

    The key holds a named ``Leaf`` for each tensor and non-primitive object; each tensor object counts its bytes
    once. Callers run it with torch functions disabled so the metadata reads don't go through the weight-access mode.
    """
    key: list[Leaf] = []
    seen: set[int] = set()
    total = 0
    for name, obj in leaves:
        if torch.is_tensor(obj):
            key.append(TensorLeaf(name, _class_name(type(obj)), obj.dtype, tuple(obj.shape)))
            if id(obj) not in seen:
                seen.add(id(obj))
                total += obj.numel() * obj.element_size()
        elif not isinstance(obj, _PRIMITIVES):
            key.append(Leaf(name, _class_name(type(obj))))
    return tuple(key), total


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
    """An open call: its node, the start of its timer, the raw key of its inputs and its call index.

    ``call_index`` is the call's position among the node's calls in start order (the same order
    ``CallTree.sequences`` records). The start is a CUDA event or a ``perf_counter`` value; ``device`` is the
    CUDA device the start event was recorded on.
    """

    node: CallNode
    start: Any
    input_key: tuple[Leaf, ...]
    call_index: int
    device: int | None = None
