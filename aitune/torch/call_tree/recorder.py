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

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Self

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallNode, CallTree
from aitune.torch.call_tree.callers import unrecorded_caller
from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container
from aitune.torch.call_tree.nodes import NodeKey, key_path
from aitune.torch.call_tree.weights.accesses import WeightAccess, WeightAccessLog, WeightAccessMode, storage_key

_NO_RESULT = object()


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
        self._nodes: dict[NodeKey, CallNode] = {}
        self._paths: dict[NodeKey, str] = {}
        self._stack: list[NodeKey] = []
        self._events: dict[NodeKey, list[tuple[Any, Any]]] = {}
        self._ms: dict[NodeKey, float] = {}
        self._sequences: list[list[NodeKey]] = []
        self._current_sequence: list[NodeKey] | None = None
        self._apply_depth = 0
        self._original_apply: Callable[..., Any] | None = None
        self._weight_access_log = WeightAccessLog()
        self._weight_access_mode: WeightAccessMode | None = None
        self._modules: list[tuple[str, nn.Module]] = []
        self._registered: set[int] = set()
        self._order = 0
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
            self._weight_access_log.start(self._modules)
            weight_access_mode = WeightAccessMode(
                self._weight_access_log, lambda: self._paths[self._stack[-1]] if self._stack else ""
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
        if id(module) not in self._registered:
            self._registered.add(id(module))
            self._modules.append((label_prefix, module))

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
        parent = self._stack[-1] if self._stack else ()
        key = (*parent, label)
        node = self._nodes.get(key)
        if node is None:
            node = CallNode(
                key=key,
                label=label,
                owner_class=type(obj).__name__,
                method=method,
                module=obj if isinstance(obj, nn.Module) else None,
                first_seen=self._order,
            )
            self._order += 1
            self._nodes[key] = node
            self._paths[key] = key_path(key)
            if parent in self._nodes:
                self._nodes[parent].children.append(key)
        node.calls += 1
        node.bytes_in += _tensor_bytes((args, kwargs))
        if self._current_sequence is not None:
            self._current_sequence.append(key)
        self._stack.append(key)
        start, device = self._start_timer()
        return _Frame(node=node, start=start, device=device)

    def end_call(self, frame: _Frame | None, result: Any = _NO_RESULT) -> None:
        """Record the end of a call started with ``begin_call``.

        Args:
            frame: The frame ``begin_call`` returned.
            result: The call's return value, to count its tensor bytes; omit it when the call raised.
        """
        if frame is None:
            return
        self._stop_timer(frame)
        if result is not _NO_RESULT:
            frame.node.bytes_out += _tensor_bytes(result)
        if self._stack and self._stack[-1] == frame.node.key:
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

        Idempotent: resolved event times accumulate into a per-key total, so calling ``result`` again returns the
        same times, and recording more calls afterwards keeps accumulating. The returned tree is a snapshot: its nodes
        are copies, so later recording does not change it.
        """
        self.flush()
        nodes = {
            key: replace(node, children=list(node.children), time_ms=self._ms.get(key, 0.0))
            for key, node in self._nodes.items()
        }
        return CallTree(
            nodes=nodes,
            iterations=len(self._sequences),
            sequences=list(self._sequences),
            weight_accesses=self._weight_access_log.snapshot(),
            modules=list(self._modules),
        )

    def flush(self) -> None:
        """Resolve pending CUDA event pairs into the accumulated totals so they are not held until the end.

        Adapters call it once per iteration; pending events stay bounded by one iteration. Each end event is
        synchronized on its own, so pairs recorded on any device resolve without a device-wide synchronization.
        """
        if not self._events:
            return
        if self._use_cuda:
            for pairs in self._events.values():
                for _, end in pairs:
                    end.synchronize()
        for key, pairs in self._events.items():
            self._ms[key] = self._ms.get(key, 0.0) + sum(start.elapsed_time(end) for start, end in pairs)
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
        key = frame.node.key
        if self._use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(frame.device))
            self._events.setdefault(key, []).append((frame.start, end))
        else:
            self._ms[key] = self._ms.get(key, 0.0) + (time.perf_counter() - frame.start) * 1000.0

    def _make_apply_probe(self, original_apply: Callable) -> Callable:
        """Build the replacement for ``nn.Module._apply`` that ``__enter__`` installs.

        ``nn.Module._apply`` is the method that ``.to()``, ``.cuda()``, ``.half()`` and similar calls use to convert a
        module's parameters and buffers. The replacement runs ``original_apply`` unchanged. While the recorder is
        recording, it first records a ``"move"`` access on every tracked weight of the module and, unless
        ``recurse=False``, of all its submodules: the innermost recorded call's key, or the code found by
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
                caller = recorder._paths[recorder._stack[-1]] if recorder._stack else unrecorded_caller()
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


def _tensor_bytes(tree: Any) -> int:
    """Total bytes of the tensor leaves of ``tree``, deduplicated by object identity.

    Containers of tensors (lists, dicts, ModelOutputs, ...) are walked; non-tensor leaves (a ``DynamicCache``, a
    plain object) are ignored. The walk runs with torch function disabled so it records no weight accesses.
    """
    with torch._C.DisableTorchFunction():
        seen: set[int] = set()
        total = 0
        for leaf in torch.utils._pytree.tree_leaves(tree):
            if isinstance(leaf, torch.Tensor) and id(leaf) not in seen:
                seen.add(id(leaf))
                total += leaf.numel() * leaf.element_size()
    return total


@dataclass
class _Frame:
    """An open call: its node and the start of its timer.

    The start is a CUDA event or a ``perf_counter`` value; ``device`` is the CUDA device the start event was
    recorded on.
    """

    node: CallNode
    start: Any
    device: int | None = None
