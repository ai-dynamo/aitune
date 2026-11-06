# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mode-independent call recorder.

The recorder builds a context-keyed call tree. It installs no hooks on modules: each inspection mode (an adapter)
wraps the methods it finds and calls ``enter`` and ``exit`` around the original call::

    frame = recorder.enter(label, obj, name, args, kwargs)
    try:
        return original(*args, **kwargs)
    finally:
        recorder.exit(frame)

The only hook the recorder owns is a process-wide probe on ``nn.Module._apply`` (``.to()``, ``.cuda()``, ...), active
inside ``recording()``. Timing uses CUDA events per call (no per-call synchronization) when CUDA is available and
``time.perf_counter`` otherwise; ``flush()`` resolves the pending events once per iteration.
"""

import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallNode, CallTree, NodeKey
from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container

OUTSIDE_ANY_CALL = "<outside any call>"

_MOVING_METHODS = frozenset({
    "_apply",
    "to",
    "cuda",
    "cpu",
    "xpu",
    "mps",
    "half",
    "float",
    "double",
    "bfloat16",
    "type",
    "to_empty",
})


def _unrecorded_caller() -> str:
    """Name the code that moved modules when no recorded call is running.

    Walks up from the ``_apply`` probe and returns ``<unrecorded: qualname>`` for the first frame that is neither in
    ``torch`` nor in this module, and is not itself a module-moving method (so a library ``.to()`` override such as
    ``PreTrainedModel.to`` is skipped and its caller, e.g. ``SentenceTransformer.encode``, is named).
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        in_torch = module == "torch" or module.startswith("torch.")
        if (
            not in_torch
            and module not in (__name__, "aitune.torch.call_tree.accesses")
            and frame.f_code.co_name not in _MOVING_METHODS
        ):
            return f"<unrecorded: {frame.f_code.co_qualname}>"
        frame = frame.f_back
    return OUTSIDE_ANY_CALL


@dataclass
class _Frame:
    """An open call: its node and the start of its timer.

    The start is a CUDA event or a ``perf_counter`` value; ``device`` is the CUDA device the start event was
    recorded on.
    """

    node: CallNode
    start: Any
    device: int | None = None


class CallRecorder:
    """Record a context-keyed call tree from enter/exit callbacks.

    Example:
        >>> recorder = CallRecorder()
        >>> with recorder.recording():
        ...     recorder.begin_iteration()
        ...     outer = recorder.enter("root.forward", object(), "forward", ())
        ...     inner = recorder.enter("root.head.forward", object(), "forward", ())
        ...     recorder.exit(inner)
        ...     recorder.exit(outer)
        ...     recorder.end_iteration()
        >>> sorted(recorder.result().nodes)
        [('root.forward',), ('root.forward', 'root.head.forward')]
    """

    # The recorder whose ``nn.Module._apply`` probe is installed; the probe is process-wide, so only one may record.
    _recording_now: "CallRecorder | None" = None

    def __init__(self) -> None:
        """Create an empty recorder."""
        self._nodes: dict[NodeKey, CallNode] = {}
        self._stack: list[NodeKey] = []
        self._events: dict[NodeKey, list[tuple[Any, Any]]] = {}
        self._ms: dict[NodeKey, float] = {}
        self._sequences: list[list[NodeKey]] = []
        self._current_sequence: list[NodeKey] | None = None
        self._external_apply: dict[int, set[str]] = {}
        self._apply_depth = 0
        self._accesses = None
        self._final_accesses: dict = {}
        self._modules: list[tuple[str, nn.Module]] = []
        self._registered: set[int] = set()
        self._order = 0
        self._active = False
        self._use_cuda = torch.cuda.is_available()

    @property
    def active(self) -> bool:
        """Whether the recorder is inside ``recording()``."""
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

    def enter(
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
            A frame to pass to ``exit``, or None if the call is not recorded (not recording, exact torch container,
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
            if parent in self._nodes:
                self._nodes[parent].children.append(key)
        node.calls += 1
        if self._current_sequence is not None:
            self._current_sequence.append(key)
        self._stack.append(key)
        start, device = self._start_timer()
        return _Frame(node=node, start=start, device=device)

    def exit(self, frame: _Frame | None) -> None:
        """Record the end of a call started with ``enter``."""
        if frame is None:
            return
        self._stop_timer(frame)
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

    @contextmanager
    def recording(self) -> Generator["CallRecorder", None, None]:
        """Record calls inside the context, with the ``nn.Module._apply`` probe installed."""
        if CallRecorder._recording_now is not None:
            raise RuntimeError("Another CallRecorder is already recording; the nn.Module._apply probe is process-wide.")
        from aitune.torch.call_tree.accesses import WeightAccessMode

        original_apply = nn.Module._apply
        CallRecorder._recording_now = self
        try:
            nn.Module._apply = self._make_apply_probe(original_apply)
            mode = WeightAccessMode(self)
            mode.__enter__()
            self._accesses = mode
            self._active = True
            yield self
        finally:
            self._active = False
            self._stack.clear()
            self._current_sequence = None
            if self._accesses is not None:
                # Accumulate over recording passes, like the nodes: a weight read only in an earlier pass still counts.
                self._final_accesses = self._weight_accesses()
                self._accesses.__exit__(None, None, None)
                self._accesses = None
            nn.Module._apply = original_apply
            CallRecorder._recording_now = None

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
            external_apply={k: set(v) for k, v in self._external_apply.items()},
            weight_accesses=self._weight_accesses(),
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

    def _access_caller(self, node_key) -> str:
        """Label of the call at ``node_key``, or the unrecorded caller when the stack is empty."""
        if node_key:
            return self._nodes[node_key].label
        return _unrecorded_caller()

    def _weight_accesses(self) -> dict:
        """Accesses of all recording passes so far, deduplicated per storage and expanded to tensor ids."""
        out: dict[int, set] = {tensor_id: set(records) for tensor_id, records in self._final_accesses.items()}
        if self._accesses is not None:
            for key, records in self._accesses.accesses.items():
                for tensor_id in self._accesses.tensor_ids(key):
                    out.setdefault(tensor_id, set()).update(records)
        return out

    def _make_apply_probe(self, original_apply: Callable) -> Callable:
        """Build the replacement for ``nn.Module._apply`` that ``recording()`` installs.

        ``nn.Module._apply`` is the method that ``.to()``, ``.cuda()``, ``.half()`` and similar calls use to convert a
        module's parameters and buffers. The replacement runs ``original_apply`` unchanged. While the recorder is
        recording, it first records who called it: the label of the innermost recorded call, or the code found by
        ``_unrecorded_caller`` if no recorded call is running. The caller is added to ``external_apply`` for the
        module and, unless ``recurse=False``, for all of its submodules. Only the outermost ``_apply`` of a
        conversion records, because ``_apply`` calls itself on each child.

        Args:
            original_apply: The ``nn.Module._apply`` to call; restored by ``recording()`` on exit.

        Returns:
            A function with the signature of ``nn.Module._apply``.
        """
        recorder = self

        def _apply(module: nn.Module, fn: Callable, *args: Any, **kwargs: Any) -> Any:
            # ``nn.Module._apply`` recurses through child ``_apply`` calls; only the outermost one records.
            if recorder._active and recorder._apply_depth == 0:
                caller = recorder._nodes[recorder._stack[-1]].label if recorder._stack else _unrecorded_caller()
                recurse = kwargs.get("recurse", args[0] if args else True)
                for submodule in module.modules() if recurse else [module]:
                    recorder._external_apply.setdefault(id(submodule), set()).add(caller)
            recorder._apply_depth += 1
            try:
                return original_apply(module, fn, *args, **kwargs)
            finally:
                recorder._apply_depth -= 1
                if recorder._apply_depth == 0 and recorder._active and recorder._accesses is not None:
                    # A move or cast gives the tensors new storage; keep tracking them under it.
                    recorder._accesses.register(module)

        return _apply
