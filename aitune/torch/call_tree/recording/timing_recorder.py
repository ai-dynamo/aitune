# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The timing pass of the call tree: the recorder that times one call-path depth per replay."""

from __future__ import annotations

import itertools
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.nodes import CallNode, CallPath
from aitune.torch.call_tree.recording.call_recorder import NO_RESULT, CallRecorder

logger = getLogger(__name__)

# ``torch._C._CudaEventBase.record``, resolved lazily on first CUDA use: a CPU-only build ships a placeholder
# ``_CudaEventBase`` without ``record``, so resolving it at import would break the import.
_event_record: Any = None


class TimingRecorder(CallRecorder):
    """Time the calls of the nodes a ``StructureRecorder`` recorded: the timing pass of the call tree.

    The pass replays each sample of the structure pass once per call-path depth its calls reach
    (``depths_per_sample``). Each replay (``time_replay``) runs the whole sample but
    times only the calls at its depth. Deeper calls run inside a timed call and are ignored (see ``_skips_calls``);
    shallower calls are only tracked to know each call's path. Together, the replays time every call once, except
    calls inside a compiled part and calls deeper than ``max_depth``.

    Timer: the pass uses CUDA events when the model may run GPU work, and ``perf_counter`` otherwise. It assumes
    GPU work when an inspected module holds a CUDA parameter or buffer, or when the process has CUDA memory
    reserved; the second check also catches models whose inputs or intermediate tensors are on the GPU. GPU work
    runs asynchronously, so ``perf_counter`` would only measure the kernel launches; events recorded on the GPU
    stream measure the work itself. For CPU work, ``perf_counter`` is exact. Events are reused from a pool per
    device, so they cost little per call.

    The timer has two known limits:

    - A CPU model can get CUDA events. That happens when the process holds unrelated CUDA tensors or did GPU work
      earlier, since the caching allocator keeps that memory reserved after the tensors are freed. CUDA events can
      time short CPU calls too long, but not too short.
    - Both events of a call are recorded on the current stream of the current CUDA device when the call starts,
      even if the call switches streams or devices. Work the call queues on another device, or on a stream that
      this stream does not wait for, is not measured, so such a call is timed too short.

    Results: when the ``with`` block ends normally and no ``time_replay`` block raised, the pass writes each
    node's measured ``total_time_ms`` into the tree in place. Entering the pass resets it, so a pass replaces
    earlier times. ``total_time_ms`` None means the node was not timed in this pass, for example a call inside a
    compiled part, never that it took no time. To keep several timed trees, time a ``tree.copy()`` each.

    Diagnostics: when the timing pass replays different calls than the structure pass saw, for example a
    non-deterministic inference such as sampling in ``generate``, some nodes end up timed for a different number
    of calls than they made. When the ``with`` block ends, one warning names the count and up to three example
    paths; their ``total_time_ms`` only counts the calls that were timed. Calls inside a compiled part and nodes
    deeper than ``max_depth`` are never timed and never counted.

    Example:
        An inference function whose calls report to the active recorder, as ``inspect``'s wrappers make them:

        >>> from torch import nn
        >>> from aitune.torch.call_tree.recording import StructureRecorder
        >>> def report(module, label, x):
        ...     return CallRecorder.active().call(module.forward, (x,), {}, label=label, obj=module, method="forward")
        >>> class Model(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.lin = nn.Linear(4, 4)
        ...
        ...     def forward(self, x):
        ...         return report(self.lin, "model.lin.forward", x).relu()
        >>> model = Model()
        >>> def infer(x):
        ...     return report(model, "model.forward", x)
        >>> samples = [torch.randn(2, 4), torch.randn(2, 4)]

        The structure pass records the tree, then the timing pass replays each sample once per depth:

        >>> structure = StructureRecorder()
        >>> with structure:
        ...     for x in samples:
        ...         with structure.record_sample():
        ...             _ = infer(x)
        >>> timer = TimingRecorder(structure.result())
        >>> timer.depths_per_sample()
        [[1, 2], [1, 2]]
        >>> with timer:
        ...     for depths, x in zip(timer.depths_per_sample(), samples, strict=True):
        ...         for depth in depths:
        ...             with timer.time_replay(depth):
        ...                 _ = infer(x)
        >>> [(node.path_str, node.total_time_ms is not None) for node in timer.tree.nodes.values()]
        [('model.forward', True), ('model.forward > model.lin.forward', True)]
    """

    def __init__(self, tree: CallTree, max_depth: int | None = None) -> None:
        """Time calls against ``tree``'s nodes; ``_start`` chooses CUDA events or ``perf_counter``.

        Args:
            tree: The structure pass's call tree.
            max_depth: The deepest call-path depth to time, or None to time every depth; deeper nodes keep
                ``total_time_ms`` None, without a warning.
        """
        self._tree = tree
        self._max_depth = max_depth
        self._use_cuda = False
        self._depth: int | None = None
        self._failed = False
        self._stack: list[_TimedCall] = []
        self._times: dict[CallPath, list[float]] = {}
        self._events: list[tuple[CallNode, Any, Any, int]] = []
        self._free_events: dict[int, list[Any]] = {}
        self._streams: dict[tuple[int, int, int], torch.cuda.Stream] = {}

    @property
    def tree(self) -> CallTree:
        """The tree the pass times into."""
        return self._tree

    def depths_per_sample(self) -> list[list[int]]:
        """For each sample, the depths the timing pass times, one replay of the sample per depth.

        Each replay runs the whole sample but times only the calls at one depth. A sample whose calls go three
        levels deep is replayed three times: once to time the root call, once its children, and once theirs.
        Depths beyond ``max_depth``, and depths whose calls all run inside compiled parts, are left out.

        Only reads the tree, so it can be called before entering the ``with`` block.

        Returns:
            One sorted list of depths per sample, ``depths_per_sample()[i]`` belonging to sample ``i``, for
            example ``[[1, 2, 3], [1, 2]]``.

        Example:
            The first sample is replayed twice, timing depth 1 and then depth 2. The second sample never calls the
            child, so it is replayed once:

            >>> root = CallNode(
            ...     path=("m.forward",), label="m.forward", owner_class="M", method="forward",
            ...     calls_per_sample=[1, 1],
            ... )
            >>> child = CallNode(
            ...     path=("m.forward", "m.child.forward"),
            ...     label="m.child.forward",
            ...     owner_class="C",
            ...     method="forward",
            ...     calls_per_sample=[1, 0],
            ... )
            >>> tree = CallTree(nodes={root.path: root, child.path: child})
            >>> TimingRecorder(tree).depths_per_sample()
            [[1, 2], [1]]
        """
        return [
            sorted({
                len(node.path)
                for node in self._tree.nodes.values()
                if node.calls_per_sample[i] > 0 and self._is_timeable(node)
            })
            for i in range(self._tree.number_of_samples)
        ]

    @contextmanager
    def time_replay(self, depth: int) -> Generator[None]:
        """Time one replay inside the ``with`` block: the calls at ``depth``.

        An exception inside the block invalidates the pass: it writes no times, as when the pass itself raises.

        Args:
            depth: The call-path depth this replay times.
        """
        self._depth = depth
        try:
            yield
        except BaseException:
            self._failed = True
            self._depth = None
            self._events.clear()  # drop the interrupted replay's pending event pairs
            raise
        self._resolve_events()

    def begin_call(
        self, label: str, obj: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ) -> _TimedCall:
        """Look up a timing-pass call in the recorded nodes; time it when it is at the depth this replay times."""
        path = (*self._stack[-1].path, label) if self._stack else (label,)
        node = self._tree.nodes.get(path)
        start = device = stream = None
        if node is not None and not node.inside_compiled_part and len(path) == self._depth:
            start, device, stream = self._start_timer()
        call = _TimedCall(
            path=path,
            node=node,
            start=start,
            device=device,
            stream=stream,
        )
        self._stack.append(call)
        return call

    def end_call(self, frame: _TimedCall | None, result: Any = NO_RESULT) -> None:
        """Stop timing a call started with ``begin_call``; pops it if it is the innermost open call."""
        if frame is None:
            return
        if frame.start is not None and frame.node is not None:
            self._stop_timer(frame.node, frame)
        if self._stack and self._stack[-1] is frame:
            self._stack.pop()

    def _resolve_events(self) -> None:
        """Resolve the current replay's pending CUDA event pairs into call times.

        ``time_replay`` calls it once per replay; pending events stay bounded by one replay. Each end event is
        synchronized on its own, so pairs recorded on any device resolve without a device-wide synchronization.
        """
        if not self._events:
            return
        if self._use_cuda:
            for _, _, end, _ in self._events:
                end.synchronize()
        for node, start, end, device in self._events:
            self._add_time(node, start.elapsed_time(end))
            pool = self._free_events.setdefault(device, [])
            pool.append(start)
            pool.append(end)
        self._events.clear()

    def _start(self) -> None:
        """Start the timing pass: choose the timer, reset the per-pass state and clear the tree's earlier times.

        CUDA events are used when an inspected module holds a parameter or buffer on a CUDA device or the process
        has CUDA memory reserved — the model's inputs or tensors can live on the GPU even when its modules do not,
        and the caching allocator keeps earlier CUDA work's memory reserved after the tensors are freed. A model
        that runs on the CPU with no CUDA memory reserved is timed with ``perf_counter``, which is exact for CPU
        work; CUDA events around CPU work only roughly follow it and can be several times too long for short
        calls.
        """
        self._use_cuda = torch.cuda.is_available() and (
            any(
                tensor.is_cuda
                for _, module in self._tree.modules
                for tensor in itertools.chain(module.parameters(), module.buffers())
            )
            or (torch.cuda.is_initialized() and torch.cuda.memory_reserved() > 0)
        )
        for node in self._tree.nodes.values():
            node.total_time_ms = None
        self._depth = None
        self._failed = False
        self._stack.clear()
        self._times.clear()
        self._events.clear()

    def _stop(self, completed: bool) -> None:
        """End the timing pass, writing the measured times into the nodes and warning on a normal exit.

        Each node the pass timed gets ``total_time_ms``, the sum of its measured calls (``CallNode``). When the
        block raised (``completed`` False) or a replay raised inside it, nothing is written: the nodes keep the
        None ``_start`` set.
        """
        if not completed or self._failed:
            return
        self._resolve_events()
        self._warn_about_diverged_replay()
        for path, times in self._times.items():
            self._tree.nodes[path].total_time_ms = sum(times)

    def _skips_calls(self) -> bool:
        """Whether the next call needs no recording at all.

        It would be deeper than the timed depth, i.e. inside a timed call, so this pass needs nothing from it —
        no stack entry.
        """
        return self._depth is not None and len(self._stack) >= self._depth

    def _is_timeable(self, node: CallNode) -> bool:
        """Whether the pass times ``node``: not inside a compiled part and not deeper than ``max_depth``."""
        return not node.inside_compiled_part and (self._max_depth is None or len(node.path) <= self._max_depth)

    def _warn_about_diverged_replay(self) -> None:
        """Warn once when the replays timed a different number of calls for a node than the structure pass recorded.

        The timing pass replays each sample once per timeable depth, so a complete pass times every call. A
        node with a different number of timed calls means the replays behaved differently than the recorded
        samples, for example a non-deterministic inference such as sampling in ``generate``. Nodes inside a
        compiled part or deeper than ``max_depth`` are never timed and never counted.
        """
        diverged = [
            node
            for node in self._tree.nodes.values()
            if self._is_timeable(node) and len(self._times.get(node.path, ())) != node.total_calls
        ]
        if diverged:
            examples = sorted(node.path_str for node in diverged)
            logger.warning(
                "The timing pass timed a different number of calls than the structure pass recorded for %d call-tree "
                "%s (for "
                "example %s%s): it replayed different calls than the structure pass saw, for example a "
                "non-deterministic inference such as sampling in generate. Their total_time_ms only counts the calls "
                "that were timed.",
                len(diverged),
                "node" if len(diverged) == 1 else "nodes",
                ", ".join(examples[:3]),
                ", ..." if len(examples) > 3 else "",
            )

    def _add_time(self, node: CallNode, milliseconds: float) -> None:
        """Record one timed call of ``node``."""
        self._times.setdefault(node.path, []).append(milliseconds)

    def _start_timer(self) -> tuple[Any, int | None, Any]:
        """Start timing a call: record a pooled CUDA event on the current stream, or read ``perf_counter``.

        Returns:
            The start of the timer and, with CUDA, the device and stream the start event was recorded on (``None``s
            without CUDA). The end event goes on the same stream, even if the call switches the current stream.
        """
        if self._use_cuda:
            device = torch.cuda.current_device()
            stream = self._current_stream(device)
            return self._record_event(device, stream), device, stream
        return time.perf_counter(), None, None

    def _stop_timer(self, node: CallNode, call: _TimedCall) -> None:
        """Stop timing ``node``'s call: queue the CUDA event pair for the end of the replay, or record the CPU time."""
        if call.device is not None:  # set exactly when the start is a CUDA event
            end = self._record_event(call.device, call.stream)
            self._events.append((node, call.start, end, call.device))
        else:
            self._add_time(node, (time.perf_counter() - call.start) * 1000.0)

    def _current_stream(self, device: int) -> torch.cuda.Stream:
        """The current stream of ``device``, cached by its raw id: ~0.1us, and stream contexts change the id."""
        key = torch._C._cuda_getCurrentStream(device)
        stream = self._streams.get(key)
        if stream is None:
            stream = self._streams[key] = torch.cuda.Stream(stream_id=key[0], device_index=key[1], device_type=key[2])
        return stream

    def _record_event(self, device: int, stream: Any) -> Any:
        """Record an event from ``device``'s pool on ``stream`` and return it."""
        pool = self._free_events.setdefault(device, [])
        event = pool.pop() if pool else torch.cuda.Event(enable_timing=True)
        _record(event, stream)
        return event


@dataclass(slots=True)
class _TimedCall:
    """An open timing-pass call: its call path, the recorded node, its timer's start and its start stream.

    ``node`` is None for a call with no recorded node. The start is a CUDA event or a ``perf_counter`` value, or
    None for an untimed call; ``device`` and ``stream`` are the CUDA device and stream the start event was
    recorded on.
    """

    path: CallPath
    node: CallNode | None
    start: Any
    device: int | None
    stream: Any


def _record(event: Any, stream: Any) -> None:
    """Record ``event`` on ``stream`` directly.

    The cheapest correct record (~1.5us), without the current-stream lookup ``Event.record()`` does (~2.2us more).
    """
    global _event_record
    if _event_record is None:
        _event_record = torch._C._CudaEventBase.record
    _event_record(event, stream)
