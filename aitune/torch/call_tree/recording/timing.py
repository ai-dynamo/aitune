# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The timing pass of the call tree: the recorder that times one call-path depth per iteration."""

import itertools
import time
from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.methods import is_compiled_module
from aitune.torch.call_tree.nodes import CallNode, CallPath, TimedIteration, format_call_path
from aitune.torch.call_tree.recording.base import NO_RESULT, CallRecorder

logger = getLogger(__name__)

# ``torch._C._CudaEventBase.record``, resolved lazily on first CUDA use: a CPU-only build ships a placeholder
# ``_CudaEventBase`` without ``record``, so resolving it at import would break the import.
_event_record: Any = None


def _record(event: Any, stream: Any) -> None:
    """Record ``event`` on ``stream`` directly.

    The cheapest correct record (~1.5us), without the current-stream lookup ``Event.record()`` does (~2.2us more).
    """
    global _event_record
    if _event_record is None:
        _event_record = torch._C._CudaEventBase.record
    _event_record(event, stream)


def _format_paths(paths: list[CallPath]) -> str:
    """The paths joined for a warning message: at most three, then an ellipsis."""
    return ", ".join(format_call_path(path) for path in paths[:3]) + (", …" if len(paths) > 3 else "")


def depth_schedule(tree: CallTree) -> list[int | None]:
    """Plan which call-path depth each timing-pass iteration times, from the calls the structure pass saw.

    Each iteration times calls at one depth only, so no timed call runs inside another and adds its timer's
    overhead to the outer time. The timing pass replays the structure pass's samples in the same order, so
    ``tree.sequences[i]`` lists the calls iteration ``i`` will make, and the whole schedule can be planned up front.

    Each iteration picks one of the depths it makes calls at:

    1. If some of its calls have no planned timing yet, it picks the depth whose untimed calls occur in the fewest
       later iterations. A call that only some samples make is then timed while it still can be.
    2. Otherwise, it picks the depth whose calls have the fewest planned timings so far. That spreads the remaining
       iterations evenly over the depths, so each node gets several measurements.

    Ties go to the shallowest depth. When every sample makes the same calls, the schedule is a plain rotation
    through the depths, shallowest first.

    Args:
        tree: The structure pass's call tree.

    Returns:
        One depth per iteration in ``tree.sequences``, or None for an iteration that made no calls.

    Example:
        The child, at depth 2, is called only in iterations 0 and 2. Iteration 0 times it, since the root has more
        chances left; the later iterations time the root:

        >>> root = CallNode(path=("m.forward",), label="m.forward", owner_class="M", method="forward", calls=4)
        >>> child = CallNode(
        ...     path=("m.forward", "m.child.forward"),
        ...     label="m.child.forward",
        ...     owner_class="C",
        ...     method="forward",
        ...     calls=2,
        ... )
        >>> root.children.append(child.path)
        >>> tree = CallTree(
        ...     nodes={root.path: root, child.path: child},
        ...     iterations=4,
        ...     sequences=[[root.path, child.path], [root.path], [root.path, child.path], [root.path]],
        ... )
        >>> depth_schedule(tree)
        [2, 1, 1, 1]
    """
    # chances[i][path]: in how many iterations after i the path is called again
    chances: list[Counter[CallPath]] = [Counter() for _ in tree.sequences]
    later: Counter[CallPath] = Counter()
    for i in range(len(tree.sequences) - 1, -1, -1):
        chances[i] = later
        later = later + Counter(set(tree.sequences[i]))  # later iterations containing the path, once each
    timed: Counter[CallPath] = Counter()  # planned timed calls per path
    schedule: list[int | None] = []
    for i, sequence in enumerate(tree.sequences):
        calls = Counter(sequence)  # path -> calls in this iteration
        if not calls:
            schedule.append(None)
            continue
        by_depth: dict[int, list[CallPath]] = {}
        for path in calls:
            by_depth.setdefault(len(path), []).append(path)
        untimed = {d: [p for p in paths if timed[p] == 0] for d, paths in by_depth.items()}
        untimed = {d: paths for d, paths in untimed.items() if paths}
        if untimed:
            depth = min(untimed, key=lambda d: (min(chances[i][p] for p in untimed[d]), d))
        else:
            depth = min(by_depth, key=lambda d: (min(timed[p] for p in by_depth[d]), d))
        schedule.append(depth)
        for path in by_depth[depth]:
            timed[path] += calls[path]
    return schedule


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


class TimingRecorder(CallRecorder):
    """Time the calls of the nodes a ``StructureRecorder`` recorded: the timing pass of the call tree.

    Each iteration times the calls at one call-path depth, the one ``depth_schedule`` plans for it. Deeper calls run
    inside a timed call and are ignored (see ``_skips_calls``). Shallower calls are not timed; the pass only tracks
    them to know each call's path.

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
    - Both events of a call are recorded on the stream that is current on the call's device when the call starts,
      even if the call switches streams or devices. Work the call queues on another device, or on a stream that
      this stream does not wait for, is not measured, so such a call is timed too short.

    Results: when the ``with`` block ends normally, the pass writes each node's measured ``timings``, one
    ``TimedIteration`` per iteration that timed it, into the tree in place. ``CallNode.time_ms`` derives the
    estimate from them. Entering the pass resets every node's ``timings``, so a pass replaces earlier times. Empty
    ``timings`` mean the node was not timed in this pass, for example a call inside a compiled part, never that it
    took no time. To keep several timed trees, time a ``tree.copy()`` each.

    Diagnostics: a call can stay untimed when ``number_of_iterations`` is too small to time every depth, or when a
    sample behaves differently than in the structure pass, for example under random sampling. When the ``with``
    block ends, one warning lists the nodes left without a time, with a hint when a depth was never timed. Calls
    inside a part compiled with ``nn.Module.compile()`` or ``torch.compile`` are expected to have no time and are
    not listed.

    Example:
        >>> from aitune.torch.call_tree.nodes import CallNode
        >>> from aitune.torch.call_tree.call_tree import CallTree
        >>> node = CallNode(path=("m.forward",), label="m.forward", owner_class="M", method="forward")
        >>> node.calls = 2
        >>> tree = CallTree(
        ...     nodes={("m.forward",): node}, iterations=2, sequences=[[("m.forward",)], [("m.forward",)]]
        ... )
        >>> timer = TimingRecorder(tree)
        >>> with timer:
        ...     for _ in range(2):
        ...         timer.begin_iteration()
        ...         timer.end_call(timer.begin_call("m.forward", None, "forward", ()))
        >>> node.time_ms is not None and node.time_ms > 0
        True
    """

    def __init__(self, tree: CallTree) -> None:
        """Time calls against ``tree``'s nodes; ``_start`` chooses CUDA events or ``perf_counter``."""
        self._tree = tree
        self._schedule = depth_schedule(tree)
        self._use_cuda = False
        self._iteration = 0
        self._depth: int | None = None
        self._timed_depths: set[int] = set()
        self._stack: list[_TimedCall] = []
        self._samples: dict[CallPath, dict[int, list[float]]] = {}
        self._events: list[tuple[CallNode, Any, Any, int, int]] = []
        self._free_events: dict[int, list[Any]] = {}
        self._streams: dict[tuple[int, int, int], torch.cuda.Stream] = {}

    @property
    def tree(self) -> CallTree:
        """The tree the pass times into."""
        return self._tree

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
            node.timings = []
        self._iteration = 0
        self._depth = None
        self._timed_depths.clear()
        self._stack.clear()
        self._samples.clear()
        self._events.clear()

    def _stop(self, completed: bool) -> None:
        """End the timing pass: on a normal exit, flush, write the measured ``timings`` into the nodes, and warn.

        Each node the pass timed gets one ``TimedIteration`` per iteration that timed it; its ``time_ms``
        estimate derives from them (``CallNode.time_ms``). When the block raised (``completed`` False), nothing
        is written: the nodes keep the empty ``timings`` ``_start`` set.
        """
        if not completed:
            return
        self.flush()
        for path, samples in self._samples.items():
            self._tree.nodes[path].timings = [
                TimedIteration(iteration, len(times), sum(times)) for iteration, times in sorted(samples.items())
            ]
        self._report_untimed_calls()

    def begin_iteration(self) -> None:
        """Start the next timing-pass iteration: time the depth the schedule plans for it.

        Call once per iteration, in the order of the structure pass's iterations; an iteration beyond the
        recorded ones times nothing.
        """
        self._depth = self._schedule[self._iteration] if self._iteration < len(self._schedule) else None
        if self._depth is not None:
            self._timed_depths.add(self._depth)
        self._iteration += 1

    def _skips_calls(self) -> bool:
        """Whether the next call needs no recording at all.

        It would be deeper than the timed depth, i.e. inside a timed call, so this pass needs nothing from it —
        no stack entry.
        """
        return self._depth is not None and len(self._stack) >= self._depth

    def begin_call(
        self, label: str, obj: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ) -> _TimedCall:
        """Look up a timing-pass call in the recorded nodes; time it when it is at the depth this iteration times."""
        path = (*self._stack[-1].path, label) if self._stack else (label,)
        node = self._tree.nodes.get(path)
        start = device = stream = None
        if node is not None and len(path) == self._depth:
            start, device, stream = self._start_timer()
        call = _TimedCall(path=path, node=node, start=start, device=device, stream=stream)
        self._stack.append(call)
        return call

    def end_call(self, frame: _TimedCall | None, result: Any = NO_RESULT) -> None:
        """Stop timing a call started with ``begin_call``; pops it if it is the innermost open call."""
        if frame is None:
            return
        if frame.start is not None:
            self._stop_timer(frame)
        if self._stack and self._stack[-1] is frame:
            self._stack.pop()

    def flush(self) -> None:
        """Resolve pending CUDA event pairs into the samples so they are not held until the end.

        Adapters call it once per iteration; pending events stay bounded by one iteration. Each end event is
        synchronized on its own, so pairs recorded on any device resolve without a device-wide synchronization.
        """
        if not self._events:
            return
        if self._use_cuda:
            for _, _, end, _, _ in self._events:
                end.synchronize()
        for node, start, end, iteration, device in self._events:
            self._add_sample(node, start.elapsed_time(end), iteration)
            pool = self._free_events.setdefault(device, [])
            pool.append(start)
            pool.append(end)
        self._events.clear()

    def _compiled_root_ids(self) -> set[int]:
        """Ids of the compiled roots in the tree.

        A module counts as compiled when ``nn.Module.compile()`` set ``_compiled_call_impl`` on it,
        ``torch.compile()`` wrapped it in an ``OptimizedModule``, or its ``forward`` is a ``torch.compile``'d
        callable (``module.forward = torch.compile(module.forward)``).
        """
        roots: set[int] = set()
        for _, module in self._tree.modules:
            if is_compiled_module(module):
                roots.add(id(module))
        return roots

    def _inside_compiled(self, path: CallPath, roots: set[int]) -> bool:
        """Whether ``path`` runs inside a compiled part, where the timing pass cannot see its calls.

        That is when a strict ancestor on the path is a compiled root's ``forward`` call: everything below it
        runs inside the compiled code — a recursive call of the root included. ``compile()`` compiles the call
        only, so the root's other methods (``encode()`` and the like) and calls below them run eagerly and are
        not inside. The root's own node is not inside, it is timed as one call; neither is an eager call of a
        module that also sits inside a compiled part, which has a different path.
        """
        for k in range(1, len(path)):
            node = self._tree.nodes.get(path[:k])
            if node is not None and node.module is not None and node.method == "forward" and id(node.module) in roots:
                return True
        return False

    def _report_untimed_calls(self) -> None:
        """Warn once about the nodes the pass left without a time.

        Nodes inside a compiled part (below a compiled module's ``forward`` call) are excepted: calls inside run
        compiled in the timing pass and are expected to have no time, while the compiled part itself is timed as
        one call. Other methods of a compiled module run eagerly, so calls below them are listed like any other.
        A hint is added when a listed node sits at a depth the pass never timed.
        """
        roots = self._compiled_root_ids()
        untimed = [
            path for path in self._tree.nodes if path not in self._samples and not self._inside_compiled(path, roots)
        ]
        if not untimed:
            return
        message = (
            f"{len(untimed)} of the call tree's {len(self._tree.nodes)} nodes have no time (time_ms is None): "
            f"{_format_paths(untimed)}."
        )
        if any(len(path) not in self._timed_depths for path in untimed):
            deepest = max(len(path) for path in self._tree.nodes)
            message += (
                f" The timing pass times one call-path depth per iteration; number_of_iterations must be at least "
                f"{deepest} to time every depth, and calls that only some samples make also need enough samples "
                f"that make them."
            )
        logger.warning("%s", message)

    def _add_sample(self, node: CallNode, milliseconds: float, iteration: int) -> None:
        """Record one timed call of ``node``, under the iteration that timed it."""
        self._samples.setdefault(node.path, {}).setdefault(iteration, []).append(milliseconds)

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

    def _stop_timer(self, call: _TimedCall) -> None:
        """Stop timing a call: queue the CUDA event pair for ``flush``, or take the CPU time as the sample."""
        node = call.node
        assert node is not None  # a call with a start time always has a node
        if self._use_cuda:
            assert call.device is not None
            end = self._record_event(call.device, call.stream)
            self._events.append((node, call.start, end, self._iteration - 1, call.device))
        else:
            self._add_sample(node, (time.perf_counter() - call.start) * 1000.0, self._iteration - 1)

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
