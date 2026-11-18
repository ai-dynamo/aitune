# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one process-wide active call recorder.

The inspection adapters install their wrappers once, and the same wrappers serve both call-tree passes —
structure and timing — so a wrapper cannot capture the recorder to report to. Instead it asks
``CallRecorder.active()`` which recorder, if any, is recording, and reports the call to its ``call``
(or calls the original unchanged when none is). Recorders activate themselves on ``__enter__`` and deactivate on
``__exit__``. The slot is a class attribute of ``CallRecorder``, not a ``ContextVar``: calls may come from any
thread, and all threads must see the active recorder.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Any, ClassVar, Self, final

from aitune.torch.call_tree.methods import is_transparent_call, is_transparent_container

NO_RESULT = object()


class CallRecorder(ABC):
    """A call recorder: the active one receives every wrapped call between ``__enter__`` and ``__exit__``.

    At most one recorder is active process-wide; the wrappers find it via ``CallRecorder.active()`` and report
    through ``call``. Subclasses implement ``begin_call``/``end_call`` and set up and tear down their
    pass in ``_start``/``_stop`` — ``StructureRecorder`` for the structure pass, ``TimingRecorder`` for the
    timing pass.

    The active recorder is kept in ``CallRecorder._active``, one slot shared by all subclasses. Whichever recorder
    is in it holds ``CallRecorder._active_lock``: ``__enter__`` acquires it without waiting, so a second recorder
    fails at once, also when two threads enter at the same time. Both are always assigned through
    ``CallRecorder``, never through ``self`` or ``type(self)``, which would give an instance or a subclass a slot
    of its own.

    Calls from any thread reach the active recorder, but each recorder keeps one stack of open calls for all
    threads. Calls on other threads are therefore recorded correctly only when they do not overlap with calls on
    another thread, for example a worker thread that runs the model while the caller waits. Calls running on
    several threads at once get each other's calls as parents, and a call that ends while another thread's call is
    innermost stays on the stack.

    Wrappers report to whichever recorder is active, not to the recorder of the inspection that installed them.
    Only one call-tree inspection may therefore run at a time per process: with two at once, for example on the
    same model in two threads, calls can be recorded twice or end up in the other inspection's tree.
    """

    _active: ClassVar[CallRecorder | None] = None
    _active_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def active(cls) -> CallRecorder | None:
        """The recorder currently recording, or None."""
        return CallRecorder._active

    def __enter__(self) -> Self:
        """Become the active recorder and start the pass.

        Raises:
            RuntimeError: If another call recorder is already recording.
        """
        if not CallRecorder._active_lock.acquire(blocking=False):
            raise RuntimeError("Another call recorder is already recording.")
        CallRecorder._active = self
        try:
            self._start()
        except BaseException:
            CallRecorder._active = None
            CallRecorder._active_lock.release()
            raise
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """End the pass and deactivate, also when the recorded code raised."""
        try:
            self._stop(completed=exc_type is None)
        finally:
            if CallRecorder._active is self:
                CallRecorder._active = None
                CallRecorder._active_lock.release()

    def _start(self) -> None:
        """Hook: set up the pass after becoming the active recorder; the default does nothing."""
        return None

    def _stop(self, completed: bool) -> None:
        """Hook: tear down the pass; ``completed`` is False when the block raised. The default does nothing."""
        return None

    def _skips_calls(self) -> bool:
        """Whether the next call needs no recording at all; checked first in ``call``, so it must be cheap."""
        return False

    @final
    def call(
        self,
        original: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        label: str,
        obj: Any,
        method: str,
    ) -> Any:
        """Run ``original(*args, **kwargs)`` and report it to this recorder as one call; return its result.

        Exact torch containers and helpers that run a callable argument run unreported, so their callees attach
        to the caller. The rule lives here, final, so that both passes apply the same rule.

        Args:
            original: The unwrapped method, ``obj.<method>`` before the adapter wrapped it.
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.
            label: The adapter's label of the call, e.g. ``root.vlm.forward``.
            obj: The object the method belongs to.
            method: The method's name.

        Returns:
            What ``original`` returned.
        """
        if self._skips_calls() or is_transparent_container(obj) or is_transparent_call(args, kwargs):
            return original(*args, **kwargs)
        frame = self.begin_call(label, obj, method, args, kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException:
            self.end_call(frame)
            raise
        self.end_call(frame, result)
        return result

    @abstractmethod
    def begin_call(
        self, label: str, obj: Any, method: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ) -> Any:
        """Record the start of a call; return a frame to pass to ``end_call``, or None to not record it."""

    @abstractmethod
    def end_call(self, frame: Any, result: Any = NO_RESULT) -> None:
        """Record the end of the call ``frame`` is for, with its return value or ``NO_RESULT`` when it raised."""
