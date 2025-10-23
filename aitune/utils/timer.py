# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Timer utilities for measuring and logging operation durations."""

import contextvars
import logging
import time

# Thread-safe context variable to track nested timer depth
_timer_depth: contextvars.ContextVar[int] = contextvars.ContextVar("timer_depth", default=0)


def _format_duration(seconds: float) -> str:
    """Format duration in a human-readable way.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string with appropriate unit (ms, s, or min)
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        minutes = seconds / 60.0
        return f"{minutes:.2f}min"


class Timer:
    """Context manager for timing and logging operations.

    Supports nested timers with automatic depth tracking and indentation,
    checkpoints for intermediate timing, and programmatic access to elapsed time.

    Example:
        # Basic usage with logging
        with Timer("Model training"):
            train_model()

        # No name - logs won't include operation name
        with Timer() as timer:
            do_work()
            print(f"Took {timer.elapsed:.2f}s")
            # Logs: "⏱️ completed in 1.23s" (no name in message)

        # With checkpoints
        with Timer("Data pipeline") as timer:
            load_data()
            timer.checkpoint("Data loaded")
            preprocess_data()
            timer.checkpoint("Preprocessing done")

        # Access elapsed time
        with Timer("Inference") as timer:
            result = model.predict(x)
            print(f"Took {timer.elapsed:.2f}s")

        # Silent mode - no logging, just timing
        with Timer("Silent operation", silent=True) as timer:
            do_work()
            print(f"Operation took {timer.elapsed:.3f}s")

        # Initial depth for custom indentation
        with Timer("Nested operation", depth=2):
            do_work()  # Will be indented 4 spaces (2 levels)

        # Enable depth tracking for nested indentation
        with Timer("Outer", track_depth=True):
            with Timer("Inner", track_depth=True):
                do_work()  # Inner will be indented relative to Outer

        # Silent mode with explicit logging at specific points
        with Timer("Processing", silent=True) as timer:
            step1()
            timer.log_elapsed("Step 1 completed")  # Explicitly log
            step2()
            timer.log_elapsed("Step 2 completed")  # Explicitly log

        # Silent mode with indented logging
        with Timer("Pipeline", silent=True) as timer:
            load_data()
            timer.log_elapsed("Data loaded", depth=0)
            with_substeps()
            timer.log_elapsed("Substep 1", depth=1)
            timer.log_elapsed("Substep 2", depth=1)
            finalize()
            timer.log_elapsed("Pipeline complete", depth=0)

        # Manual start/stop without context manager
        timer = Timer("Manual operation")
        timer.start()
        do_work()
        elapsed = timer.stop()
        print(f"Work took {elapsed:.2f}s")
    """

    def __init__(
        self,
        name: str | None = None,
        level: int = logging.INFO,
        logger: logging.Logger | None = None,
        silent: bool = False,
        depth: int = 0,
        track_depth: bool = False,
    ):
        """Initialize the Timer.

        Args:
            name: Name of the operation being timed (default: None, no name in logs)
            level: Logging level to use (default: INFO)
            logger: Optional logger instance (default: creates logger for this module)
            silent: If True, disables all logging and only tracks time (default: False)
            depth: Initial depth offset for indentation (default: 0)
            track_depth: If True, participates in nested depth tracking via context vars (default: False)
        """
        self.name = name
        self.level = level
        self.logger = logger or logging.getLogger(__name__)
        self.silent = silent
        self.initial_depth = depth
        self.track_depth = track_depth

        self._start_time: float | None = None
        self._end_time: float | None = None
        self._last_checkpoint_time: float | None = None
        self._depth: int = 0
        self._depth_token: contextvars.Token | None = None

    @property
    def elapsed(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            Elapsed time in seconds. If timer is running, returns time since start.
            If timer has finished, returns total duration.
        """
        if self._start_time is None:
            return 0.0

        end_time = self._end_time if self._end_time is not None else time.perf_counter()
        return end_time - self._start_time

    def _get_indent(self) -> str:
        """Get indentation string based on current depth plus initial depth."""
        return "  " * (self._depth + self.initial_depth)

    def _log(self, message: str, emoji: str = "⏱️", indent: str | None = None) -> None:
        """Helper to log a message, handling the case when name is None.

        Args:
            message: The message to log (e.g., "started", "completed in 1.23s")
            emoji: The emoji/prefix for the message (default: "⏱️")
            indent: Optional indent string (defaults to self._get_indent())
        """
        if self.silent:
            return

        indent_str = indent if indent is not None else self._get_indent()

        if self.name:
            self.logger.log(self.level, "%s%s %s: %s", indent_str, emoji, self.name, message)
        else:
            self.logger.log(self.level, "%s%s %s", indent_str, emoji, message)

    def start(self, log: bool = False) -> "Timer":
        """Explicitly start the timer.

        This allows using the timer without a context manager.
        Can be called multiple times to restart timing.

        Args:
            log: If True, logs the start message.

        Returns:
            Self for method chaining

        Example:
            timer = Timer("operation")
            timer.start()
            do_work()
            elapsed = timer.stop()
            print(f"Took {elapsed:.2f}s")
        """
        # Start timing
        self._start_time = time.perf_counter()
        self._end_time = None
        self._last_checkpoint_time = None

        if not self.silent and log:
            self._log(message="started")

        # Track depth for proper indentation (only if not silent and track_depth enabled)
        if not self.silent and self.track_depth:
            self._depth = _timer_depth.get()

        return self

    def stop(self) -> float:
        """Explicitly stop the timer and return elapsed time.

        This allows using the timer without a context manager.
        Returns the elapsed time and optionally logs completion.

        Returns:
            Elapsed time in seconds

        Example:
            timer = Timer("operation")
            timer.start()
            do_work()
            elapsed = timer.stop()
            print(f"Took {elapsed:.2f}s")
        """
        if self._start_time is None:
            if not self.silent:
                self.logger.warning("stop() called before timer started")
            return 0.0

        # Stop timing
        self._end_time = time.perf_counter()
        duration = self.elapsed

        # Log completion message (uses depth set by start())
        self._log(message=f"completed in {_format_duration(duration)}")

        return duration

    def log_elapsed(self, message: str | None = None, depth: int = 0) -> float:
        """Explicitly log the elapsed time, even in silent mode.

        This method allows you to log elapsed time at any point, regardless of
        the silent setting. Useful when you want silent timing but need to log
        at specific checkpoints.

        Args:
            message: Optional message to include in the log. If not provided,
                    uses the timer name.
            depth: Indentation level for the log message (default: 0).
                  Each level adds 2 spaces of indentation.

        Returns:
            Current elapsed time in seconds
        """
        elapsed = self.elapsed
        log_message = message or self.name

        # Always log, even in silent mode (so we temporarily disable silent check)
        indent_str = "  " * depth
        was_silent = self.silent
        original_name = self.name

        self.silent = False
        self.name = log_message  # Use the message as the name (can be None)
        self._log(message=_format_duration(elapsed), indent=indent_str)

        # Restore original state
        self.silent = was_silent
        self.name = original_name
        return elapsed

    def checkpoint(self, name: str) -> float:
        """Log a checkpoint with intermediate timing.

        Args:
            name: Name of the checkpoint

        Returns:
            Elapsed time since last checkpoint (or start if first checkpoint)
        """
        if self._start_time is None:
            if not self.silent:
                self.logger.warning("checkpoint() called before timer started")
            return 0.0

        current_time = time.perf_counter()

        # Calculate lap time (time since last checkpoint or start)
        if self._last_checkpoint_time is not None:
            lap_time = current_time - self._last_checkpoint_time
        else:
            lap_time = current_time - self._start_time

        # Calculate total elapsed time
        total_time = current_time - self._start_time

        # Log checkpoint with lap and total times
        checkpoint_msg = f"{name} (lap: {_format_duration(lap_time)}, total: {_format_duration(total_time)})"
        self._log(message=checkpoint_msg, emoji="├─")

        # Update last checkpoint time
        self._last_checkpoint_time = current_time

        return lap_time

    def __enter__(self) -> "Timer":
        """Enter the context and start timing."""
        # Use start() to begin timing and log (this sets self._depth to current depth)
        self.start()

        # Increment depth for nested timers, including initial_depth (only if not silent and track_depth enabled)
        if not self.silent and self.track_depth:
            self._depth_token = _timer_depth.set(self._depth + self.initial_depth + 1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and log the elapsed time."""
        # Stop timing and log completion (must happen before resetting depth)
        self.stop()

        # Restore depth for nested timers
        if not self.silent and self._depth_token is not None:
            _timer_depth.reset(self._depth_token)

        # Don't suppress exceptions
        return False
