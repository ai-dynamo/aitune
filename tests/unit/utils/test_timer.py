# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Timer utilities."""

import logging
import time

import pytest

from aitune.utils.timer import Timer, _format_duration

# Tests for _format_duration function


def test_format_milliseconds():
    """Test formatting durations less than 1 second."""
    assert _format_duration(0.001) == "1.00ms"
    assert _format_duration(0.0156) == "15.60ms"
    assert _format_duration(0.5) == "500.00ms"
    assert _format_duration(0.999) == "999.00ms"


def test_format_seconds():
    """Test formatting durations between 1 and 60 seconds."""
    assert _format_duration(1.0) == "1.00s"
    assert _format_duration(5.5) == "5.50s"
    assert _format_duration(30.25) == "30.25s"
    assert _format_duration(59.99) == "59.99s"


def test_format_minutes():
    """Test formatting durations 60 seconds and above."""
    assert _format_duration(60.0) == "1.00min"
    assert _format_duration(90.0) == "1.50min"
    assert _format_duration(120.0) == "2.00min"
    assert _format_duration(3600.0) == "60.00min"


def test_format_zero():
    """Test formatting zero duration."""
    assert _format_duration(0.0) == "0.00ms"


def test_format_edge_cases():
    """Test edge cases around boundaries."""
    # Just under 1 second
    assert "ms" in _format_duration(0.9999)
    # Exactly 1 second
    assert _format_duration(1.0) == "1.00s"
    # Just under 60 seconds
    assert "s" in _format_duration(59.999)
    # Exactly 60 seconds
    assert _format_duration(60.0) == "1.00min"


def test_timer_with_custom_sink():
    """Test Timer with custom logger."""
    custom_logger = logging.getLogger("custom")
    timer = Timer("test", sink=custom_logger.info)
    assert timer.sink == custom_logger.info


def test_basic_timer_context(caplog):
    """Test basic timer context manager usage."""
    with caplog.at_level(logging.INFO):
        with Timer("operation"):
            time.sleep(0.01)

    # Check that we have completion message (no start message by default)
    messages = [record.message for record in caplog.records]
    assert any("operation: completed in" in msg for msg in messages)


def test_timer_elapsed_property():
    """Test the elapsed property."""
    with Timer("test") as timer:
        # Elapsed should be near zero at start
        assert timer.elapsed >= 0.0
        assert timer.elapsed < 0.1

        time.sleep(0.05)

        # Elapsed should be approximately 0.05 seconds
        assert timer.elapsed >= 0.04
        assert timer.elapsed < 0.1

    # After context, elapsed should still be accessible
    assert timer.elapsed >= 0.04


def test_timer_elapsed_before_start():
    """Test elapsed property before timer is started."""
    timer = Timer("test")
    assert timer.elapsed == pytest.approx(0.0)


def test_timer_elapsed_after_completion():
    """Test elapsed property after timer completes."""
    with Timer("test") as timer:
        time.sleep(0.01)

    # Elapsed should remain constant after completion
    elapsed1 = timer.elapsed
    time.sleep(0.01)
    elapsed2 = timer.elapsed
    assert elapsed1 == elapsed2


def test_checkpoint_single(caplog):
    """Test single checkpoint."""
    with caplog.at_level(logging.INFO):
        with Timer("process") as timer:
            time.sleep(0.01)
            lap_time = timer.checkpoint("step 1")

            # Lap time should be approximately 0.01s
            assert lap_time >= 0.009
            assert lap_time < 0.05

    # Check checkpoint message
    messages = [record.message for record in caplog.records]
    assert any("step 1" in msg and "lap:" in msg and "total:" in msg for msg in messages)


def test_checkpoint_multiple(caplog):
    """Test multiple checkpoints."""
    with caplog.at_level(logging.INFO):
        with Timer("pipeline") as timer:
            time.sleep(0.01)
            lap1 = timer.checkpoint("stage 1")

            time.sleep(0.01)
            lap2 = timer.checkpoint("stage 2")

            time.sleep(0.01)
            lap3 = timer.checkpoint("stage 3")

    # Each lap should be approximately 0.01s
    assert lap1 >= 0.009 and lap1 < 0.05
    assert lap2 >= 0.009 and lap2 < 0.05
    assert lap3 >= 0.009 and lap3 < 0.05

    # Check all checkpoints were logged
    messages = [record.message for record in caplog.records]
    assert any("stage 1" in msg for msg in messages)
    assert any("stage 2" in msg for msg in messages)
    assert any("stage 3" in msg for msg in messages)


def test_checkpoint_before_start(caplog):
    """Test checkpoint called before timer starts."""
    timer = Timer("test")

    with caplog.at_level(logging.INFO):
        lap_time = timer.checkpoint("premature")

    assert lap_time == pytest.approx(0.0)
    assert any("checkpoint() called before timer started" in record.message for record in caplog.records)


def test_timer_with_exception(caplog):
    """Test timer when exception is raised."""
    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            with Timer("failing operation"):
                time.sleep(0.01)
                raise ValueError("test error")

    # Should log completion message (timer just tracks duration, not success/failure)
    messages = [record.message for record in caplog.records]
    assert any("failing operation: completed in" in msg for msg in messages)


def test_timer_exception_not_suppressed():
    """Test that exceptions are not suppressed by timer."""
    with pytest.raises(RuntimeError, match="test"):
        with Timer("test"):
            raise RuntimeError("test")


def test_nested_timers_depth(caplog):
    """Test nested timers have proper depth/indentation."""
    with caplog.at_level(logging.INFO):
        with Timer("outer", track_depth=True):
            time.sleep(0.01)
            with Timer("inner", track_depth=True):
                time.sleep(0.01)
                with Timer("innermost", track_depth=True):
                    time.sleep(0.01)

    messages = [record.message for record in caplog.records]

    # Find messages for each timer
    outer_msgs = [msg for msg in messages if "outer:" in msg]
    inner_msgs = [msg for msg in messages if "inner:" in msg and "innermost" not in msg]
    innermost_msgs = [msg for msg in messages if "innermost:" in msg]

    # Check that messages exist
    assert len(outer_msgs) >= 1
    assert len(inner_msgs) >= 1
    assert len(innermost_msgs) >= 1


def test_nested_timer_completion_order(caplog):
    """Test that nested timers complete in correct order."""
    with caplog.at_level(logging.INFO):
        with Timer("A", track_depth=True):
            with Timer("B", track_depth=True):
                with Timer("C", track_depth=True):
                    time.sleep(0.01)

    # All timers should complete successfully (no start messages by default)
    messages = [record.message for record in caplog.records]
    assert len(messages) == 3
    assert messages[0].startswith("    ⏱️ C: completed in")
    assert messages[1].startswith("  ⏱️ B: completed in")
    assert messages[2].startswith("⏱️ A: completed in")


def test_timer_get_indent():
    """Test _get_indent method."""
    timer = Timer("test")
    timer._depth = 0
    assert timer._get_indent() == ""

    timer._depth = 1
    assert timer._get_indent() == "  "

    timer._depth = 3
    assert timer._get_indent() == "      "


def test_timer_logging_levels(caplog):
    """Test that timer respects custom logging levels."""
    # Test with DEBUG level
    with caplog.at_level(logging.DEBUG):
        with Timer("debug test", sink=logging.debug):
            time.sleep(0.01)

    debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug_messages) >= 1  # Completion (no start message by default)

    caplog.clear()

    # Test with WARNING level
    with caplog.at_level(logging.WARNING):
        with Timer("warning test", sink=logging.warning):
            time.sleep(0.01)

    warning_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_messages) >= 1  # Completion (no start message by default)


def test_concurrent_timers(caplog):
    """Test multiple timers running in sequence."""
    with caplog.at_level(logging.INFO):
        with Timer("task1"):
            time.sleep(0.01)

        with Timer("task2"):
            time.sleep(0.01)

        with Timer("task3"):
            time.sleep(0.01)

    messages = [record.message for record in caplog.records]
    assert any("task1" in msg for msg in messages)
    assert any("task2" in msg for msg in messages)
    assert any("task3" in msg for msg in messages)

    # Each should have completed (no start messages by default)
    completed = [msg for msg in messages if "completed in" in msg]
    assert len(completed) == 3


def test_timer_name_preserved(caplog):
    """Test that timer name is preserved throughout lifecycle."""
    timer_name = "my custom operation"

    with caplog.at_level(logging.INFO):
        with Timer(timer_name) as timer:
            assert timer.name == timer_name
            time.sleep(0.01)
            timer.checkpoint("checkpoint")

    messages = [record.message for record in caplog.records]
    # Name should appear in all messages (checkpoint and complete, no start by default)
    assert sum(1 for msg in messages if timer_name in msg) >= 2  # checkpoint, complete


def test_checkpoint_lap_time_calculation():
    """Test that checkpoint lap times are calculated correctly."""
    with Timer("test") as timer:
        time.sleep(0.02)
        lap1 = timer.checkpoint("cp1")

        time.sleep(0.03)
        lap2 = timer.checkpoint("cp2")

        time.sleep(0.01)
        lap3 = timer.checkpoint("cp3")

    # First lap should be ~0.02s
    assert 0.015 < lap1 < 0.04

    # Second lap should be ~0.03s (since last checkpoint)
    assert 0.025 < lap2 < 0.05

    # Third lap should be ~0.01s (since last checkpoint)
    assert 0.005 < lap3 < 0.03


def test_checkpoint_total_time():
    """Test that checkpoint shows correct total time."""
    with Timer("test") as timer:
        time.sleep(0.01)
        timer.checkpoint("cp1")

        time.sleep(0.01)
        timer.checkpoint("cp2")

        # Total elapsed should be approximately 0.02s
        assert 0.015 < timer.elapsed < 0.04


# Manual start/stop without context manager


def test_timer_manual_start_stop():
    """Test using timer with explicit start/stop."""
    timer = Timer("manual operation", silent=True)

    # Start timer
    returned_timer = timer.start()
    assert returned_timer is timer  # Should return self for chaining
    assert timer._start_time is not None

    time.sleep(0.01)

    # Stop timer
    elapsed = timer.stop()
    assert elapsed >= 0.01
    assert elapsed < 0.05
    assert timer._end_time is not None

    # Elapsed should still be accessible
    assert timer.elapsed == elapsed


def test_timer_manual_start_stop_with_logging(caplog):
    """Test manual start/stop with logging."""
    with caplog.at_level(logging.INFO):
        timer = Timer("manual operation")
        timer.start(log=True)  # Explicitly request start logging
        time.sleep(0.01)
        elapsed = timer.stop()

    messages = [record.message for record in caplog.records]
    assert len(messages) == 2
    assert any("started" in msg for msg in messages)
    assert any("completed in" in msg for msg in messages)
    assert elapsed >= 0.01


def test_timer_manual_start_without_logging(caplog):
    """Test manual start/stop without start logging (default behavior)."""
    with caplog.at_level(logging.INFO):
        timer = Timer("manual operation")
        timer.start()  # Default log=False, no start message
        time.sleep(0.01)
        elapsed = timer.stop()

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1  # Only completion message
    assert not any("started" in msg for msg in messages)
    assert any("completed in" in msg for msg in messages)
    assert elapsed >= 0.01


def test_timer_manual_start_multiple_times():
    """Test restarting timer with start() method."""
    timer = Timer("operation", silent=True)

    # First timing
    timer.start()
    time.sleep(0.01)
    elapsed1 = timer.stop()

    # Restart timing
    timer.start()
    time.sleep(0.02)
    elapsed2 = timer.stop()

    # Second run should be longer
    assert elapsed2 > elapsed1
    assert elapsed2 >= 0.02


def test_timer_manual_stop_before_start(caplog):
    """Test calling stop() before start()."""
    timer = Timer("test")

    with caplog.at_level(logging.INFO):
        elapsed = timer.stop()

    assert elapsed == pytest.approx(0.0)
    assert any("stop() called before timer started" in record.message for record in caplog.records)


def test_timer_manual_with_checkpoint():
    """Test using checkpoints with manual start/stop."""
    timer = Timer("operation", silent=True)
    timer.start()

    time.sleep(0.01)
    lap1 = timer.checkpoint("checkpoint 1")

    time.sleep(0.01)
    lap2 = timer.checkpoint("checkpoint 2")

    elapsed = timer.stop()

    # Verify lap times
    assert 0.009 < lap1 < 0.05
    assert 0.009 < lap2 < 0.05
    assert elapsed >= 0.02


def test_timer_manual_with_log_elapsed():
    """Test using log_elapsed with manual start/stop."""
    timer = Timer("operation", silent=True)
    timer.start()

    time.sleep(0.01)
    elapsed1 = timer.log_elapsed("Midpoint")

    time.sleep(0.01)
    elapsed2 = timer.stop()

    assert elapsed1 < elapsed2
    assert elapsed2 >= 0.02


# Edge cases and error conditions


def test_timer_silent_mode(caplog):
    """Test timer in silent mode produces no logs but tracks time."""
    with caplog.at_level(logging.INFO):
        with Timer("silent operation", silent=True) as timer:
            time.sleep(0.01)

    # No messages should be logged
    messages = [record.message for record in caplog.records]
    assert len(messages) == 0

    # But elapsed time should still be tracked
    assert timer.elapsed >= 0.01
    assert timer.elapsed < 0.05


def test_timer_silent_checkpoint(caplog):
    """Test that checkpoints in silent mode don't log but return lap time."""
    with caplog.at_level(logging.INFO):
        with Timer("silent", silent=True) as timer:
            time.sleep(0.01)
            lap1 = timer.checkpoint("checkpoint 1")
            time.sleep(0.01)
            lap2 = timer.checkpoint("checkpoint 2")

    # No log messages should be produced
    assert len(caplog.records) == 0

    # But lap times should still be calculated
    assert 0.009 < lap1 < 0.05
    assert 0.009 < lap2 < 0.05


def test_timer_silent_with_explicit_logging(caplog):
    """Test log_elapsed() explicitly logs even in silent mode."""
    with caplog.at_level(logging.INFO):
        with Timer("silent operation", silent=True) as timer:
            time.sleep(0.01)
            # Explicitly log the elapsed time
            elapsed = timer.log_elapsed("Checkpoint reached")
            time.sleep(0.01)
            elapsed2 = timer.log_elapsed()

    # Should have exactly 2 log messages from explicit log_elapsed() calls
    messages = [record.message for record in caplog.records]
    assert len(messages) == 2
    assert "Checkpoint reached:" in messages[0]
    assert "ms" in messages[0] or "s" in messages[0]
    assert "silent operation:" in messages[1]

    # Returned elapsed times should be valid
    assert elapsed >= 0.009
    assert elapsed2 >= elapsed


def test_timer_log_elapsed_normal_mode(caplog):
    """Test log_elapsed() works in normal (non-silent) mode too."""
    with caplog.at_level(logging.INFO):
        with Timer("normal operation") as timer:
            time.sleep(0.01)
            timer.log_elapsed("Manual checkpoint")

    messages = [record.message for record in caplog.records]
    # Should have: manual checkpoint, completion (no start message by default)
    assert len(messages) >= 2
    assert any("Manual checkpoint:" in msg for msg in messages)
    assert any("completed in" in msg for msg in messages)


def test_timer_log_elapsed_with_depth(caplog):
    """Test log_elapsed() with different indentation depths."""
    with caplog.at_level(logging.INFO):
        with Timer("operation", silent=True) as timer:
            time.sleep(0.01)
            timer.log_elapsed("Level 0", depth=0)
            timer.log_elapsed("Level 1", depth=1)
            timer.log_elapsed("Level 2", depth=2)
            timer.log_elapsed("Level 3", depth=3)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 4

    # Check indentation levels
    assert messages[0].startswith("⏱️ Level 0:")  # No indent
    assert messages[1].startswith("  ⏱️ Level 1:")  # 2 spaces
    assert messages[2].startswith("    ⏱️ Level 2:")  # 4 spaces
    assert messages[3].startswith("      ⏱️ Level 3:")  # 6 spaces


def test_timer_log_elapsed_with_depth_and_nested_timers(caplog):
    """Test log_elapsed() depth parameter alongside nested timers."""
    with caplog.at_level(logging.INFO):
        with Timer("outer", track_depth=True) as outer:
            time.sleep(0.01)
            outer.log_elapsed("Outer checkpoint", depth=0)
            with Timer("inner", track_depth=True) as inner:
                time.sleep(0.01)
                inner.log_elapsed("Inner checkpoint", depth=1)

    messages = [record.message for record in caplog.records]

    # Find the explicit log_elapsed messages
    outer_log = [msg for msg in messages if "Outer checkpoint" in msg][0]
    inner_log = [msg for msg in messages if "Inner checkpoint" in msg][0]

    # Verify indentation
    assert outer_log.startswith("⏱️ Outer checkpoint:")
    assert inner_log.startswith("  ⏱️ Inner checkpoint:")


def test_timer_without_context_manager():
    """Test timer properties without using context manager."""
    timer = Timer("test")

    # Elapsed should be 0 before start
    assert timer.elapsed == pytest.approx(0.0)

    # Checkpoint before start should warn
    assert timer.checkpoint("test") == pytest.approx(0.0)
