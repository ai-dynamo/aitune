# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for logging utilities."""

from aitune.utils.logging import log_to_file, write_exception_log


def test_log_to_file_appends_message_and_exception(tmp_path):
    """Log helper should append messages and exception details without truncating."""
    log_file = tmp_path / "nested" / "build.log"
    log_file.parent.mkdir()
    log_file.write_text("existing log\n", encoding="utf-8")

    try:
        raise RuntimeError("validation failed")
    except RuntimeError as e:
        log_to_file(log_file, "Backend validation failed", exception=e)

    log_text = log_file.read_text(encoding="utf-8")
    assert "existing log" in log_text
    assert "Backend validation failed" in log_text
    assert "Exception type: RuntimeError" in log_text
    assert "Exception details: validation failed" in log_text
    assert "Full traceback:" in log_text


def test_log_to_file_preserves_large_exception_payloads(tmp_path):
    """Exception logs should preserve the complete compiler payload."""
    message = "failure context\n" + "graph node\n" * 10_000 + "root cause"
    log_file = tmp_path / "build.log"

    try:
        raise RuntimeError(message)
    except RuntimeError as exception:
        log_to_file(log_file, "Backend validation failed", exception=exception)

    log_text = log_file.read_text(encoding="utf-8")
    assert message in log_text
    assert "truncated" not in log_text


def test_write_exception_log_uses_cache_directory(tmp_path):
    cache_dir = tmp_path / "module"

    try:
        raise RuntimeError("tuning failed")
    except RuntimeError as exception:
        error_log = write_exception_log(cache_dir, exception)

    assert error_log == cache_dir / "error.log"
    assert "Traceback (most recent call last)" in error_log.read_text()
    assert "RuntimeError: tuning failed" in error_log.read_text()
