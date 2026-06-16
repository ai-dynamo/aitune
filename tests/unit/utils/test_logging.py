# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for logging utilities."""

from aitune.utils.logging import log_to_file


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
