# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for disk-space utilities."""

import errno
import logging
from collections import namedtuple
from pathlib import Path

import pytest

from aitune.utils.disk_space import (
    DEFAULT_MIN_FREE_BYTES,
    DiskSpaceError,
    check_disk_space,
    raise_if_out_of_space,
)

_DiskUsage = namedtuple("_DiskUsage", ("total", "used", "free"))


def _usage(total: int, used: int, free: int) -> _DiskUsage:
    """Build a disk-usage tuple mirroring ``shutil.disk_usage``'s return shape."""
    return _DiskUsage(total, used, free)


def test_default_min_free_bytes_is_50_gib():
    """Default minimum free space threshold is 50 GiB."""
    assert DEFAULT_MIN_FREE_BYTES == 50 * 1024**3


def test_disk_space_error_is_oserror_subclass():
    """DiskSpaceError derives from OSError so generic handlers still catch it."""
    assert issubclass(DiskSpaceError, OSError)


def test_check_disk_space_returns_free_bytes_above_threshold(mocker, tmp_path, caplog):
    """check_disk_space returns free bytes and does not warn when above threshold."""
    mocker.patch(
        "aitune.utils.disk_space.shutil.disk_usage",
        return_value=_usage(100 * 1024**3, 50 * 1024**3, 50 * 1024**3),
    )

    with caplog.at_level(logging.WARNING):
        free = check_disk_space(tmp_path, min_free_bytes=10 * 1024**3)

    assert free == 50 * 1024**3
    assert not any("disk space" in record.message.lower() for record in caplog.records)


def test_check_disk_space_warns_below_threshold(mocker, tmp_path, caplog):
    """check_disk_space logs a WARNING when free space is below threshold."""
    mocker.patch(
        "aitune.utils.disk_space.shutil.disk_usage",
        return_value=_usage(100 * 1024**3, 99 * 1024**3, 1 * 1024**3),
    )

    with caplog.at_level(logging.WARNING):
        free = check_disk_space(tmp_path, min_free_bytes=10 * 1024**3)

    assert free == 1 * 1024**3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning when free space is below threshold"
    msg = warnings[0].message
    assert str(tmp_path) in msg
    assert "1.00 GiB" in msg or "1.0 GiB" in msg
    assert "10" in msg


def test_check_disk_space_uses_nearest_existing_parent(mocker, tmp_path, caplog):
    """When the target path doesn't exist yet, check the nearest existing parent."""
    disk_usage_mock = mocker.patch(
        "aitune.utils.disk_space.shutil.disk_usage",
        return_value=_usage(100 * 1024**3, 50 * 1024**3, 50 * 1024**3),
    )
    missing = tmp_path / "does" / "not" / "exist"

    check_disk_space(missing)

    disk_usage_mock.assert_called_once_with(tmp_path)


def test_check_disk_space_env_var_overrides_default(mocker, tmp_path, caplog, monkeypatch):
    """AITUNE_CACHE_MIN_FREE_BYTES overrides the default threshold."""
    monkeypatch.setenv("AITUNE_CACHE_MIN_FREE_BYTES", str(5 * 1024**3))  # 5 GiB
    mocker.patch(
        "aitune.utils.disk_space.shutil.disk_usage",
        return_value=_usage(100 * 1024**3, 93 * 1024**3, 7 * 1024**3),
    )

    with caplog.at_level(logging.WARNING):
        check_disk_space(tmp_path)

    # 7 GiB free, threshold 5 GiB, so no warning expected.
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_check_disk_space_env_var_triggers_warning(mocker, tmp_path, caplog, monkeypatch):
    """AITUNE_CACHE_MIN_FREE_BYTES raises the bar and should trigger a warning."""
    monkeypatch.setenv("AITUNE_CACHE_MIN_FREE_BYTES", str(50 * 1024**3))  # 50 GiB
    mocker.patch(
        "aitune.utils.disk_space.shutil.disk_usage",
        return_value=_usage(100 * 1024**3, 90 * 1024**3, 10 * 1024**3),
    )

    with caplog.at_level(logging.WARNING):
        check_disk_space(tmp_path)

    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_raise_if_out_of_space_translates_enospc(tmp_path):
    """OSError with errno.ENOSPC is re-raised as DiskSpaceError with the path in the message."""
    original = OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(DiskSpaceError) as exc_info:
        raise_if_out_of_space(original, path=tmp_path)

    assert exc_info.value.__cause__ is original
    assert str(tmp_path) in str(exc_info.value)


def test_raise_if_out_of_space_ignores_other_oserrors(tmp_path):
    """Non-ENOSPC OSErrors pass through raise_if_out_of_space without raising."""
    other = OSError(errno.EACCES, "Permission denied")

    # Should not raise.
    raise_if_out_of_space(other, path=tmp_path)


def test_raise_if_out_of_space_ignores_non_oserror(tmp_path):
    """Non-OSError exceptions pass through raise_if_out_of_space without raising."""
    exc = ValueError("unrelated")

    # Should not raise.
    raise_if_out_of_space(exc, path=tmp_path)


def test_disk_space_error_message_mentions_path_and_free_bytes():
    """DiskSpaceError renders path and (optional) free-bytes context in its message."""
    err = DiskSpaceError(path=Path("/tmp/aitune_cache"), free_bytes=123 * 1024**2)

    msg = str(err)
    assert "/tmp/aitune_cache" in msg
    assert "123" in msg  # free bytes shown in some human-readable form
