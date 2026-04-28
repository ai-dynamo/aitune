# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk-space checks for the AITune cache directory.

Out-of-space errors used to fail silently during JIT tuning: the broad
``except Exception`` in the tune loop caught ``OSError(ENOSPC)`` and fell back
to eager mode without surfacing the cause. This module provides:

* :class:`DiskSpaceError` — a clear, typed error carrying path + free-bytes context.
* :func:`check_disk_space` — pre-flight warning when the cache device is low.
* :func:`raise_if_out_of_space` — translator that turns caught ``OSError(ENOSPC)``
  into :class:`DiskSpaceError` at the call site that swallowed it.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
from pathlib import Path

_logger = logging.getLogger(__name__)

DEFAULT_MIN_FREE_BYTES: int = 50 * 1024**3  # 50 GiB

_ENV_VAR = "AITUNE_CACHE_MIN_FREE_BYTES"


def _format_bytes(num_bytes: int) -> str:
    """Render a byte count using the largest binary unit that keeps it >= 1."""
    step = 1024.0
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < step or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= step
    return f"{value:.2f} TiB"  # pragma: no cover — loop always returns


class DiskSpaceError(OSError):
    """Raised when the AITune cache device has run out of (or very little) space.

    Inherits from :class:`OSError` so call sites that already handle I/O errors
    continue to catch it. The string form embeds the target path and, when
    known, the remaining free bytes — so a log line or traceback alone tells
    the operator exactly which device is full.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        free_bytes: int | None = None,
        message: str | None = None,
    ) -> None:
        """Build an ENOSPC-style error with ``path`` and ``free_bytes`` context in the message."""
        parts = [message or "No space left on AITune cache device"]
        if path is not None:
            parts.append(f"path={path}")
        if free_bytes is not None:
            parts.append(f"free={_format_bytes(free_bytes)}")
        super().__init__(errno.ENOSPC, "; ".join(parts))
        self.path = Path(path) if path is not None else None
        self.free_bytes = free_bytes


def _configured_min_free_bytes() -> int:
    """Resolve the minimum-free-bytes threshold, honouring the env override."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return DEFAULT_MIN_FREE_BYTES
    try:
        return int(raw)
    except ValueError:
        _logger.warning("Invalid %s=%r; falling back to default %d", _ENV_VAR, raw, DEFAULT_MIN_FREE_BYTES)
        return DEFAULT_MIN_FREE_BYTES


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def check_disk_space(path: Path | str, *, min_free_bytes: int | None = None) -> int:
    """Inspect free space for the filesystem that hosts ``path``.

    If ``path`` does not yet exist (the cache dir is typically created lazily),
    the nearest existing parent is probed instead — that's the device any future
    write would land on.

    A WARNING log is emitted when the free space is below ``min_free_bytes``
    (resolved from the ``AITUNE_CACHE_MIN_FREE_BYTES`` env var or the 50 GiB
    default). Returns the number of free bytes observed.
    """
    target = _nearest_existing(Path(path))
    threshold = min_free_bytes if min_free_bytes is not None else _configured_min_free_bytes()

    usage = shutil.disk_usage(target)
    if usage.free < threshold:
        _logger.warning(
            "Low disk space for AITune cache: %s free at %s (below %s threshold). "
            "Tuning may fail with DiskSpaceError if writes exceed available space.",
            _format_bytes(usage.free),
            target,
            _format_bytes(threshold),
        )
    return usage.free


def raise_if_out_of_space(exc: BaseException, path: Path | str | None = None) -> None:
    """Re-raise ``exc`` as :class:`DiskSpaceError` when it represents ENOSPC.

    Use inside ``except`` blocks that otherwise suppress or swallow exceptions,
    so out-of-space failures stop propagating silently::

        try:
            write_artifact(...)
        except Exception as e:
            raise_if_out_of_space(e, path=cache_dir)
            # ... existing fallback / best-effort handling ...
    """
    if not isinstance(exc, OSError):
        return
    if exc.errno != errno.ENOSPC:
        return

    free_bytes: int | None = None
    if path is not None:
        try:
            free_bytes = shutil.disk_usage(_nearest_existing(Path(path))).free
        except OSError:
            free_bytes = None

    raise DiskSpaceError(path=path, free_bytes=free_bytes) from exc
