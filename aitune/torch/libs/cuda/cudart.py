# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA runtime library loader and singleton accessor."""

import ctypes
import glob
import os
from logging import getLogger

logger = getLogger(__name__)


def _load_cudart() -> ctypes.CDLL:
    """Load libcudart, searching LD_LIBRARY_PATH and common CUDA install paths."""
    search_dirs = [
        *os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep),
        "/usr/local/cuda/lib64",
        "/usr/lib",
    ]
    candidates = [p for d in search_dirs for p in glob.glob(os.path.join(d, "libcudart.so*"))]
    candidates.append("libcudart.so")  # final fallback — relies on ldconfig
    for lib in candidates:
        try:
            handle = ctypes.CDLL(lib)
            logger.debug("Loaded CUDA runtime library: %s", lib)
            return handle
        except OSError:
            continue
    raise OSError("Cannot find libcudart. Ensure CUDA is installed and libcudart.so is on LD_LIBRARY_PATH.")


_CUDART: ctypes.CDLL | None = None


def cudart() -> ctypes.CDLL:
    """Return the global libcudart handle, loading it on first call."""
    global _CUDART
    if _CUDART is None:
        _CUDART = _load_cudart()
    return _CUDART
