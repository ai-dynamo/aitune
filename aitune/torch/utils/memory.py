# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch memory utilities for garbage collection and CUDA cache cleanup."""

import ctypes
import gc

import torch


def cleanup_memory() -> None:
    """Perform garbage collection and CUDA cache cleanup."""
    cpu_cleanup()
    gpu_cleanup()


def gpu_cleanup():
    """Perform CUDA cache cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cpu_cleanup():
    """Perform CPU memory cleanup.

    Note: Only supported on Linux and Windows platforms.
    """
    gc.collect()
    cpu_cleanup_low_level()


def cpu_cleanup_low_level():
    """Perform low-level CPU memory cleanup."""
    import platform

    try:
        if platform.system() == "Linux":
            ctypes.CDLL(ctypes.util.find_library("c")).malloc_trim(0)
        elif platform.system() == "Windows":
            ctypes.windll.msvcrt._malloc_trim(0)
    except (OSError, AttributeError):
        # Silently ignore if platform-specific cleanup is not available
        pass
