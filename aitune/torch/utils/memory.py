# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
