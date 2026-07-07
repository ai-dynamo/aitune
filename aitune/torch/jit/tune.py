# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module for tuning JIT models."""

from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patcher import Patcher
from aitune.torch.utils.memory import cleanup_memory


def deferred():
    """Enable deferred tuning on the next normal forward pass."""
    if config.mode != JITMode.TUNE_DEFERRED:
        raise ValueError(f"tune.deferred() requires JITMode.TUNE_DEFERRED, but current mode is {config.mode.value}")

    cleanup_memory()
    Patcher.enable_tune_deferred()
