# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-parallel configuration for Flux."""

from enum import Enum


class ContextParallelMode(str, Enum):
    """Supported context-parallel attention modes."""

    RING = "ring"
    ULYSSES = "ulysses"

    def __str__(self) -> str:
        """Return the command-line value."""
        return self.value
