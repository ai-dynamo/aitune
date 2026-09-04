# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance validation configuration."""

from enum import Enum


class PerformanceValidationMode(str, Enum):
    """Controls eager-baseline profiling and its effect on backend selection.

    Attributes:
        ENABLED: Profile performance and reject or fall back from backends that do not beat eager.
        DIAGNOSTIC: Profile and report performance without using the eager comparison for selection.
        DISABLED: Skip eager-baseline profiling and comparison.
    """

    ENABLED = "enabled"
    DIAGNOSTIC = "diagnostic"
    DISABLED = "disabled"
