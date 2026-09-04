# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance validation configuration."""

from enum import Enum


class PerformanceValidationMode(str, Enum):
    """Controls eager-baseline profiling and its effect on backend selection.

    Attributes:
        DISABLED: Skip eager-baseline profiling and comparison.
        DIAGNOSTIC: Profile and report performance without using the eager comparison for selection.
        ENFORCED: Profile performance and reject or fall back from backends that do not beat eager.
    """

    DISABLED = "disabled"
    DIAGNOSTIC = "diagnostic"
    ENFORCED = "enforced"
