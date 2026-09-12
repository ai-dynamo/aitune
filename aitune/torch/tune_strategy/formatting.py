# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep tuning progress messages consistent without depending on strategy implementations."""

import sys


def fmt_speedup_comparison(speedup: float, passed: bool) -> str:
    """Keeps per-backend speedup formatting consistent across tuning strategies."""
    indicator = "▲ faster" if passed else "▼ slower"
    if sys.stdout.isatty():
        color = "\033[92m" if passed else "\033[33m"
        indicator = f"{color}{indicator}\033[0m"
    return f"{speedup:.2f}x ({indicator})"


def fmt_speedup_msg_short(speedup: float, detail: str) -> str:
    """Returns a compact speedup line without module/backend fields."""
    if sys.stdout.isatty():
        lightning = "\033[94m⚡\033[0m"
        speedup_str = f"\033[92m\033[1m{speedup:.2f}x\033[0m"
    else:
        lightning = "⚡"
        speedup_str = f"{speedup:.2f}x"
    return f"{lightning} speedup: {speedup_str} ({detail})"


def fmt_speedup_msg(speedup: float, detail: str, name: str, backend_desc: str) -> str:
    """Returns the full speedup summary line with module/backend fields."""
    if sys.stdout.isatty():
        lightning = "\033[94m⚡\033[0m"
        speedup_str = f"\033[92m\033[1m{speedup:.2f}x\033[0m"
        name_str = f"\033[1m{name}\033[0m"
        backend_str = f"\033[96m{backend_desc}\033[0m"
    else:
        lightning = "⚡"
        speedup_str = f"{speedup:.2f}x"
        name_str = name
        backend_str = backend_desc
    return f"{lightning} {name_str} | backend: {backend_str} | speedup: {speedup_str} ({detail})"
