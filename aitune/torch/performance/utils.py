# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for performance profiles."""

from __future__ import annotations

from typing import Any


def _qualified_type_name(value: Any) -> str:
    """Return a stable-enough type name for report metadata."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
