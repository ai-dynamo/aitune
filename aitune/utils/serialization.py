# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON serialization utilities."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def json_serialize(x: Any) -> Any:
    """Convert a Python object into a JSON-serializable structure."""
    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): json_serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_serialize(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value
    if is_dataclass(x):
        return json_serialize(asdict(x))
    if isinstance(x, datetime):
        return x.isoformat()
    return str(x)
