# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialization utilities."""

import logging
from dataclasses import asdict, fields, is_dataclass
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


def drop_unsupported_fields(config_class: type, config: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that are not fields of the ``config_class`` dataclass.

    Loads configs saved with another library version, e.g. Torch-TensorRT ``CompilationSettings``
    rejects ``enabled_precisions`` since 2.15.
    """
    supported = {f.name for f in fields(config_class)}
    unsupported = sorted(set(config) - supported)
    if unsupported:
        logging.getLogger(__name__).warning(
            "Ignoring %s settings not supported by the installed version: %s", config_class.__name__, unsupported
        )
    return {name: value for name, value in config.items() if name in supported}
