# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor utility helpers."""

import re
from typing import Any, Literal

import torch


def format_tensor_name(path: int | str | tuple[int | str, ...], prefix: Literal["input", "output"]) -> str:
    """Format a semantic path as a readable, backend-safe tensor name."""
    if path == ():
        return prefix
    readable_path = re.sub(r"[^A-Za-z0-9_]+", "_", repr(path)).strip("_")
    return f"{prefix}_{readable_path}"


def none_at_tensors(obj: Any) -> Any:
    """Return a copy of *obj* with every ``torch.Tensor`` leaf replaced by ``None``.

    Preserves the type of containers (tuple, list, dict).
    """
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, dict):
        return {k: none_at_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        inner = [none_at_tensors(v) for v in obj]
        return type(obj)(inner)
    return obj
