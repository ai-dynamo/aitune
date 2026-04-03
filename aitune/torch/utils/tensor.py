# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor utility helpers."""

from typing import Any

import torch


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
