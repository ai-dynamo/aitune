# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic local models used by functional CI."""

import torch
from torch import nn


def make_mlp(*, device: torch.device, dtype: torch.dtype = torch.float32, seed: int = 0) -> nn.Module:
    """Create a deterministic 128 → 128 → 64 MLP on the requested device."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    return model.to(device=device, dtype=dtype).eval()
