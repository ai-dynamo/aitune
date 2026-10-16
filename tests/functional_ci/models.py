# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic toy models shared by functional cases."""

import torch
from torch import nn

IN_FEATURES = 128
OUT_FEATURES = 64


def make_mlp(device: torch.device, seed: int = 0) -> nn.Module:
    """Return a seeded multi-layer perceptron (MLP) `IN_FEATURES -> 128 -> OUT_FEATURES` in eval mode."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(IN_FEATURES, 128), nn.ReLU(), nn.Linear(128, OUT_FEATURES))
    return model.to(device).eval()


def make_input(device: torch.device, batch_size: int = 2, seed: int = 1) -> torch.Tensor:
    """Return a seeded `(batch_size, IN_FEATURES)` input batch."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch_size, IN_FEATURES, device=device, generator=generator)
