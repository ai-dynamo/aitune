# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic toy models shared by functional cases."""

from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn

IN_FEATURES = 128
OUT_FEATURES = 64


@contextmanager
def seeded(device: torch.device, seed: int) -> Generator[None, None, None]:
    """Seed the global RNG for the block, then restore the previous RNG state."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        yield


def make_mlp(device: torch.device, seed: int = 0) -> nn.Module:
    """Return a seeded multi-layer perceptron (MLP) `IN_FEATURES -> 128 -> OUT_FEATURES` in eval mode."""
    with seeded(device, seed):
        model = nn.Sequential(nn.Linear(IN_FEATURES, 128), nn.ReLU(), nn.Linear(128, OUT_FEATURES))
    return model.to(device).eval()


def make_image_model(device: torch.device, seed: int = 0) -> nn.Module:
    """Return a seeded CNN accepting `(batch, 3, height, width)` image inputs."""
    with seeded(device, seed):
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
    return model.to(device).eval()


def make_spatial_model(device: torch.device, seed: int = 0) -> nn.Module:
    """Return a seeded grayscale CNN accepting `(batch, height, width)` inputs."""
    with seeded(device, seed):
        model = nn.Sequential(
            # PyTorch 2.8–2.10's exported-program loader fails on symbolic 4D strides; add the channel axis internally.
            nn.Unflatten(1, (1, -1)),
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
    return model.to(device).eval()


def make_input(device: torch.device, batch_size: int = 2, seed: int = 1) -> torch.Tensor:
    """Return a seeded `(batch_size, IN_FEATURES)` input batch."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch_size, IN_FEATURES, device=device, generator=generator)
