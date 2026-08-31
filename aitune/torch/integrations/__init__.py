# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integrations module."""

from collections.abc import Callable

import torch.nn as nn

_DISTRIBUTED_MODULE_DETECTORS: dict[str, Callable[[nn.Module], bool]] = {}


def register_distributed_module_detector(name: str, detector: Callable[[nn.Module], bool]) -> None:
    """Register an integration-owned distributed module detector."""
    _DISTRIBUTED_MODULE_DETECTORS[name] = detector


def is_integration_distributed_module(module: nn.Module) -> bool:
    """Return whether any enabled integration recognizes a distributed module."""
    return any(detector(module) for detector in _DISTRIBUTED_MODULE_DETECTORS.values())
