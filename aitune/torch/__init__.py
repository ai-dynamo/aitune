# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch tuning module."""

# configuring some variables before importing other modules
from aitune.torch.config import aitune_cache_dir, config  # noqa: I001

from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.dataloader import DataLoaderFactory
from aitune.torch.inspecting import inspect, wrap
from aitune.torch.module import Module
from aitune.torch.performance import PerformanceProfile, profile
from aitune.torch.tune_strategy import (
    FirstWinsStrategy,
    LatencyBudgetStrategy,
    MaxThroughputStrategy,
    MinLatencyStrategy,
    OneBackendStrategy,
    TuneStrategy,
)
from aitune.torch.tune_data.reporting import snapshot_tuning_data
from aitune.torch.tuning import load, save, tune
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning

__all__ = [
    "aitune_cache_dir",
    "config",
    "jit_config",
    "inspect",
    "wrap",
    "tune",
    "load",
    "save",
    "snapshot_tuning_data",
    "PerformanceProfile",
    "profile",
    "Module",
    "PatchedModule",
    "TuneStrategy",
    "OneBackendStrategy",
    "FirstWinsStrategy",
    "LatencyBudgetStrategy",
    "MaxThroughputStrategy",
    "MinLatencyStrategy",
    "LocalTorchStorage",
    "DataLoaderFactory",
    "patch_for_jit_tuning",
]

if config.enable_diffusers_integration:
    import aitune.torch.integrations.diffusers  # noqa: F401

if config.enable_transformers_integration:
    import aitune.torch.integrations.transformers  # noqa: F401
