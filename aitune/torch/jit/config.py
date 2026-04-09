# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for JIT module."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from aitune.torch.backend.backend import Backend
from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.config import DEFAULT_DEVICE
from aitune.torch.utils.device import get_device
from aitune.utils.env_vars import AITUNE_JIT_CACHE_DIR as _AITUNE_JIT_CACHE_DIR


@dataclass
class Config:
    """Configuration for JIT module."""

    dry_run: bool = False  # whether to perform dry-run tuning
    dry_run_failure_probability: float = 0.2  # probability of failure in dry-run mode to imitate tuning failure
    inspect_mode: bool = False  # whether to perform inspect mode
    device: str | torch.device | None = (
        DEFAULT_DEVICE  # device to perform tuning on, if None, the device will use module device
    )

    min_samples: int = 1  # minimum number of samples recorded before tuning
    batch_axis_required: bool = True  # if True, the batch axis must detected in the input data
    max_depth_level: int = 1  # maximum depth of the module hierarchy
    min_parameters: int = 0  # minimum number of parameters to be tuned
    detect_graph_breaks: bool = False  # if True, graph break detection is enabled before tuning
    skip_modules: list[str] = field(default_factory=list)  # list of modules (class names) to skip

    cache_dir: Path = field(default_factory=lambda: _AITUNE_JIT_CACHE_DIR)
    backends: list[Backend] = field(
        default_factory=lambda: [
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]
    )  # backends to use for JIT tuning

    def __post_init__(self):
        """Post init."""
        self.device = get_device(self.device)

    def reset_to_defaults(self) -> None:
        """Reset all options to their default values (e.g. for test isolation)."""
        defaults = Config()
        for f in fields(Config):
            setattr(self, f.name, getattr(defaults, f.name))


config = Config()
