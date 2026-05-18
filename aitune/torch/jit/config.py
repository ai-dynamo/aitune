# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for JIT module."""

import enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from aitune.torch.config import DEFAULT_DEVICE
from aitune.torch.utils.device import get_device
from aitune.utils.env_vars import AITUNE_JIT_CACHE_DIR as _AITUNE_JIT_CACHE_DIR

if TYPE_CHECKING:
    from aitune.torch.tune_strategy.tune_strategy import TuneStrategy


class JITMode(enum.Enum):
    """Mode for JIT execution."""

    INSPECT = "inspect"  # inspect mode, only for inspection of model execution
    TUNE_EAGER = "tune_eager"  # tune mode, eager tuning after defined number of samples / inference calls
    TUNE_DEFERRED = "tune_deferred"  # tune mode, deferred tuning with explicit call in the code


@dataclass
class Config:
    """Configuration for JIT module."""

    mode: JITMode = JITMode.TUNE_EAGER
    dry_run: bool = False  # whether to perform dry-run tuning
    dry_run_failure_probability: float = 0.2  # probability of failure in dry-run mode to imitate tuning failure
    device: str | torch.device | None = (
        DEFAULT_DEVICE  # device to perform tuning on, if None, the device will use module device
    )

    min_samples: int = 1  # minimum number of samples recorded before tuning
    batch_axis_required: bool = True  # if True, the batch axis must detected in the input data
    max_depth_level: int = 1  # maximum depth of the module hierarchy
    min_parameters: int = 0  # minimum number of parameters to be tuned
    detect_graph_breaks: bool = False  # if True, graph break detection is enabled before tuning
    skip_modules: list[str] = field(default_factory=list)  # list of modules (class names) to skip

    # Extra package prefixes the JIT patcher must not intercept (matched via ``startswith``),
    # on top of ``_DEFAULT_PATCH_EXCLUDE_PACKAGES``. The built-in defaults always apply —
    # use this to add libraries whose internals must not be wrapped.
    extra_patch_exclude_packages: tuple[str, ...] = ()

    # Extra module classes the JIT patcher must not intercept (matched via ``isinstance``),
    # on top of ``_DEFAULT_PATCH_EXCLUDE_MODULES``. The built-in defaults always apply.
    extra_patch_exclude_modules: tuple[type[torch.nn.Module], ...] = ()

    cache_dir: Path = field(default_factory=lambda: _AITUNE_JIT_CACHE_DIR)
    strategy: "TuneStrategy | None" = None  # explicit override; when None, `resolve_strategy()` builds the default

    def __post_init__(self):
        """Post init."""
        self.device = get_device(self.device)

    def resolve_strategy(self) -> "TuneStrategy":
        """Return the tune strategy to use for JIT tuning.

        When ``strategy`` is set explicitly it is returned as-is. Otherwise the default is a
        ``FirstWinsStrategy`` covering TensorRT (with and without dynamo) and TorchInductorJit —
        kept here so the contract is visible on the config and tune-data snapshots can reflect
        what will actually run.

        Strategy and backend modules are imported lazily to keep the JIT config a thin data
        layer that doesn't pull runtime modules at import time.
        """
        if self.strategy is not None:
            return self.strategy
        from aitune.torch.backend.tensorrt.tensorrt_backend import TensorRTBackend, TensorRTBackendConfig
        from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
        from aitune.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy

        return FirstWinsStrategy(
            backends=[
                TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)),
                TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
                TorchInductorJitBackend(),
            ]
        )

    def reset_to_defaults(self) -> None:
        """Reset all options to their default values (e.g. for test isolation)."""
        defaults = Config()
        for f in fields(Config):
            setattr(self, f.name, getattr(defaults, f.name))


config = Config()
