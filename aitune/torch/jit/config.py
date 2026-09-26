# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for JIT module."""

import enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from aitune.torch.utils.device import get_device
from aitune.utils.env_vars import AITUNE_JIT_CACHE_DIR as _AITUNE_JIT_CACHE_DIR

if TYPE_CHECKING:
    from aitune.torch.tune_strategy.resolver import StrategyInput
    from aitune.torch.tune_strategy.tune_strategy import TuneStrategy


class JITMode(enum.Enum):
    """Mode for JIT execution."""

    INSPECT = "inspect"  # inspect mode, only for inspection of model execution
    TUNE_EAGER = "tune_eager"  # tune mode, eager tuning after defined number of samples / inference calls
    TUNE_DEFERRED = "tune_deferred"  # tune mode, deferred tuning enabled by an explicit marker call


@dataclass
class Config:
    """Configuration for JIT module."""

    mode: JITMode = JITMode.TUNE_EAGER
    dry_run: bool = False  # whether to perform dry-run tuning
    dry_run_failure_probability: float = 0.2  # probability of failure in dry-run mode to imitate tuning failure
    device: str | torch.device | None = None  # use the existing module or rank-local device when unspecified

    min_samples: int = 1  # minimum number of samples recorded before tuning
    batch_axis_required: bool = True  # if True, the batch axis must detected in the input data
    max_depth_level: int = 1  # maximum depth of the module hierarchy
    min_parameters: int = 0  # minimum number of parameters to be tuned
    detect_graph_breaks: bool = False  # if True, graph break detection is enabled before tuning
    skip_modules: list[str] = field(default_factory=list)  # list of modules (class names) to skip

    # Extra package prefixes or fully qualified module class names the JIT patcher must not intercept,
    # on top of the built-in defaults.
    patch_exclude: tuple[str, ...] = ()

    cache_dir: Path = field(default_factory=lambda: _AITUNE_JIT_CACHE_DIR)
    strategy: "StrategyInput | None" = None  # explicit or dynamically resolved strategy

    def __post_init__(self):
        """Post init."""
        if self.device is not None:
            self.device = get_device(self.device)

    def resolve_strategy(self, module: nn.Module) -> "TuneStrategy":
        """Return the tune strategy to use for JIT tuning.

        When ``strategy`` is set explicitly it is returned as-is. Otherwise the dynamic
        resolver selects JIT backends from the module's execution properties and builds a
        ``MaxThroughputStrategy``.

        Args:
            module: Module that will be tuned.
        """
        from aitune.torch.tune_strategy.resolver import materialize_strategy, resolve_strategy

        configured_strategy = self.strategy or resolve_strategy()
        return materialize_strategy(configured_strategy, module)

    def reset_to_defaults(self) -> None:
        """Reset all options to their default values (e.g. for test isolation)."""
        defaults = Config()
        for f in fields(Config):
            setattr(self, f.name, getattr(defaults, f.name))


config = Config()
