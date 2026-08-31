# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace configuration."""

from enum import Enum
from pathlib import Path
from typing import Any

import nvtx.nvtx as nvtx

from aitune.utils.env_vars import AITUNE_CACHE_DIR as _AITUNE_CACHE_DIR
from aitune.utils.env_vars import AITUNE_DIFFUSERS_INTEGRATION, AITUNE_TRANSFORMERS_INTEGRATION, TUNING_DATA_PATH

DEFAULT_MIN_NUM_SAMPLES = 100
DEFAULT_MAX_NUM_SAMPLES_STORED = 1  # you can set infinity if you want to store/use all samples

DEFAULT_DEVICE_AFTER_TUNING = "meta"
DEFAULT_TUNING_DATA_OUTPUT_PATH = _AITUNE_CACHE_DIR / "tuning_data" / "report.json"

# NVTX is by default enabled. Otherwise we allow overwriting it by AITUNE env variable AITUNE_NVTX_EVENTS
if not nvtx._ENABLED:
    from aitune.utils.env_vars import AITUNE_NVTX_EVENTS

    nvtx._ENABLED = AITUNE_NVTX_EVENTS


def aitune_cache_dir() -> Path:
    """Configure cache dir location based on environment variable.

    Returns:
        Cache dir from AITUNE_CACHE_DIR environment variable, or default.
    """
    return _AITUNE_CACHE_DIR


class AITuneMode(str, Enum):
    """AITune execution mode used for tuning."""

    JIT = "JIT"
    DECLARATIVE = "DECLARATIVE"


class AITuneConfig:
    """AITune configuration."""

    def __init__(self) -> None:
        """Initialize AITuneConfig."""
        self._cache_dir: Path = aitune_cache_dir()
        self._min_num_samples: int = DEFAULT_MIN_NUM_SAMPLES
        self.max_num_samples_stored: int | float = DEFAULT_MAX_NUM_SAMPLES_STORED
        self.device_after_tuning: str = DEFAULT_DEVICE_AFTER_TUNING
        # ``None`` preserves whether AITune selected the path. Explicit paths, including
        # AITUNE_TUNING_DATA_PATH, must be used verbatim in distributed runs.
        self._tuning_data_output_path: Path | None = Path(TUNING_DATA_PATH) if TUNING_DATA_PATH is not None else None
        self.strict_mode: bool = True
        self.enable_diffusers_integration: bool = AITUNE_DIFFUSERS_INTEGRATION
        self.enable_transformers_integration: bool = AITUNE_TRANSFORMERS_INTEGRATION

    @property
    def min_num_samples(self) -> int:
        """Get the minimum number of samples to collect before optimizing."""
        return self._min_num_samples

    @min_num_samples.setter
    def min_num_samples(self, min_num_samples: int) -> None:
        """Set the minimum number of samples to collect before optimizing."""
        if min_num_samples < 1:
            raise ValueError(f"min_num_samples must be greater than 0, got {min_num_samples}")
        self._min_num_samples = min_num_samples

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: str | Path) -> None:
        """Set the cache directory."""
        self._cache_dir = Path(cache_dir)

    @property
    def tuning_data_output_path(self) -> Path:
        """Get the output path for tuning data."""
        return self._tuning_data_output_path or DEFAULT_TUNING_DATA_OUTPUT_PATH

    @tuning_data_output_path.setter
    def tuning_data_output_path(self, output_path: str | Path) -> None:
        """Set the output path for tuning data."""
        self._tuning_data_output_path = Path(output_path)

    @property
    def uses_default_tuning_data_output_path(self) -> bool:
        """Return whether AITune selected the tuning data output path."""
        return self._tuning_data_output_path is None

    def to_dict(self) -> dict[str, Any]:
        """Return all configuration attributes keyed by their public name."""
        result = {k.lstrip("_"): v for k, v in vars(self).items()}
        result["tuning_data_output_path"] = self.tuning_data_output_path
        return result


config = AITuneConfig()
