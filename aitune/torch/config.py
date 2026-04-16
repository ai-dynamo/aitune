# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace configuration."""

from enum import Enum
from pathlib import Path

import nvtx.nvtx as nvtx

from aitune.utils.env_vars import AITUNE_CACHE_DIR as _AITUNE_CACHE_DIR
from aitune.utils.env_vars import TUNING_DATA_COLLECTION_ENABLE

DEFAULT_MIN_NUM_SAMPLES = 100
DEFAULT_MAX_NUM_SAMPLES_STORED = 1  # you can set infinity if you want to store/use all samples

DEFAULT_PICKLE_PROTOCOL = 5

# Profiling configuration
DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD = 0.05
DEFAULT_THROUGHPUT_BACKOFF_LIMIT = 2
DEFAULT_STABILITY_PERCENTAGE = 95
DEFAULT_WINDOW_SIZE = 10

DEFAULT_DEVICE = "cuda:0"
DEFAULT_DEVICE_AFTER_TUNING = "meta"
DEFAULT_TUNING_DATA_OUTPUT_DIR = _AITUNE_CACHE_DIR / "tuning_data"

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
        self.enable_tuning_data_collection: bool = TUNING_DATA_COLLECTION_ENABLE
        self._tuning_data_output_dir: Path = DEFAULT_TUNING_DATA_OUTPUT_DIR
        self.strict_mode: bool = True
        self.enable_hf_integrations: bool = True

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
    def tuning_data_output_dir(self) -> Path:
        """Get the output directory for tuning data."""
        return self._tuning_data_output_dir

    @tuning_data_output_dir.setter
    def tuning_data_output_dir(self, output_dir: str | Path) -> None:
        """Set the output directory for tuning data."""
        self._tuning_data_output_dir = Path(output_dir)


config = AITuneConfig()
