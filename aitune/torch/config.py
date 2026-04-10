# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inplace configuration."""

from pathlib import Path

import nvtx.nvtx as nvtx

from aitune.utils.env_vars import AITUNE_CACHE_DIR as _AITUNE_CACHE_DIR

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aitune"
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

# NVTX is by default enabled. Otherwise we allow overwriting it by AITUNE env variable AITUNE_NVTX_EVENTS
if not nvtx._ENABLED:
    from aitune.utils.env_vars import AITUNE_NVTX_EVENTS

    nvtx._ENABLED = AITUNE_NVTX_EVENTS


def aitune_cache_dir() -> Path:
    """Configure cache dir location based on environment variable.

    Returns:
        Cache dir from environment variable or DEFAULT_CACHE_DIR.
    """
    return _AITUNE_CACHE_DIR


class AITuneConfig:
    """AITune configuration."""

    def __init__(self) -> None:
        """Initialize AITuneConfig."""
        self._cache_dir: Path = aitune_cache_dir()
        self._min_num_samples: int = DEFAULT_MIN_NUM_SAMPLES
        self._max_num_samples_stored: int | float = DEFAULT_MAX_NUM_SAMPLES_STORED
        self._device_after_tuning: str = DEFAULT_DEVICE_AFTER_TUNING
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
    def max_num_samples_stored(self) -> int | float:
        """Get the minimum number of samples to collect before optimizing."""
        return self._max_num_samples_stored

    @max_num_samples_stored.setter
    def max_num_samples_stored(self, max_num_samples_stored: int) -> None:
        """Set the minimum number of samples to collect before optimizing."""
        self._max_num_samples_stored = max_num_samples_stored

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: str | Path) -> None:
        """Set the cache directory."""
        self._cache_dir = Path(cache_dir)

    @property
    def device_after_tuning(self) -> str:
        """Get the device to use after tuning."""
        return self._device_after_tuning

    @device_after_tuning.setter
    def device_after_tuning(self, device_after_tuning: str) -> None:
        """Set the device to use after tuning."""
        self._device_after_tuning = device_after_tuning


config = AITuneConfig()
