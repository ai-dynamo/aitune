# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inplace configuration."""

import os
from pathlib import Path

import nvtx.nvtx as nvtx

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

# Disable NVTX by default, enable with NVTX_ENABLE=1
NVTX_ENABLE = os.getenv("NVTX_ENABLE") in ("1", "true", "True", "yes", "Yes", "YES")
if not nvtx._ENABLED:
    nvtx._ENABLED = NVTX_ENABLE

# Console output configuration
CONSOLE_OUTPUT_ENABLE = os.getenv("AITUNE_CONSOLE_OUTPUT") in ("1", "true", "True", "yes", "Yes", "YES")
SYSTEM_MONITOR_ENABLE = os.getenv("AITUNE_SYSTEM_MONITOR") in ("1", "true", "True", "yes", "Yes", "YES")


def aitune_cache_dir() -> Path:
    """Configure cache dir location based on environment variable.

    Returns:
        Cache dir from environment variable or DEFAULT_CACHE_DIR.
    """
    cache_dir = os.environ.get("AITUNE_CACHE_DIR", DEFAULT_CACHE_DIR)
    return Path(cache_dir)


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


def get_bool_env_variable(env_variable: str, default: bool) -> bool:
    """Get a boolean environment variable."""
    value = os.environ.get(env_variable)
    if value is None:
        return default
    return value in ["1", "true", "True", "yes", "Yes", "YES"]


config = AITuneConfig()
