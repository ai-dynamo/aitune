# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics providers for hardware metrics."""

import atexit
from abc import ABC, abstractmethod
from typing import Any

import psutil
import torch
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


class AbstractHardwareMetricProvider(ABC):
    """Abstract base class for hardware metric providers."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Gets the metrics."""
        return {}


class HostMetricProvider(AbstractHardwareMetricProvider):
    """Host metric provider."""

    def get_metrics(self) -> dict[str, Any]:
        """Gets the metrics."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return {
            "host_cpu_percent": cpu_percent,
            "host_memory_used": memory.used,
            "host_memory_free": memory.available,
        }


class TorchMetricProvider(AbstractHardwareMetricProvider):
    """Torch metric provider."""

    def get_metrics(self) -> dict[str, Any]:
        """Gets the metrics."""
        return {
            "torch_allocated": torch.cuda.memory_allocated(),
            "torch_reserved": torch.cuda.memory_reserved(),
        }


class NvmlMetricProvider(AbstractHardwareMetricProvider):
    """Nvml metric provider."""

    def __init__(self):
        """Initializes the NvmlMetricProvider."""
        nvmlInit()
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())]
        self._closed = False
        atexit.register(self.close)

    def get_metrics(self) -> dict[str, Any]:
        """Gets the metrics."""
        metrics = {}
        for index, handle in enumerate(self.handles):
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            # NVML returns power in milliwatts (mW). Convert to watts: power_W = power_mW / 1000
            power_usage_milliwatts = nvmlDeviceGetPowerUsage(handle)
            metrics[f"cuda:{index}_memory_used"] = memory_info.used
            metrics[f"cuda:{index}_memory_free"] = memory_info.free
            metrics[f"cuda:{index}_memory_total"] = memory_info.total
            metrics[f"cuda:{index}_utilization"] = utilization.gpu
            metrics[f"cuda:{index}_power_usage_milliwatts"] = power_usage_milliwatts
        return metrics

    def close(self):
        """Closes the NvmlMetricProvider."""
        if self._closed:
            return
        self._closed = True
        self.handles = []
        nvmlShutdown()
