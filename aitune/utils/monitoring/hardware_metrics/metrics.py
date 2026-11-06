# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metrics providers for hardware metrics."""

import atexit
import os
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
    nvmlDeviceGetUUID,
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


def _resolve_visible_device(entry: str, uuids: list[str], by_uuid: bool) -> int | None:
    """Returns the NVML index of one CUDA_VISIBLE_DEVICES entry, or None if CUDA would stop parsing there."""
    if by_uuid:
        matches = [index for index, uuid in enumerate(uuids) if entry.startswith("GPU-") and uuid.startswith(entry)]
        return matches[0] if len(matches) == 1 else None
    if entry.isdecimal() and int(entry) < len(uuids):
        return int(entry)
    return None


def _visible_device_indices() -> list[int]:
    """Returns the NVML indices of the GPUs visible to CUDA, in CUDA device order.

    NVML enumerates every GPU in the machine and ignores CUDA_VISIBLE_DEVICES, while
    ``cuda:N`` is the N-th visible GPU. Entries are device indices or (unique prefixes of)
    ``GPU-`` UUIDs, and the first entry decides which. Like CUDA, parsing stops at the first
    invalid entry, and a repeated entry hides every GPU. Index entries assume that CUDA
    enumerates GPUs in PCI bus order, as NVML does; this holds for identical GPUs or with
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``. UUID entries are always exact.
    """
    count = nvmlDeviceGetCount()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return list(range(count))
    uuids = []
    for index in range(count):
        uuid = nvmlDeviceGetUUID(nvmlDeviceGetHandleByIndex(index))
        uuids.append(uuid.decode() if isinstance(uuid, bytes) else uuid)
    entries = [entry.strip() for entry in visible.split(",")]
    by_uuid = entries[0].startswith("GPU-")
    indices = []
    for entry in entries:
        index = _resolve_visible_device(entry, uuids, by_uuid)
        if index is None:
            break
        if index in indices:
            return []
        indices.append(index)
    return indices


class NvmlMetricProvider(AbstractHardwareMetricProvider):
    """Nvml metric provider.

    Metrics are keyed by CUDA device (``cuda:N``), so only GPUs visible through
    CUDA_VISIBLE_DEVICES are reported.
    """

    def __init__(self):
        """Initializes the NvmlMetricProvider."""
        nvmlInit()
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in _visible_device_indices()]
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
