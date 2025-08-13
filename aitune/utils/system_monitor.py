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
"""System monitoring utilities for GPU and CPU usage."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
    nvmlSystemGetDriverVersion,
)

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """Data class containing GPU statistics."""

    device_index: int
    device_name: str
    memory_used_bytes: int
    memory_free_bytes: int
    memory_total_bytes: int
    utilization_percent: int
    power_usage_milliwatts: int

    @property
    def memory_used_mb(self) -> float:
        """Return memory used in MB."""
        return self.memory_used_bytes / 1024 / 1024

    @property
    def memory_free_mb(self) -> float:
        """Return memory free in MB."""
        return self.memory_free_bytes / 1024 / 1024

    @property
    def memory_total_mb(self) -> float:
        """Return total memory in MB."""
        return self.memory_total_bytes / 1024 / 1024

    @property
    def memory_free_percent(self) -> float:
        """Return percentage of free memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_free_bytes / self.memory_total_bytes) * 100

    @property
    def memory_used_percent(self) -> float:
        """Return percentage of used memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_used_bytes / self.memory_total_bytes) * 100

    @property
    def power_usage_watts(self) -> float:
        """Return power usage in watts."""
        return self.power_usage_milliwatts / 1000


@dataclass
class CPUStats:
    """Data class containing CPU statistics."""

    utilization_percent: float
    memory_used_bytes: int
    memory_free_bytes: int
    memory_total_bytes: int

    @property
    def memory_used_mb(self) -> float:
        """Return memory used in MB."""
        return self.memory_used_bytes / 1024 / 1024

    @property
    def memory_free_mb(self) -> float:
        """Return memory free in MB."""
        return self.memory_free_bytes / 1024 / 1024

    @property
    def memory_total_mb(self) -> float:
        """Return total memory in MB."""
        return self.memory_total_bytes / 1024 / 1024

    @property
    def memory_free_percent(self) -> float:
        """Return percentage of free memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_free_bytes / self.memory_total_bytes) * 100

    @property
    def memory_used_percent(self) -> float:
        """Return percentage of used memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_used_bytes / self.memory_total_bytes) * 100


class SystemMonitor:
    """Utility class for monitoring GPU and CPU usage.

    This class uses pynvml to collect GPU statistics and psutil for CPU statistics.
    It provides properties for accessing current system metrics and logging helpers
    for structured output.
    """

    def __init__(self):
        """Initialize the SystemMonitor and set up NVML."""
        try:
            nvmlInit()
            self._driver_version = nvmlSystemGetDriverVersion()
            self._gpu_count = nvmlDeviceGetCount()
            self._nvml_initialized = True
            logger.debug("NVML initialized. Driver version: %s", self._driver_version)
            logger.debug("Found %d GPU devices", self._gpu_count)
        except Exception as e:
            logger.debug("Failed to initialize NVML: %s", e)
            self._nvml_initialized = False
            self._gpu_count = 0
            self._driver_version = None

    def __del__(self):
        """Clean up NVML when the instance is destroyed."""
        if hasattr(self, "_nvml_initialized") and self._nvml_initialized:
            try:
                nvmlShutdown()
                logger.debug("NVML shutdown complete")
            except Exception as e:
                logger.debug("Failed to shutdown NVML: %s", e)

    @property
    def driver_version(self) -> str | None:
        """Return the NVIDIA driver version."""
        return self._driver_version if self._nvml_initialized else None

    @property
    def gpu_count(self) -> int:
        """Return the number of GPU devices."""
        return self._gpu_count

    def get_gpu_stats(self, device_index: int = 0) -> GPUStats | None:
        """Get current statistics for the specified GPU.

        Args:
            device_index: Index of the GPU device (default: 0)

        Returns:
            GPUStats object with current statistics or None if unavailable
        """
        if not self._nvml_initialized or device_index >= self._gpu_count:
            return None

        try:
            handle = nvmlDeviceGetHandleByIndex(device_index)
            name = nvmlDeviceGetName(handle)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            power_usage = nvmlDeviceGetPowerUsage(handle)

            return GPUStats(
                device_index=device_index,
                device_name=name,
                memory_used_bytes=memory_info.used,
                memory_free_bytes=memory_info.free,
                memory_total_bytes=memory_info.total,
                utilization_percent=utilization.gpu,
                power_usage_milliwatts=power_usage,
            )
        except Exception as e:
            logger.debug("Failed to get GPU stats for device %s: %s", device_index, e)
            return None

    def get_all_gpu_stats(self) -> list[GPUStats]:
        """Get statistics for all available GPUs.

        Returns:
            List of GPUStats objects for all available GPUs
        """
        return [stats for stats in (self.get_gpu_stats(i) for i in range(self._gpu_count)) if stats is not None]

    @property
    def cpu_stats(self) -> CPUStats:
        """Get current CPU statistics.

        Returns:
            CPUStats object with current CPU statistics
        """
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        return CPUStats(
            utilization_percent=cpu_percent,
            memory_used_bytes=memory.used,
            memory_free_bytes=memory.available,
            memory_total_bytes=memory.total,
        )

    def log_gpu_stats(self, device_index: int | None = None, logger_func=None):
        """Log GPU statistics in a structured format.

        Args:
            device_index: GPU device index to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
        """
        if logger_func is None:
            logger_func = logger.debug

        if not self._nvml_initialized:
            logger_func("GPU statistics unavailable - NVML not initialized")
            return

        if device_index is not None:
            stats = self.get_gpu_stats(device_index)
            if stats:
                self._log_single_gpu_stats(stats, logger_func)
            else:
                logger_func("GPU statistics unavailable for device %s", device_index)
        else:
            for i in range(self._gpu_count):
                stats = self.get_gpu_stats(i)
                if stats:
                    self._log_single_gpu_stats(stats, logger_func)

    def _log_single_gpu_stats(self, stats: GPUStats, logger_func):
        """Log statistics for a single GPU.

        Args:
            stats: GPUStats object to log
            logger_func: Logging function to use
        """
        logger_func(f"GPU {stats.device_index} ({stats.device_name}):")
        logger_func(
            f"  Memory: {stats.memory_used_mb:.1f} MB used ({stats.memory_used_percent:.1f}%), "
            f"{stats.memory_free_mb:.1f} MB free ({stats.memory_free_percent:.1f}%), "
            f"{stats.memory_total_mb:.1f} MB total"
        )
        logger_func(f"  Utilization: {stats.utilization_percent}%")
        logger_func(f"  Power Usage: {stats.power_usage_watts:.2f} W")

    def log_cpu_stats(self, logger_func=None):
        """Log CPU statistics in a structured format.

        Args:
            logger_func: Optional custom logging function, defaults to logger.debug
        """
        if logger_func is None:
            logger_func = logger.debug

        stats = self.cpu_stats
        logger_func("CPU:")
        logger_func("  Utilization: %.1f%%", stats.utilization_percent)
        logger_func(
            "  Memory: %.1f MB used (%.1f%%), %.1f MB free (%.1f%%), %.1f MB total",
            stats.memory_used_mb,
            stats.memory_used_percent,
            stats.memory_free_mb,
            stats.memory_free_percent,
            stats.memory_total_mb,
        )

    def log_system_stats(self, gpu_indices: list[int] | None = None, logger_func=None, log_label: str | None = None):
        """Log both CPU and GPU statistics in a structured format.

        Args:
            gpu_indices: List of GPU indices to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
            log_label: Optional label to identify what stage/operation these stats are for
        """
        if logger_func is None:
            logger_func = logger.debug

        header = "=== System Resource Usage ==="
        if log_label:
            header = f"=== System Resource Usage: {log_label} ==="

        logger_func(header)
        self.log_cpu_stats(logger_func)

        if gpu_indices is not None:
            for idx in gpu_indices:
                self.log_gpu_stats(idx, logger_func)
        else:
            self.log_gpu_stats(None, logger_func)

        footer = "============================="
        if log_label:
            footer = "=" * len(header)

        logger_func(footer)

    @contextmanager
    def system_stats_context(
        self, gpu_indices: list[int] | None = None, logger_func=None, log_label: str | None = None
    ):
        """Context manager that logs system statistics on enter and exit.

        Args:
            gpu_indices: List of GPU indices to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
            log_label: Optional label to identify what stage/operation these stats are for

        Yields:
            None
        """
        entry_label = f"{log_label} - Start" if log_label else "Start"
        self.log_system_stats(gpu_indices, logger_func, entry_label)

        try:
            yield
        finally:
            exit_label = f"{log_label} - End" if log_label else "End"
            self.log_system_stats(gpu_indices, logger_func, exit_label)


def system_resource_monitor(name: str | None = None, parent_monitor: SystemMonitor | None = None, logger_func=None):
    """Decorator that logs system statistics before and after function execution.

    This decorator wraps a function to log system resource usage before and after
    the function is called, providing insights into resource consumption.

    Args:
        name: Optional name to identify the decorated function in logs
                If None, the function's name will be used
        parent_monitor: Optional parent monitor to use for logging
        logger_func: Optional custom logging function, defaults to logger.debug
    Returns:
        Decorator function

    Example:
        >>> @system_resource_monitor("my_operation")
        ... def resource_intensive_function():
        ...     pass
        >>> resource_intensive_function()

        >>> @system_resource_monitor()  # Uses function name in logs
        ... def another_function():
        ...     pass
        >>> another_function()
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if no name provided
            label = name if name is not None else func.__name__

            # Get the SystemMonitor instance if not provided
            monitor = parent_monitor or SystemMonitor()

            with monitor.system_stats_context(log_label=label, logger_func=logger_func):
                return func(*args, **kwargs)

        return wrapper

    return decorator
