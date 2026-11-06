# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hardware metric providers."""

from types import SimpleNamespace

import pytest
import torch

import aitune.utils.monitoring.hardware_metrics.metrics as metrics_module
from aitune.utils.monitoring.hardware_metrics.metrics import (
    HostMetricProvider,
    NvmlMetricProvider,
    TorchMetricProvider,
)

_FAKE_UUIDS = [f"GPU-{index}a{index}b0000-0000-0000-0000-00000000000{index}" for index in range(8)]


@pytest.fixture
def fake_nvml(monkeypatch):
    """Replaces NVML with an 8-GPU fake whose handles are the NVML indices."""
    monkeypatch.setattr(metrics_module, "nvmlInit", lambda: None)
    monkeypatch.setattr(metrics_module, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetCount", lambda: len(_FAKE_UUIDS))
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetUUID", lambda handle: _FAKE_UUIDS[handle])
    monkeypatch.setattr(
        metrics_module,
        "nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(used=handle, free=100 - handle, total=100),
    )
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=handle))
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetPowerUsage", lambda handle: 1000 * handle)


@pytest.mark.parametrize(
    "visible_devices, expected",
    [
        (None, list(range(8))),
        ("4", [4]),
        ("6,4", [6, 4]),
        (" 5 , 7 ", [5, 7]),
        (_FAKE_UUIDS[3], [3]),
        (f"GPU-6a6,{_FAKE_UUIDS[2]}", [6, 2]),
        ("4,9,5", [4]),
        ("4,4", []),
        (f"4,{_FAKE_UUIDS[5]}", [4]),
        (f"{_FAKE_UUIDS[6]},4", [6]),
        ("4,GPU-", [4]),
        ("4,MIG-1234", [4]),
        ("-1", []),
        ("", []),
    ],
)
def test_visible_device_indices_follow_cuda_visible_devices(fake_nvml, monkeypatch, visible_devices, expected):
    """Test that NVML devices are selected and ordered the way CUDA parses CUDA_VISIBLE_DEVICES."""
    if visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    assert metrics_module._visible_device_indices() == expected


def test_visible_device_indices_accept_bytes_uuids(fake_nvml, monkeypatch):
    """Test that UUIDs returned as bytes by older NVML bindings are matched."""
    monkeypatch.setattr(metrics_module, "nvmlDeviceGetUUID", lambda handle: _FAKE_UUIDS[handle].encode())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _FAKE_UUIDS[1])
    assert metrics_module._visible_device_indices() == [1]


def test_nvml_metric_provider_keys_metrics_by_cuda_device(fake_nvml, monkeypatch):
    """Test that cuda:N metrics describe the N-th visible GPU, not the N-th GPU in the machine."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,4")
    provider = NvmlMetricProvider()
    try:
        metrics = provider.get_metrics()
    finally:
        provider.close()

    assert metrics == {
        "cuda:0_memory_used": 6,
        "cuda:0_memory_free": 94,
        "cuda:0_memory_total": 100,
        "cuda:0_utilization": 6,
        "cuda:0_power_usage_milliwatts": 6000,
        "cuda:1_memory_used": 4,
        "cuda:1_memory_free": 96,
        "cuda:1_memory_total": 100,
        "cuda:1_utilization": 4,
        "cuda:1_power_usage_milliwatts": 4000,
    }


def test_host_metric_provider_get_metrics_has_expected_keys():
    """Test that HostMetricProvider.get_metrics returns expected keys."""
    provider = HostMetricProvider()
    metrics = provider.get_metrics()
    assert "host_cpu_percent" in metrics
    assert "host_memory_used" in metrics
    assert "host_memory_free" in metrics

    # CPU percent should be between 0 and 100
    assert 0 <= metrics["host_cpu_percent"] <= 100

    # Memory values should be positive
    assert metrics["host_memory_used"] >= 0
    assert metrics["host_memory_free"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_metric_provider_get_metrics_has_expected_keys():
    """Test that TorchMetricProvider.get_metrics returns expected keys."""
    provider = TorchMetricProvider()
    metrics = provider.get_metrics()
    assert "torch_allocated" in metrics
    assert "torch_reserved" in metrics

    # Memory values should be non-negative
    assert metrics["torch_allocated"] >= 0
    assert metrics["torch_reserved"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvml_metric_provider_get_metrics_has_expected_keys():
    """Test that NvmlMetricProvider.get_metrics returns expected keys for each GPU."""
    provider = NvmlMetricProvider()
    metrics = provider.get_metrics()

    num_gpus = len(provider.handles)
    assert num_gpus == torch.cuda.device_count()
    for i in range(num_gpus):
        assert f"cuda:{i}_memory_used" in metrics
        assert f"cuda:{i}_memory_free" in metrics
        assert f"cuda:{i}_memory_total" in metrics
        assert f"cuda:{i}_utilization" in metrics
        assert f"cuda:{i}_power_usage_milliwatts" in metrics

        assert metrics[f"cuda:{i}_memory_used"] >= 0
        assert metrics[f"cuda:{i}_memory_free"] >= 0
        assert metrics[f"cuda:{i}_memory_total"] >= 0

        # Utilization should be 0-100
        assert 0 <= metrics[f"cuda:{i}_utilization"] <= 100

        # Power usage in milliwatts should be non-negative
        assert metrics[f"cuda:{i}_power_usage_milliwatts"] >= 0
