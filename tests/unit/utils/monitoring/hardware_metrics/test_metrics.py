# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hardware metric providers."""

import pytest
import torch

from aitune.utils.monitoring.hardware_metrics.metrics import (
    HostMetricProvider,
    NvmlMetricProvider,
    TorchMetricProvider,
)


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
