# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PyTorch SDPA kernel provider."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn.attention import SDPBackend

from aitune.torch.backend.kernels.kernel_provider import KernelProviderState, TorchSDPAKernelProvider


def test_torch_sdpa_provider_prepare_and_serialization_round_trip():
    provider = TorchSDPAKernelProvider(SDPBackend.MATH)

    assert provider.supported_function == "scaled_dot_product_attention"
    assert provider.prepare([]) is True
    assert provider.state is KernelProviderState.READY

    state_dict = provider.to_dict()
    restored = TorchSDPAKernelProvider.from_dict(state_dict)

    assert state_dict == {
        "type": "TorchSDPAKernelProvider",
        "backend": "MATH",
    }
    assert restored.state is KernelProviderState.READY
    assert restored.backend is SDPBackend.MATH


def test_torch_sdpa_provider_is_callable_on_cpu():
    query = torch.randn(2, 4, 8, 16)
    key = torch.randn(2, 4, 8, 16)
    value = torch.randn(2, 4, 8, 16)
    sample = ((query, key, value), {})
    provider = TorchSDPAKernelProvider(SDPBackend.MATH)

    assert provider.prepare([sample]) is True

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = provider(query, key, value)

    torch.testing.assert_close(actual, expected)


def test_torch_sdpa_provider_name():
    provider = TorchSDPAKernelProvider(SDPBackend.MATH)

    assert provider.name == "PyTorch SDPA MATH"
    assert repr(provider) == provider.name
