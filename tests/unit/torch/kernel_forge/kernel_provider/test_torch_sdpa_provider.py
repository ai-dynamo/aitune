# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PyTorch SDPA kernel provider."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn.attention import SDPBackend

from aitune.torch.kernel_forge.kernel_provider import KernelProviderState, TorchSDPAKernelProvider
from tests.utilities.helpers import requires_cuda


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


@requires_cuda
def test_torch_sdpa_flash_provider_is_callable(torch_device):
    query = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=torch_device)
    key = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=torch_device)
    value = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=torch_device)
    sample = ((query, key, value), {})
    provider = TorchSDPAKernelProvider(SDPBackend.FLASH_ATTENTION)

    assert provider.prepare([sample]) is True

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = provider(query, key, value)

    torch.testing.assert_close(actual, expected)


def test_torch_sdpa_provider_name():
    provider = TorchSDPAKernelProvider(SDPBackend.MATH)

    assert provider.name == "PyTorch SDPA MATH"
    assert repr(provider) == provider.name
