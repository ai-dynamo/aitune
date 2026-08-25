# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["flash-attn-4==4.0.0b27"]
# scope = "always"
# ///

"""Functional test for the FlashAttention-4 kernel provider.

Requires the ``flash-attn-4`` package and CUDA.
"""

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from aitune.torch.backend.kernels.kernel_provider import FlashAttention4KernelProvider


def _assert_flash_attention4_matches_sdpa(*, query_heads: int, key_value_heads: int, enable_gqa: bool):
    """Verify FlashAttention-4 matches PyTorch SDPA for one head layout."""
    torch.manual_seed(0)
    provider = FlashAttention4KernelProvider()
    query = torch.randn(1, query_heads, 16, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(1, key_value_heads, 16, 64, device="cuda", dtype=torch.float16)
    value = torch.randn(1, key_value_heads, 16, 64, device="cuda", dtype=torch.float16)
    kwargs = {"is_causal": True, "scale": 0.125}
    if enable_gqa:
        kwargs["enable_gqa"] = True
    sample = ((query, key, value), kwargs)

    assert provider.prepare([sample])
    actual = provider(query, key, value)
    expected = scaled_dot_product_attention(query, key, value, **kwargs)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_flash_attention4_kernel_provider_uses_installed_implementation():
    """Verify the provider calls the installed FlashAttention-4 implementation."""
    pytest.importorskip("flash_attn.cute")

    _assert_flash_attention4_matches_sdpa(query_heads=4, key_value_heads=4, enable_gqa=False)
    _assert_flash_attention4_matches_sdpa(query_heads=4, key_value_heads=2, enable_gqa=True)
    _assert_flash_attention4_matches_sdpa(query_heads=4, key_value_heads=1, enable_gqa=True)


if __name__ == "__main__":
    test_flash_attention4_kernel_provider_uses_installed_implementation()
