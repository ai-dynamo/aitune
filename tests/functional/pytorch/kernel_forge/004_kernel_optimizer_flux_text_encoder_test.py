# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["flash-attn-4==4.0.0b29"]
# scope = "always"
# ///

"""Functional test for FlashAttention-4 kernel optimization."""

import logging
from logging import basicConfig

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernel_selector_backend import _default_kernel_providers
from aitune.torch.kernel_forge import KernelOptimizer
from aitune.torch.kernel_forge.kernel_provider import (
    DiffusersAttentionBackend,
    DiffusersAttentionKernelProvider,
    KernelProvider,
)

# These tests run both as pytest modules and as standalone scripts in CI or manually.
# Standalone execution adds this directory, rather than the repository root, to sys.path.
if __package__:
    from .kernel_utils_for_test import (
        CountingKernelProvider,
        PreferProviderKernelUtils,
        assert_provider_was_used,
        selected_counting_provider,
    )
else:
    from kernel_utils_for_test import (
        CountingKernelProvider,
        PreferProviderKernelUtils,
        assert_provider_was_used,
        selected_counting_provider,
    )


def get_sample(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a representative SDPA sample supported by FlashAttention-4."""
    batch_size, num_heads, sequence_length, head_dim = 2, 8, 256, 64

    query = torch.randn(batch_size, num_heads, sequence_length, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(batch_size, num_heads, sequence_length, head_dim, device="cuda", dtype=dtype)
    value = torch.randn(batch_size, num_heads, sequence_length, head_dim, device="cuda", dtype=dtype)

    return query, key, value


class AttentionModule(torch.nn.Module):
    """Test module using PyTorch SDPA."""

    def forward(self, query, key, value):
        """Run causal scaled dot-product attention."""
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)


@pytest.mark.parametrize("provider", _default_kernel_providers(), ids=lambda provider: provider.name)
def test_kernel_optimizer_flash_attention(provider: KernelProvider):
    """Replace SDPA with the selected attention implementation."""
    logging.info("Testing %s kernel optimization", provider.name)

    dtype = torch.float16
    net = AttentionModule()
    sample = get_sample(dtype)
    data = [(sample, {})]

    provider = CountingKernelProvider(provider)
    optimizer = KernelOptimizer(
        kernel_providers=[provider],
        kernel_utils=PreferProviderKernelUtils(),
    )
    plan = optimizer.make_plan(net, data, module=net)
    assert {provider.supported_function for provider in plan.providers} == {"scaled_dot_product_attention"}
    provider = selected_counting_provider(plan)

    with plan.apply(net):
        result = net(*get_sample(dtype))

        assert_provider_was_used(provider)
    assert result.shape == sample[0].shape


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)

    for provider in _default_kernel_providers():
        if (
            isinstance(provider, DiffusersAttentionKernelProvider)
            and provider.backend is DiffusersAttentionBackend.FLASH_3_HUB
        ):
            # skip testing hub version, not available in the CI environment
            continue
        test_kernel_optimizer_flash_attention(provider=provider)
