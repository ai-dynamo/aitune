# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["sageattention"]
# scope = "always"
# ///

"""Functional test for KernelOptimizer for attention.

Requires the ``sageattention`` package and CUDA.
"""

import logging
from importlib.metadata import version
from logging import basicConfig

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version

from aitune.torch.kernel_forge import KernelOptimizer
from aitune.torch.kernel_forge.kernel_provider import (
    KernelProvider,
    SageAttentionKernelProvider,
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
    # those are the values for the CogVideoX-5b pipeline, so that sage attn supports it
    bs, num_heads, seq_len, head_dim = 2, 48, 17776, 64

    q = torch.randn(bs, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(bs, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(bs, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    return q, k, v


class AttentionModule(torch.nn.Module):
    """Test module for attention."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=False)


def test_kernel_optimizer_attention(provider: KernelProvider):
    pytest.importorskip("sageattention")
    provider = CountingKernelProvider(provider)

    net = AttentionModule()
    data = [(get_sample(torch.float16), {})]
    optimizer = KernelOptimizer(
        kernel_providers=provider,
        kernel_utils=PreferProviderKernelUtils(),
    )
    plan = optimizer.make_plan(net, data=data, module=net)
    provider = selected_counting_provider(plan)
    with plan.apply(net):
        # verify inference works
        result = net(*get_sample(torch.float16))
        assert_provider_was_used(provider)
        assert result.shape == (2, 48, 17776, 64)
        # verify compilation works
        if Version(version("sageattention")).major < 2:
            compiled_net = torch.compile(net, fullgraph=True)
        else:
            # version 2 has graph breaks
            compiled_net = torch.compile(net, fullgraph=False)
        result = compiled_net(*get_sample(torch.float16))
        assert result.shape == (2, 48, 17776, 64)


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)

    for provider in [
        SageAttentionKernelProvider(),
        # DiffusersAttentionKernelProvider(DiffusersAttentionBackend.SAGE),  # not working with v1
        # DiffusersAttentionKernelProvider(DiffusersAttentionBackend.SAGE_HUB),  # not working
    ]:
        test_kernel_optimizer_attention(provider=provider)
