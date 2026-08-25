# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the kernel generator interface."""

import pytest
import torch.nn.functional as F  # noqa: N812

from aitune.torch.backend.kernels.kernel_provider import KernelGenerationResult, TorchSDPAKernelProvider


@pytest.mark.parametrize(
    ("provider", "error"),
    [
        (None, None),
        (TorchSDPAKernelProvider(), "generation failed"),
    ],
)
def test_generation_result_requires_exactly_one_outcome(provider, error):
    with pytest.raises(ValueError, match="exactly one of a provider or an error"):
        KernelGenerationResult(
            function="scaled_dot_product_attention",
            provider=provider,
            description="invalid",
            error=error,
        )


def test_generation_result_rejects_raw_callable():
    with pytest.raises(TypeError, match="provider must be a KernelProvider"):
        KernelGenerationResult(
            function="linear",
            provider=F.linear,  # type: ignore[arg-type]
            description="invalid",
        )
