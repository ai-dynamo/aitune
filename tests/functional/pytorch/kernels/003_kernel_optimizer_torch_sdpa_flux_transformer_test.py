# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from logging import basicConfig

import diffusers
import torch
from torch.nn.attention import SDPBackend

from aitune.torch.backend.kernels import KernelOptimizer
from aitune.torch.backend.kernels.kernel_provider import TorchSDPAKernelProvider

# These tests run both as pytest modules and as standalone scripts in CI or manually.
# Standalone execution adds this directory, rather than the repository root, to sys.path.
if __package__:
    from .kernel_utils import (
        CountingKernelProvider,
        PreferProviderKernelUtils,
        assert_provider_was_used,
        selected_counting_provider,
    )
else:
    from kernel_utils import (
        CountingKernelProvider,
        PreferProviderKernelUtils,
        assert_provider_was_used,
        selected_counting_provider,
    )


def make_generator():
    """Make a generator for the pipeline."""
    return torch.Generator(device="cuda").manual_seed(1234)


def test_torch_sdpa_in_tiny_flux(dtype: torch.dtype = torch.bfloat16) -> None:
    """Test the TorchSDPAKernelProvider."""
    logging.info("Testing with dtype: %s", dtype)

    model_id = "hf-internal-testing/tiny-flux-pipe"
    pipe = diffusers.FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to("cuda", dtype=dtype)

    ref_prompt = "A futuristic cityscape with neon lights and flying cars"
    data = [((ref_prompt,), {})]
    ref_result = pipe(ref_prompt, generator=make_generator())

    providers = [
        CountingKernelProvider(TorchSDPAKernelProvider(SDPBackend.MATH)),
        CountingKernelProvider(TorchSDPAKernelProvider(SDPBackend.EFFICIENT_ATTENTION)),
        CountingKernelProvider(TorchSDPAKernelProvider(SDPBackend.CUDNN_ATTENTION)),
        CountingKernelProvider(TorchSDPAKernelProvider(SDPBackend.FLASH_ATTENTION)),
    ]
    optimizer = KernelOptimizer(
        top_k=5,
        kernel_providers=providers,
        kernel_utils=PreferProviderKernelUtils(),
    )
    plan = optimizer.make_plan(pipe, data, module=pipe.transformer)
    provider = selected_counting_provider(plan)

    with plan.apply(pipe.transformer):
        opt_result = pipe(ref_prompt, generator=make_generator())
        assert_provider_was_used(provider)

    assert opt_result.images[0].size == ref_result.images[0].size


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        test_torch_sdpa_in_tiny_flux(dtype)
