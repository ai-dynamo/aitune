# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel selection for scaled dot-product attention delegated to Torch Inductor JIT."""

import pytest
import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

import aitune.torch as ait
from aitune.torch.backend import KernelSelectorBackend
from tests.functional_ci.lifecycle import Tolerance, pinned_strategy, selected_backend

FP16_ATTENTION = Tolerance(rtol=2e-3, atol=2e-3)


class Attention(nn.Module):
    """Single scaled dot-product attention call."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Return causal attention output."""
        return functional.scaled_dot_product_attention(query, key, value, is_causal=True)


def _qkv(device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    """Return seeded fp16 `(batch, heads, sequence, head_dim)` query, key, and value tensors."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(2, 8, 256, 64, device=device, dtype=torch.float16, generator=generator) for _ in range(3))


@pytest.mark.matrix(
    "aot_tuning.models.tensor-inputs/kernel-selector",
    "aot_tuning.models.static-shapes/kernel-selector",
    "aot_tuning.models.single-graph/kernel-selector",
    "modes.aot-tune/kernel-selector",
    "topology.kernel-selector/single-gpu",
)
@pytest.mark.parametrize("repeat", range(100))
def test_kernel_selector_attention(device: torch.device, repeat: int, caplog, record_property):
    """The selector evaluates attention providers, builds the delegate, and matches eager attention."""
    backend = KernelSelectorBackend()
    tuned = ait.Module(Attention().eval(), "attention", strategy=pinned_strategy(backend))
    with torch.no_grad():
        tuned(*_qkv(device, seed=0))
        # Default dispatch already runs the provider kernels, so pin a slower eager baseline to make selection stable.
        with sdpa_kernel(SDPBackend.MATH):
            tuned.tune(device=device)

        for record in caplog.records:
            if record.msg.startswith("Kernel candidates results"):
                improvement = record.args[1]["Improvement"]
                record_property("max_improvement", float(improvement.max()))
                record_property("improvements", improvement.round(3).to_dict())

        selector = selected_backend(tuned, KernelSelectorBackend)
        assert selector._runtime is not None and selector._runtime.plan.providers, "no attention provider selected"
        sample = _qkv(device, seed=1)
        FP16_ATTENTION.assert_close(tuned(*sample), Attention()(*sample))
