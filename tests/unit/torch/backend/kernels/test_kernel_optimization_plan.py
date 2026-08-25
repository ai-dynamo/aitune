# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for serializable kernel optimization plans."""

from torch.nn.attention import SDPBackend

from aitune.torch.backend.kernels.kernel_optimization_plan import KernelOptimizationPlan
from aitune.torch.backend.kernels.kernel_provider import KernelProviderState, TorchSDPAKernelProvider


def test_plan_serialization_round_trip_restores_concrete_providers():
    provider = TorchSDPAKernelProvider(SDPBackend.MATH)
    assert provider.prepare([])
    plan = KernelOptimizationPlan((provider,))

    state_dict = plan.to_dict()
    restored = KernelOptimizationPlan.from_dict(state_dict)

    assert state_dict == {
        "providers": [{"type": "TorchSDPAKernelProvider", "backend": "MATH"}],
    }
    assert len(restored.providers) == 1
    restored_provider = restored.providers[0]
    assert isinstance(restored_provider, TorchSDPAKernelProvider)
    assert restored_provider.backend is SDPBackend.MATH
    assert restored_provider.state is KernelProviderState.READY
