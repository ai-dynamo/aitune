# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["sageattention"]
# scope = "always"
# ///

"""Functional test for KernelSelectorBackend with an attention provider."""

import logging
import sys
from logging import basicConfig
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from aitune.torch import load, save
from aitune.torch.backend import (
    KernelSelectorBackend,
    KernelSelectorBackendConfig,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.backend.backend import Backend, BackendState, BuildMode
from aitune.torch.kernel_forge.kernel_provider import SageAttentionKernelProvider
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

# This test runs both as a pytest module and as a standalone script in CI or manually.
# Standalone execution adds this directory, rather than the repository root, to sys.path.
if __package__:
    from .kernel_forge.kernel_utils_for_test import PreferProviderKernelUtils
else:
    sys.path.insert(0, str(Path(__file__).parent / "kernel_forge"))
    from kernel_utils_for_test import PreferProviderKernelUtils


def get_sample(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create an SDPA sample supported by SageAttention."""
    batch_size, num_heads, sequence_length, head_dimension = 2, 48, 17776, 64
    shape = (batch_size, num_heads, sequence_length, head_dimension)
    query = torch.randn(*shape, device="cuda", dtype=dtype)
    key = torch.randn(*shape, device="cuda", dtype=dtype)
    value = torch.randn(*shape, device="cuda", dtype=dtype)
    return query, key, value


class AttentionModule(torch.nn.Module):
    """Module containing an SDPA call eligible for provider optimization."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(64, 64)

    def forward(self, query, key, value):
        attention = F.scaled_dot_product_attention(query, key, value, enable_gqa=False)
        return self.projection(attention)


@pytest.mark.parametrize(
    "delegate_backend",
    [
        TorchInductorJitBackend(),
        TorchInductorAotBackend(),
        TorchTensorRTJitBackend(
            TorchTensorRTJitBackendConfig(
                compile_config=TorchTensorRTConfig(
                    enabled_precisions={torch.float16},
                    # SageAttention v1 runs through Triton, leaving only the projection eligible for TensorRT.
                    # Lower the default threshold so this single-operation partition is compiled instead of skipped.
                    min_block_size=1,
                )
            )
        ),
    ],
    ids=["inductor-jit", "inductor-aot", "torch-tensorrt-jit"],
)
def test_kernel_selector_backend_for_attention(delegate_backend: Backend):
    pytest.importorskip("sageattention")
    sample = get_sample(torch.float16)
    model = AttentionModule().to(device="cuda", dtype=torch.float16)
    with torch.no_grad():
        expected = model(*sample)

    # -------------------- backend --------------------
    backend = KernelSelectorBackend(
        KernelSelectorBackendConfig(
            kernel_providers=SageAttentionKernelProvider(),
        ),
        delegate_backend=delegate_backend,
    )

    # -------------------- strategy --------------------
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)

    module = Module(model, "functional-kernel-optimizer-attention", strategy=strategy)
    module(*sample)

    # ------------------ kernel optimizer --------------
    create_optimizer = KernelSelectorBackend._create_optimizer
    selected_plans = []

    def create_preferred_optimizer(self):
        """Create an optimizer that deterministically prefers valid providers.

        Additionally, records the selected plan for later inspection.
        """
        optimizer = create_optimizer(self)
        optimizer.kernel_utils = PreferProviderKernelUtils()
        make_plan = optimizer.make_plan

        def make_plan_and_record(*args, **kwargs):
            plan = make_plan(*args, **kwargs)
            selected_plans.append(plan)
            return plan

        optimizer.make_plan = make_plan_and_record
        return optimizer

    # -------------------- tune --------------------
    with patch.object(KernelSelectorBackend, "_create_optimizer", create_preferred_optimizer):
        module.tune(device=torch.device("cuda"))

    # assert that the kernel optimizer was used
    built_backend = next(iter(module.module.backends.values()))
    assert isinstance(built_backend, KernelSelectorBackend)
    assert len(selected_plans) == 1
    selected_providers = selected_plans[0].providers
    actual = module(*sample)

    assert len(selected_providers) == 1
    assert isinstance(selected_providers[0], SageAttentionKernelProvider)
    if built_backend.build_mode is BuildMode.JUST_IN_TIME:
        assert built_backend._runtime is not None
        assert built_backend._runtime.plan.to_dict() == selected_plans[0].to_dict()
    else:
        assert built_backend._runtime is None
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    # -------------------- checkpoint --------------------
    with TemporaryDirectory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "kernel-optimizer-attention.ait"
        save(module, checkpoint_path)
        restored = load(
            AttentionModule().to(dtype=torch.float16),
            checkpoint_path,
            device_map={"": torch.device("cuda")},
        )

    restored_backend = next(iter(restored.module.backends.values()))
    assert isinstance(restored_backend, KernelSelectorBackend)
    restored_actual = restored(*sample)

    assert restored_backend.state is BackendState.DEPLOYED
    if restored_backend.build_mode is BuildMode.JUST_IN_TIME:
        assert restored_backend._runtime is not None
        restored_providers = restored_backend._runtime.plan.providers
        assert len(restored_providers) == 1
        assert isinstance(restored_providers[0], SageAttentionKernelProvider)
    else:
        assert restored_backend._runtime is None
    torch.testing.assert_close(restored_actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)
    inductor_jit = TorchInductorJitBackend()
    inductor_aot = TorchInductorAotBackend()
    tensorrt_jit = TorchTensorRTJitBackend(
        TorchTensorRTJitBackendConfig(
            compile_config=TorchTensorRTConfig(
                enabled_precisions={torch.float16},
                # SageAttention v1 runs through Triton, leaving only the projection eligible for TensorRT.
                # Lower the default threshold so this single-operation partition is compiled instead of skipped.
                min_block_size=1,
            )
        )
    )

    for delegate_backend in (inductor_jit, inductor_aot, tensorrt_jit):
        test_kernel_selector_backend_for_attention(delegate_backend)
