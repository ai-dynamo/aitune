# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# allow_failure = false
# ///

from logging import DEBUG, basicConfig

import torch
import torch.nn as nn

from aitune.torch import DynamicDim, Module, OneBackendStrategy, tune
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig, TorchTensorRTConfig
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.module_registry import MODULE_REGISTRY


class StridedSpatialModel(nn.Module):
    """Small model whose spatial range requires inferred export guards."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.head = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.conv(x))).flatten(1)


def test_tune_torch_tensorrt_aot_with_dynamic_hints_fallback():
    """Compile a guarded spatial range by relaxing rejected explicit dimensions to bounded hints."""
    device = torch.device("cuda")
    model = StridedSpatialModel().to(device).eval()
    dataset = [torch.randn(3, 128, 128, device=device)]
    validation_inputs = [
        torch.randn(1, 3, 128, 128, device=device),
        torch.randn(1, 3, 192, 192, device=device),
    ]

    with torch.no_grad():
        expected_outputs = [model(value) for value in validation_inputs]

    spatial = DynamicDim("spatial", min=128, opt=128, max=192)
    backend = TorchTensorRTAotBackend(
        TorchTensorRTAotBackendConfig(
            compile_config=TorchTensorRTConfig(
                min_block_size=1,
                assume_dynamic_shape_support=True,
            )
        )
    )
    strategy = OneBackendStrategy(backend)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    tuned_model = Module(
        model,
        "strided-spatial-dynamic-hints",
        strategy=strategy,
        dynamic_shapes={"x": (1, 3, spatial, spatial)},
    )

    try:
        tune(
            tuned_model,
            dataset,
            batch_sizes=[1],
            device=device,
            ignore_failing_modules=False,
        )

        assert tuned_model.state == ModuleState.TUNED, f"tuning failed, module state is {tuned_model.state}"
        with torch.no_grad():
            for value, expected in zip(validation_inputs, expected_outputs, strict=True):
                torch.testing.assert_close(tuned_model(value), expected, rtol=1e-3, atol=1e-4)
    finally:
        tuned_model.deactivate()
        MODULE_REGISTRY.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_torch_tensorrt_aot_with_dynamic_hints_fallback()
