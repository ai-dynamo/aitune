# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# # dynamic shapes works on torch 2.9
# dependencies = ["timm"]
# ///

from logging import INFO, basicConfig, getLogger
from pathlib import Path

import timm
import torch

from aitune.torch.backend import (
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
)
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = getLogger(Path(__file__).stem)


def do_test(backend: TorchTensorRTAotBackend, dtype: torch.dtype, device: str):
    # given
    model = timm.create_model("resnet18", pretrained=False)
    model.to(device, dtype=dtype)
    model.eval()

    data = torch.randn((3, 224, 224), device=device).to(dtype)
    sample = torch.randn((4, 3, 224, 224), device=device).to(dtype)

    with torch.no_grad():
        out = model(sample)

    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(model, "functional-resnet18", strategy=strategy)

    # when
    tune(module, [data], batch_sizes=[1, 2, 4, 8], dry_run=False, device=device, disable_external_logging=False)

    # then - verify tuning
    out = module(sample)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)

    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-2, atol=1e-2)

    # try dynamic shapes
    module(data.repeat(8, 1, 1, 1))
    module(data.repeat(4, 1, 1, 1))
    module(data.repeat(2, 1, 1, 1))

    MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)

    do_test(
        TorchTensorRTAotBackend(
            config=TorchTensorRTAotBackendConfig(
                compile_config=TorchTensorRTConfig(
                    enabled_precisions={torch.float32, torch.float16, torch.bfloat16},
                    use_python_runtime=False,
                    assume_dynamic_shape_support=True,
                ),
            ),
        ),
        dtype=torch.float32,
        device="cuda",
    )

    logger.info("Done")
