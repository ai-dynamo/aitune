# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///

from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

import timm
import torch

from aitune.torch.backend.torch_tensorrt_jit_backend import (
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = getLogger(Path(__file__).stem)


def do_test(backend: TorchTensorRTJitBackend, dtype: torch.dtype):
    # given
    model = timm.create_model("resnet18", pretrained=False)
    model.to("cuda", dtype=dtype)
    model.eval()
    data = torch.randn((3, 224, 224), device="cuda").to(dtype)
    sample = torch.randn((2, 3, 224, 224), device="cuda").to(dtype)

    with torch.no_grad():
        out = model(sample)
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    module = Module(model, "functional-resnet18", strategy=strategy)
    # when
    tune(module, data, batch_sizes=[2, 1], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(sample)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-2, atol=1e-2)

    MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)

    do_test(
        TorchTensorRTJitBackend(
            config=TorchTensorRTJitBackendConfig(
                compile_config=TorchTensorRTConfig(
                    enabled_precisions={torch.float32, torch.float16, torch.bfloat16},
                ),
            ),
        ),
        torch.float32,
    )

    logger.info("Done")
