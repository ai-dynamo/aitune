# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# ///


from logging import DEBUG, basicConfig

import timm
import torch

from aitune.torch import BatchDim, DynamicDim, Module, tune
from aitune.torch.backend.torch_inductor_aot_backend import TorchInductorAotBackend
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


def test_tune_resnet_torch_inductor_aot_user_dynamic_shapes():
    """Tune one shape and run unseen batch and spatial sizes allowed by the user configuration."""
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False).to(device).eval()
    data_224 = torch.randn((3, 224, 224), device=device)
    data_256 = torch.randn((2, 3, 256, 256), device=device)

    with torch.no_grad():
        expected_256 = torch.nn.functional.softmax(model(data_256), dim=1)

    batch = BatchDim("batch", min=1, opt=1, max=2)
    spatial = DynamicDim("spatial", min=224, opt=224, max=256)
    strategy = OneBackendStrategy(TorchInductorAotBackend())
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)

    try:
        module = Module(
            model,
            "functional-resnet18-user-dynamic",
            strategy=strategy,
            dynamic_shapes={"x": (batch, 3, spatial, spatial)},
        )
        tune(
            module,
            [data_224],
            batch_sizes=[1],
            dry_run=False,
            disable_external_logging=False,
        )

        # Without this the numerical check below compares the eager model against itself
        # and passes even when tuning fell back to the original module.
        assert module.state == ModuleState.TUNED, f"tuning failed, module state is {module.state}"

        actual_256 = torch.nn.functional.softmax(module(data_256), dim=1)
        torch.testing.assert_close(actual_256, expected_256, rtol=1e-3, atol=1e-4)
    finally:
        MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_torch_inductor_aot_user_dynamic_shapes()
