# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = true
# ///


from logging import DEBUG, basicConfig, getLogger

import timm
import torch

from aitune.torch import tune
from aitune.torch.backend.torch_inductor_aot_backend import TorchInductorAotBackend, TorchInductorAotBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def do_test(backend: TorchInductorAotBackend):
    # given
    device = torch.device("cuda")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_validate_against_baseline(False)
    module = Module(model, "functional-resnet18", strategy=strategy)
    # when
    tune(module, data, batch_sizes=[2, 1], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-3, atol=1e-4)


def test_tune_resnet_torch_inductor_aot():
    errors = []
    configs = [
        TorchInductorAotBackendConfig(),
        TorchInductorAotBackendConfig(inductor_configs={"max_autotune": True}),
    ]

    for config in configs:
        try:
            logger.info("Testing config: %s", config.describe() or "default")
            do_test(TorchInductorAotBackend(config=config))
        except Exception as e:
            logger.error("Error with config %s: %s", config.describe(), e)
            errors.append(f"Error with config {config.describe()!r}: {e}")
        finally:
            MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_resnet_torch_inductor_aot()
