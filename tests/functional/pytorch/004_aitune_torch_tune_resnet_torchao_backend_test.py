# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# additional_tags = ["gpu/rtx-a6000"]
# ///

from itertools import product
from logging import INFO, basicConfig, getLogger

import timm
import torch
from torchao.utils import is_sm_at_least_89

from aitune.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = getLogger(__name__)


def do_test(backend: TorchAOBackend, dtype: torch.dtype):
    # given
    model = timm.create_model("resnet18", pretrained=False)
    model.to("cuda", dtype=dtype)
    model.eval()

    data = torch.randn((3, 224, 224), device="cuda").to(dtype)
    sample = torch.randn((4, 3, 224, 224), device="cuda").to(dtype)

    with torch.no_grad():
        out = model(sample)

    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    strategy = OneBackendStrategy(backend)
    strategy.enable_validate_against_baseline(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "functional-resnet18",
        strategy=strategy,
    )
    # when
    tune(module, data, batch_sizes=[1, 2, 4], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(sample)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-2, atol=1e-2)


def test_tune_resnet_torchao():
    errors = []
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    quantizations = TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys()

    for dtype, quantization in product(dtypes, quantizations):
        try:
            if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
                continue  # fp8wo and fp8dq are not supported on this device
            logger.info("Testing %s and %s", quantization, dtype)
            config = TorchAOBackendConfig(quantization=quantization)  # type: ignore
            do_test(TorchAOBackend(config=config), dtype)  # type: ignore
            logger.info("Successfully quantized %s and %s", quantization, dtype)
        except Exception as e:
            logger.error("Error with %s and %s: %s", quantization, dtype, e)
            errors.append(f"Error with {quantization} and {dtype}: {e}")
        finally:
            MODULE_REGISTRY.clear()
    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_tune_resnet_torchao()
