# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# ///

from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

import timm
import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import (
    ProfileMode,
    TensorRTBackend,
    TensorRTBackendConfig,
    TensorRTProfile,
)
from aitune.torch.config import config as global_config
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = getLogger(Path(__file__).stem)


def do_test(backend: TensorRTBackend, dtype: torch.dtype, device: str, batch_sizes: list[int] | None = None):
    # given
    batch_sizes = batch_sizes or [2, 4, 1]
    batch_size = min(batch_sizes)
    batch_size = max(batch_size, 2)

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device, dtype=dtype)
    model.eval()

    data = torch.randn((3, 224, 224), device=device).to(dtype)
    sample = torch.randn((batch_size, 3, 224, 224), device=device).to(dtype)

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
    tune(module, data, batch_sizes=batch_sizes, dry_run=False, device=device, disable_external_logging=False)

    # then - verify tuning
    out = module(sample)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)

    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-2, atol=1e-2)

    MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)

    do_test(
        TensorRTBackend(
            config=TensorRTBackendConfig(),
        ),
        dtype=torch.float32,
        device="cuda",
    )

    do_test(
        TensorRTBackend(
            config=TensorRTBackendConfig(),
        ),
        dtype=torch.float16,
        device="cuda",
    )

    do_test(
        TensorRTBackend(
            config=TensorRTBackendConfig(),
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )

    global_config.max_num_samples_stored = float("inf")

    # multi profile, user provided profiles
    do_test(
        TensorRTBackend(
            config=TensorRTBackendConfig(
                profiles=[
                    TensorRTProfile().add_input_shape(
                        "args_0", (16, 3, 224, 224), (16, 3, 224, 224), (16, 3, 224, 224)
                    ),
                    TensorRTProfile().add_input_shape(
                        "args_0", (32, 3, 224, 224), (32, 3, 224, 224), (32, 3, 224, 224)
                    ),
                ],
            ),
        ),
        dtype=torch.float32,
        device="cuda",
        batch_sizes=[16, 32],
    )

    # multi profile, auto generated profiles with batch sizes and samples
    do_test(
        TensorRTBackend(
            config=TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED),
        ),
        dtype=torch.bfloat16,
        device="cuda",
        batch_sizes=[16, 32, 64],
    )

    logger.info("Done")
