# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on Stable Diffusion 1.5."""

# /// script
# dependencies = ["diffusers", "transformers"]
# scope = "always"
# allow_failure = false
# use_gated_hf_token = true
# additional_tags = ["mem/80g"]
# ///

import re
from logging import INFO, basicConfig, getLogger
from time import perf_counter

import torch
from diffusers import StableDiffusionPipeline

from aitune.torch.jit.config import JITMode, config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning
from aitune.torch.jit.tune import deferred as tune_deferred

logger = getLogger(__name__)


@patch_for_jit_tuning
def create_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipe.to("cuda")
    return pipe


def test_jit_sd15():
    pipe = create_model()

    prompt = "A futuristic cityscape with neon lights and flying cars"

    config.mode = JITMode.TUNE_DEFERRED

    pipe([prompt], num_inference_steps=50, height=256, width=256)

    tune_deferred()

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))
    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*CLIPTextModel.*state=tuned.*TensorRTBackend", history[1])
    assert re.match(r".*UNet2DConditionModel.*state=tuned.*TensorRTBackend", history[2])
    assert re.match(r".*Conv2d.*", history[3])
    assert re.match(r".*Decoder.*state=tuned.*TensorRTBackend", history[4])
    assert re.match(r".*StableDiffusionSafetyChecker.*state=tuned.*TorchInductorJitBackend", history[5])

    logger.info("Testing inference with batch_size=1")
    start = perf_counter()
    pipe(
        [prompt],
        height=256,
        width=256,
        num_inference_steps=50,
    )
    end = perf_counter()
    logger.info("Batch_size=1, res=256, steps=50, inference duration: %.2f seconds", end - start)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_sd15()
