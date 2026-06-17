# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on FLUX."""

# /// script
# dependencies = ["diffusers>=0.25.0,<0.35","transformers"]
# scope = "always"
# allow_failure = false
# use_gated_hf_token = true
# ///
import re
from logging import getLogger
from time import perf_counter

import torch
from _tuning_data_artifacts import collect_tuning_data
from diffusers import FluxPipeline

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning

logger = getLogger(__name__)


@patch_for_jit_tuning
def get_flux_pipeline(model_name: str = "hf-internal-testing/tiny-flux-pipe", device: str = "cuda"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        FluxPipeline: The loaded Flux pipeline
    """
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device, dtype=torch.float16)
    torch.cuda.empty_cache()
    return pipe


@collect_tuning_data(__file__)
def test_jit_flux():
    pipe = get_flux_pipeline()

    prompt = "A fluffy, orange tabby cat with bright green eyes is captured mid-air, pouncing playfully on a vibrant red ball of yarn"

    config.dry_run = False
    config.min_samples = 4
    config.max_depth_level = 1
    config.detect_graph_breaks = True

    def batch():
        for size in [256, 512]:
            pipe([prompt] * 1, num_inference_steps=1, height=size, width=size)
            pipe([prompt] * 2, num_inference_steps=1, height=size, width=size)

    for _ in range(5):
        batch()

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))
    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    # assert re.match(r".*CLIPTextModel.*state=tuned.*TensorRTBackend", history[1])
    assert re.match(r".*T5EncoderModel.*state=tuned.*TensorRTBackend", history[2])
    assert re.match(r".*FluxTransformer2DModel.*state=tuned.*TensorRTBackend", history[3])
    # assert re.match(r".*Decoder.*state=tuned.*TensorRTBackend", history[4])

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
    # basicConfig(level=INFO, force=True)
    test_jit_flux()
