# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on Wan2.2-T2V-A14B-Diffusers."""

# /// script
# dependencies = ["diffusers>0.35","transformers","accelerate","ftfy"]
# scope = "nightly"
# allow_failure = false
# additional_tags = ["mem/80g"]
# ///
import re
from logging import INFO, basicConfig, getLogger
from time import perf_counter

import torch
from diffusers import WanPipeline

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning

logger = getLogger(__name__)


@patch_for_jit_tuning
def get_wan_pipeline(model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers", device: str = "cuda"):
    """Get a pretrained Wan model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        WanPipeline: The loaded Wan pipeline
    """
    pipe = WanPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def test_jit_wan():
    pipe = get_wan_pipeline()

    prompt = """The camera rushes from far to near in a low-angle shot, revealing a white ferret on a log. It plays, leaps into the water,
    and emerges, as the camera zooms in for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
    Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic shadows and warm highlights.
    Medium composition, front view, low angle, with depth of field."""

    negative_prompt = """Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray,
    worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed,
    disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"""

    config.dry_run = False
    config.min_samples = 68
    config.batch_axis_required = False
    config.max_depth_level = 1
    config.detect_graph_breaks = True

    with torch.no_grad():
        pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50)

    # Capture the print_hierarchy output
    history = []
    PatchedModule.print_hierarchy(sink=lambda s: history.append(s))
    print("\n".join(history))
    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in history[0]
    assert re.match(r".*UMT5EncoderModel.*state=recording", history[1])
    assert re.match(r".*WanTransformer3DModel.*state=recording", history[2])
    assert re.match(r".*WanTransformer3DModel.*state=tuned.*TensorRTBackend", history[3]) or re.match(
        r".*WanTransformer3DModel.*state=tuned.*TorchInductorJitBackend", history[3]
    )
    assert re.match(r".*WanCausalConv3d.*state=recording", history[4])
    assert re.match(r".*WanDecoder3d.*state=recording", history[5])
    assert re.match(r".*WanCausalConv3d.*state=recording", history[6])
    assert re.match(r".*WanCausalConv3d.*state=recording", history[7])

    logger.info("Testing inference with batch_size=1")
    start = perf_counter()
    pipe(prompt, negative_prompt=negative_prompt)
    end = perf_counter()
    logger.info("Batch_size=1, res=256, steps=50, inference duration: %.2f seconds", end - start)


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_wan()
