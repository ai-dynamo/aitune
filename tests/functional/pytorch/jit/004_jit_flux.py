# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test JIT tuning with patch decorator on FLUX."""

# /// script
# dependencies = ["diffusers>=0.25.0,<0.35","transformers"]
# scope = "always"
# allow_failure = false
# use_gated_hf_token = true
# additional_tags = ["mem/80g"]
# ///
import re
from logging import INFO, basicConfig, getLogger
from time import perf_counter

import torch
from diffusers import FluxPipeline

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning

logger = getLogger(__name__)


@patch_for_jit_tuning
def get_flux_pipeline(model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "cuda"):
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


def test_jit_flux():
    pipe = get_flux_pipeline()

    prompt = "A fluffy, orange tabby cat with bright green eyes is captured mid-air, pouncing playfully on a vibrant red ball of yarn"

    config.dry_run = False
    config.min_samples = 4
    config.max_depth_level = 1
    config.detect_graph_breaks = True

    def batch():
        with torch.no_grad():
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
    basicConfig(level=INFO, force=True)
    test_jit_flux()
