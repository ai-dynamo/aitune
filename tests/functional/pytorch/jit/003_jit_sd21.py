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
"""Test JIT tuning with patch decorator on Stable Diffusion 2.1."""

# /// script
# dependencies = ["diffusers", "transformers"]
# scope = "always"
# allow_failure = false
# ///

import re
from logging import INFO, basicConfig

import torch
from diffusers import StableDiffusionPipeline

from aitune.torch.jit.config import config
from aitune.torch.jit.patched_module import PRINT_HIERARCHY_HEADER, PatchedModule
from aitune.torch.jit.patcher import patch_for_jit_tuning


class TestSink:
    """Sink for capturing output from PatchedModule.print_hierarchy."""

    def __init__(self):
        self.output = []

    def write(self, text):
        self.output.append(text)


@patch_for_jit_tuning
def create_model():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe


def test_jit_sd21():
    pipe = create_model()

    prompt = "A fluffy, orange tabby cat with bright green eyes is captured mid-air, pouncing playfully on a vibrant red ball of yarn"

    config.dry_run = False
    config.min_samples = 4
    config.max_depth_level = 1
    config.detect_graph_breaks = True

    def batch():
        with torch.no_grad():
            pipe([prompt] * 1, num_inference_steps=1)
            pipe([prompt] * 2, num_inference_steps=1)

    for _ in range(5):
        batch()

    # Capture the print_hierarchy output
    sink = TestSink()
    PatchedModule.print_hierarchy(sink=sink.write)
    print("\n".join(sink.output))
    # Assert the expected output
    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*CLIPTextModel.*state=tuned.*TensorRTBackend", sink.output[1])
    assert re.match(r".*UNet2DConditionModel.*state=tuned.*TensorRTBackend", sink.output[2])
    assert re.match(r".*Conv2d.*state=tuned.*TensorRTBackend", sink.output[3])
    assert re.match(r".*Decoder.*state=tuned.*TensorRTBackend", sink.output[4])


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_sd21()
