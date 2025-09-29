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

import os
import re
from logging import INFO, basicConfig
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from aitune.torch.jit.config import config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patcher import patch_for_jit_tuning


@patch_for_jit_tuning
def create_model():
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe


def test_jit_sd21():
    prompt = "A fluffy, orange tabby cat with bright green eyes is captured mid-air, pouncing playfully on a vibrant red ball of yarn"
    config.dry_run = False
    config.inspect_mode = True
    config.min_samples = 4
    config.max_depth_level = 3
    config.detect_graph_breaks = True

    pipe = create_model()

    def batch():
        with torch.no_grad():
            pipe([prompt] * 1, num_inference_steps=1)
            pipe([prompt] * 2, num_inference_steps=1)

    for _ in range(5):
        batch()

    output_dir = Path(os.environ.get("AITUNE_OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "inspect_sd21.html"
    InspectModule.export_history_to_html(html_path, "SD21")

    assert html_path.exists(), f"HTML file {html_path} was not created"

    with html_path.open(encoding="utf-8") as f:
        html_content = f.read()

    # Assert basic HTML structure
    assert "<!DOCTYPE html>" in html_content, "HTML file missing DOCTYPE declaration"
    assert '<html lang="en">' in html_content, "HTML file missing html tag"
    assert "<head>" in html_content, "HTML file missing head section"
    assert "<body>" in html_content, "HTML file missing body section"

    # Assert model name appears in HTML
    assert "Model: SD21" in html_content, "Model name 'SD21' not found in HTML"

    # Assert title contains expected text
    assert "AITune Model Inspector" in html_content, "Expected title not found in HTML"

    # Assert CSS styles are present
    assert "<style>" in html_content, "CSS styles not found in HTML"
    assert "background: linear-gradient" in html_content, "Expected CSS styling not found"

    # Assert JavaScript functionality is present
    assert "<script>" in html_content, "JavaScript not found in HTML"
    assert "function toggleModule" in html_content, "Toggle functionality not found in JavaScript"
    assert "function searchModules" in html_content, "Search functionality not found in JavaScript"

    # Assert module hierarchy data is present (should have at least one module)
    assert "module-item" in html_content, "No module items found in HTML"
    assert "module-header" in html_content, "No module headers found in HTML"
    assert "module-details" in html_content, "No module details found in HTML"

    # Assert that there are modules with execution data
    assert "Call Count" in html_content, "Call count data not found in HTML"
    assert "Execution Details" in html_content, "Execution data not found in HTML"

    # Assert that we have encountered modules (count module names)
    module_names = re.findall(r'<div class="module-name">([^<]+)</div>', html_content)
    assert len(module_names) >= 5, f"Expected at least 5 modules, but found only {len(module_names)}"


if __name__ == "__main__":
    basicConfig(level=INFO, force=True)
    test_jit_sd21()
