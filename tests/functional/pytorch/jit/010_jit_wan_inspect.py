# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on Wan2.2-T2V-A14B-Diffusers."""

# /// script
# dependencies = ["diffusers>0.35","transformers","accelerate","ftfy"]
# scope = "nightly"
# allow_failure = false
# additional_tags = ["mem/80g"]
# ///

import os
import re
from logging import INFO, basicConfig
from pathlib import Path

import torch
from diffusers import WanPipeline

import aitune.torch.jit.enable_inspection as inspection  # noqa: F401


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


def test_jit_wan_inspect():
    prompt = """The camera rushes from far to near in a low-angle shot, revealing a white ferret on a log. It plays, leaps into the water,
    and emerges, as the camera zooms in for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
    Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic shadows and warm highlights.
    Medium composition, front view, low angle, with depth of field."""

    negative_prompt = """Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray,
    worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed,
    disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"""

    pipe = get_wan_pipeline()

    with torch.no_grad():
        pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=10, height=64, width=128, num_frames=21)

    output_dir = Path(os.environ.get("AITUNE_OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "inspect_wan.html"
    inspection.save_report(html_path, "WAN")

    assert html_path.exists(), f"HTML file {html_path} was not created"

    with html_path.open(encoding="utf-8") as f:
        html_content = f.read()

    # Assert basic HTML structure
    assert "<!DOCTYPE html>" in html_content, "HTML file missing DOCTYPE declaration"
    assert '<html lang="en">' in html_content, "HTML file missing html tag"
    assert "<head>" in html_content, "HTML file missing head section"
    assert "<body>" in html_content, "HTML file missing body section"

    # Assert model name appears in HTML
    assert "Model: WAN" in html_content, "Model name 'WAN' not found in HTML"

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
    test_jit_wan_inspect()
