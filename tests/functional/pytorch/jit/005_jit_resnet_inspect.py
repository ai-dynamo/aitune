# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test JIT tuning with patch decorator on resnet."""
# /// script
# dependencies = ["timm"]
# scope = "always"
# allow_failure = false
# ///

import os
import re
from logging import INFO, basicConfig
from pathlib import Path

import timm
import torch

import aitune.torch.jit.enable_inspection as inspection  # noqa: F401


def create_resnet():
    """Create a ResNet18 model.

    The decorator will make this model tunable.
    """
    return timm.create_model("resnet18", pretrained=False).to("cuda")


def test_jit_resnet():
    resnet = create_resnet()

    def batch():
        # we are calling two times with different batch sizes to recognize dynamic axes
        resnet(torch.randn(2, 3, 224, 224, device="cuda"))
        resnet(torch.randn(16, 3, 224, 224, device="cuda"))

    for _ in range(5):
        batch()

    output_dir = Path(os.environ.get("AITUNE_OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "inspect_resnet.html"
    inspection.save_report(html_path, "ResNet")

    assert html_path.exists(), f"HTML file {html_path} was not created"

    with html_path.open(encoding="utf-8") as f:
        html_content = f.read()

    # Assert basic HTML structure
    assert "<!DOCTYPE html>" in html_content, "HTML file missing DOCTYPE declaration"
    assert '<html lang="en">' in html_content, "HTML file missing html tag"
    assert "<head>" in html_content, "HTML file missing head section"
    assert "<body>" in html_content, "HTML file missing body section"

    # Assert model name appears in HTML
    assert "Model: ResNet" in html_content, "Model name 'ResNet' not found in HTML"

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
    test_jit_resnet()
