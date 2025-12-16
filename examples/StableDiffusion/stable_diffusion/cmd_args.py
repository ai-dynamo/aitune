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
"""Common command line arguments for Stable Diffusion."""

import argparse


def parse_sizes(sizes_str: str) -> list[tuple[int, int]]:
    """Parse sizes string into list of width and height tuples.

    Args:
        sizes_str: String with multiple size combinations separated by spaces
                  (e.g., "128,128 256,256" or "512,512")

    Returns:
        List of (width, height) tuples as integers

    Raises:
        ValueError: If the format is invalid or values are not positive integers
    """
    try:
        size_combinations = sizes_str.split()
        result = []

        for combo in size_combinations:
            parts = combo.split(",")
            if len(parts) != 2:
                raise ValueError(f"Size combination '{combo}' must be in format 'width,height' (e.g., '512,512')")

            width = int(parts[0].strip())
            height = int(parts[1].strip())

            if width <= 0 or height <= 0:
                raise ValueError(f"Width and height in '{combo}' must be positive integers")

            result.append((width, height))

        return result
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("Width and height must be valid integers") from e
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tuned Stable Diffusion model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape with mountains and a lake",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry",
        help="Negative text prompt",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[(512, 512), (1024, 1024)],
        help="Image dimensions as space-separated width,height pairs (e.g., '128,128 256,256' or '512,512')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=str,
        default="stable_diffusion.ait",
        help="Path to save the tuned model",
    )
    args = parser.parse_args()
    return args
