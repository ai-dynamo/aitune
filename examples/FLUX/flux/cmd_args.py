# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common command line arguments for Flux."""

import argparse

from flux.context_parallel import ContextParallelMode
from flux.defaults import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
)


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
    parser = argparse.ArgumentParser(description="Tune or run a FLUX pipeline")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name or path")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
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
        default=[DEFAULT_IMAGE_SIZE],
        help="Image dimensions as space-separated width,height pairs (default: '1024,1024')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Guidance scale",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=str,
        default="flux-dev.ait",
        help="Path to save the tuned model",
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use Diffusers context parallelism")
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Include TorchAO NVFP4 and FP8 transformer backends during tuning",
    )
    parser.add_argument(
        "--context-parallel",
        type=ContextParallelMode,
        choices=ContextParallelMode,
        default=ContextParallelMode.ULYSSES,
        help="Context-parallel attention mode for multi-GPU execution",
    )

    args = parser.parse_args()
    return args
