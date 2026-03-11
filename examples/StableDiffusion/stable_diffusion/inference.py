# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference script for Stable Diffusion model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

from aitune.torch import load
from stable_diffusion.cmd_args import parse_args
from stable_diffusion.model import get_pipeline

logger = getLogger(__name__)


def do_inference(
    model_name,
    prompt,
    sizes,
    steps,
    tuned_model_path,
    output_dir,
):
    """Do inference with the Stable Diffusion model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        tuned_model_path: Path to load the tuned model
        output_dir: Directory to save the generated image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipe = get_pipeline(model_name=model_name)
    pipe.to("cuda")

    logger.info("Generating images on original pipeline")
    for height, width in sizes:
        images = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)
        image_path = output_path / f"stable_diffusion_output_orig_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        images[0][0].save(image_path)
        logger.info("Generated image saved to: %s", image_path)

    logger.info("Loading tuned pipeline")
    pipe = load(pipe, tuned_model_path)

    logger.info("Generating images on tuned pipeline")
    for height, width in sizes:
        images = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)
        image_path = output_path / f"stable_diffusion_output_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        images[0][0].save(image_path)
        logger.info("Generated image saved to: %s", image_path)


def main():
    """Entry point for the script."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    do_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        tuned_model_path=args.tuned_model_path,
        output_dir=os.environ.get("AITUNE_OUTPUT_DIR", "output"),
    )


if __name__ == "__main__":
    main()
