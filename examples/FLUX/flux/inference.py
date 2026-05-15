# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference script for Flux model."""

import os
import time
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from aitune_examples_common.checkpoint import relocated_checkpoint_path

import aitune.torch as ait
from aitune.utils.monitoring import annotate
from flux.cmd_args import parse_args
from flux.model import get_pipeline

logger = getLogger(__name__)


def _run_pipeline(pipe, prompt, sizes, steps, guidance_scale, max_sequence_length, output_path, filename_prefix):
    for height, width in sizes:
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "generator": torch.Generator("cpu").manual_seed(0),
        }
        logger.info("Warmup generation")
        pipe(**kwargs)
        logger.info("Warmup generation done")

        logger.info("%s generation", filename_prefix)
        start = time.perf_counter()
        images = pipe(**kwargs)
        logger.info("%s generation time (%dx%d): %.4f s", filename_prefix, height, width, time.perf_counter() - start)

        image_path = output_path / f"flux_output_{filename_prefix}_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        images[0][0].save(image_path)
        logger.info("Generated image saved to: %s", image_path)


@annotate(name="inference", color="green")
def do_inference(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    output_dir,
):
    """Do inference with the Flux model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        guidance_scale: Guidance scale
        max_sequence_length: Maximum sequence length
        tuned_model_path: Path to load the tuned model
        output_dir: Directory to save the generated image
    """
    pipe = get_pipeline(model_name=model_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating images on original pipeline")
    _run_pipeline(pipe, prompt, sizes, steps, guidance_scale, max_sequence_length, output_path, "orig")

    logger.info("Loading tuned pipeline")
    pipe = ait.load(pipe, tuned_model_path)

    logger.info("Generating images on tuned pipeline")
    _run_pipeline(pipe, prompt, sizes, steps, guidance_scale, max_sequence_length, output_path, "tuned")


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    do_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=relocated_checkpoint_path(args.tuned_model_path),
        output_dir=os.environ.get("AITUNE_OUTPUT_DIR", "output"),
    )


if __name__ == "__main__":
    main()
