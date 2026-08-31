# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare original and AITune-tuned FLUX image generation."""

import os
import time
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from aitune_examples_common.checkpoint import relocated_checkpoint_path

import aitune.torch as ait
from flux.cmd_args import parse_args
from flux.context_parallel import ContextParallelMode
from flux.distributed import distributed_output_path, is_rank_zero, synchronize
from flux.distributed import initialize as initialize_distributed
from flux.distributed import shutdown as shutdown_distributed
from flux.model import get_pipeline

logger = getLogger(__name__)


def _run_pipeline(pipeline, prompt, sizes, steps, guidance_scale, max_sequence_length, filename_prefix):
    generated_images = []
    for width, height in sizes:
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "generator": torch.Generator("cpu").manual_seed(0),
            "output_type": "pil" if is_rank_zero() else "pt",
        }
        if is_rank_zero():
            logger.info("Warmup generation")
        pipeline(**kwargs)
        if is_rank_zero():
            logger.info("Warmup generation done")

        synchronize()
        if is_rank_zero():
            logger.info("%s generation", filename_prefix)
        start = time.perf_counter()
        images = pipeline(**kwargs)
        synchronize()

        if is_rank_zero():
            logger.info(
                "%s generation time (%dx%d): %.4f s",
                filename_prefix,
                width,
                height,
                time.perf_counter() - start,
            )
            generated_images.append((images[0][0], width, height))
    return generated_images


def _save_images(output_path, prompt, filename_prefix, generated_images):
    output_path.mkdir(parents=True, exist_ok=True)
    for image, width, height in generated_images:
        image_path = output_path / f"flux_output_{filename_prefix}_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        image.save(image_path)
        logger.info("Generated image saved to: %s", image_path)


def do_inference(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    output_dir,
    multi_gpu=False,
    context_parallel=ContextParallelMode.ULYSSES,
):
    """Generate and save images before and after loading an AITune checkpoint.

    Args:
        model_name: Hugging Face model name or path.
        prompt: Text prompt for image generation.
        sizes: List of ``(width, height)`` tuples.
        steps: Number of inference steps.
        guidance_scale: Guidance scale.
        max_sequence_length: Maximum sequence length.
        tuned_model_path: Path from which to load the tuned model.
        output_dir: Directory in which to save generated images.
        multi_gpu: Whether to use Diffusers context parallelism.
        context_parallel: Context-parallel attention mode.
    """
    pipeline = get_pipeline(model_name=model_name, multi_gpu=multi_gpu, context_parallel=context_parallel)

    output_path = Path(output_dir)
    if is_rank_zero():
        logger.info("Generating images on original pipeline")
    original_images = _run_pipeline(pipeline, prompt, sizes, steps, guidance_scale, max_sequence_length, "orig")
    if is_rank_zero():
        _save_images(output_path, prompt, "orig", original_images)
    synchronize()

    if is_rank_zero():
        logger.info("Loading tuned pipeline")
    pipeline = ait.load(pipeline, tuned_model_path)

    if is_rank_zero():
        logger.info("Generating images on tuned pipeline")
    tuned_images = _run_pipeline(pipeline, prompt, sizes, steps, guidance_scale, max_sequence_length, "tuned")
    if is_rank_zero():
        _save_images(output_path, prompt, "tuned", tuned_images)
    synchronize()


def run_example(args) -> None:
    """Generate images before and after loading the configured checkpoint."""
    tuned_model_path = distributed_output_path(args.tuned_model_path)
    do_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=relocated_checkpoint_path(tuned_model_path),
        output_dir=os.environ.get("AITUNE_OUTPUT_DIR", "output"),
        multi_gpu=args.multi_gpu,
        context_parallel=args.context_parallel,
    )


def main():
    """Initialize distributed execution, run the example, and release its resources."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    initialize_distributed(args.multi_gpu)
    try:
        run_example(args)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
