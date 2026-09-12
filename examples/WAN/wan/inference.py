# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare original and AITune-tuned WAN video generation."""

import os
import time
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from aitune_examples_common.checkpoint import relocated_checkpoint_path
from diffusers.utils import export_to_video

import aitune.torch as ait
from wan.cmd_args import parse_args
from wan.context_parallel import ContextParallelMode
from wan.distributed import distributed_output_path, is_rank_zero, synchronize
from wan.distributed import initialize as initialize_distributed
from wan.distributed import shutdown as shutdown_distributed
from wan.model import get_pipeline

logger = getLogger(__name__)


def _pipeline_kwargs(prompt, negative_prompt, height, width, num_frames, steps, guidance_scale, max_sequence_length):
    """Build deterministic WAN generation arguments."""
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "output_type": "np",
    }


def _generate(pipeline, kwargs, label):
    """Warm up and time one video generation."""
    logger.info("Warming up %s generation", label)
    pipeline(**kwargs, generator=torch.Generator("cpu").manual_seed(0))
    synchronize()

    logger.info("Starting %s generation", label)
    start = time.perf_counter()
    frames = pipeline(**kwargs, generator=torch.Generator("cpu").manual_seed(0)).frames[0]
    synchronize()
    if is_rank_zero():
        logger.info("%s generation time: %.4f s", label, time.perf_counter() - start)
        return frames
    return None


def do_inference(
    model_name,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    steps,
    guidance_scale,
    max_sequence_length,
    fps,
    tuned_model_path,
    output_dir,
    multi_gpu=False,
    context_parallel=ContextParallelMode.ULYSSES,
):
    """Generate original and tuned WAN videos and save them on rank zero."""
    pipeline = get_pipeline(model_name=model_name, multi_gpu=multi_gpu, context_parallel=context_parallel)
    kwargs = _pipeline_kwargs(
        prompt, negative_prompt, height, width, num_frames, steps, guidance_scale, max_sequence_length
    )

    original_frames = _generate(pipeline, kwargs, "original")
    if is_rank_zero():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        export_to_video(original_frames, output_path / "wan_original.mp4", fps=fps)

    synchronize()
    logger.info("Loading tuned WAN pipeline")
    pipeline = ait.load(pipeline, tuned_model_path)
    tuned_frames = _generate(pipeline, kwargs, "tuned")
    if is_rank_zero():
        export_to_video(tuned_frames, Path(output_dir) / "wan_tuned.mp4", fps=fps)
    synchronize()


def run_example(args, multi_gpu: bool) -> None:
    """Generate videos before and after loading the configured checkpoint."""
    checkpoint_path = distributed_output_path(args.tuned_model_path)
    do_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        fps=args.fps,
        tuned_model_path=relocated_checkpoint_path(checkpoint_path),
        output_dir=os.environ.get("AITUNE_OUTPUT_DIR", "output"),
        multi_gpu=multi_gpu,
        context_parallel=args.context_parallel,
    )


def main():
    """Initialize distributed execution, run inference, and release its resources."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    multi_gpu = initialize_distributed()
    try:
        run_example(args, multi_gpu)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
