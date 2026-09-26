# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect and tune a WAN text-to-video pipeline on one or multiple GPUs."""

import os
from logging import basicConfig, getLogger

import torch
from aitune_examples_common.checkpoint import copy_checkpoint_to_tmp

import aitune.torch as ait
from wan.cmd_args import parse_args
from wan.context_parallel import ContextParallelMode
from wan.defaults import DEFAULT_BATCH_SIZE
from wan.distributed import distributed_output_path, synchronize
from wan.distributed import initialize as initialize_distributed
from wan.distributed import shutdown as shutdown_distributed
from wan.model import get_pipeline

logger = getLogger(__name__)


def tune_model(
    model_name,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    multi_gpu=False,
    context_parallel=ContextParallelMode.ULYSSES,
):
    """Inspect, tune, and save the WAN transformer modules."""
    pipeline = get_pipeline(model_name=model_name, multi_gpu=multi_gpu, context_parallel=context_parallel)

    def run_pipeline(*args, **kwargs):
        return pipeline(
            *args,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="latent",
            **kwargs,
        )

    input_data = [{"prompt": prompt}]
    modules_info = ait.inspect(
        pipeline,
        input_data,
        inference_function=run_pipeline,
        number_of_iterations=1,
        warmup_iterations=1,
    )
    modules_info.describe()

    transformer_names = ["transformer"]
    if getattr(pipeline, "transformer_2", None) is not None:
        transformer_names.append("transformer_2")
    for name in transformer_names:
        setattr(
            pipeline,
            name,
            ait.module.Module(
                getattr(pipeline, name),
                name=name,
            ),
        )

    logger.info("Tuning WAN transformer modules: %s", model_name)
    ait.tune(run_pipeline, input_data, batch_sizes=[DEFAULT_BATCH_SIZE])
    logger.info("Tuning completed")

    output_path = distributed_output_path(tuned_model_path) if multi_gpu else tuned_model_path
    ait.save(pipeline, output_path)
    logger.info("Model saved to %s", output_path)
    relocated_path = copy_checkpoint_to_tmp(output_path)
    logger.info("Checkpoint copied to %s", relocated_path)


def run_example(args, multi_gpu: bool) -> None:
    """Inspect, tune, and save the configured WAN pipeline."""
    tune_model(
        model_name=args.model_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=args.tuned_model_path,
        multi_gpu=multi_gpu,
        context_parallel=args.context_parallel,
    )
    synchronize()


def main():
    """Initialize distributed execution, run tuning, and release its resources."""
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
