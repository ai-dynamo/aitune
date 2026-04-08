# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune Stable Diffusion model."""

import logging
import os

from aitune.torch import FirstWinsStrategy, inspect, save, tune, wrap
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, TorchEagerBackend, TorchInductorJitBackend
from stable_diffusion.cmd_args import parse_args
from stable_diffusion.model import get_pipeline

logger = logging.getLogger(__name__)


def tune_model(model_name, prompt, sizes, steps, tuned_model_path, batch_sizes=None, strategy=None):
    """Tune the Stable Diffusion model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        tuned_model_path: Path to save the tuned model
        batch_sizes: List of batch sizes to tune, if None, tune only with batch size 1
    """
    batch_sizes = batch_sizes or [1]
    pipeline = get_pipeline(model_name=model_name)

    input_data = [{"prompt": prompt}]

    # Inspect pipeline to get modules
    modules_info = inspect(pipeline, input_data)

    # Define strategy
    if strategy is None:
        strategy = FirstWinsStrategy(
            backends=[
                TensorRTBackend(),
                TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
                TorchInductorJitBackend(),
                TorchEagerBackend(),
            ]
        )
        strategy.enable_find_max_batch_size(enable=False)

    # Wrap all modules with AITune Module
    modules = modules_info.get_modules(min_execution_percentage=0.05)
    pipeline = wrap(pipeline, modules, strategy=strategy)

    def call_wrapper(*args, **kwargs):
        for height, width in sizes:
            print(f"Generating image with height={height} and width={width}")  # noqa: T201
            pipeline(
                *args,
                height=height,
                width=width,
                num_inference_steps=steps,
                **kwargs,
            )

    logger.info("Tuning module: %s", model_name)
    tune(call_wrapper, input_data, batch_sizes=batch_sizes)
    logger.info("Tuning completed.")

    save(pipeline, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True
    )
    args = parse_args()
    tune_model(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
