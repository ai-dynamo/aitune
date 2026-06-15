# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune Flux model."""

import os
from logging import basicConfig, getLogger

import torch
from aitune_examples_common.checkpoint import copy_checkpoint_to_tmp

import aitune.torch as ait
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchInductorJitBackend,
    TorchQuantizationConfig,
)
from flux.cmd_args import parse_args
from flux.model import get_pipeline

logger = getLogger(__name__)


def filter_fn(mod, fqn):
    """Filter function for Flux.

    Adapter from:
    - Source code: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/9fefb0744ca6eef03d558728f4ee74304978da76/benchmark.py#L194
    - Blog post: https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
    """
    import torch

    if not isinstance(mod, torch.nn.Linear):
        return False
    elif "embed" in fqn:
        return False
    elif fqn == "norm_out.linear":
        return False
    elif fqn == "proj_out":
        return False
    elif mod.in_features < 1024 or mod.out_features < 1024:
        return False
    return True


def _nvfp4_strategy():
    strategy_nvfp4 = ait.FirstWinsStrategy(
        backends=[
            TorchAOBackend(TorchAOBackendConfig(quantization="nvfp4dq", filter_fn=filter_fn)),
            TensorRTBackend(
                TensorRTBackendConfig(
                    quantization_config=TorchQuantizationConfig(
                        quantization_config="FP8_DEFAULT_CFG",
                        device="cuda",
                    ),
                )
            ),
            TensorRTBackend(
                TensorRTBackendConfig(
                    quantization_config=TorchQuantizationConfig(
                        quantization_config="FP8_DEFAULT_CFG",
                        device="cuda",
                    ),
                    use_dynamo=False,
                )
            ),
            TorchAOBackend(TorchAOBackendConfig(quantization="fp8dq")),
            TensorRTBackend(TensorRTBackendConfig(use_dynamo=True)),
            TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]
    )

    return strategy_nvfp4


def _fp8_strategy():
    strategy_fp8 = ait.FirstWinsStrategy(
        backends=[
            TensorRTBackend(
                TensorRTBackendConfig(
                    quantization_config=TorchQuantizationConfig(
                        quantization_config="FP8_DEFAULT_CFG",
                        device="cuda",
                    ),
                )
            ),
            TensorRTBackend(
                TensorRTBackendConfig(
                    quantization_config=TorchQuantizationConfig(
                        quantization_config="FP8_DEFAULT_CFG",
                        device="cuda",
                    ),
                    use_dynamo=False,
                )
            ),
            TorchAOBackend(TorchAOBackendConfig(quantization="fp8dq")),
            TensorRTBackend(TensorRTBackendConfig(use_dynamo=True)),
            TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]
    )
    return strategy_fp8


def tune_model(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    batch_sizes=None,
):
    """Tune the Flux model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        guidance_scale: Guidance scale
        max_sequence_length: Maximum sequence length
        tuned_model_path: Path to save the tuned model
        batch_sizes: List of batch sizes to tune
    """
    pipe = get_pipeline(model_name=model_name)

    def call_wrapper(*args, **kwargs):
        for height, width in sizes:
            print(f"Generating image with height={height} and width={width}")  # noqa: T201
            pipe(
                *args,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
                **kwargs,
            )

    input_data = [{"prompt": prompt}]

    # Inspect pipeline to get modules
    modules_info = ait.inspect(
        pipe,
        input_data,
        inference_function=call_wrapper,
        number_of_iterations=1,
        warmup_iterations=2,
    )
    modules_info.describe()

    # Define strategy with NVFP4 support
    strategy_nvfp4 = _nvfp4_strategy()
    strategy_nvfp4.enable_find_max_batch_size(enable=False)

    # Define strategy with FP8 support
    strategy_fp8 = _fp8_strategy()
    strategy_fp8.enable_find_max_batch_size(enable=False)

    # Wrap all modules except transformer with fp8 strategy
    modules = [m for m in modules_info.get_modules() if m.name != "transformer"]
    pipe = ait.wrap(pipe, modules, strategy=strategy_fp8)

    # Wrap transformer separately with nvfp4 strategy
    pipe.transformer = ait.module.Module(pipe.transformer, name="transformer", strategy=strategy_nvfp4)

    # First do a dry run for testing
    logger.info("Tuning module: %s", model_name)
    ait.tune(call_wrapper, input_data, batch_sizes=[1] if batch_sizes is None else batch_sizes)
    logger.info("Tuning completed.")

    ait.save(pipe, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)
    relocated_path = copy_checkpoint_to_tmp(tuned_model_path)
    logger.info("Checkpoint copied to %s", relocated_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    tune_model(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
