# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect and tune a FLUX pipeline on one or multiple GPUs."""

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
    TorchInductorJitBackendConfig,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from flux.cmd_args import parse_args
from flux.context_parallel import ContextParallelMode
from flux.defaults import DEFAULT_BATCH_SIZE
from flux.distributed import distributed_output_path, is_rank_zero, synchronize
from flux.distributed import initialize as initialize_distributed
from flux.distributed import shutdown as shutdown_distributed
from flux.model import get_pipeline

logger = getLogger(__name__)


def filter_fn(mod, fqn):
    """Select large transformer linear layers for TorchAO quantization.

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


def _transformer_strategy(
    sizes: list[tuple[int, int]],
    multi_gpu=False,
    quantization=False,
):
    dynamic = len(sizes) > 1
    backends = []
    if quantization:
        backends = [
            TorchAOBackend(config=TorchAOBackendConfig(quantization="nvfp4dq", filter_fn=filter_fn)),
            TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8dq", filter_fn=filter_fn)),
        ]

    backends += [
        TorchTensorRTJitBackend(
            config=TorchTensorRTJitBackendConfig(
                dynamic=dynamic,
                compile_config=TorchTensorRTConfig(
                    use_distributed_mode_trace=multi_gpu, min_block_size=50, truncate_double=True
                ),
            )
        ),
        TorchInductorJitBackend(config=TorchInductorJitBackendConfig(dynamic=dynamic)),
    ]

    strategy = ait.MaxThroughputStrategy(backends=backends)
    return strategy


def _pipeline_strategy():
    """Compare backends used by tunable modules outside the transformer."""
    return ait.MaxThroughputStrategy(
        backends=[
            TensorRTBackend(),
            TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=False)),
            TorchInductorJitBackend(),
        ]
    )


def tune_model(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    multi_gpu=False,
    context_parallel=ContextParallelMode.ULYSSES,
    quantization=False,
):
    """Inspect, tune, and save the FLUX pipeline.

    Args:
        model_name: Hugging Face model name or path.
        prompt: Text prompt for image generation.
        sizes: List of ``(width, height)`` tuples.
        steps: Number of inference steps.
        guidance_scale: Guidance scale.
        max_sequence_length: Maximum sequence length.
        tuned_model_path: Path used to save the tuned model.
        multi_gpu: Whether to use Diffusers context parallelism.
        context_parallel: Context-parallel attention mode.
        quantization: Whether to include TorchAO quantization backends for the transformer.
    """
    pipeline = get_pipeline(model_name=model_name, multi_gpu=multi_gpu, context_parallel=context_parallel)

    def run_pipeline(*args, **kwargs):
        for width, height in sizes:
            print(f"Generating image with height={height} and width={width}")  # noqa: T201
            pipeline(
                *args,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
                output_type="pil" if is_rank_zero() else "pt",
                **kwargs,
            )

    input_data = [{"prompt": prompt}]

    # Inspect the end-to-end pipeline to discover tunable modules.
    modules_info = ait.inspect(
        pipeline,
        input_data,
        inference_function=run_pipeline,
        number_of_iterations=1,
        warmup_iterations=2,
    )
    modules_info.describe()

    transformer_strategy = _transformer_strategy(sizes=sizes, multi_gpu=multi_gpu, quantization=quantization)
    transformer_strategy.enable_find_max_batch_size(enable=False)

    default_strategy = _pipeline_strategy()
    default_strategy.enable_find_max_batch_size(enable=False)

    # Leave all modules except the transformer unquantized.
    modules = [m for m in modules_info.get_modules() if m.name != "transformer"]
    pipeline = ait.wrap(pipeline, modules, strategy=default_strategy)

    # Tune the transformer separately so quantization can be enabled independently.
    pipeline.transformer = ait.module.Module(
        pipeline.transformer,
        name="transformer",
        strategy=transformer_strategy,
    )

    # Record the generation workload and tune every wrapped module.
    logger.info("Tuning pipeline: %s", model_name)
    ait.tune(run_pipeline, input_data, batch_sizes=[DEFAULT_BATCH_SIZE])
    logger.info("Tuning completed.")

    if multi_gpu:
        tuned_model_path = distributed_output_path(tuned_model_path)
    ait.save(pipeline, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)
    relocated_path = copy_checkpoint_to_tmp(tuned_model_path)
    logger.info("Checkpoint copied to %s", relocated_path)


def run_example(args) -> None:
    """Inspect, tune, and save the configured FLUX pipeline."""
    tune_model(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=args.tuned_model_path,
        multi_gpu=args.multi_gpu,
        context_parallel=args.context_parallel,
        quantization=args.quantization,
    )
    synchronize()


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
