# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect and tune a FLUX pipeline on one or multiple GPUs."""

from __future__ import annotations

import os
import time
from logging import basicConfig, getLogger
from typing import TYPE_CHECKING

import torch

import aitune.torch as ait
from aitune.torch.backend import (
    Backend,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)

from .checkpoint import save_checkpoint
from .config import GenerationInput, Recipe, argument_parser, load_recipe
from .distributed import initialize, launch, shutdown
from .model import generate, get_pipeline

if TYPE_CHECKING:
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

logger = getLogger(__name__)


def filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
    """Select large transformer linear layers for TorchAO quantization.

    Adapter from:
    - Source code: https://github.com/sayakpaul/diffusers-blackwell-quants/blob/9fefb0744ca6eef03d558728f4ee74304978da76/benchmark.py#L194
    - Blog post: https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
    """
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
    multi_gpu: bool = False,
    quantization: bool = False,
) -> ait.MaxThroughputStrategy:
    backends: list[Backend] = []
    if quantization:
        backends = [
            TorchAOBackend(config=TorchAOBackendConfig(quantization="nvfp4dq", filter_fn=filter_fn)),
            TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8dq", filter_fn=filter_fn)),
        ]

    backends += [
        TorchTensorRTJitBackend(
            config=TorchTensorRTJitBackendConfig(
                dynamic=False,
                compile_config=TorchTensorRTConfig(
                    use_distributed_mode_trace=multi_gpu, min_block_size=50, truncate_double=True
                ),
            )
        ),
        TorchInductorJitBackend(config=TorchInductorJitBackendConfig(dynamic=False)),
    ]

    strategy = ait.MaxThroughputStrategy(backends=backends)
    return strategy


def tune_model(recipe: Recipe) -> None:
    """Inspect all tuning samples, select backends, and save rank-specific artifacts."""
    pipeline = get_pipeline(recipe)
    samples = recipe.samples("tune")
    logger.info("Tuning samples: %s", [sample["id"] for sample in samples])

    def run_pipeline(prompt: str, seed: int | torch.Tensor) -> FluxPipelineOutput:
        return generate(pipeline, recipe, {"prompt": prompt, "seed": int(seed)})

    input_data: list[GenerationInput] = [{"prompt": sample["prompt"], "seed": sample["seed"]} for sample in samples]
    modules_info = ait.inspect(
        pipeline,
        input_data,
        inference_function=run_pipeline,
        number_of_iterations=len(samples),
        warmup_iterations=recipe["benchmark"]["warmup_iterations"],
    )
    modules_info.describe()
    transformer_strategy = _transformer_strategy(
        multi_gpu=recipe.gpu_count > 1,
        quantization=recipe["tuning"]["quantization"],
    ).enable_find_max_batch_size(False)
    default_strategy = ait.resolve_strategy().enable_find_max_batch_size(False)
    modules = [module for module in modules_info.get_modules() if module.name != "transformer"]
    pipeline = ait.wrap(pipeline, modules, strategy=default_strategy)
    pipeline.transformer = ait.module.Module(pipeline.transformer, name="transformer", strategy=transformer_strategy)
    started = time.perf_counter()
    ait.tune(run_pipeline, input_data, batch_sizes=[1])
    tuning_seconds = time.perf_counter() - started
    save_checkpoint(pipeline, recipe, tuning_seconds)
    logger.info("Saved %s; tuning took %.3f seconds", recipe.artifact_path, tuning_seconds)


def run(recipe: Recipe) -> None:
    basicConfig(level=os.environ.get("AITUNE_LOG_LEVEL", "INFO"), force=True)
    launch(recipe, "flux_schnell.tune")
    initialize()
    try:
        tune_model(recipe)
    finally:
        shutdown()


def main() -> None:
    run(load_recipe(argument_parser(__doc__).parse_args().recipe))


if __name__ == "__main__":
    main()
