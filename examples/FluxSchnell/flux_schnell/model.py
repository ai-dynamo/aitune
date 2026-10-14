# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load and execute the pinned FLUX pipeline described by a recipe."""

import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, FluxPipeline

from .config import Recipe
from .distributed import is_rank_zero


def get_pipeline(recipe: Recipe):
    pipeline = FluxPipeline.from_pretrained(
        recipe["model"]["base_model"],
        revision=recipe["model"]["revision"],
        torch_dtype=torch.bfloat16,
    ).to(torch.device("cuda", torch.cuda.current_device()))
    pipeline.set_progress_bar_config(disable=True)
    if recipe.gpu_count > 1:
        pipeline.transformer.set_attention_backend("_native_cudnn")
        mode = recipe["execution"]["context_parallel"]
        pipeline.transformer.enable_parallelism(
            config=ContextParallelConfig(**{f"{mode}_degree": dist.get_world_size()})
        )
    torch.cuda.empty_cache()
    return pipeline


def generate(pipeline, recipe: Recipe, sample: dict):
    """Reset the sample seed for every invocation, including warmup and measurement."""
    with torch.no_grad():
        return pipeline(
            prompt=sample["prompt"],
            **recipe["workload"],
            generator=torch.Generator("cpu").manual_seed(sample["seed"]),
            output_type="pil" if is_rank_zero() else "pt",
            return_dict=True,
        )
