# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a FLUX pipeline for single-GPU or context-parallel execution."""

import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, FluxPipeline

from flux.context_parallel import ContextParallelMode
from flux.defaults import DEFAULT_MODEL_NAME

MODEL_NAME = DEFAULT_MODEL_NAME


def get_pipeline(
    model_name: str = MODEL_NAME,
    device: str = "cuda",
    multi_gpu: bool = False,
    context_parallel: ContextParallelMode = ContextParallelMode.ULYSSES,
):
    """Load a pretrained FLUX pipeline from Hugging Face.

    Args:
        model_name: Hugging Face model name or path.
        device: Device on which to load the pipeline.
        multi_gpu: Whether to apply Diffusers context parallelism.
        context_parallel: Context-parallel attention mode.

    Returns:
        The loaded FLUX pipeline.
    """
    pipeline = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if multi_gpu:
        pipeline.transformer.set_attention_backend("_native_cudnn")
        world_size = dist.get_world_size()
        context_parallel_configs = {
            ContextParallelMode.RING: ContextParallelConfig(ring_degree=world_size),
            ContextParallelMode.ULYSSES: ContextParallelConfig(ulysses_degree=world_size),
        }
        pipeline.transformer.enable_parallelism(config=context_parallel_configs[context_parallel])
    torch.cuda.empty_cache()
    return pipeline
