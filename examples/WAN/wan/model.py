# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a WAN pipeline for single-GPU or context-parallel execution."""

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, ContextParallelConfig, WanPipeline

from wan.context_parallel import ContextParallelMode
from wan.defaults import DEFAULT_MODEL_NAME


def get_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    multi_gpu: bool = False,
    context_parallel: ContextParallelMode = ContextParallelMode.ULYSSES,
):
    """Load a pretrained WAN text-to-video pipeline.

    Args:
        model_name: Hugging Face model name or path.
        device: Device on which to load the pipeline.
        multi_gpu: Whether to apply Diffusers context parallelism.
        context_parallel: Context-parallel attention mode.

    Returns:
        The loaded WAN pipeline.
    """
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
    pipeline = WanPipeline.from_pretrained(model_name, vae=vae, torch_dtype=torch.bfloat16).to(device)
    pipeline.set_progress_bar_config(disable=True)

    if multi_gpu:
        world_size = dist.get_world_size()
        configs = {
            ContextParallelMode.RING: ContextParallelConfig(ring_degree=world_size),
            ContextParallelMode.ULYSSES: ContextParallelConfig(ulysses_degree=world_size),
        }
        transformers = (pipeline.transformer, getattr(pipeline, "transformer_2", None))
        for transformer in filter(None, transformers):
            transformer.set_attention_backend("_native_cudnn")
            transformer.enable_parallelism(config=configs[context_parallel])

    torch.cuda.empty_cache()
    return pipeline
