# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load a FLUX pipeline for single-GPU or context-parallel execution."""

import torch
import torch.distributed as dist
from diffusers import (
    AutoencoderKL,
    ContextParallelConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

from flux.context_parallel import ContextParallelMode
from flux.defaults import DEFAULT_MODEL_NAME

MODEL_NAME = DEFAULT_MODEL_NAME


def _get_meta_pipeline(model_name: str) -> FluxPipeline:
    """Build the FLUX pipeline from configs on the meta device; weights come from an AITune checkpoint."""
    with torch.device("meta"):
        pipeline = FluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler"),
            vae=AutoencoderKL.from_config(AutoencoderKL.load_config(model_name, subfolder="vae")),
            text_encoder=CLIPTextModel(CLIPTextConfig.from_pretrained(model_name, subfolder="text_encoder")),
            tokenizer=CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer"),
            text_encoder_2=T5EncoderModel(T5Config.from_pretrained(model_name, subfolder="text_encoder_2")),
            tokenizer_2=T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2"),
            transformer=FluxTransformer2DModel.from_config(
                FluxTransformer2DModel.load_config(model_name, subfolder="transformer")
            ),
        )
    return pipeline.to(dtype=torch.bfloat16)


def get_pipeline(
    model_name: str = MODEL_NAME,
    device: str = "cuda",
    multi_gpu: bool = False,
    context_parallel: ContextParallelMode = ContextParallelMode.ULYSSES,
):
    """Load a pretrained FLUX pipeline from Hugging Face.

    Args:
        model_name: Hugging Face model name or path.
        device: Device on which to load the pipeline. ``"meta"`` builds it without weights for loading an AITune checkpoint.
        multi_gpu: Whether to apply Diffusers context parallelism.
        context_parallel: Context-parallel attention mode.

    Returns:
        The loaded FLUX pipeline.
    """
    pipeline = (
        _get_meta_pipeline(model_name)
        if device == "meta"
        else FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    )
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
