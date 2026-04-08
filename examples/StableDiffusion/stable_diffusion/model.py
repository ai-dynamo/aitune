# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model utilities for Stable Diffusion pipeline."""

import torch
from diffusers import DiffusionPipeline

MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"


def get_pipeline(model_name: str = MODEL_NAME, device: str = "cuda"):
    """Get a pretrained Stable Diffusion model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        DiffusionPipeline: The loaded Stable Diffusion pipeline
    """
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device, dtype=torch.float16)
    return pipe
