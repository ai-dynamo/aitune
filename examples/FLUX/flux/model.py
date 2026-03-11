# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model utilities for Flux pipeline."""

import torch
from diffusers import FluxPipeline


def get_pipeline(model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "cuda"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        FluxPipeline: The loaded Flux pipeline
    """
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device, dtype=torch.float16)
    torch.cuda.empty_cache()
    return pipe
