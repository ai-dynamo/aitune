# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model utilities for Stable Diffusion pipeline."""

import torch
from diffusers import StableDiffusionPipeline


def get_pipeline(model_name: str = "stabilityai/stable-diffusion-2-1", device: str = "cuda"):
    """Get a pretrained Stable Diffusion model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        StableDiffusionPipeline: The loaded Stable Diffusion pipeline
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device)
    return pipe
