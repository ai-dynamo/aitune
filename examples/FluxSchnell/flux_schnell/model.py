# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load and execute the pinned FLUX pipeline described by a recipe."""

from logging import getLogger
from typing import Literal, cast

import numpy as np
import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from numpy.typing import NDArray
from PIL import Image

from .config import GenerationInput, Recipe, Sample
from .distributed import is_rank_zero, synchronize

logger = getLogger(__name__)

Variant = Literal["original", "tuned"]
ImageSamples = dict[str, NDArray[np.float32]]


def get_pipeline(recipe: Recipe) -> FluxPipeline:
    pipeline = FluxPipeline.from_pretrained(
        recipe.model.base_model,
        revision=recipe.model.revision,
        torch_dtype=torch.bfloat16,
    ).to(torch.device("cuda", torch.cuda.current_device()))
    pipeline.set_progress_bar_config(disable=True)
    if recipe.gpu_count > 1:
        pipeline.transformer.set_attention_backend("_native_cudnn")
        mode = recipe.execution.context_parallel
        pipeline.transformer.enable_parallelism(
            config=ContextParallelConfig(**{f"{mode}_degree": dist.get_world_size()})
        )
    torch.cuda.empty_cache()
    return pipeline


def generate(pipeline: FluxPipeline, recipe: Recipe, sample: GenerationInput | Sample) -> FluxPipelineOutput:
    """Reset the sample seed for every invocation, including warmup and measurement."""
    with torch.no_grad():
        return pipeline(
            prompt=sample.prompt,
            **recipe.workload.model_dump(),
            generator=torch.Generator("cpu").manual_seed(sample.seed),
            output_type="pil" if is_rank_zero() else "pt",
            return_dict=True,
        )


def generate_samples(
    pipeline: FluxPipeline, recipe: Recipe, split: Literal["inference", "validation"], variant: Variant
) -> ImageSamples:
    images: ImageSamples = {}
    for sample in recipe.samples(split):
        logger.info("%s %s sample: %s (seed=%s)", variant, split, sample.id, sample.seed)
        output = generate(pipeline, recipe, sample)
        synchronize()
        if is_rank_zero():
            image = cast(Image.Image, output.images[0])  # Rank zero requests PIL output.
            path = recipe.output_dir / split / variant / f"{sample.id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            images[sample.id] = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return images
