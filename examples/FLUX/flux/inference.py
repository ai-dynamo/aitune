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
"""Inference script for Flux model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

import torch

from aitune.torch import load
from flux.cmd_args import parse_args
from flux.model import get_pipeline

logger = getLogger(__name__)


def do_inference(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    output_dir,
):
    """Do inference with the Flux model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        guidance_scale: Guidance scale
        max_sequence_length: Maximum sequence length
        tuned_model_path: Path to load the tuned model
        output_dir: Directory to save the generated image
    """
    pipe = get_pipeline(model_name=model_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating images on original pipeline")
    for height, width in sizes:
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
        )

        # Save the generated image
        image_path = output_path / f"flux_output_orig_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        images[0][0].save(image_path)
        logger.info("Generated image saved to: %s", image_path)

    logger.info("Loading tuned pipeline")
    pipe = load(pipe, tuned_model_path)

    logger.info("Generating images on tuned pipeline")
    for height, width in sizes:
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(0),
        )

        # Save the generated image
        image_path = output_path / f"flux_output_{prompt[:20].replace(' ', '_')}_{width}x{height}.jpg"
        images[0][0].save(image_path)
        logger.info("Generated image saved to: %s", image_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    do_inference(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=args.tuned_model_path,
        output_dir=os.environ.get("AITUNE_OUTPUT_DIR", "output"),
    )


if __name__ == "__main__":
    main()
