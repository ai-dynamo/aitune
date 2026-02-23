# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Example of JIT tuning Diffusion pipeline."""

import argparse
from logging import basicConfig, getLogger
from pathlib import Path
from time import perf_counter

import diffusers
import torch
from diffusers import AutoPipelineForText2Image
from torch.utils.data import DataLoader

from aitune.torch.jit.config import config as ait_jit_config

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "prompts.txt"

logger = getLogger(__name__)

models = [
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-dev",
]


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Diffusion pipeline JIT tuning benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=models[0],
        choices=models,
        help=f"Model to use for inference (default: {models[1]})",
    )
    parser.add_argument(
        "--file-with-prompts",
        type=str,
        default=str(DEFAULT_PROMPTS_FILE),
        help=f"File containing prompts, one per line (default: {DEFAULT_PROMPTS_FILE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory for generated images (default: outputs)"
    )
    parser.add_argument("--max-batch-size", type=int, default=2, help="Maximum batch size for inference (default: 2)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    return args


def initialize_logger(log_level):
    basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    diffusers.logging.disable_progress_bar()


def load_pipeline(model: str):
    """Load pipeline"""
    logger.info("Loading pipeline...")
    start_time = perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    logger.info("Loading pipeline took %.1f seconds", perf_counter() - start_time)
    return pipe


def get_prompts(file_with_prompts):
    """Get prompts from a file."""
    if not Path(file_with_prompts).exists():
        raise FileNotFoundError(f"File with prompts not found: {file_with_prompts}")
    prompts = [line.strip() for line in open(file_with_prompts)]
    if not prompts:
        raise ValueError("No prompts found in the file")
    return prompts


def warm_up_pipe(pipe: AutoPipelineForText2Image, prompts: list[str], max_batch_size: int):
    """Warmup phase for JIT tuning.

    We run model with bs=1 and max_batch_size so that JIT know min/max shapes.
    """
    logger.info("Starting warmup")
    start_time = perf_counter()
    pipe(prompts[:1], num_inference_steps=1)  # min shapes
    pipe(prompts[:max_batch_size], num_inference_steps=1)  # max shapes
    logger.info("Warmup took %.1f seconds", perf_counter() - start_time)


def run(
    model: str,
    file_with_prompts: str,
    output_dir: str,
    max_batch_size: int,
    log_level: str,
):
    ait_jit_config.min_samples = 2

    initialize_logger(log_level)
    pipe = load_pipeline(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = get_prompts(file_with_prompts)
    file_index = 0
    dataloader = DataLoader(prompts, batch_size=max_batch_size, shuffle=False, num_workers=0)
    durations = []

    warm_up_pipe(pipe, prompts, max_batch_size)

    logger.info("Generating %s images and saving them to folder: %s", len(prompts), output_dir)
    for batch in dataloader:
        start_time = perf_counter()
        with torch.no_grad():
            images = pipe(batch).images
        durations.append(perf_counter() - start_time)

        for prompt, image in zip(batch, images, strict=True):
            filename = f"{file_index:03d}_{prompt[:20].replace(' ', '_')}.png"
            file_index += 1
            image.save(output_dir / filename)

    logger.info("Average batch generation time: %.1f seconds", sum(durations) / len(durations))


def main():
    args = get_cmd_args()
    run(
        model=args.model,
        file_with_prompts=args.file_with_prompts,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
