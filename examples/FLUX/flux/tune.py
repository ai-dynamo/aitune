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
"""Tune Flux model."""

import os
from logging import basicConfig, getLogger

import torch

import aitune.torch as ait
from aitune.torch.backend import (
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchEagerBackend,
    TorchInductorBackend,
)
from aitune.torch.backend.tensorrt.torch_quantization import TorchQuantizationConfig
from flux.cmd_args import parse_args
from flux.model import get_pipeline

logger = getLogger(__name__)


def tune_model(
    model_name,
    prompt,
    sizes,
    steps,
    guidance_scale,
    max_sequence_length,
    tuned_model_path,
    batch_sizes=None,
    strategy=None,
):
    """Tune the Flux model.

    Args:
        model_name: HuggingFace model name or path
        prompt: Text prompt for image generation
        sizes: List of (height, width) tuples
        steps: Number of inference steps
        guidance_scale: Guidance scale
        max_sequence_length: Maximum sequence length
        tuned_model_path: Path to save the tuned model
        batch_sizes: List of batch sizes to tune
        strategy: AITune strategy to use
    """
    pipe = get_pipeline(model_name=model_name)

    input_data = [{"prompt": prompt}]

    # Inspect pipeline to get modules
    modules_info = ait.inspect(pipe, input_data, number_of_iterations=1, warmup_iterations=1)

    # Define strategy if not provided
    if strategy is None:
        strategy = ait.FirstWinsStrategy(
            backends=[
                TensorRTBackend(
                    TensorRTBackendConfig(
                        quantization_config=TorchQuantizationConfig(
                            quantization_config="FP8_DEFAULT_CFG",
                            device="cuda",
                        ),
                        use_dynamo=False,
                    )
                ),
                # TensorRTBackend(TensorRTBackendConfig(use_dynamo=True)),
                TensorRTBackend(TensorRTBackendConfig(use_dynamo=False)),
                TorchInductorBackend(),
                TorchEagerBackend(),
            ]
        )
    strategy.enable_find_max_batch_size(enable=False)

    # Wrap all modules with AITune Module
    modules = modules_info.get_modules()
    pipe = ait.wrap(pipe, modules, strategy=strategy)

    def call_wrapper(*args, **kwargs):
        for height, width in sizes:
            print(f"Generating image with height={height} and width={width}")  # noqa: T201
            pipe(
                *args,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(0),
                **kwargs,
            )

    # First do a dry run for testing
    logger.info("Tuning module: %s", model_name)
    ait.tune(call_wrapper, input_data, batch_sizes=[1] if batch_sizes is None else batch_sizes)
    logger.info("Tuning completed.")

    ait.save(pipe, tuned_model_path)
    logger.info("Model saved to %s", tuned_model_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    tune_model(
        model_name=args.model_name,
        prompt=args.prompt,
        sizes=args.sizes,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
