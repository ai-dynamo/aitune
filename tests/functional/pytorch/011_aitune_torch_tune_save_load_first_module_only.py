# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# use_gated_hf_token = true
# additional_tags = ["mem/80g"]
# ///

import gc
import pathlib
import tempfile
from logging import INFO, basicConfig, getLogger

import torch
from diffusers import StableDiffusionPipeline

from aitune.torch.backend import TensorRTBackend, TorchInductorJitBackend, TorchTensorRTAotBackend
from aitune.torch.module import Module
from aitune.torch.tune_strategy import FirstWinsStrategy
from aitune.torch.tuning import load, save, tune

basicConfig(level=INFO, force=True)
logger = getLogger(__name__)

PROMPT = "a photo of a cat sitting on a windowsill, natural lighting, detailed fur, photorealistic"


def _get_pipeline(model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", device: str = "cuda"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        FluxPipeline: The loaded Flux pipeline
    """
    model = StableDiffusionPipeline.from_pretrained(model_name)
    model.to(device)
    return model


def _tune_and_save(save_path: pathlib.Path, device: str = "cuda"):
    pipeline = _get_pipeline(device=device)

    strategy = FirstWinsStrategy(backends=[TensorRTBackend(), TorchTensorRTAotBackend(), TorchInductorJitBackend()])
    strategy.enable_validate_against_baseline(False)
    pipeline.text_encoder = Module(
        pipeline.text_encoder,
        "text_encoder",
        strategy=strategy,
    )

    tune(
        pipeline,
        [PROMPT],
        batch_sizes=[1],
        max_num_batches_per_batch_size=1,
        device=device,
        disable_external_logging=False,
    )

    save(pipeline, save_path)


def _load_and_infer(load_path: pathlib.Path, device: str = "cuda"):
    pipeline = _get_pipeline(device=device)

    load(pipeline, load_path, disable_external_logging=False)

    res = pipeline(PROMPT, generator=torch.Generator(device="cuda").manual_seed(42))
    opt_image = res[0][0]

    return opt_image


def _clean_up():
    gc.collect()
    torch.cuda.empty_cache()


def test_pipeline_serialization_with_first_module_only():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = pathlib.Path(temp_dir) / "sd-dev-tuned.pt"
        _tune_and_save(save_path)
        _clean_up()
        opt_image = _load_and_infer(save_path)

    assert opt_image is not None


if __name__ == "__main__":
    test_pipeline_serialization_with_first_module_only()
