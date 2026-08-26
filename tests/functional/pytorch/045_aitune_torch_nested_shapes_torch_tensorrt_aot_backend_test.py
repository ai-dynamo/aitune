# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers>5", "diffusers>=0.40"]
# docker_image = "nvcr.io/nvidia/pytorch:26.06-py3"
# scope = "always"
# use_gated_hf_token = true
# ///

import gc
from logging import DEBUG, basicConfig

import torch
from diffusers import FluxTransformer2DModel

from aitune.torch import Module, OneBackendStrategy, tune
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig, TorchTensorRTConfig
from aitune.torch.dataloader import DataLoaderFactory
from aitune.torch.module_registry import MODULE_REGISTRY

MODEL_ID = "hf-internal-testing/tiny-flux-pipe"
IMAGE_SEQUENCE_LENGTH = 6
TEXT_SEQUENCE_LENGTH = 5


def _load_transformer(device: torch.device) -> FluxTransformer2DModel:
    return FluxTransformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer").eval().to(device)


def _make_dataset(model: FluxTransformer2DModel, device: torch.device) -> list[dict]:
    config = model.config
    inner_dim = config.num_attention_heads * config.attention_head_dim
    img_ids = torch.zeros(IMAGE_SEQUENCE_LENGTH, 3, device=device)
    img_ids[:, 1] = torch.arange(IMAGE_SEQUENCE_LENGTH, device=device)
    txt_ids = torch.zeros(TEXT_SEQUENCE_LENGTH, 3, device=device)

    def make_sample() -> dict:
        sample = {
            "hidden_states": torch.randn(IMAGE_SEQUENCE_LENGTH, config.in_channels, device=device),
            "encoder_hidden_states": torch.randn(TEXT_SEQUENCE_LENGTH, config.joint_attention_dim, device=device),
            "pooled_projections": torch.randn(config.pooled_projection_dim, device=device),
            "timestep": torch.tensor(0.5, device=device),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "controlnet_block_samples": [torch.randn(IMAGE_SEQUENCE_LENGTH, inner_dim, device=device)],
        }
        if config.guidance_embeds:
            sample["guidance"] = torch.tensor(1.0, device=device)
        return sample

    return [make_sample() for _ in range(4)]


def _collate_nested_inputs(samples: list[dict]) -> dict:
    inputs = {
        "hidden_states": torch.stack([sample["hidden_states"] for sample in samples]),
        "encoder_hidden_states": torch.stack([sample["encoder_hidden_states"] for sample in samples]),
        "pooled_projections": torch.stack([sample["pooled_projections"] for sample in samples]),
        "timestep": torch.stack([sample["timestep"] for sample in samples]),
        "img_ids": samples[0]["img_ids"],
        "txt_ids": samples[0]["txt_ids"],
        "controlnet_block_samples": [
            torch.stack([sample["controlnet_block_samples"][index] for sample in samples])
            for index in range(len(samples[0]["controlnet_block_samples"]))
        ],
    }
    if "guidance" in samples[0]:
        inputs["guidance"] = torch.stack([sample["guidance"] for sample in samples])
    return inputs


def _tune_and_validate(batch_sizes: list[int]) -> None:
    device = torch.device("cuda")
    model = _load_transformer(device)
    dataset = _make_dataset(model, device)
    validation_inputs = _collate_nested_inputs(dataset[: max(batch_sizes)])

    with torch.no_grad():
        expected = model(**validation_inputs)

    # Flux rotary embeddings create float64 constants, which TensorRT must truncate to a supported precision.
    backend = TorchTensorRTAotBackend(
        TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTConfig(truncate_double=True))
    )
    strategy = OneBackendStrategy(backend)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(False)
    shape_mode = "static" if len(batch_sizes) == 1 else "dynamic"
    tuned_model = Module(model, f"tiny-flux-transformer-{shape_mode}", strategy=strategy)

    try:
        tune(
            tuned_model,
            DataLoaderFactory(dataset, collate_fn=_collate_nested_inputs),
            batch_sizes=batch_sizes,
            max_num_batches_per_batch_size=1,
            device=device,
            ignore_failing_modules=False,
        )
        actual = tuned_model(**validation_inputs)
        torch.testing.assert_close(actual.sample, expected.sample, rtol=1e-2, atol=1e-2)
    finally:
        tuned_model.deactivate()
        MODULE_REGISTRY.clear()
        del tuned_model, model, dataset, validation_inputs, expected
        gc.collect()
        torch.cuda.empty_cache()


def test_tune_huggingface_flux_transformer_with_nested_static_and_dynamic_shapes():
    """Tune a real Hugging Face Flux transformer whose ControlNet residual input is a list of tensors."""
    _tune_and_validate([2])
    _tune_and_validate([2, 4])


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    test_tune_huggingface_flux_transformer_with_nested_static_and_dynamic_shapes()
