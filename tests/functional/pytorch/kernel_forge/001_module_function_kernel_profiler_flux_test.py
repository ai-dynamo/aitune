# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers", "diffusers"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# use_gated_hf_token = true
# ///

"""Functional test for ModuleFunctionKernelProfiler on a tiny Flux pipeline."""

import logging
from logging import basicConfig

import diffusers
from torch import nn

from aitune.torch import ModuleFunctionKernelProfiler


def profile_module(profiler: ModuleFunctionKernelProfiler, pipe, module: nn.Module, name: str):
    def inference_fn():
        pipe("A futuristic cityscape with neon lights and flying cars", num_inference_steps=1)

    profiling_df, function_data = profiler.profile(inference_fn, module=module)
    df = profiler.describe_results(profiling_df, function_data)

    assert len(df) > 0
    assert len(function_data) > 0

    for row in df.itertuples(index=False):
        expected_num_distinct_samples = len(function_data[row.function_name])
        assert row.num_distinct_samples == expected_num_distinct_samples, (
            f"{name}.{row.function_name}: expected {expected_num_distinct_samples} distinct samples, "
            f"got {row.num_distinct_samples}"
        )

    logging.info("Module: %s, summary: %s", name, df.to_string())


def test_profile_and_describe_in_tiny_flux():
    model_id = "hf-internal-testing/tiny-flux-pipe"
    pipe = diffusers.FluxPipeline.from_pretrained(model_id)
    pipe.to("cuda")
    profiler = ModuleFunctionKernelProfiler()

    profile_module(profiler, pipe, pipe.transformer, "transformer")
    profile_module(profiler, pipe, pipe.text_encoder, "text_encoder")
    profile_module(profiler, pipe, pipe.vae, "vae")


if __name__ == "__main__":
    basicConfig(level=logging.INFO, format="%(message)s", force=True)
    test_profile_and_describe_in_tiny_flux()
