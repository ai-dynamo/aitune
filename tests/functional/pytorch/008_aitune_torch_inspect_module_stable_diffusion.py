# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["transformers", "diffusers"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# use_gated_hf_token = true
# additional_tags = ["mem/80g"]
# ///


import diffusers

from aitune.torch import inspect


def test_inspect_stable_diffusion():
    # given
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cuda")

    prompt = "A futuristic cityscape with neon lights and flying cars"
    input_data = [{"prompt": prompt}]

    num_inference_steps = 10

    def inference_function(prompt):
        return pipe(prompt, num_inference_steps=num_inference_steps)

    # when
    number_of_iterations = 1
    warmup_iterations = 1
    modules_info = inspect(pipe, input_data, inference_function, number_of_iterations, warmup_iterations)

    # then - verify inspection
    modules_info.describe()

    assert len(modules_info.get_modules()) == 5

    expected_module_names = {"decoder", "unet", "text_encoder", "post_quant_conv", "safety_checker"}
    modules = modules_info.get_modules()

    assert len(modules) == len(expected_module_names)
    names = {module.name for module in modules}
    assert names == expected_module_names, f"Expected {expected_module_names} but got {names}"

    top_modules = modules_info.get_modules(min_execution_ratio=0.6)
    assert len(top_modules) == 1
    assert top_modules[0].name == "unet"
    assert top_modules[0].execution_count == num_inference_steps * number_of_iterations + 1
    assert top_modules[0].total_execution_time > 0
    assert top_modules[0].average_execution_time > 0
    assert top_modules[0].total_execution_time < modules_info._total_execution_time

    top_modules = modules_info.get_modules(limit=1)
    assert len(top_modules) == 1
    assert top_modules[0].name == "unet"


if __name__ == "__main__":
    test_inspect_stable_diffusion()
