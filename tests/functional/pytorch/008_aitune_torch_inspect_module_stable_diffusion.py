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

# /// script
# dependencies = ["transformers", "diffusers"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///


import diffusers

from aitune.torch import inspect


def test_inspect_stable_diffusion():
    # given
    model_id = "stabilityai/stable-diffusion-2-1"
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

    assert len(modules_info.get_modules()) == 4

    expected_module_names = {"unet", "decoder", "text_encoder", "post_quant_conv"}
    modules = modules_info.get_modules()

    assert len(modules) == len(expected_module_names)
    names = {module.name for module in modules}
    assert names == expected_module_names, f"Expected {expected_module_names} but got {names}"

    top_modules = modules_info.get_modules(min_execution_percentage=0.6)
    assert len(top_modules) == 1
    assert top_modules[0].name == "unet"
    assert top_modules[0].execution_count == num_inference_steps * number_of_iterations
    assert top_modules[0].total_execution_time > 0
    assert top_modules[0].average_execution_time > 0
    assert top_modules[0].total_execution_time < modules_info._total_execution_time

    top_modules = modules_info.get_modules(limit=1)
    assert len(top_modules) == 1
    assert top_modules[0].name == "unet"


if __name__ == "__main__":
    test_inspect_stable_diffusion()
