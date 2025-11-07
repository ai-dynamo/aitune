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
# dependencies = ["transformers", "diffusers<0.35"]
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# use_gated_hf_token = true
# ///


import diffusers

from aitune.torch import inspect


def test_inspect_flux():
    # given
    # model_id = "black-forest-labs/FLUX.1-dev"
    model_id = "hf-internal-testing/tiny-flux-pipe"
    pipe = diffusers.FluxPipeline.from_pretrained(model_id)
    pipe.to("cuda")

    prompt = "A futuristic cityscape with neon lights and flying cars"
    input_data = [{"prompt": prompt}]

    # when
    number_of_iterations = 1
    warmup_iterations = 1
    modules_info = inspect(pipe, input_data, None, number_of_iterations, warmup_iterations)

    # then - verify inspection
    modules_info.describe()

    assert len(modules_info.get_modules()) == 4

    expected_module_names = ["transformer", "decoder", "text_encoder", "text_encoder_2"]
    modules = modules_info.get_modules()

    assert len(modules) == len(expected_module_names)
    for module in modules:
        assert module.name in expected_module_names

    top_modules = modules_info.get_modules(min_execution_percentage=0.50)
    assert len(top_modules) == 1
    assert top_modules[0].name == "transformer"
    assert top_modules[0].execution_count == 28 * number_of_iterations
    assert top_modules[0].total_execution_time > 0
    assert top_modules[0].average_execution_time > 0
    assert top_modules[0].total_execution_time < modules_info._total_execution_time

    top_modules = modules_info.get_modules(limit=1)
    assert len(top_modules) == 1
    assert top_modules[0].name == "transformer"


if __name__ == "__main__":
    test_inspect_flux()
