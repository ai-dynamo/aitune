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

# /// script
# dependencies = ["transformers", "diffusers", "sentencepiece"]
# scope = "always"
# allow_failure = true
# [environment]
# TQDM_DISABLE=1
# ///
"""Testing exporting issues with FLUX using tiny model.

Problem encountered:
Traceback (most recent call last):
  File "ai-tune/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_core.py", line 707, in _translate_fx_graph
    _handle_call_function_node_with_lowering(
  File "ai-tune/.venv/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_core.py", line 521, in _handle_call_function_node_with_lowering
    raise _errors.GraphConstructionError(
torch.onnx._internal.exporter._errors.GraphConstructionError: Error when calling function
        'TracedOnnxFunction(<function aten_repeat_interleave_self_int at 0x7f0170328d60>)'
        with args '[
                SymbolicTensor(name='cos_2', type=Tensor(DOUBLE), shape=Shape([SymbolicDim(s0 + 128), 8]), producer='node_Cos_256', index=0), 2, 1]'
       and kwargs
           '{'output_size': 16}'

## Exception summary

<class 'TypeError'>: aten_repeat_interleave_self_int() got an unexpected keyword argument 'output_size'
⬆️
<class 'torch.onnx._internal.exporter._errors.GraphConstructionError'>:
    Error when calling function 'TracedOnnxFunction(<function aten_repeat_interleave_self_int at 0x7f0170328d60>)'
        with args '[SymbolicTensor(name='cos_2', type=Tensor(DOUBLE), shape=Shape([SymbolicDim(s0 + 128), 8]), producer='node_Cos_256', index=0), 2, 1]'
        and kwargs '{'output_size': 16}'
⬆️
<class 'torch.onnx._internal.exporter._errors.ConversionError'>: Error when translating node %repeat_interleave :
    [num_users=1] = call_function[target=torch.ops.aten.repeat_interleave.self_int](args = (%cos_2, 2, 1), kwargs = {output_size: 16}). See the stack trace for more information.


    Related issue:
        https://github.com/onnx/onnx/issues/7101
"""

import logging

import diffusers
import torch

from aitune.torch import Module, OneBackendStrategy, tune
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig


def test_flux_like_tiny():
    """Test exporting FLUX like tiny model."""
    device = torch.device("cuda")

    pipe = diffusers.FluxPipeline.from_pretrained("hf-internal-testing/tiny-flux-pipe")
    # pipe = diffusers.FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipe.to(device)

    def call_wrapper(*args, **kwargs):
        for height, width in ((256, 256), (384, 384)):
            pipe(
                *args,
                height=height,
                width=width,
                num_inference_steps=3,
                guidance_scale=3.5,
                max_sequence_length=128,
                generator=torch.Generator("cpu").manual_seed(0),
                **kwargs,
            )

    prompt = "A futuristic cityscape with neon lights and flying cars"
    input_data = [{"prompt": prompt}]
    pipe(prompt=prompt, width=512, height=512, num_inference_steps=4)

    strategy = OneBackendStrategy(TensorRTBackend(config=TensorRTBackendConfig(use_dynamo=True)))
    strategy.enable_find_max_batch_size(enable=False)

    pipe.transformer = Module(pipe.transformer, "transformer", strategy=strategy)

    tune(call_wrapper, input_data, batch_sizes=[1, 2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    test_flux_like_tiny()
