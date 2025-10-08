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
# scope = "always"
# ///


import logging
from pathlib import Path

import torch

from aitune.torch.backend.tensorrt import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = logging.getLogger(Path(__file__).stem)


class AddOneModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_custom_module_with_cuda_graphs():
    """Test custom module with TensorRT backend using CUDA graphs for optimized inference."""
    # given
    device = torch.device("cuda")

    model = AddOneModule()
    model.to(device)
    model.eval()
    data = torch.tensor([[1, 1]], device=device)

    # Create TensorRT backend with CUDA graphs enabled
    config = TensorRTBackendConfig(use_cuda_graphs=True)
    backend = TensorRTBackend(config=config)

    # when - create module and tune with CUDA graphs enabled
    module = Module(
        model,
        "functional-custom-module-cuda-graphs",
        strategy=OneBackendStrategy(backend).enable_find_max_batch_size(False),
    )

    # Verify recording works
    module(data)

    # Tune with CUDA graphs enabled
    tune(module, [data[0]], batch_sizes=[1, 2, 4], device=device)

    # then - verify inference works and produces correct results

    # graph capture
    output1 = module(data)
    # print(output1)

    # graph replay
    output2 = module(data)
    # print(output2)

    # graph replay with different memory address
    for i in range(3, 10):
        output3 = module(torch.full_like(data, i))
        # print(output3)

    # graph replay with different shape
    output4 = module(torch.zeros(4, 2, dtype=torch.int64) + 2)
    # print(output4)

    torch.testing.assert_close(output1, torch.tensor([[2, 2]], device=device))
    torch.testing.assert_close(output2, torch.tensor([[2, 2]], device=device))
    torch.testing.assert_close(output3, torch.tensor([[10, 10]], device=device))
    torch.testing.assert_close(output4, torch.tensor([[3, 3], [3, 3], [3, 3], [3, 3]], device=device))

    # Verify CUDA graphs are created in the backend
    trtre_backend = next(iter(module._self_wrapper.backends.values()))
    assert trtre_backend._cuda_graph is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    test_custom_module_with_cuda_graphs()
    logger.info("Done")
