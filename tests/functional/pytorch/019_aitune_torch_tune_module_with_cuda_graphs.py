# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    config = TensorRTBackendConfig(use_dynamo=False, use_cuda_graphs=True, opset_version=20)
    backend = TensorRTBackend(config=config)

    # when - create module and tune with CUDA graphs enabled
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "functional-custom-module-cuda-graphs",
        strategy=strategy,
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
