# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# ///


import logging
from pathlib import Path
from unittest.mock import patch

import torch

from aitune.torch.backend.tensorrt import ProfileMode, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.config import config as global_config
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = logging.getLogger(Path(__file__).stem)
BATCH_SIZES = [1, 2, 4]


class AddOneModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


# Retain one tuning sample per static profile, restoring the global setting afterward.
@patch.object(global_config, "max_num_samples_stored", len(BATCH_SIZES))
def test_custom_module_with_cuda_graphs():
    """Test custom module with TensorRT backend using CUDA graphs for optimized inference."""
    # given
    device = torch.device("cuda")

    model = AddOneModule()
    model.to(device)
    model.eval()
    data = torch.tensor([[1, 1]], device=device)

    # Create TensorRT backend with CUDA graphs enabled
    config = TensorRTBackendConfig(
        use_dynamo=False, use_cuda_graphs=True, opset_version=20, profiles=ProfileMode.SAMPLES_USED
    )
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

    # Tune with CUDA graphs enabled
    tune(module, [data[0]], batch_sizes=BATCH_SIZES, max_num_batches_per_batch_size=1, device=device)

    # Exercise alternating static profiles after tuning has warmed their graphs.
    trtre_backend = next(iter(module._self_wrapper.backends.values()))
    assert len(trtre_backend._trt_optimization_profiles) == len(BATCH_SIZES)
    assert trtre_backend._cuda_graphs.static_profile_indices == set(range(len(BATCH_SIZES)))
    module(data)
    first_graph = trtre_backend._cuda_graphs.active.graph
    assert first_graph is not None
    larger = torch.full((4, 2), 2, dtype=torch.int64, device=device)
    module(larger)
    second_graph = trtre_backend._cuda_graphs.active.graph
    assert second_graph is not None and second_graph is not first_graph
    for _ in range(4):
        module(data.clone())
        assert trtre_backend._cuda_graphs.active.graph is first_graph
        module(larger.clone())
        assert trtre_backend._cuda_graphs.active.graph is second_graph
    assert not trtre_backend._cuda_graphs.capture_failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    test_custom_module_with_cuda_graphs()
    logger.info("Done")
