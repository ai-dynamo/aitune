# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# ///


import logging
import time
from pathlib import Path

import numpy as np
import timm
import torch

from aitune.torch.backend.tensorrt import TensorRTBackend, TensorRTBackendConfig
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = logging.getLogger(Path(__file__).stem)


def test_resnet50_with_cuda_graphs_invalidation():
    """Test ResNet50 with CUDA graphs handles different batch sizes (shape changes) by creating a new CUDA graph."""
    # given
    device = torch.device("cuda")

    data_bs2 = torch.randn((2, 3, 224, 224), device=device)
    data_bs4 = torch.randn((4, 3, 224, 224), device=device)

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()

    sample = torch.randn((3, 224, 224), device=device)
    with torch.no_grad():
        vanilla_benchmark = simple_benchmark(model, sample, batch_sizes=[1, 2, 4, 8, 16, 32, 64])

    # Create TensorRT backend with CUDA graphs enabled
    config = TensorRTBackendConfig(use_cuda_graphs=False)
    backend = TensorRTBackend(config=config)

    # when - create module and tune with CUDA graphs enabled
    strategy = OneBackendStrategy(backend)
    strategy.enable_validate_against_baseline(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "functional-resnet50-cuda-graphs-shapes",
        strategy=strategy,
    )

    tune(module, [data_bs2[0]], batch_sizes=[1, 2, 4, 8, 16, 32, 64], dry_run=False, device=device)

    # Test with different batch sizes to trigger CUDA graph invalidation
    trtre_backend = next(iter(module._self_wrapper.backends.values()))

    output1 = module(data_bs2)
    cuda_graph_bs2 = trtre_backend._cuda_graph

    output2 = module(data_bs2)
    assert cuda_graph_bs2 is trtre_backend._cuda_graph

    # New graph
    output3 = module(data_bs4)
    cuda_graph_bs4 = trtre_backend._cuda_graph
    if trtre_backend._config.use_cuda_graphs:
        assert cuda_graph_bs2 is not cuda_graph_bs4

    output4 = module(data_bs4)
    assert cuda_graph_bs4 is trtre_backend._cuda_graph  # No change

    torch.testing.assert_close(output1, output2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output3, output4, atol=1e-5, rtol=1e-5)

    tuned_benchmark = simple_benchmark(module, sample, batch_sizes=[1, 2, 4, 8, 16, 32, 64])

    print(vanilla_benchmark.to_markdown(), "vanilla")
    print(tuned_benchmark.to_markdown(), "tunned cuda_graph=", trtre_backend._config.use_cuda_graphs)


def simple_benchmark(module, sample, num_runs=100, num_warmup=10, batch_sizes=None):
    """Simple benchmark."""
    import pandas as pd

    from aitune.torch.dataloader import DataLoaderFactory

    batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64]
    results = {}
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size: {batch_size}")

        data = next(iter(DataLoaderFactory([sample]).create_dataloader(batch_size)))[0]

        # Warmup
        for _ in range(num_warmup):
            module(data)

        # Time it
        t = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            module(data)
            torch.cuda.synchronize()
            t.append(time.perf_counter() - start_time)

        results[batch_size] = {
            "latency": np.mean(t),
            "throughput": 1 / np.mean(t) * batch_size,
        }

    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    test_resnet50_with_cuda_graphs_invalidation()
    logger.info("Done")
