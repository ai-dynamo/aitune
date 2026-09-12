# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# scope = "always"
# ///


import logging
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import timm
import torch

from aitune.torch.backend.tensorrt import ProfileMode, TensorRTBackend, TensorRTBackendConfig
from aitune.torch.config import config as global_config
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = logging.getLogger(Path(__file__).stem)
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


@patch.object(global_config, "max_num_samples_stored", len(BATCH_SIZES))
def test_resnet50_with_cuda_graph_profiles():
    """Measure ResNet50 performance and verify graph reuse across static batch profiles."""
    # given
    device = torch.device("cuda")

    data_bs2 = torch.randn((2, 3, 224, 224), device=device)
    data_bs4 = torch.randn((4, 3, 224, 224), device=device)

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()

    sample = torch.randn((3, 224, 224), device=device)
    with torch.no_grad():
        vanilla_benchmark = simple_benchmark(model, sample, batch_sizes=BATCH_SIZES)

    # Create TensorRT backend with CUDA graphs enabled
    config = TensorRTBackendConfig(use_cuda_graphs=True, profiles=ProfileMode.SAMPLES_USED)
    backend = TensorRTBackend(config=config)

    # when - create module and tune with CUDA graphs enabled
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "functional-resnet50-cuda-graphs-shapes",
        strategy=strategy,
    )

    tune(module, [data_bs2[0]], batch_sizes=BATCH_SIZES, max_num_batches_per_batch_size=1, device=device)

    trtre_backend = next(iter(module._self_wrapper.backends.values()))
    graphs = trtre_backend._cuda_graphs
    assert len(trtre_backend._trt_optimization_profiles) == len(BATCH_SIZES)
    assert graphs.static_profile_indices == set(range(len(BATCH_SIZES)))

    module(data_bs2)
    assert graphs.active is not None and graphs.active.graph is not None
    cuda_graph_bs2 = graphs.active.graph
    module(data_bs4)
    assert graphs.active is not None and graphs.active.graph is not None
    cuda_graph_bs4 = graphs.active.graph
    assert cuda_graph_bs2 is not cuda_graph_bs4

    # Every request changes shape and returns to an already captured profile.
    for _ in range(4):
        module(data_bs2)
        assert graphs.active.graph is cuda_graph_bs2
        module(data_bs4)
        assert graphs.active.graph is cuda_graph_bs4
    assert not graphs.capture_failed

    tuned_benchmark = simple_benchmark(module, sample, batch_sizes=BATCH_SIZES)

    assert not graphs.capture_failed
    assert len(graphs.profiles) == len(BATCH_SIZES)
    print(vanilla_benchmark.to_markdown(), "eager")
    print(tuned_benchmark.to_markdown(), "tuned with cached cuda graphs")


def simple_benchmark(module, sample, num_runs=100, num_warmup=10, batch_sizes=None):
    """Simple benchmark."""
    import pandas as pd

    from aitune.torch.dataloader import DataLoaderFactory

    batch_sizes = batch_sizes or BATCH_SIZES
    results = {}
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size: {batch_size}")

        data = next(iter(DataLoaderFactory([sample]).create_dataloader(batch_size)))[0]

        # Warmup
        for _ in range(num_warmup):
            module(data)

        # Synchronize warmup before starting the wall-clock measurement.
        torch.cuda.synchronize()
        t = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            module(data)
            torch.cuda.synchronize()
            t.append(time.perf_counter() - start_time)

        assert np.isfinite(t).all() and (np.asarray(t) > 0).all()
        results[batch_size] = {
            "latency": np.mean(t),
            "throughput": 1 / np.mean(t) * batch_size,
        }

    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    test_resnet50_with_cuda_graph_profiles()
    logger.info("Done")
