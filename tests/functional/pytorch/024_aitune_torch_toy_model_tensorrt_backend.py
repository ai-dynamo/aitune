# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

import torch

from aitune.torch.backend.tensorrt.tensorrt_backend import (
    ProfileMode,
    TensorRTBackend,
    TensorRTBackendConfig,
)
from aitune.torch.config import config as global_config
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from aitune.torch.tuning import tune

logger = getLogger(Path(__file__).stem)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.1


def testing_multi_profile_with_samples():
    device = "cuda"
    dtype = torch.float32

    # create a model that process images of size 224x224 and 448x448
    model = ToyModel()
    model.to(device, dtype=dtype)
    model.eval()

    # create two samples images of size 224x224 and 448x448
    data1 = torch.randn((3, 224, 224), device=device).to(dtype)
    data2 = torch.randn((3, 448, 448), device=device).to(dtype)

    # set max_num_samples_stored to 4 to generate all profiles 2 samples x 2 batch sizes
    global_config.max_num_samples_stored = 4

    # configure the backend to use multi-profile mode, and auto generate profiles from samples used for tuning
    backend = TensorRTBackend(TensorRTBackendConfig(profiles=ProfileMode.SAMPLES_USED))
    strategy = OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    module = Module(
        model,
        "toy-model",
        strategy=strategy,
    )

    # tune the model with the two samples, need to use DynamicShapeDataset as samples are different shapes
    tune(module, DynamicShapeDataset([data1, data2]), batch_sizes=[2, 8], device=device)

    # testing model with different samples and batch sizes
    module(data1.repeat(8, 1, 1, 1))
    module(data2.repeat(8, 1, 1, 1))

    try:
        module(data1.repeat(4, 1, 1, 1))
    except RuntimeError:
        pass  # expected exception when passing sample with shape that is not in any profile
    else:
        raise AssertionError("Expected exception when passing sample with shape that is not in any profile")

    # still runs after error
    module(data2.repeat(2, 1, 1, 1))

    # check profiles are saved and loaded correctly
    active_backend = next(iter(module.module.backends.values()))
    profiles = active_backend._trt_optimization_profiles

    shapes = {tuple(profile["args_0"].min) for profile in profiles}

    assert (8, 3, 448, 448) in shapes
    assert (2, 3, 448, 448) in shapes

    assert (8, 3, 224, 224) in shapes
    assert (2, 3, 224, 224) in shapes

    MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    testing_multi_profile_with_samples()
