# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["timm"]
# ///

from logging import DEBUG, basicConfig, getLogger

import pytest
import timm
import torch

from aitune.torch.backend.torch_eager import TorchEagerBackend
from aitune.torch.backend.torch_inductor_jit_backend import TorchInductorJitBackend
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.min_latency_strategy import MinLatencyStrategy
from aitune.torch.tuning import tune

logger = getLogger(__name__)

_DEVICE = torch.device("cuda")
_BATCH_SIZES = [1, 2, 4]


def _get_backends():
    return [TorchInductorJitBackend(), TorchEagerBackend()]


def _resnet18():
    model = timm.create_model("resnet18", pretrained=False)
    model.to(_DEVICE)
    model.eval()
    return model


@pytest.mark.parametrize(
    "strategy_cls,module_name",
    [
        (MaxThroughputStrategy, "functional-resnet18-max-throughput"),
        (MinLatencyStrategy, "functional-resnet18-min-latency"),
    ],
)
def test_profiling_strategy_resnet18(strategy_cls, module_name):
    model = _resnet18()
    data = torch.randn((3, 224, 224), device=_DEVICE)
    sample = data.unsqueeze(0)

    with torch.no_grad():
        expected = torch.nn.functional.softmax(model(sample)[0], dim=0)

    strategy = strategy_cls(backends=_get_backends())
    module = Module(model, module_name, strategy=strategy)

    try:
        tune(module, data, batch_sizes=_BATCH_SIZES, dry_run=False, disable_external_logging=False)

        actual = torch.nn.functional.softmax(module(sample)[0], dim=0)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        assert len(strategy.perf_validation_results) > 0
        assert all(r.metric > 0 for r in strategy.perf_validation_results)
        assert all(r.baseline_metric > 0 for r in strategy.perf_validation_results)
    finally:
        MODULE_REGISTRY.clear()


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    for cls, name in [
        (MaxThroughputStrategy, "functional-resnet18-max-throughput"),
        (MinLatencyStrategy, "functional-resnet18-min-latency"),
    ]:
        test_profiling_strategy_resnet18(cls, name)
