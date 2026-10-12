# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# allow_failure = false
# ///

"""Select TorchEager after a TorchInductorAot build moved the shared module to the CPU.

Both flows activate TorchEager after `TorchInductorAotBackend._build` calls `offload(module, "cpu")`:
the eager candidate wins over a slower candidate, or the eager baseline wins when performance validation is on.
"""

import time
from logging import DEBUG, basicConfig

import pytest
import torch
from torch import nn

from aitune.torch.backend import Backend, TorchEagerBackend, TorchInductorAotBackend
from aitune.torch.backend.backend import BackendState
from aitune.torch.module.wrapper_module import Module
from aitune.torch.module_registry import MODULE_REGISTRY
from aitune.torch.tune_strategy.max_throughput_strategy import MaxThroughputStrategy
from aitune.torch.tune_strategy.performance_validation import PerformanceValidationMode
from aitune.torch.tuning import tune

_DEVICE = torch.device("cuda")


class SlowInductorAotBackend(TorchInductorAotBackend):
    """Real Inductor AOT build, which offloads the module, with slow inference so that TorchEager wins."""

    _build_mode = TorchInductorAotBackend._build_mode
    _execution_modes = TorchInductorAotBackend._execution_modes

    def _infer(self, *args, **kwargs):
        time.sleep(0.01)
        return super()._infer(*args, **kwargs)


def _tune_and_check(backends: list[Backend], mode: PerformanceValidationMode, name: str):
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to(_DEVICE).eval()
    data = torch.randn((2, 64), device=_DEVICE)
    with torch.no_grad():
        expected = model(data)

    strategy = MaxThroughputStrategy(backends=backends)
    strategy.enable_performance_validation(mode)
    strategy.enable_find_max_batch_size(False)
    module = Module(model, name, strategy=strategy)
    try:
        tune(
            module,
            list(data.unbind(0)),
            batch_sizes=[2],
            device=_DEVICE,
            ignore_failing_modules=False,
        )

        (selected,) = module.module.backends.values()
        assert type(selected) is TorchEagerBackend
        assert selected.state == BackendState.ACTIVE

        assert next(model.parameters()).device.type == _DEVICE.type
        with torch.no_grad():
            torch.testing.assert_close(module(data), expected)
    finally:
        MODULE_REGISTRY.clear()


@pytest.mark.parametrize(
    "backends,mode",
    [
        pytest.param(
            lambda: [TorchEagerBackend(), SlowInductorAotBackend()],
            PerformanceValidationMode.DISABLED,
            id="eager-candidate-wins",
        ),
        pytest.param(
            lambda: [SlowInductorAotBackend()],
            PerformanceValidationMode.ENABLED,
            id="eager-baseline-wins",
        ),
    ],
)
def test_eager_selected_after_inductor_aot_offload(backends, mode, request):
    _tune_and_check(backends(), mode, f"functional-{request.node.callspec.id}")


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)
    _tune_and_check(
        [TorchEagerBackend(), SlowInductorAotBackend()],
        PerformanceValidationMode.DISABLED,
        "candidate",
    )
    _tune_and_check([SlowInductorAotBackend()], PerformanceValidationMode.ENABLED, "baseline")
