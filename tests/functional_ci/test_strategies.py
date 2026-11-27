# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy selection, fallback, performance validation, max-batch discovery, and dry-run."""

import time
from collections.abc import Callable

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import (
    Backend,
    TensorRTBackend,
    TorchEagerBackend,
    TorchInductorAotBackend,
)
from aitune.torch.backend.backend import BackendState
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.tune_strategy.performance_validation import PerformanceValidationMode
from aitune.torch.tune_strategy.profiling_tune_strategy import ProfilingTuneStrategy
from tests.functional_ci.backends import BACKENDS
from tests.functional_ci.lifecycle import TENSORRT_FP32, assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp
from tests.toy_backends import BuildFailsBackend


class SlowInductorAotBackend(TorchInductorAotBackend):
    """Real Inductor AOT build, which offloads the module to the CPU, with slow inference so that eager wins."""

    _build_mode = TorchInductorAotBackend._build_mode
    _execution_modes = TorchInductorAotBackend._execution_modes

    def _infer(self, *args, **kwargs):
        time.sleep(0.01)
        return super()._infer(*args, **kwargs)


PROFILING_STRATEGIES: dict[str, Callable[[list[Backend]], ProfilingTuneStrategy]] = {
    "max-throughput": ait.MaxThroughputStrategy,
    "min-latency": ait.MinLatencyStrategy,
    "latency-budget": lambda backends: ait.LatencyBudgetStrategy(latency_budget_ms=10_000, backends=backends),
}


def _tune(strategy, device: torch.device, dry_run: bool = False, batch_sizes: tuple[int, ...] = (2,)) -> ait.Module:
    """Tune the toy MLP on `batch_sizes` with `strategy`; one batch size means static shapes."""
    tuned = ait.Module(make_mlp(device), "mlp", strategy=strategy)
    ait.tune(tuned, list(make_input(device).unbind(0)), batch_sizes=list(batch_sizes), device=device, dry_run=dry_run)
    return tuned


@pytest.mark.matrix("strategies.first-wins/aot-tune", "strategies.first-wins/selection-checked")
def test_first_wins_falls_back_after_build_failure(device: torch.device):
    """A backend that fails to build is skipped and the next candidate is selected."""
    strategy = ait.FirstWinsStrategy([BuildFailsBackend(RuntimeError), TorchInductorAotBackend()])
    strategy.enable_performance_validation(False).enable_find_max_batch_size(False)
    tuned = _tune(strategy, device)
    selected_backend(tuned, TorchInductorAotBackend)
    assert_matches_eager(tuned, make_mlp(device), device, TENSORRT_FP32)


@pytest.mark.matrix("strategies.{strategy_id}/aot-tune", "strategies.validation-diagnostic/aot-tune")
@pytest.mark.parametrize("strategy_id", PROFILING_STRATEGIES)
def test_profiling_strategy_diagnostic(strategy_id: str, device: torch.device):
    """Profiling strategies pick one of the candidates and record the eager comparison in diagnostic mode."""
    strategy = PROFILING_STRATEGIES[strategy_id]([
        TorchEagerBackend(),
        TorchInductorAotBackend(),
        BACKENDS["tensorrt"].factory(),
    ])
    strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC).enable_find_max_batch_size(False)
    tuned = _tune(strategy, device)
    assert type(selected_backend(tuned)) in {TorchEagerBackend, TorchInductorAotBackend, TensorRTBackend}
    assert strategy.perf_validation_results
    assert_matches_eager(tuned, make_mlp(device), device, TENSORRT_FP32)


@pytest.mark.matrix("strategies.find-max-batch-size/aot-tune", "strategies.find-max-batch-size/selection-checked")
def test_find_max_batch_size(device: torch.device):
    """Max-batch discovery extends the dynamic TensorRT range beyond the recorded batch sizes."""
    strategy = ait.OneBackendStrategy(BACKENDS["tensorrt"].factory()).enable_performance_validation(False)
    tuned = _tune(strategy, device, batch_sizes=(1, 2))
    (graph_spec,) = tuned.graph_specs
    max_batch_size = graph_spec.get_max_batch_size()
    assert max_batch_size > 2
    assert_matches_eager(tuned, make_mlp(device), device, TENSORRT_FP32, batch_sizes=(max_batch_size,))


@pytest.mark.matrix("strategies.dry-run/aot-tune")
def test_dry_run_keeps_eager(device: torch.device):
    """A dry run checks the strategy without building or activating a backend."""
    tuned = _tune(pinned_strategy(TorchInductorAotBackend()), device, dry_run=True)
    assert tuned.state != ModuleState.TUNED
    assert_matches_eager(tuned, make_mlp(device), device, TENSORRT_FP32)


@pytest.mark.matrix("strategies.eager-after-offload/aot-tune", "strategies.eager-after-offload/selection-checked")
@pytest.mark.parametrize(
    ("backends", "mode"),
    [
        pytest.param(
            lambda: [TorchEagerBackend(), SlowInductorAotBackend()],
            PerformanceValidationMode.DISABLED,
            id="eager-candidate-wins",
        ),
        pytest.param(lambda: [SlowInductorAotBackend()], PerformanceValidationMode.ENABLED, id="eager-baseline-wins"),
    ],
)
def test_eager_selected_after_inductor_aot_offload(backends, mode, device: torch.device):
    """Eager wins after the Inductor AOT build offloaded the module, as a candidate or as the validation baseline."""
    strategy = ait.MaxThroughputStrategy(backends())
    strategy.enable_performance_validation(mode).enable_find_max_batch_size(False)
    tuned = _tune(strategy, device)
    assert selected_backend(tuned, TorchEagerBackend).state == BackendState.ACTIVE
    assert next(tuned.parameters()).device == device
    assert_matches_eager(tuned, make_mlp(device), device, BACKENDS["torch-eager"].tolerance)
