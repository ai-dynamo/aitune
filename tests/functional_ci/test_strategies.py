# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy selection, fallback, performance validation, max-batch discovery, and dry-run."""

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
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.tune_strategy.performance_validation import PerformanceValidationMode
from tests.functional_ci.backends import BACKENDS
from tests.functional_ci.lifecycle import TENSORRT_FP32, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp
from tests.toy_backends import BuildFailsBackend

PROFILING_STRATEGIES: dict[str, Callable[[list[Backend]], ait.MaxThroughputStrategy]] = {
    "max-throughput": ait.MaxThroughputStrategy,
    "min-latency": ait.MinLatencyStrategy,
    "latency-budget": lambda backends: ait.LatencyBudgetStrategy(latency_budget_ms=10_000, backends=backends),
}


def _tune(strategy, device: torch.device, dry_run: bool = False, batch_sizes: tuple[int, ...] = (2,)) -> ait.Module:
    """Tune the toy MLP on `batch_sizes` with `strategy`; one batch size means static shapes."""
    tuned = ait.Module(make_mlp(device), "mlp", strategy=strategy)
    ait.tune(tuned, list(make_input(device).unbind(0)), batch_sizes=list(batch_sizes), device=device, dry_run=dry_run)
    return tuned


def _assert_matches_eager(tuned: ait.Module, device: torch.device, batch_size: int = 2):
    """Compare the tuned module with a fresh eager MLP."""
    value = make_input(device, batch_size=batch_size, seed=5)
    with torch.no_grad():
        TENSORRT_FP32.assert_close(tuned(value), make_mlp(device)(value))


def test_first_wins_falls_back_after_build_failure(device: torch.device):
    """A backend that fails to build is skipped and the next candidate is selected."""
    strategy = ait.FirstWinsStrategy([BuildFailsBackend(RuntimeError), TorchInductorAotBackend()])
    strategy.enable_performance_validation(False).enable_find_max_batch_size(False)
    tuned = _tune(strategy, device)
    assert type(selected_backend(tuned)) is TorchInductorAotBackend
    _assert_matches_eager(tuned, device)


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
    _assert_matches_eager(tuned, device)


def test_find_max_batch_size(device: torch.device):
    """Max-batch discovery extends the dynamic TensorRT range beyond the recorded batch sizes."""
    strategy = ait.OneBackendStrategy(BACKENDS["tensorrt"].factory()).enable_performance_validation(False)
    tuned = _tune(strategy, device, batch_sizes=(1, 2))
    (graph_spec,) = tuned.graph_specs
    max_batch_size = graph_spec.get_max_batch_size()
    assert max_batch_size > 2
    _assert_matches_eager(tuned, device, batch_size=max_batch_size)


def test_dry_run_keeps_eager(device: torch.device):
    """A dry run checks the strategy without building or activating a backend."""
    tuned = _tune(pinned_strategy(TorchInductorAotBackend()), device, dry_run=True)
    assert tuned.state != ModuleState.TUNED
    _assert_matches_eager(tuned, device)
