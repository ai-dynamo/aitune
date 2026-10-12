# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-API lifecycle shared by functional cases: inspect, wrap, tune, infer, save, load, infer."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import Backend
from aitune.torch.module import Module


@dataclass
class Tolerance:
    """Allowed difference between tuned and eager outputs."""

    rtol: float = 1e-4
    atol: float = 1e-5


@dataclass
class LifecycleResult:
    """Modules produced by `run_lifecycle`."""

    tuned: Module
    restored: Module


def selected_backend(module: Module) -> Backend:
    """Return the only backend selected for a single-graph module."""
    backends = module.module.backends
    assert len(backends) == 1, f"expected one selected backend, found {len(backends)}"
    return next(iter(backends.values()))


def _model_with_zeroed_parameters(model_factory: Callable[[], nn.Module]) -> nn.Module:
    """Return a fresh model whose outputs match the eager model only if `ait.load` restores the checkpoint weights."""
    model = model_factory()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def run_lifecycle(
    backend: Backend,
    model_factory: Callable[[], nn.Module],
    value: torch.Tensor,
    checkpoint: Path,
    tolerance: Tolerance,
) -> LifecycleResult:
    """Tune `backend` on a fresh model, check outputs, then reload the checkpoint into a zero-initialized model."""
    strategy = ait.OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    model = model_factory()
    with torch.no_grad():
        expected = model(value)

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    tuned = ait.wrap(model, inspected.get_modules(), strategy=strategy)
    ait.tune(tuned, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=value.device)

    actual = selected_backend(tuned)
    assert type(actual) is type(backend)
    assert actual.key() == backend.key()
    assert actual.device == value.device
    assert len(tuned.graph_specs) == 1
    with torch.no_grad():
        torch.testing.assert_close(tuned(value), expected, rtol=tolerance.rtol, atol=tolerance.atol)

    ait.save(tuned, checkpoint)
    tuned.deactivate()
    restored = ait.load(_model_with_zeroed_parameters(model_factory), checkpoint)
    assert restored is not tuned
    assert type(selected_backend(restored)) is type(backend)
    with torch.no_grad():
        torch.testing.assert_close(restored(value), expected, rtol=tolerance.rtol, atol=tolerance.atol)
    return LifecycleResult(tuned=tuned, restored=restored)
