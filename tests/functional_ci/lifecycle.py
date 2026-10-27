# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-API lifecycle shared by functional cases: inspect, wrap, tune, infer, save, load, infer."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import Backend, ONNXExecutionProvider, ONNXRuntimeBackend
from aitune.torch.module import Module
from tests.functional_ci.models import make_input


@dataclass(frozen=True)
class Tolerance:
    """Allowed difference between tuned and eager outputs."""

    rtol: float = 1e-4
    atol: float = 1e-5

    def assert_close(self, actual, expected) -> None:
        """Assert that `actual` matches `expected` within this tolerance."""
        torch.testing.assert_close(actual, expected, rtol=self.rtol, atol=self.atol)


# TensorRT and ONNX Runtime fp32 on the SM120 PR runner differ from eager by about 3e-4 even with TF32 disabled.
TENSORRT_FP32 = Tolerance(rtol=1e-3, atol=1e-3)


def assert_tensorrt_provider_active(backend: ONNXRuntimeBackend) -> None:
    """Check that the session did not fall back from the TensorRT provider."""
    # ONNX Runtime falls back to the CUDA provider if the TensorRT provider cannot start.
    assert backend._session is not None, "ONNX Runtime session is not active"
    providers = backend._session.get_providers()
    assert providers[0] == "TensorrtExecutionProvider", providers


def selected_backend(module: Module, expected: type[Backend] | None = None) -> Backend:
    """Return the only backend selected for a single-graph module; check its type when `expected` is given."""
    backends: dict[Any, Backend] = getattr(module.module, "backends", {})
    assert len(backends) == 1, f"expected one selected backend, found {len(backends)}"
    (selected,) = backends.values()
    assert expected is None or type(selected) is expected, f"selected {type(selected)}, expected {expected}"
    if (
        isinstance(selected, ONNXRuntimeBackend)
        and selected._config.execution_provider == ONNXExecutionProvider.TENSORRT
    ):
        assert_tensorrt_provider_active(selected)
    return selected


def assert_matches_eager(
    tuned: Callable,
    reference: Callable,
    device: torch.device,
    tolerance: Tolerance,
    batch_sizes: Sequence[int] = (2,),
    seeds: Sequence[int] = (5,),
) -> None:
    """Compare `tuned` with `reference` on a seeded `make_input` batch for each batch size and seed."""
    with torch.no_grad():
        for batch_size in batch_sizes:
            for seed in seeds:
                value = make_input(device, batch_size=batch_size, seed=seed)
                tolerance.assert_close(tuned(value), reference(value))


def pinned_strategy(backend: Backend) -> ait.OneBackendStrategy:
    """Select exactly `backend` without performance validation or max-batch discovery."""
    strategy = ait.OneBackendStrategy(backend)
    strategy.enable_performance_validation(False)
    strategy.enable_find_max_batch_size(False)
    return strategy


def run_lifecycle(
    backend: Backend,
    model_factory: Callable[..., nn.Module],
    value: torch.Tensor,
    checkpoint: Path,
    tolerance: Tolerance,
) -> Module:
    """Tune `backend` on a fresh model, check outputs, then return the checkpoint reloaded into another seeded model."""
    strategy = pinned_strategy(backend)
    model = model_factory()
    with torch.no_grad():
        expected = model(value)

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    tuned = cast(Module, ait.wrap(model, inspected.get_modules(), strategy=strategy))
    ait.tune(tuned, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=value.device)

    actual = selected_backend(tuned, type(backend))
    assert actual.key() == backend.key()
    assert actual.device == value.device
    assert len(tuned.graph_specs) == 1
    with torch.no_grad():
        tolerance.assert_close(tuned(value), expected)

    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # Other weights than the tuned model, so outputs match only if `ait.load` restores the checkpoint weights.
    restored = cast(Module, ait.load(model_factory(seed=1), checkpoint))
    assert restored is not tuned
    selected_backend(restored, type(backend))
    with torch.no_grad():
        tolerance.assert_close(restored(value), expected)
    return restored
