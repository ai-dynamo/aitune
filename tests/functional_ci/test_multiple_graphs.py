# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional coverage for one module called with several input signatures."""

from typing import Any, cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import Backend, TorchTensorRTJitBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy
from tests.functional_ci.models import make_input, make_mlp


class SingleGraphsMLP(nn.Module):
    """Accept flat or sequence inputs, with an optional residual."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x if residual is None else x + residual)


def _split_input_for_multiple_graphs(x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Return three signatures: flat `(B, F)`, sequence `(B, 3, F)`, and `(x, residual)`."""
    return (x,), (x.unsqueeze(1).repeat(1, 3, 1),), (x, x * 0.5)


class MultipleGraphsMLP(nn.Module):
    """Call the same child with all three signatures on every parent forward."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.encoder: nn.Module | ait.Module = SingleGraphsMLP(device, seed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(*args) for args in _split_input_for_multiple_graphs(x))


def _assert_backends(module: TunedModule, backend: Backend) -> None:
    """Each of the three graphs has its own copy of the pinned backend."""
    backends = list(module.backends.values())
    assert [type(selected) for selected in backends] == [type(backend)] * 3
    assert len({id(selected) for selected in backends}) == 3
    assert all(selected is not backend for selected in backends)


@pytest.mark.matrix("aot_tuning.models.multiple-graphs")
@with_backends("base")
def test_multiple_graphs_aot_tunning(case: BackendCase, device: torch.device):
    """AOT tuning makes one graph per signature of the shared child."""
    backend = case.factory()
    model = MultipleGraphsMLP(device)
    model.encoder = ait.Module(cast(nn.Module, model.encoder), "encoder", strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    assert len(model.encoder.graph_specs) == 3
    _assert_backends(cast(TunedModule, model.encoder.module), backend)
    assert_matches_eager(model, MultipleGraphsMLP(device), device, case.tolerance, batch_sizes=(2,))


@pytest.mark.matrix("jit_tuning.models.multiple-graphs")
@with_backends("base")
def test_multiple_graphs_jit_tunning(case: BackendCase, device: torch.device):
    """JIT tuning records every signature, then tunes one graph per signature."""
    backend = case.factory()
    if isinstance(backend, TorchTensorRTJitBackend):
        # Raise instead of silently running the graph in PyTorch when TensorRT cannot build it.
        cast(Any, backend._config.compile_config).pass_through_build_failures = True
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 6
    jit_config.strategy = pinned_strategy(backend)
    reference = SingleGraphsMLP(device)
    # Patch the child directly: a patched parent would be the head, with only one signature.
    with prepare_for_jit_tuning():
        model = SingleGraphsMLP(device)

    with torch.no_grad():
        for batch_size in (1, 4):
            for args in _split_input_for_multiple_graphs(make_input(device, batch_size=batch_size)):
                model(*args)

    (head,) = PatchedModule.heads
    assert head._name.split(" ")[0] == "SingleGraphsMLP"
    assert head._state == ModuleState.TUNED
    _assert_backends(cast(TunedModule, head._wrapper), backend)
    assert_matches_eager(
        lambda value: tuple(model(*args) for args in _split_input_for_multiple_graphs(value)),
        lambda value: tuple(reference(*args) for args in _split_input_for_multiple_graphs(value)),
        device,
        case.tolerance,
        batch_sizes=(1, 4),
        seeds=(7,),
    )
