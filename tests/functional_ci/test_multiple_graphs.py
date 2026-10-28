# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch one shared module across ranks and optional tensor arguments."""

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
    return (x,), (x.unsqueeze(1).repeat(1, 3, 1),), (x, x * 0.5)


def _assert_backends(module: TunedModule, expected: type[Backend], prototype: Backend | None = None) -> None:
    backends = list(module.backends.values())
    assert [type(selected) for selected in backends] == [expected] * 3
    assert len({id(selected) for selected in backends}) == 3
    assert all(selected is not prototype for selected in backends)


class MultipleGraphsMLP(nn.Module):
    """Call the same child with all three signatures on every parent forward."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.encoder: nn.Module | ait.Module = SingleGraphsMLP(device, seed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(*args) for args in _split_input_for_multiple_graphs(x))


@pytest.mark.matrix("models.multiple-graphs", "modes.aot-tune", "modes.aot-save-load")
@with_backends("save_load")
def test_multiple_graphs_aot(case: BackendCase, device: torch.device, tmp_path):
    """Tune each layout independently and reload all graphs into different weights."""
    backend = case.factory()
    reference = MultipleGraphsMLP(device)
    model = MultipleGraphsMLP(device)
    model.encoder = ait.Module(cast(nn.Module, model.encoder), "encoder", strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    assert len(model.encoder.graph_specs) == 3
    _assert_backends(cast(TunedModule, model.encoder.module), type(backend), backend)
    assert_matches_eager(model, reference, device, case.tolerance, batch_sizes=(2,))

    checkpoint = tmp_path / "layouts.ait"
    ait.save(model, checkpoint)
    model.encoder.deactivate()
    restored = ait.load(MultipleGraphsMLP(device, seed=1), checkpoint)
    restored_encoder = cast(ait.Module, restored.encoder)
    _assert_backends(cast(TunedModule, restored_encoder.module), type(backend))
    assert_matches_eager(restored, reference, device, case.tolerance, batch_sizes=(2,))


@pytest.mark.matrix("models.multiple-graphs", "modes.jit-tune_eager")
@with_backends("jit")
def test_multiple_graphs_jit(case: BackendCase, device: torch.device):
    """Record every signature before tuning the JIT head, then dispatch each graph."""
    backend = case.factory()
    if isinstance(backend, TorchTensorRTJitBackend):
        cast(Any, backend._config.compile_config).pass_through_build_failures = True
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 6
    jit_config.strategy = pinned_strategy(backend)
    reference = SingleGraphsMLP(device)
    with prepare_for_jit_tuning():
        model = SingleGraphsMLP(device)

    with torch.no_grad():
        for batch_size in (1, 4):
            for args in _split_input_for_multiple_graphs(make_input(device, batch_size=batch_size)):
                model(*args)

        (head,) = PatchedModule.heads
        assert head._name.split(" ")[0] == "SingleGraphsMLP"
        assert head._state == ModuleState.TUNED
        _assert_backends(cast(TunedModule, head._wrapper), type(backend), backend)
        assert_matches_eager(
            lambda value: tuple(model(*args) for args in _split_input_for_multiple_graphs(value)),
            lambda value: tuple(reference(*args) for args in _split_input_for_multiple_graphs(value)),
            device,
            case.tolerance,
            batch_sizes=(1, 4),
            seeds=(7,),
        )


@pytest.mark.matrix("models.multiple-graphs", "modes.aot-tune")
@with_backends("multigraph")
def test_multiple_graphs_unrecorded_layout(case: BackendCase, device: torch.device):
    """Existing multigraph backends reject an unrecorded rank rather than running eager."""
    backend = case.factory()
    model = MultipleGraphsMLP(device)
    model.encoder = ait.Module(cast(nn.Module, model.encoder), "encoder", strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)
    _assert_backends(cast(TunedModule, model.encoder.module), type(backend), backend)
    with torch.no_grad(), pytest.raises(RuntimeError, match="No backend found"):
        model.encoder(value[:, None, None, :])
