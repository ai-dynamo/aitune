# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch one shared module across ranks and optional tensor arguments."""

from typing import Any, cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.backend import TorchTensorRTJitBackend
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy
from tests.functional_ci.models import make_input, make_mlp


class LayoutMLP(nn.Module):
    """Accept flat or sequence inputs, with an optional residual."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x if residual is None else x + residual)


def _layouts(x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], ...]:
    return (x,), (x.unsqueeze(1).repeat(1, 3, 1),), (x, x * 0.5)


class SharedLayouts(nn.Module):
    """Call the same child with all three signatures on every parent forward."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.encoder: nn.Module | ait.Module = LayoutMLP(device, seed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(*args) for args in _layouts(x))


@pytest.mark.matrix("models.multiple-graphs", "modes.aot-tune", "modes.aot-save-load")
@with_backends("save_load")
def test_multiple_graphs_aot(case: BackendCase, device: torch.device, tmp_path):
    """Tune each layout independently and reload all graphs into different weights."""
    backend = case.factory()
    reference = SharedLayouts(device)
    model = SharedLayouts(device)
    model.encoder = ait.Module(cast(nn.Module, model.encoder), "encoder", strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    encoder = cast(ait.Module, model.encoder)
    assert len(encoder.graph_specs) == 3
    assert [type(selected) for selected in cast(TunedModule, encoder.module).backends.values()] == [type(backend)] * 3
    assert_matches_eager(model, reference, device, case.tolerance, batch_sizes=(2,))

    checkpoint = tmp_path / "layouts.ait"
    ait.save(model, checkpoint)
    encoder.deactivate()
    restored = ait.load(SharedLayouts(device, seed=1), checkpoint)
    restored_encoder = cast(ait.Module, restored.encoder)
    assert [type(selected) for selected in cast(TunedModule, restored_encoder.module).backends.values()] == [
        type(backend)
    ] * 3
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
    reference = LayoutMLP(device)
    with prepare_for_jit_tuning():
        model = LayoutMLP(device)

    with torch.no_grad():
        for batch_size in (1, 4):
            for args in _layouts(make_input(device, batch_size=batch_size)):
                model(*args)

        (head,) = PatchedModule.heads
        assert head._name.split(" ")[0] == "LayoutMLP"
        assert head._state == ModuleState.TUNED
        assert [type(selected) for selected in cast(TunedModule, head._wrapper).backends.values()] == [
            type(backend)
        ] * 3
        assert_matches_eager(
            lambda value: tuple(model(*args) for args in _layouts(value)),
            lambda value: tuple(reference(*args) for args in _layouts(value)),
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
    model = SharedLayouts(device)
    model.encoder = ait.Module(cast(nn.Module, model.encoder), "encoder", strategy=pinned_strategy(backend))
    value = make_input(device)
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)
    encoder = cast(ait.Module, model.encoder)
    assert [type(selected) for selected in cast(TunedModule, encoder.module).backends.values()] == [type(backend)] * 3
    with torch.no_grad(), pytest.raises(RuntimeError, match="No backend found"):
        model.encoder(value[:, None, None, :])
