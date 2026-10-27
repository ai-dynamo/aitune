# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional coverage for graph breaks in model forward methods."""

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
from aitune.torch.module import Module
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import Tolerance, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp


class BranchingModel(nn.Module):
    """Run one of two seeded MLPs, picked by the sign of the batch mean."""

    def __init__(self, device: torch.device):
        """Build the two seeded branches on `device`."""
        super().__init__()
        self.positive = make_mlp(device)
        self.negative = make_mlp(device, seed=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the branch selected by the batch mean."""
        # Data-dependent branch: torch.compile breaks the graph here and torch.export rejects it.
        if x.mean() > 0:
            return self.positive(x)
        return self.negative(x)


def _wrap(backend: Backend, device: torch.device) -> Module:
    """Wrap a `BranchingModel` pinned to `backend`."""
    return ait.Module(BranchingModel(device), "branching", strategy=pinned_strategy(backend))


def _tune(tuned: Module, positive: torch.Tensor, **kwargs: Any) -> None:
    """Tune on the positive branch only."""
    ait.tune(tuned, list(positive.unbind(0)), batch_sizes=[positive.shape[0]], device=positive.device, **kwargs)


def _assert_both_branches(model: nn.Module | Module, positive: torch.Tensor, tolerance: Tolerance) -> None:
    """Both branches match eager, including the negative branch that tuning never saw."""
    reference = BranchingModel(positive.device)
    with torch.no_grad():
        for branch_input in (positive, -positive):
            tolerance.assert_close(model(branch_input), reference(branch_input))


def _assert_export_fails(case: BackendCase, device: torch.device) -> None:
    """Tuning re-raises the export error for the data-dependent branch."""
    with pytest.raises(Exception, match="(?i)data-dependent"):
        _tune(_wrap(case.factory(), device), make_input(device).abs(), ignore_failing_modules=False)


@pytest.mark.matrix("models.graph-breaks")
@with_backends("graph_breaks")
def test_graph_break_aot(case: BackendCase, device: torch.device):
    """AOT backends without export compile around the branch."""
    backend = case.factory()
    tuned = _wrap(backend, device)
    positive = make_input(device).abs()
    _tune(tuned, positive)
    selected_backend(tuned, type(backend))
    _assert_both_branches(tuned, positive, case.tolerance)


@pytest.mark.matrix("models.graph-breaks")
@with_backends("torch_export")
def test_graph_break_torch_export(case: BackendCase, device: torch.device):
    """`torch.export` rejects the data-dependent branch."""
    _assert_export_fails(case, device)


@pytest.mark.matrix("models.graph-breaks")
@with_backends("onnx_export")
def test_graph_break_onnx(case: BackendCase, device: torch.device):
    """Dynamo ONNX export rejects the data-dependent branch."""
    _assert_export_fails(case, device)


@pytest.mark.matrix("models.graph-breaks")
@with_backends("jit")
def test_graph_break_jit(case: BackendCase, device: torch.device):
    """JIT backends compile the head around the branch."""
    backend = case.factory()
    if isinstance(backend, TorchTensorRTJitBackend):
        # Raise instead of silently running the graph in PyTorch when TensorRT cannot build it.
        cast(Any, backend._config.compile_config).pass_through_build_failures = True
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 2
    jit_config.strategy = ait.OneBackendStrategy(backend).enable_performance_validation(False)
    with prepare_for_jit_tuning():
        model = BranchingModel(device)

    positive = make_input(device).abs()
    with torch.no_grad():
        model(positive[:1])
        model(positive)

    (head,) = PatchedModule.heads
    assert head._name.split(" ")[0] == "BranchingModel"
    assert head._state == ModuleState.TUNED
    assert all(type(selected) is type(backend) for selected in cast(TunedModule, head._wrapper).backends.values())
    _assert_both_branches(model, positive, case.tolerance)
