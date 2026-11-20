# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward hooks on the tuned module: the pre-hook supplies the input and the post-hook changes the output."""

from typing import cast

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.config import config as jit_config
from aitune.torch.jit.patched_module import ModuleState, PatchedModule
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from aitune.torch.module.tuned_module import TunedModule
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp
from tests.functional_ci.test_jit import JitMLP


def _add_hooks(model: nn.Module, value: torch.Tensor) -> nn.Module:
    """The pre-hook replaces the call arguments with `value`; the post-hook reduces the output to its row maximum."""
    model.register_forward_pre_hook(lambda module, args: (value,))
    model.register_forward_hook(lambda module, args, output: output.amax(dim=1))
    return model


@pytest.mark.matrix("aot_tuning.models.forward-hooks")
@with_backends("base")
def test_forward_hooks_aot_tuning(case: BackendCase, device: torch.device):
    """Record and serve calls without arguments; `tune(device=None)` infers the device from the model."""
    backend = case.factory()
    value = make_input(device)
    reference = _add_hooks(make_mlp(device), value)
    tuned = ait.Module(_add_hooks(make_mlp(device), value), "hooked-mlp", strategy=pinned_strategy(backend))

    with torch.no_grad():
        tuned()
        assert len(tuned.graph_specs) == 1
        tuned.tune(device=None)

        assert selected_backend(tuned, type(backend)).device == device
        actual = tuned()
        assert actual.shape == (value.shape[0],)
        case.tolerance.assert_close(actual, reference())


@pytest.mark.matrix("jit_tuning.models.forward-hooks")
@with_backends("base")
def test_forward_hooks_jit_tuning(case: BackendCase, device: torch.device):
    """Hooks registered after patching run on recorded and tuned calls; the unset JIT device is inferred."""
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.min_samples = 2
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    value = make_input(device)
    reference = _add_hooks(JitMLP(make_mlp(device)), value)
    with prepare_for_jit_tuning():
        model = _add_hooks(JitMLP(make_mlp(device)), value)

    with torch.no_grad():
        model()
        model()

        (head,) = PatchedModule.heads
        assert head._state == ModuleState.TUNED
        (selected,) = cast(TunedModule, head._wrapper).backends.values()
        assert type(selected) is type(backend)
        assert selected.device == device
        actual = model()
        assert actual.shape == (value.shape[0],)
        case.tolerance.assert_close(actual, reference())
