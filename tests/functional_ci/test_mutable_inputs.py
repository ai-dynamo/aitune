# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-visible input state changes through tuning and checkpoint reload."""

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

FEATURES = 4


class InplaceCacheModel(nn.Module):
    """Accumulate into a caller-provided cache without returning that cache."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state["cache"].add_(self.increment)
        return {"value": x + state["cache"]}


class ReplacingCacheModel(nn.Module):
    """Replace the cached tensor and return the caller's state container."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        state["cache"] = state["cache"] + self.increment
        return {"value": x + state["cache"], "state": state}


@pytest.mark.matrix("aot_tuning.models.mutable-inputs", "aot_tuning.models.mutable-inputs-inplace")
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.xfail(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
            raises=AssertionError,
            strict=True,
        ),
    },
)
def test_inplace_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Unreturned in-place cache updates survive consecutive calls and save/load."""
    _check_mutable_input_lifecycle(case, device, tmp_path, InplaceCacheModel, returns_state=False)


@pytest.mark.matrix("aot_tuning.models.mutable-inputs", "aot_tuning.models.mutable-inputs-with-return-state")
@with_backends("base")
def test_replacing_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Cache replacement and the returned container alias survive consecutive calls and save/load."""
    _check_mutable_input_lifecycle(case, device, tmp_path, ReplacingCacheModel, returns_state=True)


@pytest.mark.matrix("jit_tuning.models.mutable-inputs", "jit_tuning.models.mutable-inputs-inplace")
@with_backends(
    "base",
    marks={
        "torch-tensorrt-jit": pytest.mark.xfail(
            reason="Torch-TensorRT JIT drops in-place input writes; related upstream fix: pytorch/TensorRT#4894",
            raises=AssertionError,
            strict=True,
        ),
    },
)
def test_inplace_mutable_inputs_jit(case: BackendCase, device: torch.device):
    """JIT tuning preserves unreturned cache updates across consecutive calls."""
    _check_jit_mutable_inputs(case, device, InplaceCacheModel, returns_state=False)


@pytest.mark.matrix("jit_tuning.models.mutable-inputs", "jit_tuning.models.mutable-inputs-with-return-state")
@with_backends("base")
def test_replacing_mutable_inputs_jit(case: BackendCase, device: torch.device):
    """JIT tuning preserves cache replacement and returned caller-owned identity."""
    _check_jit_mutable_inputs(case, device, ReplacingCacheModel, returns_state=True)


def _check_jit_mutable_inputs(case: BackendCase, device: torch.device, model_factory, returns_state):
    """Tune from a fixed-shape live call, then compare accumulated state with eager."""
    backend = case.factory()
    jit_config.mode = JITMode.TUNE_EAGER
    jit_config.device = device
    jit_config.min_samples = 1
    jit_config.batch_axis_required = False
    jit_config.strategy = pinned_strategy(backend)
    reference = model_factory(device)
    with prepare_for_jit_tuning():
        model = model_factory(device)

    with torch.no_grad():
        value = torch.ones(2, FEATURES, device=device)
        # Recording starts from fresh state; inference below uses a separate accumulating cache.
        model(value, {"cache": torch.zeros_like(value)})

        (head,) = PatchedModule.heads
        assert head._state == ModuleState.TUNED
        tuned = head._wrapper
        assert isinstance(tuned, TunedModule)
        (selected,) = tuned.backends.values()
        assert type(selected) is type(backend)

        state = {"cache": torch.full_like(value, 0.25)}
        reference_state = {"cache": torch.full_like(value, 0.25)}
        for step in (1, 2, 3):
            value = torch.full((2, FEATURES), float(step), device=device)
            actual = model(value, state)
            expected = reference(value, reference_state)
            case.tolerance.assert_close(state["cache"], reference_state["cache"])
            case.tolerance.assert_close(actual["value"], expected["value"])
            if returns_state:
                assert actual["state"] is state


def _check_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path, model_factory, returns_state):
    """Outputs, accumulated input state, and container aliases survive tuning and save/load."""
    backend = case.factory()
    tuned = ait.Module(model_factory(device), "mutable-inputs", strategy=pinned_strategy(backend))

    def pipeline(value: torch.Tensor):
        # Each recording call starts with fresh state; inference below must carry updates between calls.
        state = {"cache": torch.zeros_like(value)}
        return tuned(value, state)

    ait.tune(
        pipeline,
        list(torch.ones(8, FEATURES, device=device).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    def check_state_updates(model):
        selected_backend(model, type(backend))
        # Scalar increments isolate state propagation from matrix multiplication and quantization accuracy.
        # Start with nonzero state so a reset or a missing update fails the accumulation check.
        state = {"cache": torch.full((2, FEATURES), 0.25, device=device)}
        with torch.no_grad():
            for step in (1, 2, 3):
                value = torch.full((2, FEATURES), float(step), device=device)
                actual = model(value, state)
                expected_cache = torch.full_like(value, 0.25 + step)
                case.tolerance.assert_close(state["cache"], expected_cache)
                case.tolerance.assert_close(actual["value"], value + expected_cache)
                # Backends may install their own result tensors; container identity is the shared contract.
                if returns_state:
                    assert actual["state"] is state

    check_state_updates(tuned)
    checkpoint = tmp_path / "mutable_inputs.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # A different initial increment ensures reload restores the model buffer as well as mutation metadata.
    restored = ait.load(model_factory(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    check_state_updates(restored)
