# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-visible input state changes through tuning and checkpoint reload."""

from functools import partial

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend
from tests.toy_models.stateful import update_tensor

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


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
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


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@with_backends("base")
def test_replacing_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path):
    """Cache replacement and the returned container alias survive consecutive calls and save/load."""
    _check_mutable_input_lifecycle(case, device, tmp_path, ReplacingCacheModel, returns_state=True)


def _check_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path, model_factory, returns_state):
    """Outputs, accumulated input state, and container aliases survive tuning and save/load."""
    backend = case.factory()
    tuned = ait.Module(model_factory(device), "mutable-inputs", strategy=pinned_strategy(backend))

    def pipeline(value: torch.Tensor):
        # Nonzero state and inputs expose every operation, including multiplication and zeroing.
        # Each recording call starts fresh; inference below must retain updates between calls.
        state = {"cache": torch.full_like(value, 0.25)}
        return tuned(value, state)

    ait.tune(
        pipeline,
        list(torch.full((8, FEATURES), 2.0, device=device).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    def check_state_updates(model):
        selected_backend(model, type(backend))
        eager = model_factory(device).eval()
        # Interleave independent callers. A later inference must not overwrite another cache.
        states = [{"cache": torch.full((2, FEATURES), initial, device=device)} for initial in (0.25, 1.5)]
        eager_states = [{"cache": state["cache"].clone()} for state in states]
        with torch.no_grad():
            for cache_index, scale in ((0, 1.0), (1, 7.0), (0, 3.0)):
                state = states[cache_index]
                value = torch.arange(2 * FEATURES, device=device).reshape(2, FEATURES).float() / 8 + scale
                expected = eager(value, eager_states[cache_index])
                actual = model(value, state)
                case.tolerance.assert_close(actual["value"], expected["value"])
                for actual_state, expected_state in zip(states, eager_states, strict=True):
                    case.tolerance.assert_close(actual_state["cache"], expected_state["cache"])
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


class TensorMutationModel(nn.Module):
    """Exercise indexed writes and arithmetic updates without returning the cache."""

    def __init__(self, device: torch.device, seed: int = 0, *, operation: str):
        super().__init__()
        self.operation = operation
        self.register_buffer("scale", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        index = torch.arange(x.shape[0], device=x.device).flip(0)
        if self.operation == "scatter_":
            index = index[:, None].expand_as(x)
        update_tensor(state["cache"], x * self.scale, index, self.operation)
        return {"value": x + state["cache"]}


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@pytest.mark.parametrize("operation", ["mul_", "index_copy_", "scatter_", "zero_"])
@with_backends("base")
def test_tensor_mutation_lifecycle(case: BackendCase, device: torch.device, tmp_path, operation):
    """Numerical input updates survive repeated calls, interleaved caches, and checkpoint reload."""
    _check_mutable_input_lifecycle(
        case, device, tmp_path, partial(TensorMutationModel, operation=operation), returns_state=False
    )
