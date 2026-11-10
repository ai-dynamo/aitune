# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-visible input state changes through tuning and checkpoint reload."""

import pytest
import torch
from torch import nn

import aitune.torch as ait
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


@pytest.mark.matrix("aot_tuning.models.mutable-inputs")
@with_backends("base")
@pytest.mark.parametrize(
    ("model_factory", "returns_state"),
    [
        pytest.param(InplaceCacheModel, False, id="inplace-unreturned-state"),
        pytest.param(ReplacingCacheModel, True, id="replacement-returned-state"),
    ],
)
def test_mutable_input_lifecycle(case: BackendCase, device: torch.device, tmp_path, model_factory, returns_state):
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
