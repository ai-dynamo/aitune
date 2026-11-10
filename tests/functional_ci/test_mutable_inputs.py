# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-visible input state changes through tuning and checkpoint reload."""

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import OUT_FEATURES, make_input, make_mlp


class InplaceCacheModel(nn.Module):
    """Accumulate into a caller-provided cache without returning that cache."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        delta = self.mlp(x)
        state["cache"].add_(delta)
        return {"value": delta + 0.5 * state["cache"]}


class ReplacingCacheModel(nn.Module):
    """Replace the cached tensor and return the caller's state container."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.mlp = make_mlp(device, seed)

    def forward(self, x: torch.Tensor, state: dict[str, torch.Tensor]):
        state["cache"] = state["cache"] + self.mlp(x)
        return {"value": 0.5 * state["cache"], "state": state}


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
    reference = model_factory(device)
    tuned = ait.Module(model_factory(device), "mutable-inputs", strategy=pinned_strategy(backend))

    def pipeline(value: torch.Tensor):
        # Each recording call starts with fresh state; inference below must carry updates between calls.
        state = {"cache": torch.zeros(value.shape[0], OUT_FEATURES, device=device)}
        return tuned(value, state)

    ait.tune(
        pipeline,
        list(make_input(device, batch_size=8).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    def check_state_updates(model):
        selected_backend(model, type(backend))
        # Keep eager and tuned containers independent, and start with nonzero state to catch resets.
        expected_state = {"cache": torch.full((2, OUT_FEATURES), 0.25, device=device)}
        state = {"cache": expected_state["cache"].clone()}

        def infer(value):
            actual = model(value, state)
            # Backends may install their own result tensors; container identity is the shared contract.
            if returns_state:
                assert actual["state"] is state
            return actual

        for seed in (5, 6, 7):
            assert_matches_eager(
                infer,
                lambda value: reference(value, expected_state),
                device,
                case.tolerance,
                batch_sizes=(2,),
                seeds=(seed,),
            )
            case.tolerance.assert_close(state["cache"], expected_state["cache"])

    check_state_updates(tuned)
    checkpoint = tmp_path / "mutable_inputs.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    # Different initial weights ensure reload restores model weights as well as mutation metadata.
    restored = ait.load(model_factory(device, seed=1), checkpoint)
    assert isinstance(restored, ait.Module)
    check_state_updates(restored)
