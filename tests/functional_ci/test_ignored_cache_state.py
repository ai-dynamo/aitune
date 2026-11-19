# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ignored cache payload replacements through tuning, inference, and checkpoint reload."""

from copy import deepcopy

import pytest
import torch
from torch import nn

import aitune.torch as ait
from aitune.torch.module.locator import Locator
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend

FEATURES = 4


class IgnoredPayload(dict):
    """Opaque state that mutation capture must neither traverse nor copy."""

    def __deepcopy__(self, memo):
        raise AssertionError("Ignored cache payload must not be copied")


class CustomCache:
    """Caller-owned tensor state with a payload replaced by forward."""

    def __init__(self, values: torch.Tensor, payload: object):
        self.values = values
        self.payload = payload

    def update(self, increment: torch.Tensor) -> None:
        self.values = self.values + increment
        self.payload = IgnoredPayload(child={"value": "after"})


class CacheModel(nn.Module):
    """Update cache tensors while installing an ignored Python payload."""

    def __init__(self, device: torch.device, seed: int = 0):
        super().__init__()
        self.register_buffer("increment", torch.tensor(float(seed + 1), device=device))

    def forward(self, x: torch.Tensor, cache: CustomCache) -> torch.Tensor:
        cache.update(self.increment)
        return x + cache.values


@pytest.fixture
def registered_cache_types():
    # Observe the original non-tensor payload paths so replacing them exercises ignored-type filtering.
    Locator.register_user_type(CustomCache, only_tensors=False)
    Locator.ignore_type(IgnoredPayload)
    try:
        yield
    finally:
        Locator.unregister_user_type(CustomCache)
        Locator.unignore_type(IgnoredPayload)


@pytest.mark.matrix("aot_tuning.models.ignored-cache-state")
@with_backends("base")
@pytest.mark.parametrize(
    "initial_payload",
    ["pending", {"child": "before"}, {"child": {"value": "before"}}],
    ids=["ignored-leaf", "ignored-parent", "ignored-ancestor"],
)
def test_ignored_cache_state_lifecycle(
    case: BackendCase, device: torch.device, tmp_path, registered_cache_types, initial_payload
):
    """Skip ignored replacements while preserving tensor updates before and after save/load."""
    backend = case.factory()
    tuned = ait.Module(CacheModel(device), "ignored-cache-state", strategy=pinned_strategy(backend))

    def pipeline(value: torch.Tensor):
        cache = CustomCache(torch.zeros_like(value), deepcopy(initial_payload))
        return tuned(value, cache)

    ait.tune(
        pipeline,
        list(torch.ones(8, FEATURES, device=device).unbind(0)),
        batch_sizes=[2],
        device=device,
        ignore_failing_modules=False,
    )

    # Recording metadata is available on the tuned module, but is not retained by checkpoint reload.
    assert len(tuned.graph_specs) == 1
    assert [mutation.locator.path for mutation in tuned.graph_specs[0].forward_contract.input_mutations] == [
        ("cache", "values")
    ]

    def check_cache_updates(model):
        selected_backend(model, type(backend))
        reference = CacheModel(device)
        cache = CustomCache(torch.full((2, FEATURES), 0.25, device=device), None)
        reference_cache = CustomCache(cache.values.clone(), None)
        with torch.no_grad():
            for step in (1, 2, 3):
                # Reset only the payload: ignored Python assignments need not be replayed by compiled backends.
                # Tensor state persists across calls, and every call has the recorded input structure.
                cache.payload = deepcopy(initial_payload)
                reference_cache.payload = deepcopy(initial_payload)
                value = torch.full((2, FEATURES), float(step), device=device)
                expected = reference(value, reference_cache)
                actual = model(value, cache)
                case.tolerance.assert_close(actual, expected)
                case.tolerance.assert_close(cache.values, reference_cache.values)

    try:
        check_cache_updates(tuned)
        checkpoint = tmp_path / "ignored_cache_state.ait"
        ait.save(tuned, checkpoint)
        tuned.deactivate()
        # Different initial weights make checkpoint parameter restoration observable.
        restored = ait.load(CacheModel(device, seed=1), checkpoint)
        assert isinstance(restored, ait.Module)
        try:
            check_cache_updates(restored)
        finally:
            restored.deactivate()
    finally:
        tuned.deactivate()
