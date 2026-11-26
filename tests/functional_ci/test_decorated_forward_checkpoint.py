# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise decorated-forward checkpoint loading after clearing Dynamo caches."""

import pytest
import torch

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy, selected_backend
from tests.toy_models.decorated_forward_models import ToyDecoratedSpeechModel


@pytest.mark.matrix("aot_tuning.models.decorated-forward", "modes.aot-save-load")
@with_backends("base")
def test_decorated_forward_checkpoint(case: BackendCase, device: torch.device, tmp_path):
    """Inspect, tune, and save; clear Dynamo caches and load into an uninspected model."""
    # Exercise the same model on every backend, with no fallback or expected failures,
    # so each result shows whether that backend supports the complete lifecycle.
    backend = case.factory()
    model = ToyDecoratedSpeechModel().to(device).eval()
    value = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16
    validation_values = (value, value + 0.5)
    with torch.no_grad():
        expected = [model(sample).cpu() for sample in validation_values]

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    tuned = ait.wrap(model, inspected.get_modules(), strategy=pinned_strategy(backend))
    assert isinstance(tuned, ait.Module)
    ait.tune(
        tuned,
        list(value.unbind(0)),
        batch_sizes=[value.shape[0]],
        device=device,
        ignore_failing_modules=False,
    )
    selected_backend(tuned, type(backend))
    with torch.no_grad():
        for sample, reference in zip(validation_values, expected, strict=True):
            case.tolerance.assert_close(tuned(sample).cpu(), reference)

    checkpoint = tmp_path / "decorated.ait"
    ait.save(tuned, checkpoint)
    tuned.deactivate()
    torch._dynamo.reset()

    # Load without inspection; different weights also verify checkpoint state restoration.
    fresh_model = ToyDecoratedSpeechModel(seed=1).to(device).eval()
    restored = ait.load(fresh_model, checkpoint)
    assert isinstance(restored, ait.Module)
    selected_backend(restored, type(backend))
    with torch.no_grad():
        for sample, reference in zip(validation_values, expected, strict=True):
            case.tolerance.assert_close(restored(sample).cpu(), reference)
