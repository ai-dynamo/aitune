# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial wrapping of a shared submodule that produces several graphs."""

import pytest
import torch
from torch import nn

import aitune.torch as ait
from tests.functional_ci.backends import BackendCase, with_backends
from tests.functional_ci.lifecycle import pinned_strategy
from tests.functional_ci.models import OUT_FEATURES, make_input, make_mlp, seeded


class TwoTower(nn.Module):
    """Calls a shared encoder with two input layouts, then a head; `unused` is never called."""

    def __init__(self, device: torch.device):
        """Build seeded submodules on `device`."""
        super().__init__()
        self.encoder = make_mlp(device)
        with seeded(device, 2):
            self.head = nn.Linear(OUT_FEATURES, 8).to(device).eval()
            self.unused = nn.Linear(4, 4).to(device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the flat and the sequence-layout encodings."""
        return self.head(self.encoder(x) + self.encoder(x.unsqueeze(1)).squeeze(1))


@pytest.mark.matrix("models.multiple-graphs", "modes.aot-inspect-wrap")
@with_backends("multigraph")
def test_partial_wrap_multigraph(case: BackendCase, device: torch.device, tmp_path):
    """Wrap only the shared encoder; it tunes one graph per layout, hooks still fire, and the checkpoint reloads."""
    backend, tolerance = case.factory(), case.tolerance
    value = make_input(device)
    model = TwoTower(device)
    with torch.no_grad():
        expected = model(value)

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1, min_depth=1)
    found = inspected.get_modules()
    assert not any(info.module is model.unused for info in found)
    (encoder_info,) = (info for info in found if info.module is model.encoder)
    ait.wrap(model, [encoder_info], strategy=pinned_strategy(backend))
    ait.tune(model, list(value.unbind(0)), batch_sizes=[value.shape[0]], device=device)

    assert isinstance(model.encoder, ait.Module)
    assert isinstance(model.head, nn.Linear)
    assert len(model.encoder.graph_specs) == 2
    assert [type(selected) for selected in model.encoder.module.backends.values()] == [type(backend)] * 2

    calls = []
    model.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        tolerance.assert_close(model(value), expected)
    assert calls == [1]

    ait.save(model, tmp_path / "model.ait")
    model.encoder.deactivate()
    restored = ait.load(TwoTower(device), tmp_path / "model.ait")
    with torch.no_grad():
        tolerance.assert_close(restored(value), expected)
