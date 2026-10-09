# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional backend lifecycle cases."""

from collections.abc import Callable

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import Backend, TorchEagerBackend

from .fixtures.assertions import selected_backend
from .fixtures.models import make_mlp


@pytest.mark.backend_lifecycle
@pytest.mark.parametrize(
    ("backend_factory", "device_mode"),
    [
        pytest.param(
            TorchEagerBackend,
            "explicit",
            id="torch-eager-fp32",
            marks=pytest.mark.functional_case(backend="TorchEagerBackend", variant="fp32-explicit"),
        )
    ],
)
def test_backend_lifecycle(
    functional_device: torch.device,
    tmp_path,
    backend_factory: Callable[[], Backend],
    device_mode: str,
):
    """Tune, infer, save, and restore the explicit Torch Eager configuration."""
    requested_backend = backend_factory()
    strategy = ait.OneBackendStrategy(requested_backend).enable_performance_validation(False)
    model = make_mlp(device=functional_device)
    generator = torch.Generator(device=functional_device).manual_seed(1)
    value = torch.randn(2, 128, device=functional_device, generator=generator)
    with torch.no_grad():
        expected = model(value)

    inspected = ait.inspect(model, [value], number_of_iterations=1, warmup_iterations=1)
    module = ait.wrap(model, inspected.get_modules(), strategy=strategy)
    tune_device = functional_device if device_mode == "explicit" else None
    ait.tune(module, list(value.unbind(0)), batch_sizes=[2], device=tune_device)

    backend = selected_backend(module)
    assert type(backend) is type(requested_backend)
    assert backend.key() == requested_backend.key()
    graph_spec = module.graph_specs[0]
    assert len(module.graph_specs) == 1
    assert graph_spec.input_spec.tensor_data[0][1].shape == [2, 128]
    assert graph_spec.output_spec.tensor_data[0][1].shape == [2, 64]
    torch.testing.assert_close(module(value), expected, rtol=1e-4, atol=1e-5)

    checkpoint = tmp_path / "torch-eager.ait"
    ait.save(module, checkpoint)
    module.deactivate()
    restored = ait.load(make_mlp(device=functional_device), checkpoint)
    assert restored is not module
    assert selected_backend(restored) is not backend
    torch.testing.assert_close(restored(value), expected, rtol=1e-4, atol=1e-5)
