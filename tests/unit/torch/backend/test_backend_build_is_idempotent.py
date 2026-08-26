# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend import (
    TorchEagerBackend,
    TorchInductorJitBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torchao_backend import TorchAOBackend
from tests.utilities.helpers import make_graph_spec, make_sample_store, requires_cuda


class Latch:
    def __init__(self):
        self.latched = False

    def signal(self):
        if self.latched:
            raise RuntimeError("Latch already latched - should be called only once.")
        self.latched = True


class ToyModel(nn.Module):
    def forward(self, x: torch.Tensor, latch: Latch):
        latch.signal()
        return x


def build_backend(backend: Backend, torch_device: torch.device, tmp_path: Path):
    model = ToyModel()
    x = torch.randn(1, 2)
    args = (x, Latch())
    kwargs = {}
    graph_spec = make_graph_spec(model.forward, (args, kwargs), x, name="test_model")
    samples = make_sample_store([(args, kwargs)], tmp_path)
    backend.build(model, graph_spec, samples, device=torch_device, cache_dir=tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "backend_class",
    [
        TorchEagerBackend,
        TorchAOBackend,
        TorchInductorJitBackend,
        TorchTensorRTJitBackend,
        # TensorRTBackend, - not supported yet
        # TorchTensorRTAotBackend,- not supported yet
    ],
)
def test_backend_build_is_idempotent(backend_class, torch_device, tmp_path):
    backend = backend_class()
    build_backend(backend, torch_device, tmp_path)
