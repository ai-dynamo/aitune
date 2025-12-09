# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aitune.torch.backend import (
    TorchEagerBackend,
    TorchInductorBackend,
    TorchTensorRTJitBackend,
)
from aitune.torch.backend.backend import Backend
from aitune.torch.backend.torchao_backend import TorchAOBackend
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.utilities.helpers import requires_cuda


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
    input_spec = SampleMetadata.from_inputs(args, kwargs)
    output_spec = SampleMetadata.from_outputs(x)
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)
    backend.build(model, graph_spec, [(args, kwargs)], device=torch_device, cache_dir=tmp_path)


@requires_cuda
@pytest.mark.parametrize(
    "backend_class",
    [
        TorchEagerBackend,
        TorchAOBackend,
        TorchInductorBackend,
        TorchTensorRTJitBackend,
        # TensorRTBackend, - not supported yet
        # TorchTensorRTAotBackend,- not supported yet
    ],
)
def test_backend_build_is_idempotent(backend_class, torch_device, tmp_path):
    backend = backend_class()
    build_backend(backend, torch_device, tmp_path)
