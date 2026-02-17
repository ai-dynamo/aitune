# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Test the patched module."""

import re

import torch

from aitune.torch.jit.config import config
from aitune.torch.jit.inspect_module import InspectModule
from aitune.torch.jit.patched_module import (
    PRINT_HIERARCHY_HEADER,
)
from aitune.torch.jit.patcher import prepare_for_jit_tuning
from tests.utilities.helpers import TestSink, requires_cuda


class TestType:
    """Just a class to mimic unsupported type."""


class TestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x, test_type=None):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@requires_cuda
def test_jit_inspect_module_hooks(torch_device):
    config.inspect_mode = True
    config.detect_graph_breaks = False

    with prepare_for_jit_tuning():
        pipeline = TestNet().to(torch_device)

    hooks_history = []

    def pre_hook(module, input):  # noqa: A002
        hooks_history.append("pre_hook")
        return input

    def hook(module, input, output):  # noqa: A002
        hooks_history.append("forward_hook")
        return output

    pipeline.register_forward_hook(hook)
    pipeline.register_forward_pre_hook(pre_hook)

    x = torch.randn(2, 10, device=torch_device)
    pipeline(x, TestType())
    assert hooks_history == ["pre_hook", "forward_hook"]
    hooks_history.clear()
    pipeline(x, TestType())
    assert hooks_history == ["pre_hook", "forward_hook"]

    assert len(InspectModule.heads) == 1

    sink = TestSink()
    InspectModule.print_hierarchy(sink=sink.write)

    assert PRINT_HIERARCHY_HEADER in sink.output[0]
    assert re.match(r".*TestNet.*state=inspect.*call_count=2", sink.output[1])
    assert re.match(r".*Linear.*state=inspect.*call_count=2", sink.output[2])
    assert re.match(r".*Linear.*state=inspect.*call_count=2", sink.output[3])
