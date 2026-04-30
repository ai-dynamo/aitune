# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the patched module."""

import re

import torch

from aitune.torch.jit.config import JITMode, config
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
    config.mode = JITMode.INSPECT
    config.detect_graph_breaks = False
    config.max_depth_level = 2  # the test exercises the TestNet → Linear hierarchy; default 1 filters the children out

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
