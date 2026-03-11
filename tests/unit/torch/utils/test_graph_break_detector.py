# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphBreakDetector."""

import pytest
import torch
import torch.nn as nn

from aitune.torch.utils.graph_break_detector import GraphBreakDetector
from tests.toy_models.torch_models import ToyTorchModel


class ModelWithIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x - 1
        else:
            return x


class ModelWithWhileLoop(nn.Module):
    def forward(self, x, threshold=1):
        y = x.clone()
        while y.mean() < threshold:
            y = y + 1
        return y - 1


class ModelWithInternalState(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = []

    def forward(self, x):
        if x.sum() > 0:
            self.history.append(x.sum().item())
        return x * 2


class ModelWithPrint(nn.Module):
    def forward(self, x):
        print(f"Current x sum: {x.sum()}")  # This will cause a graph break
        return x + 1


class ModelWithNonzero(nn.Module):
    def forward(self, x):
        # torch.nonzero often causes breaks due to dynamic output shape
        mask = torch.nonzero(x > 0)
        return x[mask[:, 0]]


@pytest.mark.parametrize(
    "model_class",
    [
        ModelWithIf,
        ModelWithWhileLoop,
        ModelWithInternalState,
        ModelWithPrint,
        # ModelWithNonzero, # This test stop working with torch 2.10
    ],
)
def test_detect_graph_breaks(model_class):
    detector = GraphBreakDetector()
    detector.detect(model_class(), torch.randn(1, 10), {})
    assert detector.has_graph_breaks()


def test_there_is_no_graph_breaks():
    detector = GraphBreakDetector()
    model = ToyTorchModel()
    args, kwargs = ((model.sample(),), {})
    detector.detect(model, args, kwargs)
    assert not detector.has_graph_breaks()
