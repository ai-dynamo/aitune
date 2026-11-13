# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for exporter-neutral tensor preparation."""

import torch

from aitune.torch.libs.exporting import prepare_tensor_export
from aitune.torch.module.recording_module import RecordingModule
from tests.toy_models import TOY_EXPORT_MODELS


class CountingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return x + 1


def test_prepare_moves_nested_tensor_leaves_to_device():
    model = dict(TOY_EXPORT_MODELS)["complex"]().eval()
    sample = model.samples(batch_sizes=[2])[0]
    graph_spec = model.graph_spec(batch_sizes=[2])
    model.to("meta")

    preparation = prepare_tensor_export(model, sample, graph_spec, device="meta")

    assert preparation.flat_inputs
    assert preparation.adapter.training is model.training
    assert all(tensor.device.type == "meta" for tensor in preparation.flat_inputs)


def test_prepare_uses_recorded_graph_spec_without_calling_module(tmp_path):
    model = CountingModule()
    recording = RecordingModule(model, "counting", cache_dir_resolver=lambda: tmp_path)
    x = torch.ones(2)
    recording(x)

    graph_spec = recording.graph_specs[0]
    sample = recording.samples_for_graph_spec(graph_spec)[0]
    preparation = prepare_tensor_export(model, sample, graph_spec)

    assert model.call_count == 1
    torch.testing.assert_close(preparation.flat_inputs, (x,))
