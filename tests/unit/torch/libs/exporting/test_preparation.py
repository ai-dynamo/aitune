# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for exporter-neutral tensor preparation."""

import torch

from aitune.torch.libs.exporting import prepare_tensor_export
from aitune.torch.module.recording_module import RecordingModule
from tests.toy_models import TOY_EXPORT_MODELS


def test_prepare_moves_nested_tensor_leaves_to_device(mocker):
    model = dict(TOY_EXPORT_MODELS)["complex"]().eval()
    sample = model.samples(batch_sizes=[2])[0]
    graph_spec = model.graph_spec(batch_sizes=[2])
    model.to("meta")

    preparation = prepare_tensor_export(model, sample, graph_spec, device="meta")

    assert preparation.flat_inputs
    assert preparation.adapter.training is model.training
    assert all(tensor.device.type == "meta" for tensor in preparation.flat_inputs)

    forward = mocker.spy(model, "forward")
    preparation.adapter(*preparation.flat_inputs)

    forward.assert_called_once()
    args, kwargs = forward.call_args
    assert len(args) == len(sample[0]) == 2
    assert kwargs.keys() == sample[1].keys()
    assert args[0].device.type == "meta"
    assert args[1][0].device.type == "meta"
    assert kwargs["options"]["residuals"][0].device.type == "meta"
    assert kwargs["scale"].device.type == "meta"
    assert args[1][1] is None
    assert kwargs["options"]["residuals"][1] is None
    assert kwargs["options"]["metadata"] == {"mode": "add"}


def test_prepare_preserves_distributed_tensor_placement(mocker):
    model = dict(TOY_EXPORT_MODELS)["simple"]().eval()
    sample = model.samples(batch_sizes=[2])[0]
    graph_spec = model.graph_spec(batch_sizes=[2])
    tensor = sample[0][0]
    mocker.patch("aitune.torch.utils.module.DTensor", torch.Tensor)

    preparation = prepare_tensor_export(model, sample, graph_spec, device="meta")

    prepared_tensor = preparation.flat_inputs[0]
    assert prepared_tensor.device == tensor.device
    assert prepared_tensor is not tensor


def test_prepare_uses_recorded_graph_spec_without_calling_module(tmp_path):
    class CountingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def forward(self, x):
            self.call_count += 1
            return x + 1

    model = CountingModule()
    recording = RecordingModule(model, "counting", cache_dir_resolver=lambda: tmp_path)
    x = torch.ones(2)
    recording(x)

    graph_spec = recording.graph_specs[0]
    sample = recording.samples_for_graph_spec(graph_spec)[0]
    preparation = prepare_tensor_export(model, sample, graph_spec)

    assert model.call_count == 1
    torch.testing.assert_close(preparation.flat_inputs, (x,))
