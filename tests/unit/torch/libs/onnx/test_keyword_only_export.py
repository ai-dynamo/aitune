# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real ONNX tracing of the keyword-only call contract required by NeMo."""

from typing import Any, cast

import onnxruntime as ort
import pytest
import torch
import wrapt

from aitune.torch.libs.exporting import ONNXExporter as TensorONNXExporter
from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
from aitune.torch.utils.tensor import format_tensor_name
from tests.utilities.helpers import make_graph_spec


@wrapt.decorator
def typed_forward(wrapped, instance, args, kwargs):
    # Match NeMo's public call restriction without requiring its model weights or CUDA.
    if args:
        raise TypeError("All arguments must be passed by kwargs only for typed methods")
    return wrapped(**kwargs)


class SpeechModel(torch.nn.Module):
    # wrapt preserves the wrapped method signature and binding at runtime.
    @cast(Any, typed_forward)
    def forward(
        self,
        input_signal: torch.Tensor | None = None,
        input_signal_length: torch.Tensor | None = None,
        processed_signal: torch.Tensor | None = None,
        processed_signal_length: torch.Tensor | None = None,
    ):
        if processed_signal is not None:
            return processed_signal + 1, processed_signal_length
        if input_signal is None:
            raise ValueError("An input signal is required")
        return input_signal + 1, input_signal_length


@pytest.mark.parametrize("exporter_class", [ONNXExporter, TensorONNXExporter], ids=["legacy", "flat-tensor"])
@pytest.mark.parametrize("signal_kind", ["input", "processed"])
def test_trace_export_preserves_keyword_only_calls(tmp_path, exporter_class, signal_kind):
    model = SpeechModel().eval()
    inputs = {
        f"{signal_kind}_signal": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        f"{signal_kind}_signal_length": torch.tensor([3, 3]),
    }
    expected = model(**inputs)
    sample = ((), inputs)
    graph_spec = make_graph_spec(model.forward, sample, expected)
    output_path = tmp_path / "speech.onnx"

    exporter_class(output_path, use_dynamo=False, opset_version=17).export(model, sample, graph_spec)

    # Validate the exported product, including signal lengths, through a real CPU runtime.
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {format_tensor_name(name, "input"): value.numpy() for name, value in inputs.items()})
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(output), reference)
