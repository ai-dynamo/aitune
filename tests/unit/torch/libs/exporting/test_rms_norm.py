# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RMSNorm ONNX export."""

import onnx
import onnxruntime as ort
import pytest
import torch

from aitune.torch.libs.exporting import ONNXExporter
from aitune.torch.libs.exporting.rms_norm import rms_norm_symbolic
from aitune.torch.utils.tensor import format_tensor_name
from tests.utilities.helpers import make_graph_spec


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape: list[int], elementwise_affine: bool, eps: float | None):
        super().__init__()
        self.norm = torch.nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            torch.nn.init.uniform_(self.norm.weight, 0.5, 1.5)

    def forward(self, x):
        return self.norm(x)


class FusedRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, 16))

    def forward(self, x):
        return torch.ops.aten._fused_rms_norm(x, [16], self.weight, 1e-6)[0]


def _export_and_run(tmp_path, model: torch.nn.Module, x: torch.Tensor, use_dynamo: bool) -> torch.Tensor:
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))
    output_path = ONNXExporter(tmp_path / "model.onnx", use_dynamo=use_dynamo).export(model, sample, graph_spec)
    session = ort.InferenceSession(output_path.as_posix(), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {format_tensor_name("x", "input"): x.numpy()})
    return torch.from_numpy(actual)


@pytest.mark.parametrize("use_dynamo", [False, True], ids=["trace", "dynamo"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    ("normalized_shape", "elementwise_affine", "eps"),
    [([16], True, 1e-6), ([4, 16], False, None)],
    ids=["last-dim-weight", "two-dims-default-eps"],
)
def test_export_matches_torch_rms_norm(tmp_path, use_dynamo, dtype, normalized_shape, elementwise_affine, eps):
    model = RMSNorm(normalized_shape, elementwise_affine, eps).to(dtype).eval()
    x = torch.randn(2, 4, 16, dtype=dtype)

    actual = _export_and_run(tmp_path, model, x, use_dynamo)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, model(x))


@pytest.mark.parametrize(
    ("model", "use_dynamo"),
    [
        (RMSNorm([16], elementwise_affine=True, eps=1e-6), False),
        (RMSNorm([16], elementwise_affine=True, eps=1e-6), True),
        pytest.param(
            FusedRMSNorm(),
            True,
            marks=pytest.mark.skipif(
                not hasattr(torch.ops.aten, "_fused_rms_norm"), reason="Requires aten._fused_rms_norm"
            ),
        ),
    ],
    ids=["trace", "dynamo", "dynamo-fused"],
)
def test_export_keeps_bfloat16_output(tmp_path, model, use_dynamo):
    """ONNX Runtime cannot take BF16 inputs from NumPy, so check only the exported output type."""
    model = model.to(torch.bfloat16).eval()
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    sample = ((x,), {})
    graph_spec = make_graph_spec(model.forward, sample, model(x))

    output_path = ONNXExporter(tmp_path / "model.onnx", use_dynamo=use_dynamo).export(model, sample, graph_spec)

    (output,) = onnx.load(output_path).graph.output
    assert output.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16


@pytest.mark.skipif(not hasattr(torch.ops.aten, "_fused_rms_norm"), reason="Requires aten._fused_rms_norm")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_dynamo_export_matches_torch_fused_rms_norm(tmp_path, dtype):
    model = FusedRMSNorm().to(dtype).eval()
    x = torch.randn(2, 4, 16, dtype=dtype)

    actual = _export_and_run(tmp_path, model, x, use_dynamo=True)

    torch.testing.assert_close(actual, model(x))


def test_rms_norm_symbolic_is_removed_after_block(tmp_path):
    model = RMSNorm([16], elementwise_affine=True, eps=1e-6).eval()
    x = torch.randn(2, 16)

    with rms_norm_symbolic():
        torch.onnx.export(model, (x,), (tmp_path / "inside.onnx").as_posix(), dynamo=False)

    # Torch keeps an empty function group after unregistering, so the lookup can fail with ValueError.
    with pytest.raises((torch.onnx.errors.UnsupportedOperatorError, ValueError)):
        torch.onnx.export(model, (x,), (tmp_path / "outside.onnx").as_posix(), dynamo=False)
