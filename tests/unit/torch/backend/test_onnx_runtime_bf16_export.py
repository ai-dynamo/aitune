# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 graph regressions for the ONNX Runtime backend."""

import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from aitune.records import DType
from aitune.torch.artifact import bounded_tensor_specs
from aitune.torch.backend.onnx_runtime_backend import ONNXRuntimeBackend, ONNXRuntimeBackendConfig
from aitune.torch.libs.onnx.runtime import run_onnx
from aitune.torch.module.sample_store import SampleStore
from tests.utilities.helpers import make_recorded_graph_spec


class Permute(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, x):
        return torch.einsum(self.equation, x)


def _export(module, x, tmp_path):
    sample = ((x,), {})
    graph_spec = make_recorded_graph_spec(module, sample)
    samples = SampleStore.from_samples([sample], tmp_path, "samples")
    backend = ONNXRuntimeBackend()
    backend._set_device(x.device)
    backend._export_onnx(module, graph_spec, samples, tmp_path)
    path = tmp_path / "model_raw.onnx"
    onnx.checker.check_model(str(path), full_check=True)
    return path, graph_spec


@pytest.mark.parametrize("ndim", [2, 3])
def test_bf16_conv_schema(ndim, tmp_path):
    conv = (nn.Conv2d if ndim == 2 else nn.Conv3d)(2, 3, 1).bfloat16()
    x = torch.ones((1, 2, *([4] * ndim)), dtype=torch.bfloat16)
    path, graph_spec = _export(conv, x, tmp_path)
    model = onnx.load(path)
    assert next(op.version for op in model.opset_import if op.domain == "") == 22
    assert "Conv" in {node.op_type for node in model.graph.node}
    assert not any(node.op_type in ("Cast", "CastLike") for node in model.graph.node)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    for kind in ("input", "output"):
        assert bounded_tensor_specs(graph_spec, kind)[0].dtype == DType.BFLOAT16


@pytest.mark.parametrize("equation", ["nchpwq->nhwcpq", "nhwcpq->nchpwq"])
def test_bf16_permutation_einsum_matches_eager(equation, tmp_path):
    module = Permute(equation)
    x = torch.arange(96, dtype=torch.bfloat16).reshape(1, 2, 3, 2, 4, 2)
    path, _ = _export(module, x, tmp_path)
    model = onnx.load(path)
    assert [node.op_type for node in model.graph.node] == ["Transpose"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = module(x)
    result = torch.empty(expected.shape, dtype=x.dtype)
    binding = session.io_binding()
    binding.bind_input(session.get_inputs()[0].name, "cpu", 0, onnx.TensorProto.BFLOAT16, tuple(x.shape), x.data_ptr())
    binding.bind_output(
        session.get_outputs()[0].name, "cpu", 0, onnx.TensorProto.BFLOAT16, tuple(result.shape), result.data_ptr()
    )
    session.run_with_iobinding(binding)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("dynamo", "requested", "dtype", "expected"),
    [
        (True, None, torch.float32, None),
        (True, 20, torch.bfloat16, 20),
        (False, None, torch.bfloat16, None),
        (False, 21, torch.bfloat16, 21),
    ],
)
def test_opset_respects_config(dynamo, requested, dtype, expected):
    module = nn.Identity()
    graph_spec = make_recorded_graph_spec(module, ((torch.zeros(2, dtype=dtype),), {}))
    config = ONNXRuntimeBackendConfig(use_dynamo=dynamo, opset_version=requested)
    assert ONNXRuntimeBackend(config)._export_opset(module, graph_spec) == expected
    assert config.opset_version == requested


def test_bf16_nested_buffer_selects_opset_without_changing_config():
    module = nn.Sequential(nn.Identity())
    module[0].register_buffer("state", torch.ones(1, dtype=torch.bfloat16))
    graph_spec = make_recorded_graph_spec(module, ((torch.ones(2),), {}))
    config = ONNXRuntimeBackendConfig()
    assert ONNXRuntimeBackend(config)._export_opset(module, graph_spec) == 22
    assert config.opset_version is None
    trace = ONNXRuntimeBackend(ONNXRuntimeBackendConfig(use_dynamo=False, opset_version=22))
    with pytest.raises(ValueError, match="use_dynamo=True"):
        trace._export_opset(module, graph_spec)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires BF16 CUDA")
@pytest.mark.parametrize("ndim", [2, 3])
def test_bf16_conv_cuda_matches_eager(ndim, tmp_path):
    conv = (nn.Conv2d if ndim == 2 else nn.Conv3d)(2, 3, 3, padding=1).cuda().bfloat16().eval()
    x = torch.randn((1, 2, *([4] * ndim)), device="cuda", dtype=torch.bfloat16)
    expected = conv(x).detach()
    path, _ = _export(conv, x, tmp_path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
    actual = run_onnx(session, {session.get_inputs()[0].name: x}, x.device)
    torch.testing.assert_close(next(iter(actual.values())), expected, atol=0.02, rtol=0.02)
