# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnxruntime
import pytest
import torch
from onnx import TensorProto, helper

from aitune.torch.libs.onnx.runtime import run_onnx


def _cast_session(input_type, output_type, provider):
    graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["y"], to=output_type)],
        "cast",
        [helper.make_tensor_value_info("x", input_type, [2, 2])],
        [helper.make_tensor_value_info("y", output_type, [2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    return onnxruntime.InferenceSession(model.SerializeToString(), providers=[provider])


@pytest.mark.parametrize(
    ("dtype", "element_type"),
    [(torch.bfloat16, TensorProto.BFLOAT16), (torch.float16, TensorProto.FLOAT16), (torch.float32, TensorProto.FLOAT)],
)
def test_bind_floating_point_input(dtype, element_type):
    # An FP32 output isolates input binding from BF16 output collection.
    session = _cast_session(element_type, TensorProto.FLOAT, "CPUExecutionProvider")
    x = torch.tensor([[1.5, -2.25], [3.75, 0.5]], dtype=dtype).T
    y = run_onnx(session, {"x": x}, x.device)["y"]
    torch.testing.assert_close(y, x.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_collect_bfloat16_cuda_output():
    session = _cast_session(TensorProto.FLOAT, TensorProto.BFLOAT16, "CUDAExecutionProvider")
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    # Signed zero and values outside FP16's range distinguish a bit-exact BF16 copy from an FP16 round trip.
    x = torch.tensor([[-0.0, 1e20], [1e-20, -1.5]], dtype=torch.float32, device="cuda")
    y = run_onnx(session, {"x": x}, x.device)["y"]
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.view(torch.int16), x.bfloat16().view(torch.int16), rtol=0, atol=0)
