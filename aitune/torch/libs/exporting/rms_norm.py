# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export RMSNorm to ONNX as standard operators.

Torch has no ONNX translation for RMSNorm below opset 23. The legacy tracer rejects ``aten::rms_norm``,
and Dynamo export on CUDA keeps ``aten._fused_rms_norm``, which has no ONNX translation in Torch 2.14.
Both translations follow Torch's decomposition: they compute in float32, or in float64 for float64 inputs,
and cast the result back to the input type.
"""

import contextlib
from collections.abc import Callable, Generator, Sequence
from typing import Any

import onnx_ir as ir
import onnxscript
import torch
from torch.onnx import symbolic_helper

# ReduceMean takes axes as an input from opset 18. The legacy exporter uses this function for opsets 18 to 20.
_RMS_NORM_SYMBOLIC_OPSET = 18

# onnxscript types its operators for script mode, but the Dynamo exporter traces them with ONNX IR values.
op: Any = onnxscript.opset18


def _epsilon(eps: float | None, is_double: bool) -> float:
    """Return Torch's default epsilon for the computation type if the call has none."""
    if eps is not None:
        return eps
    return torch.finfo(torch.float64 if is_double else torch.float32).eps


@symbolic_helper.parse_args("v", "is", "v", "f")
def _rms_norm_symbolic(g, input, normalized_shape: list[int], weight, eps: float | None):  # noqa: A002
    """Build RMSNorm for the legacy ONNX tracer."""
    is_double = input.type().scalarType() == "Double"
    computation_type = ir.DataType.DOUBLE if is_double else ir.DataType.FLOAT
    x = g.op("Cast", input, to_i=computation_type)
    axes = g.op("Constant", value_t=torch.tensor(range(-len(normalized_shape), 0)))
    mean_square = g.op("ReduceMean", g.op("Mul", x, x), axes, keepdims_i=1)
    epsilon = g.op(
        "Constant", value_t=torch.tensor(_epsilon(eps, is_double), dtype=torch.float64 if is_double else torch.float32)
    )
    result = g.op("Mul", x, g.op("Reciprocal", g.op("Sqrt", g.op("Add", mean_square, epsilon))))
    if not weight.node().mustBeNone():
        result = g.op("Mul", result, g.op("Cast", weight, to_i=computation_type))
    return g.op("CastLike", result, input)


@contextlib.contextmanager
def rms_norm_symbolic() -> Generator[None, None, None]:
    """Register the RMSNorm translation with the legacy ONNX tracer for the duration of the block."""
    torch.onnx.register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, _RMS_NORM_SYMBOLIC_OPSET)
    try:
        yield
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::rms_norm", _RMS_NORM_SYMBOLIC_OPSET)


def _fused_rms_norm(input, normalized_shape: Sequence[int], weight=None, eps: float | None = None):  # noqa: A002
    """Build ``aten._fused_rms_norm`` and its inverse RMS output for the Dynamo exporter."""
    is_double = input.dtype == ir.DataType.DOUBLE
    computation_type = ir.DataType.DOUBLE if is_double else ir.DataType.FLOAT
    x = op.Cast(input, to=computation_type)
    axes = op.Constant(value_ints=list(range(-len(normalized_shape), 0)))
    mean_square = op.ReduceMean(op.Mul(x, x), axes, keepdims=1)
    epsilon = op.Constant(value=ir.tensor(_epsilon(eps, is_double), dtype=computation_type))
    inverse_rms = op.Reciprocal(op.Sqrt(op.Add(mean_square, epsilon)))
    result = op.Mul(x, inverse_rms)
    if weight is not None:
        result = op.Mul(result, op.Cast(weight, to=computation_type))
    return op.CastLike(result, input), inverse_rms


def rms_norm_translations() -> dict[Callable, Callable]:
    """Return Dynamo custom translations for the RMSNorm operators that this Torch version has."""
    if not hasattr(torch.ops.aten, "_fused_rms_norm"):
        return {}
    return {torch.ops.aten._fused_rms_norm.default: _fused_rms_norm}
