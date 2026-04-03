# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ONNXModelInfo and ONNXPrecision."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aitune.torch.libs.onnx.onnx_model_info import ONNX_DTYPE_TO_PRECISION, ONNXModelInfo, ONNXPrecision
from tests.toy_models.onnx_models import ToyOnnxModel

# ---------------------------------------------------------------------------
# ONNXPrecision enum
# ---------------------------------------------------------------------------


def test_precision_enum_values():
    assert ONNXPrecision.FP16.value == "fp16"
    assert ONNXPrecision.INT8.value == "int8"
    assert ONNXPrecision.FP8.value == "fp8"
    assert ONNXPrecision.INT4.value == "int4"


def test_precision_enum_is_str():
    assert isinstance(ONNXPrecision.FP16, str)


# ---------------------------------------------------------------------------
# ONNX_DTYPE_TO_PRECISION mapping
# ---------------------------------------------------------------------------


def test_dtype_to_precision_keys():
    assert ONNX_DTYPE_TO_PRECISION[10] == ONNXPrecision.FP16  # FLOAT16
    assert ONNX_DTYPE_TO_PRECISION[3] == ONNXPrecision.INT8  # INT8
    assert ONNX_DTYPE_TO_PRECISION[17] == ONNXPrecision.FP8  # FLOAT8E4M3FN
    assert ONNX_DTYPE_TO_PRECISION[22] == ONNXPrecision.INT4  # INT4


def test_dtype_to_precision_fp32_not_present():
    # dtype 1 is FLOAT (FP32) — should NOT be in the quantization map
    assert 1 not in ONNX_DTYPE_TO_PRECISION


# ---------------------------------------------------------------------------
# Helpers for building mock ONNX model protos (used for edge cases only)
# ---------------------------------------------------------------------------


def _make_value_info(name: str, dims):
    """Build a mock ValueInfoProto with the given tensor dims.

    Each element of dims can be:
      - int   → static dim_value
      - str   → symbolic dim_param
      - None  → unknown dim (neither field set)
    """
    val = MagicMock()
    val.name = name
    val.type.HasField.return_value = True  # tensor_type and shape are present

    mock_dims = []
    for d in dims:
        dim = MagicMock()
        if isinstance(d, int):
            dim.HasField.side_effect = lambda field, _d=d: field == "dim_value"
            dim.dim_value = d
        elif isinstance(d, str):
            dim.HasField.side_effect = lambda field, _s=d: field == "dim_param"
            dim.dim_param = d
        else:
            dim.HasField.return_value = False
        mock_dims.append(dim)

    val.type.tensor_type.shape.dim = mock_dims
    return val


def _make_initializer(data_type: int):
    init = MagicMock()
    init.data_type = data_type
    return init


def _make_opset_entry(version: int):
    entry = MagicMock()
    entry.version = version
    return entry


def _make_model(
    input_specs=None,
    output_specs=None,
    initializers=None,
    opset_versions=None,
    producer_name="",
    producer_version="",
    model_version=0,
    doc_string="",
):
    model = MagicMock()
    model.graph.input = input_specs or []
    model.graph.output = output_specs or []
    model.graph.initializer = initializers or []
    model.opset_import = [_make_opset_entry(v) for v in (opset_versions or [17])]
    model.producer_name = producer_name
    model.producer_version = producer_version
    model.model_version = model_version
    model.doc_string = doc_string
    return model


# ---------------------------------------------------------------------------
# ONNXModelInfo — real model tests (toy_linear.onnx)
# ---------------------------------------------------------------------------


def test_model_path_property():
    path = ToyOnnxModel().path
    info = ONNXModelInfo(path)
    assert info.model_path == path


def test_input_names():
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.input_names == ["input"]


def test_output_names():
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.output_names == ["output"]


def test_opset_version():
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.opset_version == 12


def test_producer_info():
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.producer_name == "pytorch"
    assert info.producer_version == "2.7.0"


def test_model_version_and_doc_string():
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.model_version == 0
    assert info.doc_string == ""


# ---------------------------------------------------------------------------
# ONNXModelInfo — input_shapes (real models)
# ---------------------------------------------------------------------------


def test_input_shapes_dynamic_batch_dim():
    # toy_linear.onnx has input shape [batch_size, 256]
    info = ONNXModelInfo(ToyOnnxModel(is_linear=True).path)
    assert info.input_shapes == {"input": ["batch_size", 256]}


def test_input_shapes_mixed_dims():
    # toy_conv.onnx has input shape [batch_size, 1, 129, 129]
    info = ONNXModelInfo(ToyOnnxModel(is_linear=False).path)
    assert info.input_shapes == {"input": ["batch_size", 1, 129, 129]}


# ---------------------------------------------------------------------------
# ONNXModelInfo — input_shapes (edge cases via mock)
# ---------------------------------------------------------------------------


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_input_shapes_static_dims(mock_load):
    inputs = [_make_value_info("x", [2, 3, 224, 224])]
    mock_load.return_value = _make_model(input_specs=inputs)
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.input_shapes == {"x": [2, 3, 224, 224]}


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_input_shapes_unknown_dim(mock_load):
    inputs = [_make_value_info("x", [None, 3])]
    mock_load.return_value = _make_model(input_specs=inputs)
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.input_shapes == {"x": [None, 3]}


# ---------------------------------------------------------------------------
# ONNXModelInfo — opset edge cases
# ---------------------------------------------------------------------------


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_opset_version_none_when_no_opset(mock_load):
    model = _make_model()
    model.opset_import = []
    mock_load.return_value = model
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.opset_version is None


# ---------------------------------------------------------------------------
# ONNXModelInfo — precision detection
# ---------------------------------------------------------------------------


def test_precision_fp32_model_returns_none():
    # toy_linear.onnx uses FP32 weights — no quantization precision
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.precision is None


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_fp16_model(mock_load):
    inits = [_make_initializer(10)] * 5  # dtype 10 = FLOAT16
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.precision == ONNXPrecision.FP16


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_int8_model(mock_load):
    inits = [_make_initializer(3)] * 10  # dtype 3 = INT8
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.precision == ONNXPrecision.INT8


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_lowest_wins(mock_load):
    # FP16 + INT8 present → INT8 is lower precision, should be reported
    inits = [_make_initializer(10)] * 10 + [_make_initializer(3)] * 3
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.precision == ONNXPrecision.INT8


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_no_initializers_returns_none(mock_load):
    mock_load.return_value = _make_model(initializers=[])
    info = ONNXModelInfo(Path("/tmp/model.onnx"))
    assert info.precision is None


# ---------------------------------------------------------------------------
# ONNXModelInfo — error handling
# ---------------------------------------------------------------------------


def test_import_error_propagates(monkeypatch):
    import aitune.torch.libs.onnx.onnx_model_info as module_under_test

    monkeypatch.setattr(module_under_test.onnx, "load", MagicMock(side_effect=ImportError("no onnx")))
    with pytest.raises(ImportError):
        ONNXModelInfo(Path("/tmp/model.onnx"))


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_load_error_propagates(mock_load):
    mock_load.side_effect = RuntimeError("corrupt model")
    with pytest.raises(RuntimeError, match="corrupt model"):
        ONNXModelInfo(Path("/tmp/model.onnx"))
