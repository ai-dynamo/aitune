# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ONNXModelInfo and ONNXPrecision."""

from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

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
    val.type.tensor_type.elem_type = TensorProto.FLOAT

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
    entry.domain = ""
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


def test_graph_metadata_preserves_operator_domains_and_declared_io(tmp_path):
    path = tmp_path / "custom.onnx"
    graph = helper.make_graph(
        [helper.make_node("CustomOp", ["x"], ["y"], domain="example")],
        "custom",
        [helper.make_tensor_value_info("x", TensorProto.INT64, [None, "length"])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [None, "length"])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("example", 1), helper.make_opsetid("", 17)]),
        path,
    )

    info = ONNXModelInfo(path)
    assert info.opsets == {"example": 1, "": 17}
    assert info.opset_version == 17
    assert info.operators == {"example::CustomOp"}
    assert info.input_dtypes == {"x": "INT64"}
    assert info.output_dtypes == {"y": "INT64"}
    assert info.output_shapes == {"y": [None, "length"]}


def test_standard_domain_alias_is_normalized(tmp_path):
    path = tmp_path / "standard-alias.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"], domain="ai.onnx")],
        "double",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("ai.onnx", 17)], ir_version=8), path)

    info = ONNXModelInfo(path)
    assert info.opsets == {"": 17}
    assert info.opset_version == 17
    assert info.operators == {"Add"}


@pytest.mark.parametrize("alias_version", [17, 18])
def test_both_standard_domain_spellings(tmp_path, alias_version):
    path = tmp_path / "both-standard-domains.onnx"
    graph = helper.make_graph([], "empty", [], [])
    onnx.save(
        helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx", alias_version)],
            ir_version=8,
        ),
        path,
    )

    if alias_version == 17:
        info = ONNXModelInfo(path)
        assert info.opsets == {"": 17}
        assert info.opset_version == 17
    else:
        with pytest.raises(ValueError, match="Conflicting opset versions for ONNX domain ''"):
            ONNXModelInfo(path)


def test_operators_include_nested_graphs(tmp_path):
    path = tmp_path / "nested.onnx"
    branch = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "branch", [], [])
    graph = helper.make_graph([helper.make_node("If", ["condition"], ["y"], then_branch=branch)], "nested", [], [])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    assert ONNXModelInfo(path).operators == {"If", "Relu"}


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
def test_input_shapes_static_dims(mock_load, tmp_path):
    inputs = [_make_value_info("x", [2, 3, 224, 224])]
    mock_load.return_value = _make_model(input_specs=inputs)
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.input_shapes == {"x": [2, 3, 224, 224]}


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_input_shapes_unknown_dim(mock_load, tmp_path):
    inputs = [_make_value_info("x", [None, 3])]
    mock_load.return_value = _make_model(input_specs=inputs)
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.input_shapes == {"x": [None, 3]}


# ---------------------------------------------------------------------------
# ONNXModelInfo — opset edge cases
# ---------------------------------------------------------------------------


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_opset_version_none_when_no_opset(mock_load, tmp_path):
    model = _make_model()
    model.opset_import = []
    mock_load.return_value = model
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.opset_version is None


# ---------------------------------------------------------------------------
# ONNXModelInfo — precision detection
# ---------------------------------------------------------------------------


def test_precision_fp32_model_returns_none():
    # toy_linear.onnx uses FP32 weights — no quantization precision
    info = ONNXModelInfo(ToyOnnxModel().path)
    assert info.precision is None


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_fp16_model(mock_load, tmp_path):
    inits = [_make_initializer(10)] * 5  # dtype 10 = FLOAT16
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.precision == ONNXPrecision.FP16


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_int8_model(mock_load, tmp_path):
    inits = [_make_initializer(3)] * 10  # dtype 3 = INT8
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.precision == ONNXPrecision.INT8


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_lowest_wins(mock_load, tmp_path):
    # FP16 + INT8 present → INT8 is lower precision, should be reported
    inits = [_make_initializer(10)] * 10 + [_make_initializer(3)] * 3
    mock_load.return_value = _make_model(initializers=inits)
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.precision == ONNXPrecision.INT8


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_precision_no_initializers_returns_none(mock_load, tmp_path):
    mock_load.return_value = _make_model(initializers=[])
    info = ONNXModelInfo(tmp_path / "model.onnx")
    assert info.precision is None


# ---------------------------------------------------------------------------
# ONNXModelInfo — error handling
# ---------------------------------------------------------------------------


def test_empty_source_is_rejected(tmp_path):
    path = tmp_path / "empty.onnx"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="does not contain a graph"):
        ONNXModelInfo(path)


def test_graph_metadata_does_not_load_external_weights(tmp_path):
    path = tmp_path / "external.onnx"
    weights = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="weights")
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weights"], ["y"])],
        "external",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        [weights],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    (tmp_path / "weights.data").unlink()

    info = ONNXModelInfo(path)
    assert info.operators == {"Add"}
    assert info.input_dtypes == {"x": "FLOAT"}


def test_import_error_propagates(monkeypatch, tmp_path):
    import aitune.torch.libs.onnx.onnx_model_info as module_under_test

    monkeypatch.setattr(module_under_test.onnx, "load", MagicMock(side_effect=ImportError("no onnx")))
    with pytest.raises(ImportError):
        ONNXModelInfo(tmp_path / "model.onnx")


@patch("aitune.torch.libs.onnx.onnx_model_info.onnx.load")
def test_load_error_propagates(mock_load, tmp_path):
    mock_load.side_effect = RuntimeError("corrupt model")
    with pytest.raises(RuntimeError, match="corrupt model"):
        ONNXModelInfo(tmp_path / "model.onnx")
