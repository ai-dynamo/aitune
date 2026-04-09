# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import aitune.utils.monitoring.parse_metrics as parse_metrics
from aitune.utils.monitoring.parse_metrics import format_backend_label_for_display


def test_parse_memory_metrics():
    """Test parse_memory_metrics aggregates max memory per module and backend in GB."""
    df = pd.DataFrame({
        "module_name": ["a", "a", "b"],
        "backend": ["tensorrt", "tensorrt", "tensorrt"],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3, 512 * 1024**2],
    })
    result = parse_metrics.parse_memory_metrics(df)
    expected = pd.DataFrame(
        {
            "Cuda:0\nMem [GB]": [2.0, 0.5],
        },
        index=pd.MultiIndex.from_tuples([("a", "tensorrt"), ("b", "tensorrt")], names=["module_name", "backend"]),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_parse_memory_metrics_no_columns():
    """Test parse_memory_metrics returns empty-frame MultiIndex when no memory columns."""
    df = pd.DataFrame({"module_name": ["a", "b"], "backend": ["x", "y"], "other": [1, 2]})
    result = parse_metrics.parse_memory_metrics(df)
    assert result.index.tolist() == [("a", "x"), ("b", "y")]
    assert len(result.columns) == 0


def test_parse_utilization_metrics():
    """Test parse_utilization_metrics aggregates mean and max utilization per module and backend."""
    df = pd.DataFrame({
        "module_name": ["a", "a", "b"],
        "backend": ["tensorrt", "tensorrt", "tensorrt"],
        "cuda:0_utilization": [10.0, 50.0, 80.0],
    })
    result = parse_metrics.parse_utilization_metrics(df)
    assert result.index.tolist() == [("a", "tensorrt"), ("b", "tensorrt")]
    assert "mean" in str(result.columns[0]).lower() or "Cuda" in str(result.columns[0])
    assert result.loc[("a", "tensorrt")].iloc[0] == 30.0  # mean of 10 and 50
    assert result.loc[("a", "tensorrt")].iloc[1] == 50.0  # max
    assert result.loc[("b", "tensorrt")].iloc[1] == 80.0


def test_parse_utilization_metrics_no_columns():
    """Test parse_utilization_metrics returns empty-frame MultiIndex when no utilization columns."""
    df = pd.DataFrame({"module_name": ["a"], "backend": ["tensorrt"], "other": [1]})
    result = parse_metrics.parse_utilization_metrics(df)
    assert result.index.tolist() == [("a", "tensorrt")]
    assert len(result.columns) == 0


def test_parse_power_metrics():
    """Test parse_power_metrics aggregates mean and max power in watts."""
    df = pd.DataFrame({
        "module_name": ["a", "a", "b"],
        "backend": ["tensorrt", "tensorrt", "tensorrt"],
        "cuda:0_power_usage_milliwatts": [100_000, 200_000, 150_000],
    })
    result = parse_metrics.parse_power_metrics(df)
    assert result.index.tolist() == [("a", "tensorrt"), ("b", "tensorrt")]
    # a: mean 150W, max 200W; b: 150W for both
    assert result.loc[("a", "tensorrt")].tolist() == [150.0, 200.0]
    assert result.loc[("b", "tensorrt")].tolist() == [150.0, 150.0]


def test_parse_power_metrics_no_columns():
    """Test parse_power_metrics returns empty-frame MultiIndex when no power columns."""
    df = pd.DataFrame({"module_name": ["a"], "backend": ["tensorrt"], "other": [1]})
    result = parse_metrics.parse_power_metrics(df)
    assert result.index.tolist() == [("a", "tensorrt")]
    assert len(result.columns) == 0


def test_get_metrics_summary_combines_all():
    """Test get_metrics_summary concatenates memory, utilization, and power summaries."""
    df = pd.DataFrame({
        "module_name": ["m1", "m1", "m2"],
        "backend": ["tensorrt", "tensorrt", "tensorrt"],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3, 1024**3],
        "cuda:0_utilization": [50.0, 70.0, 80.0],
        "cuda:0_power_usage_milliwatts": [100_000, 120_000, 90_000],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is not None
    assert result.index.names == ["Module", "Backend"]
    assert result.index.tolist() == [("m1", "tensorrt"), ("m2", "tensorrt")]
    # m1 max memory 2 GB
    assert result.loc[("m1", "tensorrt")].iloc[0] == 2.0
    # m1 utilization mean 60, max 70
    assert result.loc[("m1", "tensorrt")].iloc[2] == 70.0
    # m1 power mean 110W, max 120W
    assert result.loc[("m1", "tensorrt")].iloc[4] == 120.0


def test_get_metrics_summary_index_names():
    """Test get_metrics_summary renames MultiIndex levels to Module and Backend."""
    df = pd.DataFrame({
        "module_name": ["ModA", "ModA"],
        "backend": ["BackendX", "BackendX"],
        "cpu_memory_used": [1024**3, 2 * 1024**3],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is not None
    assert result.index.names == ["Module", "Backend"]


def test_get_metrics_summary_empty_returns_none():
    """Test get_metrics_summary returns None for empty DataFrame."""
    assert parse_metrics.get_metrics_summary(pd.DataFrame()) is None


def test_get_metrics_summary_no_module_name_returns_none():
    """Test get_metrics_summary returns None when module_name column is missing."""
    df = pd.DataFrame({"other": [1, 2]})
    assert parse_metrics.get_metrics_summary(df) is None


def test_get_metrics_summary_none_module_name_drops_rows():
    """Test get_metrics_summary drops rows where module_name is None."""
    df = pd.DataFrame({
        "module_name": [None, None],
        "backend": ["tensorrt", "tensorrt"],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3],
        "cuda:0_utilization": [50.0, 70.0],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is None


def test_get_metrics_summary_none_backend_drops_rows():
    """Test get_metrics_summary drops rows where backend is None."""
    df = pd.DataFrame({
        "module_name": ["m1", "m1"],
        "backend": [None, None],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is None


def test_get_metrics_summary_partial_none_backend_keeps_valid_rows():
    """Test get_metrics_summary drops only rows where backend is None, keeps the rest."""
    df = pd.DataFrame({
        "module_name": ["m1", "m1", "m1"],
        "backend": ["tensorrt", None, "tensorrt"],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3, 3 * 1024**3],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is not None
    assert result.index.tolist() == [("m1", "tensorrt")]
    assert result.iloc[0, 0] == 3.0  # max of 1GB and 3GB (None row dropped)


def test_format_backend_label_for_display_no_params():
    """Test that labels with empty parentheses are returned unchanged."""
    assert format_backend_label_for_display("TorchEagerBackend()") == "TorchEagerBackend()"


def test_format_backend_label_for_display_single_param():
    """Test single-param label is formatted to multi-line."""
    result = format_backend_label_for_display("TorchInductorBackend(mode=max-autotune-no-cudagraphs)")
    assert result == "TorchInductorBackend(\n    mode=max-autotune-no-cudagraphs\n)"


def test_format_backend_label_for_display_multiple_params():
    """Test multi-param label puts each parameter on its own indented line."""
    result = format_backend_label_for_display("TorchInductorBackend(mode=max-autotune,dynamic=True)")
    assert result == "TorchInductorBackend(\n    mode=max-autotune,\n    dynamic=True\n)"


def test_format_backend_label_for_display_nested_params():
    """Test that nested config objects are formatted recursively with increased indentation."""
    result = format_backend_label_for_display(
        "TensorRTBackend(use_dynamo=False,quantization_config=ONNXQuantizationConfig(precision='int8',calibration_method='max'))"
    )
    assert result == (
        "TensorRTBackend(\n"
        "    use_dynamo=False,\n"
        "    quantization_config=ONNXQuantizationConfig(\n"
        "        precision='int8',\n"
        "        calibration_method='max'\n"
        "    )\n"
        ")"
    )


def test_format_backend_label_for_display_empty_nested_params():
    """Test that nested config objects with no params stay on one line."""
    result = format_backend_label_for_display("TorchAOBackend(quantization_config=Int8WeightOnlyConfig())")
    assert result == ("TorchAOBackend(\n    quantization_config=Int8WeightOnlyConfig()\n)")


def test_format_backend_label_for_display_no_parentheses():
    """Test that labels without parentheses are returned unchanged."""
    assert format_backend_label_for_display("DummyBackend") == "DummyBackend"


def test_get_metrics_summary_partial_none_module_name_keeps_valid_rows():
    """Test get_metrics_summary drops only rows where module_name is None, keeps the rest."""
    df = pd.DataFrame({
        "module_name": ["m1", None, "m1"],
        "backend": ["tensorrt", None, "tensorrt"],
        "cuda:0_memory_used": [1024**3, 2 * 1024**3, 3 * 1024**3],
    })
    result = parse_metrics.get_metrics_summary(df)
    assert result is not None
    assert result.index.tolist() == [("m1", "tensorrt")]
    assert result.iloc[0, 0] == 3.0  # max of 1GB and 3GB
