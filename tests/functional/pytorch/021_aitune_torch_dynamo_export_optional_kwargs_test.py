# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = []
#
# # Optional, default "always", determines how often test is generated, always, nightly, weekly, monthly
# scope = "always"
# ///

from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from aitune.torch.libs.onnx.onnx_exporter import ONNXExporter
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata

logger = getLogger(Path(__file__).stem)


class ModelWithOptionalKwargs(nn.Module):
    """Test model with optional kwargs including None defaults."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x, optional_flag=None, optional_value=None):
        """Forward with optional kwargs.

        Args:
            x: Input tensor
            optional_flag: Optional flag (None or tensor)
            optional_value: Optional value (None or tensor)
        """
        out = self.linear(x)
        if optional_flag is not None:
            out = out * optional_flag
        if optional_value is not None:
            out = out + optional_value
        return out


def test_export_with_none_kwargs(tmp_path: Path, device: str = "cpu"):
    """Test dynamo export with None-valued kwargs - should filter them out."""
    logger.info("Test 1: Export with None kwargs")

    model = ModelWithOptionalKwargs().eval().to(device)
    x = torch.randn(2, 10, device=device)

    # Sample with None kwargs
    sample = ((x,), {})

    # Run model to get output
    args, kwargs = sample
    output = model(*args, **kwargs)

    # Create graph spec
    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    # Export with dynamo
    output_path = tmp_path / "model_none_kwargs.onnx"
    exporter = ONNXExporter(output_path=output_path, use_dynamo=True)
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    # Should only have 1 input (x), None kwargs should be filtered out
    assert len(onnx_model.graph.input) == 1, f"Expected 1 input, got {len(onnx_model.graph.input)}"
    assert onnx_model.graph.input[0].name == "args_0"

    logger.info("✓ Test 1 passed: None kwargs filtered correctly")


def test_export_with_kwargs_wrong_order(tmp_path: Path, device: str = "cpu"):
    """Test dynamo export with kwargs in wrong order - should reorder to match forward signature."""
    logger.info("Test 2: Export with kwargs in wrong order")

    model = ModelWithOptionalKwargs().eval().to(device)
    x = torch.randn(2, 10, device=device)
    flag = torch.tensor(2.0, device=device)
    value = torch.tensor(1.0, device=device)

    # Sample with kwargs in reverse order (wrong order)
    sample = ((x,), {"optional_value": value, "optional_flag": flag})

    # Run model to get output
    args, kwargs = sample
    output = model(*args, **kwargs)

    # Create graph spec
    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    # Export with dynamo
    output_path = tmp_path / "model_wrong_order.onnx"
    exporter = ONNXExporter(output_path=output_path, use_dynamo=True)
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    # Should have 3 inputs in correct order: x, optional_flag, optional_value
    assert len(onnx_model.graph.input) == 3, f"Expected 3 inputs, got {len(onnx_model.graph.input)}"
    assert onnx_model.graph.input[0].name == "args_0"  # x
    assert onnx_model.graph.input[1].name == "kwargs_optional_value"
    assert onnx_model.graph.input[2].name == "kwargs_optional_flag"

    logger.info("✓ Test 2 passed: Kwargs reordered correctly")


def test_export_with_mixed_none_and_non_none_kwargs(tmp_path: Path, device: str = "cpu"):
    """Test dynamo export with mixed None and non-None kwargs."""
    logger.info("Test 3: Export with mixed None and non-None kwargs")

    model = ModelWithOptionalKwargs().eval().to(device)
    x = torch.randn(2, 10, device=device)
    flag = torch.tensor(2.0, device=device)

    # Sample with one None and one non-None kwarg
    sample = ((x,), {"optional_flag": flag, "optional_value": None})

    # Run model to get output
    args, kwargs = sample
    output = model(*args, **kwargs)

    # Create graph spec
    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    # Export with dynamo
    output_path = tmp_path / "model_mixed_kwargs.onnx"
    exporter = ONNXExporter(output_path=output_path, use_dynamo=True)
    onnx_path = exporter.export(module=model, sample=sample, graph_spec=graph_spec)

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    # Should have 2 inputs: x and optional_flag (optional_value=None should be filtered)
    assert len(onnx_model.graph.input) == 2, f"Expected 2 inputs, got {len(onnx_model.graph.input)}"
    assert onnx_model.graph.input[0].name == "args_0"  # x
    assert onnx_model.graph.input[1].name == "kwargs_optional_flag"  # optional_flag

    logger.info("✓ Test 3 passed: Mixed None and non-None kwargs handled correctly")


if __name__ == "__main__":
    basicConfig(level=DEBUG, force=True)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        test_export_with_none_kwargs(tmp_path, device="cuda")
        test_export_with_kwargs_wrong_order(tmp_path, device="cuda")
        test_export_with_mixed_none_and_non_none_kwargs(tmp_path, device="cuda")

    logger.info("All tests passed! ✓")
