# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export modules to ONNX through a flat tensor interface."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import cast

import onnx
import torch
import torch.nn as nn

from aitune.torch.dynamic_shapes import DynamicDim
from aitune.torch.libs.exporting.dynamic_shapes import build_dynamic_shapes, log_dynamic_shapes
from aitune.torch.libs.exporting.preparation import TensorExportPreparation, prepare_tensor_export
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.tensor import format_tensor_name

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export a module to ONNX through AITune's flat tensor interface."""

    def __init__(self, output_path: Path, use_dynamo: bool = False, opset_version: int | None = None):
        """Initialize the output path and ONNX export mode."""
        self.use_dynamo = use_dynamo
        self.opset_version = opset_version
        self.output_path = output_path

    def export(
        self,
        module: nn.Module,
        sample: Sample,
        graph_spec: GraphSpec,
        verbose: bool = False,
        *,
        device: torch.device | str | None = None,
    ) -> Path:
        """Prepare a tensor-only example call and export it to ONNX."""
        logger.info("Exporting PyTorch module to ONNX: %s", self.output_path)

        try:
            preparation = prepare_tensor_export(module, sample, graph_spec, device=device)
            if self.use_dynamo:
                self._export_dynamo(preparation, graph_spec, verbose)
            else:
                self._export_trace(preparation, graph_spec, verbose)
            self.verify_model(self.output_path)
        except Exception as error:
            logger.error("Failed to export model to ONNX: %s", error)
            if self.output_path.exists():
                try:
                    self.output_path.unlink()
                except Exception as cleanup_error:
                    logger.info("Failed to remove incomplete ONNX file: %s", cleanup_error)
            raise

        logger.info("Successfully exported and verified ONNX model: %s", self.output_path)
        return self.output_path

    @staticmethod
    def verify_model(onnx_path: str | Path) -> None:
        """Validate an ONNX artifact."""
        onnx.checker.check_model(onnx_path)

    def _export_dynamo(
        self,
        preparation: TensorExportPreparation,
        graph_spec: GraphSpec,
        verbose: bool | None = None,
    ) -> None:
        """Export the prepared adapter with the Dynamo-based ONNX exporter."""
        dynamic_shapes = build_dynamic_shapes(graph_spec)
        log_dynamic_shapes(dynamic_shapes)
        torch.onnx.export(
            preparation.adapter,
            args=preparation.flat_inputs,
            kwargs=None,
            f=self.output_path.as_posix(),
            dynamo=True,
            opset_version=self.opset_version,
            input_names=self._input_names(graph_spec),
            output_names=list(graph_spec.output_names()),
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
        )

    def _export_trace(
        self,
        preparation: TensorExportPreparation,
        graph_spec: GraphSpec,
        verbose: bool | None = None,
    ) -> None:
        """Export the prepared adapter with the legacy ONNX tracer."""
        torch.onnx.export(
            preparation.adapter,
            args=preparation.flat_inputs,
            kwargs=None,
            f=self.output_path.as_posix(),
            dynamo=False,
            input_names=self._input_names(graph_spec),
            output_names=list(graph_spec.output_names()),
            opset_version=self.opset_version,
            dynamic_axes=self._create_dynamic_axes(graph_spec),
            verbose=verbose,
        )

    @staticmethod
    def _input_names(graph_spec: GraphSpec) -> list[str]:
        """Return names for flat inputs in their recorded order."""
        return [format_tensor_name(locator.path, "input") for locator, _ in graph_spec.input_spec.tensor_data]

    @staticmethod
    def _create_dynamic_axes(graph_spec: GraphSpec) -> dict:
        """Create dynamic-axis rules for the legacy ONNX tracer."""
        # Inferred axes use index lists; explicit definitions use axis-name dictionaries.
        dynamic_axes: defaultdict[str, list[int] | dict[int, str]] = defaultdict(list)

        for locator, tensor_spec in graph_spec.input_spec.tensor_data:
            definition = graph_spec.get_shape_definition(locator)
            if definition is not None:
                axes = {
                    axis: dimension.name
                    for axis, dimension in enumerate(definition)
                    if isinstance(dimension, DynamicDim)
                }
                if axes:
                    dynamic_axes[format_tensor_name(locator.path, "input")] = axes
                continue

            for axis, (min_size, max_size) in enumerate(
                zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)
            ):
                if min_size != max_size:
                    cast(list[int], dynamic_axes[format_tensor_name(locator.path, "input")]).append(axis)

        for locator, tensor_spec in graph_spec.output_spec.tensor_data:
            for axis, (min_size, max_size) in enumerate(
                zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)
            ):
                if min_size != max_size:
                    cast(list[int], dynamic_axes[format_tensor_name(locator.path, "output")]).append(axis)

        return dynamic_axes
