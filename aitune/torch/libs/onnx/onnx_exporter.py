# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX Exporter module for TensorRT backend."""

import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from aitune.torch.dynamic_shapes import DynamicDim
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.torch.utils.shapes import build_dynamic_shapes, log_dynamic_shapes
from aitune.torch.utils.tensor import format_tensor_name

# torch.onnx.export(dynamo=True, fallback=...) was removed in 2.11 (and some nightlies before).
# Inspect the signature directly rather than relying on version string parsing.
_ONNX_FALLBACK_SUPPORTED = "fallback" in inspect.signature(torch.onnx.export).parameters

# File extension constants
ONNX_FILE_EXTENSION = ".onnx"

# Setup logger
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Class for exporting PyTorch modules to ONNX format."""

    def __init__(self, output_path: Path, use_dynamo: bool = False, opset_version: int | None = None):
        """Initialize the ONNX exporter.

        Args:
            output_path: Directory to save the ONNX files
            use_dynamo: Whether to use torch.dynamo for export
            opset_version: ONNX opset version to use
        """
        self.use_dynamo = use_dynamo
        self.opset_version = opset_version
        self.output_path = output_path

    def export(
        self,
        module: nn.Module,
        sample: Sample,
        graph_spec: GraphSpec,
        verbose: bool = False,
    ) -> Path:
        """Export the module to ONNX.

        Args:
            module: PyTorch module to export
            sample: Example input for the model
            graph_spec: GraphSpec object
            verbose: Whether to print verbose output

        Returns:
            Path to the ONNX file
        """
        logger.info("Exporting PyTorch module to ONNX: %s", self.output_path)

        try:
            if self.use_dynamo:
                logger.info("Using ONNX dynamo for exporting to ONNX")
                self._export_dynamo(module, sample, graph_spec, self.output_path, verbose)
            else:
                logger.info("Using ONNX trace for exporting to ONNX")
                self._export_trace(module, sample, graph_spec, self.output_path, verbose)

            # Verify the model
            self.verify_model(onnx_path=self.output_path)
            logger.info("Successfully exported and verified ONNX model: %s", self.output_path)

            return self.output_path
        except Exception as e:
            logger.error("Failed to export model to ONNX: %s", e)
            if self.output_path.exists():
                logger.info("Removing incomplete ONNX file: %s", self.output_path)
                try:
                    self.output_path.unlink()
                except Exception as delete_error:
                    logger.info("Failed to remove incomplete ONNX file: %s", delete_error)
            raise e

    def verify_model(self, onnx_path: str | Path):
        """Verify the ONNX model.

        Args:
            onnx_path: Path to the ONNX model
        """
        logger.info("Verifying ONNX model: %s", onnx_path)
        try:
            onnx.checker.check_model(onnx_path)
            logger.info("ONNX model verification successful: %s", onnx_path)
        except Exception as e:
            logger.error("ONNX model verification failed: %s", e)
            raise e

    def _export_dynamo(
        self, module: nn.Module, sample: Sample, graph_spec: GraphSpec, onnx_path: Path, verbose: bool | None = None
    ):
        """Export the module to ONNX using torch.dynamo."""
        args, kwargs = sample

        if graph_spec.input_spec.has_batch_axis():
            min_batch_size = graph_spec.get_min_batch_size() or 2
            batch_size = max(min_batch_size, 2)
            args, kwargs = graph_spec.make_batch(args, kwargs, batch_size=batch_size)

        dynamic_shapes = self._create_dynamic_shapes((args, kwargs), graph_spec)
        input_names = [format_tensor_name(locator.path, "input") for locator, _ in graph_spec.input_spec.tensor_data]
        output_names = [format_tensor_name(locator.path, "output") for locator, _ in graph_spec.output_spec.tensor_data]

        log_dynamic_shapes(dynamic_shapes)

        export_kwargs = {}
        if _ONNX_FALLBACK_SUPPORTED:
            export_kwargs["fallback"] = False

        torch.onnx.export(
            module,
            f=onnx_path.as_posix(),
            args=args,
            kwargs=kwargs,
            dynamo=True,
            opset_version=self.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
            **export_kwargs,
        )
        logger.info("ONNX model exported successfully: %s", onnx_path)

    def _export_trace(
        self, module: nn.Module, sample: Sample, graph_spec: GraphSpec, onnx_path: Path, verbose: bool | None = None
    ):
        """Export the module to ONNX using torch.onnx.trace."""
        # Use standard torch ONNX export
        dynamic_axes = self._create_dynamic_axes(graph_spec)

        input_names = [format_tensor_name(locator.path, "input") for locator, _ in graph_spec.input_spec.tensor_data]
        output_names = [format_tensor_name(locator.path, "output") for locator, _ in graph_spec.output_spec.tensor_data]

        logger.info("Input names: %s", input_names)
        logger.info("Output names: %s", output_names)
        logger.info("Dynamic axes: %s", dynamic_axes)

        args, kwargs = sample
        torch.onnx.export(
            module,
            args=args,
            kwargs=kwargs,
            f=onnx_path.as_posix(),
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=self.opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )
        logger.info("ONNX model exported successfully: %s", onnx_path)

    @staticmethod
    def _create_dynamic_shapes(
        sample: Sample,
        graph_spec: GraphSpec,
        use_auto: bool = True,
    ) -> dict[str, Any] | None:
        """Build dynamic shapes for the normalized export sample.

        Delegates to ``aitune.torch.utils.shapes.build_dynamic_shapes``.

        Args:
            sample: Normalized args and kwargs used for export.
            graph_spec: Input graph spec.
            use_auto: When ``True`` (default), use ``Dim.AUTO`` for non-batch varying
                axes so torch.export infers divisibility constraints automatically.
                Pass ``False`` to get explicit ``Dim(name, min, max)`` instances. The
                ONNX graph stores dynamic axes as named symbols and leaves shape
                resolution to the runtime, so the default suits every export here.

        Returns:
            Dynamic shapes matching the normalized forward input structure, or ``None``
            when all inputs are static.
        """
        logger.debug("Extracting dynamic shapes")
        return build_dynamic_shapes(sample, graph_spec, use_auto=use_auto)

    def _create_dynamic_axes(self, graph_spec: GraphSpec) -> dict:
        """Create dynamic axes for the ONNX trace model.

        Args:
            graph_spec: Input graph spec
        Returns:
            Dict of dynamic axes
        """
        dynamic_axes = defaultdict(list)

        # Process input dynamic axes
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

            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    dynamic_axes[format_tensor_name(locator.path, "input")].append(ax)

        # Process output dynamic axes
        for locator, tensor_spec in graph_spec.output_spec.tensor_data:
            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    dynamic_axes[format_tensor_name(locator.path, "output")].append(ax)

        return dynamic_axes
