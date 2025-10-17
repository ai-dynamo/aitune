# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ONNX Exporter module for TensorRT backend."""

import inspect
import logging
from collections import defaultdict
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.utils.system_monitor import SystemMonitor

# File extension constants
ONNX_FILE_EXTENSION = ".onnx"

# Setup logger
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Class for exporting PyTorch modules to ONNX format."""

    def __init__(self, output_path: Path, use_dynamo: bool = False, opset_version: int | None = 20):
        """Initialize the ONNX exporter.

        Args:
            output_path: Directory to save the ONNX files
            use_dynamo: Whether to use torch.dynamo for export
            opset_version: ONNX opset version to use
        """
        self.use_dynamo = use_dynamo
        self.opset_version = opset_version
        self.system_monitor = SystemMonitor()
        self.output_path = output_path

    def export(
        self,
        module: nn.Module,
        sample: Sample,
        graph_spec: GraphSpec,
        verbose: bool | None = None,
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
        logger.debug("Exporting PyTorch module to ONNX: %s", self.output_path)

        try:
            if self.use_dynamo:
                logger.debug("Using ONNX dynamo for exporting to ONNX")
                self._export_dynamo(module, sample, graph_spec, self.output_path, verbose)
            else:
                logger.debug("Using ONNX trace for exporting to ONNX")
                self._export_trace(module, sample, graph_spec, self.output_path, verbose)

            # Verify the model
            with self.system_monitor.system_stats_context(log_label="ONNX model verification"):
                self.verify_model(onnx_path=self.output_path)

            logger.debug("Successfully exported and verified ONNX model: %s", self.output_path)

            return self.output_path
        except Exception as e:
            logger.debug("Failed to export model to ONNX", exc_info=e)
            if self.output_path.exists():
                logger.debug("Removing incomplete ONNX file: %s", self.output_path)
                try:
                    self.output_path.unlink()
                except Exception as delete_error:
                    logger.debug("Failed to remove incomplete ONNX file: %s", delete_error)
            raise e

    def verify_model(self, onnx_path: str | Path):
        """Verify the ONNX model.

        Args:
            onnx_path: Path to the ONNX model
        """
        logger.debug("Verifying ONNX model: %s", onnx_path)
        try:
            onnx.checker.check_model(onnx_path)
            logger.debug("ONNX model verification successful: %s", onnx_path)
        except Exception as e:
            logger.debug("ONNX model verification failed: %s", e)
            raise e

    def _export_dynamo(
        self, module: nn.Module, sample: Sample, graph_spec: GraphSpec, onnx_path: Path, verbose: bool | None = None
    ):
        """Export the module to ONNX using torch.dynamo."""
        with self.system_monitor.system_stats_context(log_label="Torch.dynamo ONNX export and save"):
            dynamic_shapes = self._create_dynamic_shapes(graph_spec)
            dynamic_axes = self._create_dynamic_axes(graph_spec)  # dynamic axes required for fallback=True
            input_names = graph_spec.input_spec.get_names()
            output_names = graph_spec.output_spec.get_names()

            args, kwargs = graph_spec.input_spec.make_batch(sample, batch_size=2)

            dynamic_shapes += [None] * (
                len(args) + len(kwargs) - len(dynamic_shapes)
            )  # WAR: For missing shapes for None value args and kwargs

            exported_program = torch.onnx.export(
                module,
                args=args,
                kwargs=kwargs,
                f=onnx_path.as_posix(),
                dynamo=True,
                input_names=input_names,
                dynamic_shapes=dynamic_shapes,
                dynamic_axes=dynamic_axes,
                fallback=True,
                verbose=verbose,
            )
            exported_program.save(onnx_path.as_posix())
            self._modify_onnx_io_names(onnx_path, input_names, output_names, onnx_path)

    def _export_trace(
        self, module: nn.Module, sample: Sample, graph_spec: GraphSpec, onnx_path: Path, verbose: bool | None = None
    ):
        """Export the module to ONNX using torch.onnx.trace."""
        with self.system_monitor.system_stats_context(log_label="Standard ONNX export"):
            # Use standard torch ONNX export
            dynamic_axes = self._create_dynamic_axes(graph_spec)

            args_mapping, kwargs_mapping = graph_spec.input_spec.get_names_mapping()

            # Use inspect.signature instead of getfullargspec for more complete parameter information
            forward_signature = inspect.signature(module.forward)
            forward_params = list(forward_signature.parameters.keys())

            params_list = list(forward_signature.parameters.values())
            forward_kwargs = [
                p.name
                for p in params_list
                if p.kind == inspect.Parameter.KEYWORD_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]

            for argname in kwargs_mapping:
                assert argname in forward_kwargs, f"""Argument {argname} is not in forward argspec {forward_params}.
                Collected args mapping: {args_mapping}
                Collected kwargs mapping: {kwargs_mapping}
                """

            input_names = []
            for args_names in args_mapping:
                input_names.append(args_names)

            for argname in forward_kwargs:
                if argname in kwargs_mapping:
                    input_names.extend(kwargs_mapping[argname])

            output_names = graph_spec.output_spec.get_names()

            args, kwargs = sample
            torch.onnx.export(
                module,
                args=args,
                kwargs=kwargs,
                f=onnx_path.as_posix(),
                input_names=input_names,
                output_names=output_names,
                opset_version=self.opset_version,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
            )

    def _create_dynamic_shapes(self, graph_spec: GraphSpec) -> list[dict[int, torch.export.Dim]]:
        """Using graph spec to infer how does input look like.

        Args:
            graph_spec (GraphSpec): Input graph spec

        Returns:
            List of dynamic shapes
        """
        logger.debug("Extracting dynamic shapes")
        dynamic_shapes = []
        for tensor_spec in graph_spec.input_spec.tensor_specs:
            dynamic_shape_map = {}
            for idx, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=True)):
                if d1 != d2:
                    dynamic_shape_map[idx] = torch.export.Dim(f"{tensor_spec.name}_dim_{idx}", min=d1, max=d2)

            dynamic_shapes.append(dynamic_shape_map)

        logger.debug("Extracted dynamic shapes: %s", dynamic_shapes)
        return dynamic_shapes

    def _create_dynamic_axes(self, graph_spec: GraphSpec) -> dict:
        """Create dynamic axes for the ONNX trace model.

        Args:
            graph_spec: Input graph spec

        Returns:
            Dict of dynamic axes
        """
        input_dynamic_axes = defaultdict(list)
        for tensor_spec in graph_spec.input_spec.tensor_specs:
            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    input_dynamic_axes[tensor_spec.name].append(ax)

        output_dynamic_axes = defaultdict(list)
        for tensor_spec in graph_spec.output_spec.tensor_specs:
            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    output_dynamic_axes[tensor_spec.name].append(ax)

        return dict(**input_dynamic_axes, **output_dynamic_axes)

    def _modify_onnx_io_names(self, model_path, new_input_names, new_output_names, output_path):
        """Modify the input and output names of the ONNX model."""
        graph = gs.import_onnx(onnx.load(model_path, load_external_data=False))

        # Check if the number of new input names matches the number of inputs in the graph
        if len(new_input_names) != len(graph.inputs):
            raise ValueError(
                f"Number of new input names must match the number of inputs in the ONNX graph. Got: {new_input_names} and {graph.inputs}"
            )

        # Modify the input names
        for i, _input in enumerate(graph.inputs):
            _input.name = new_input_names[i]

        # Check if the number of new output names matches the number of outputs in the graph
        if len(new_output_names) != len(graph.outputs):
            raise ValueError(
                f"Number of new output names must match the number of outputs in the ONNX graph. Got: {new_output_names} and {graph.outputs}"
            )

        # Modify the output names
        for i, _output in enumerate(graph.outputs):
            _output.name = new_output_names[i]

        # Save the modified model
        graph.cleanup()
        onnx.save(gs.export_onnx(graph), output_path)
