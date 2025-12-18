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

import logging
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.recording_module import Sample
from aitune.torch.utils.module import get_forward_arguments_names

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
        forward_args = get_forward_arguments_names(module.forward)
        dynamic_shapes = self._create_dynamic_shapes(graph_spec, forward_args)
        input_names = graph_spec.input_spec.get_names()
        output_names = graph_spec.output_spec.get_names()

        args, kwargs = sample

        if graph_spec.input_spec.has_batch_axis():
            min_batch_size = graph_spec.get_min_batch_size() or 2
            batch_size = max(min_batch_size, 2)

            args, kwargs = graph_spec.input_spec.make_batch(args, kwargs, batch_size=batch_size)

        ordered_kwargs = {}
        for kwarg in forward_args[1]:
            if kwarg in kwargs:
                ordered_kwargs[kwarg] = kwargs[kwarg]

                # WAR: Non-tensor kwargs have to be mentioned in dynamic shapes - adding missing ones
                if kwarg not in dynamic_shapes:
                    dynamic_shapes[kwarg] = {} if isinstance(kwargs[kwarg], (dict, list)) else None

        _print_dynamic_shapes(dynamic_shapes)

        # drop root level names in dynamic shapes as we relay on input_names to set the names
        # NOTE: dynamic shapes could be created as list but we need names for WAR above
        dynamic_shapes = list(dynamic_shapes.values())

        torch.onnx.export(
            module,
            f=onnx_path.as_posix(),
            args=args,
            kwargs=ordered_kwargs,
            dynamo=True,
            opset_version=self.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            fallback=False,
            verbose=verbose,
        )
        logger.info("ONNX model exported successfully: %s", onnx_path)

    def _export_trace(
        self, module: nn.Module, sample: Sample, graph_spec: GraphSpec, onnx_path: Path, verbose: bool | None = None
    ):
        """Export the module to ONNX using torch.onnx.trace."""
        # Use standard torch ONNX export
        dynamic_axes = self._create_dynamic_axes(graph_spec)

        # Use standard torch ONNX export
        args_mapping, kwargs_mapping = graph_spec.input_spec.get_names_mapping()
        _, forward_kwargs = get_forward_arguments_names(module.forward)

        for argname in kwargs_mapping:
            assert argname in forward_kwargs, f"""Argument {argname} is not in forward argspec {forward_kwargs}.
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
    def _create_dynamic_shapes(graph_spec: GraphSpec, forward_arguments: tuple[list[str], list[str]]) -> dict[str, Any]:
        """Using graph spec and forward arguments to infer how does dynamic shapes look like.

        NOTE: Dynamic shapes are ordered by forward arguments and kwargs.
        See: https://docs.pytorch.org/docs/2.7/export.html#torch.export.export:~:text=If%20you%20are%20specifying%20dynamism%20on%20keyword%20args%2C%20you%20will%20need%20to%20pass%20them%20in%20the%20order%20that%20is%20defined%20in%20the%20original%20function%20signature.
        NOTE: We use dict[str, dict[int, torch.export.Dim]] structure to store dynamic shapes.
        NOTE: We support single level nested tensors, tensor in dict.
        NOTE: Missing dimensions are not filled by default - assuming missing is a static dimension.

        See docs: https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html

        Args:
            graph_spec (GraphSpec): Input graph spec
            forward_arguments: Tuple of forward arguments and kwargs
        Returns:
            dynamic shapes
        """
        logger.debug("Extracting dynamic shapes")
        forward_args, forward_kwargs = forward_arguments
        dynamic_shapes = _create_dynamic_shapes_from_tensor_spec(graph_spec.input_spec.tensor_specs)
        input_args, input_kwargs = _create_inputs_mapping(graph_spec.input_spec)
        _war_for_positional_arguments(input_args, forward_args, forward_kwargs)
        return _create_ordered_dynamic_shapes(forward_args, forward_kwargs, input_args, input_kwargs, dynamic_shapes)

    def _create_dynamic_axes(self, graph_spec: GraphSpec) -> dict:
        """Create dynamic axes for the ONNX trace model.

        Args:
            graph_spec: Input graph spec
        Returns:
            Dict of dynamic axes
        """
        dynamic_axes = defaultdict(list)

        # Process input dynamic axes
        for tensor_spec in graph_spec.input_spec.tensor_specs:
            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    dynamic_axes[tensor_spec.name].append(ax)

        # Process output dynamic axes
        for tensor_spec in graph_spec.output_spec.tensor_specs:
            for ax, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False)):
                if d1 != d2:
                    dynamic_axes[tensor_spec.name].append(ax)

        return dynamic_axes


def _raise_on_locator_user_type(locator: Locator) -> None:
    """Raise an exception if the locator is a user type."""
    if any(T in [ObjectType.USER_TYPE, ObjectType.DATACLASS] for _, T in locator.path_iter()):
        raise ValueError("Dynamic shapes does not support user types or dataclasses.")


def _create_nested_structure(locator: Locator, root: Any = None, last: Any = None) -> Any:
    """As locator does not create parent objects, we need to create them manually.

    Args:
        locator: Locator object
        root: Root object in the path
        last: Last object in the path
    Returns:
        Root object in the path
    """
    parents = [[] if T == ObjectType.SEQUENCE else {} for _, T in locator.path_iter()]
    accessors = [a for a, _ in locator.path_iter()]

    if root is not None:
        parents[0] = root

    parent = parents[0]
    for accessor, current in zip(accessors, parents[1:] + [last], strict=True):
        if isinstance(parent, list):
            if len(parent) <= accessor:
                parent.extend([None] * (accessor - len(parent) + 1))

        if accessor in parent:
            current = parent[accessor]

        parent[accessor] = current
        parent = current

    return parents[0]


def _create_dynamic_shapes_from_tensor_spec(
    tensor_specs, _fill_missing_dims: bool = False, _fill_missing_dims_with: Any = None, _use_cached_dims: bool = False
) -> dict[str, Any]:
    """Produces flat structure of dynamic shapes, tensor_spec name to dynamic shapes.

    Args:
        tensor_specs: List of tensor specs

        # Experimental arguments
        _fill_missing_dims: Whether to non dynamic dimensions
        _fill_missing_dims_with: what to fill missing dimensions with
        other options:
            torch.export.Dim.STATIC
            torch.export.Dim.AUTO
        _use_cached_dims: use cached dimensions across input tensors
    """
    dynamic_shapes = {}
    dim_cache = {}
    for tensor_spec in tensor_specs:
        dynamic_shapes[tensor_spec.name] = {}
        for idx, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=True)):
            if d1 != d2:
                dim = torch.export.Dim(f"{tensor_spec.name}_dim_{idx}", min=d1, max=d2)
                if _use_cached_dims:
                    key = (d1, d2)
                    dim = dim_cache.setdefault(key, dim)

                dynamic_shapes[tensor_spec.name][idx] = dim
            elif _fill_missing_dims:
                dynamic_shapes[tensor_spec.name][idx] = _fill_missing_dims_with
    return dynamic_shapes


def _create_inputs_mapping(input_spec) -> tuple[dict, dict]:
    """Creates yet another input names mapping but with depth of nested tensors.

    NOTE: NOT using graph_spec.input_spec.get_names_mapping() as we require depth of nested objects
    """
    input_args, input_kwargs = {}, {}
    for locator, tensor_spec in input_spec.tensor_data:
        if tensor_spec.name.startswith("args"):
            input_args[locator.leaf_name] = tensor_spec.name
            continue

        # saving key with depth to avoid ambiguity with nested objects
        input_kwargs[(locator.leaf_name, locator.depth)] = (locator, tensor_spec.name)
    return input_args, input_kwargs


def _war_for_positional_arguments(input_args, forward_args, forward_kwargs):
    """Checks if we got positional argument that python marked as kwarg and adds it to input_args.

    It happens when argument is passed as positional and function signature does not have clear separation
    between positional and keyword arguments like with / separator: def forward(x, y, z, w, /): return x, y, z, w

    Name of the argument is stored in leaf of the locator.
    """
    for i, (arg_n, arg_name) in enumerate(zip_longest(list(input_args.items()), list(forward_args), fillvalue=None)):
        if arg_name is not None:
            # assuming it is ok
            break
        arg_name = forward_kwargs[i]
        forward_args.append(arg_name)
        input_args[arg_name] = arg_n[1]


def _create_ordered_dynamic_shapes(forward_args, forward_kwargs, input_args, input_kwargs, dynamic_shapes) -> dict:
    """Creates ordered dynamic shapes with python function signature argument names."""
    ordered_dynamic_shapes = {}
    for fwd_arg in forward_args:
        if tensor_spec_name := input_args.get(fwd_arg):
            ordered_dynamic_shapes[fwd_arg] = dynamic_shapes[tensor_spec_name]

    for kwarg in forward_kwargs:
        if loc_and_name := input_kwargs.get((kwarg, 1)):
            _locator, tensor_spec_name = loc_and_name
            ordered_dynamic_shapes[kwarg] = dynamic_shapes[tensor_spec_name]

    # by utilizing locator we can create a structure of nested dynamic shapes
    for _, (locator, tensor_spec_name) in input_kwargs.items():
        if locator.depth > 1:
            # need to provide a parents as they are not created by set_value
            _raise_on_locator_user_type(locator)
            ordered_dynamic_shapes = _create_nested_structure(locator, root=ordered_dynamic_shapes)
            locator.set_value(ordered_dynamic_shapes, dynamic_shapes[tensor_spec_name])

    return ordered_dynamic_shapes


def _print_dynamic_shapes(dynamic_shapes: dict) -> None:
    """Print the dynamic shapes."""
    logger.debug("Extracted dynamic shapes:")
    for key, value in dynamic_shapes.items():
        logger.debug("%s:", key)
        for idx, ds in (value or {}).items():
            if isinstance(ds, torch.export.dynamic_shapes._Dim):
                logger.debug("  %s: min:%d, max:%d, %s", idx, ds.min, ds.max, str(ds).split(".")[-1])
            else:
                logger.debug("  %s: %s", idx, ds)
