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
"""PyTorch Model Info module for extracting information from PyTorch models."""

import enum
import inspect
import logging
from typing import Any

import torch
from torch import nn

from aitune.torch.module.recording_module import Sample
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.utils.system_monitor import SystemMonitor

# Setup logger
logger = logging.getLogger(__name__)


class OutputFormat(enum.Enum):
    """Enum representing the output format of a PyTorch model."""

    TENSOR = "tensor"  # Single tensor output
    TUPLE = "tuple"  # Tuple of tensors
    LIST = "list"  # List of tensors
    DICT = "dict"  # Dictionary of named tensors
    OBJECT = "object"  # Custom object with tensor attributes
    UNKNOWN = "unknown"  # Unknown format


class TorchModelInfo:
    """Helper class to analyze PyTorch models and provide access to model information.

    This class performs inference on a PyTorch model during initialization, extracts
    all needed information about inputs and outputs, and makes this information
    accessible as properties.
    """

    def __init__(
        self,
        model: nn.Module,
        sample: Sample,
    ):
        """Initialize the PyTorch model info by analyzing the model.

        Args:
            model: PyTorch model to analyze
            sample: Sample input to use for model inference.
                A tuple where the first element is a tuple of positional arguments,
                and the second element is a dictionary of keyword arguments.
        """
        self.system_monitor = SystemMonitor()

        # Store original training mode and set to eval for inference
        training_mode = model.training
        model.eval()

        args, kwargs = sample

        try:
            logger.debug("Analyzing PyTorch model")

            with self.system_monitor.system_stats_context(log_label="Model analysis"):
                # Analyze inputs
                param_names = self._get_model_parameter_names(model)
                self._analyze_inputs(sample, param_names)

                # Run inference and analyze outputs
                outputs = self._run_inference(model, args, kwargs)

                self._output_class = outputs.__class__
                self._analyze_outputs(outputs)

            # Log results
            logger.debug("Extracted information for model:")
            logger.debug("  Inputs: %s", self._input_names)
            logger.debug("  Outputs: %s", self._output_names)
            if self._output_class:
                logger.debug("  Output class: %s", self._output_class)

        except Exception as e:
            logger.debug("Failed to extract information from PyTorch model: %s", e)
            raise e
        finally:
            # Restore original training mode
            model.train(training_mode)

    def _get_model_parameter_names(self, model: nn.Module) -> list[str]:
        """Extract parameter names from model's forward method.

        Args:
            model: PyTorch model to analyze

        Returns:
            List of parameter names
        """
        forward_signature = inspect.signature(model.forward)
        param_names = list(forward_signature.parameters.keys())

        # Skip 'self' parameter if present
        if param_names and param_names[0] == "self":
            param_names = param_names[1:]

        return param_names

    def _analyze_inputs(self, sample: Sample, param_names: list[str]) -> None:
        """Analyze input tensor information.

        Args:
            sample: Sample input
            param_names: Parameter names from the model's forward method
        """
        args, kwargs = sample

        # Check for kwargs that don't match any parameter name
        unknown_kwargs = set(kwargs.keys()) - set(param_names)
        if unknown_kwargs:
            logger.warning("Found unexpected kwargs that don't match any parameter: %s", unknown_kwargs)

        # Create sample metadata to help with input analysis
        sample_metadata = SampleMetadata.from_sample(sample, prefix="input")
        flattened_sample = sample_metadata.flatten_sample(sample)

        # Extract input names from sample metadata
        self._input_names = []
        self._input_shapes = {}
        self._input_dtypes = {}

        # Map tensor specs to parameter names when possible
        tensor_specs = sample_metadata.tensor_specs

        # For args, use the corresponding parameter name if available
        args_count = 0
        for tensor_spec in tensor_specs:
            tensor = flattened_sample[tensor_spec.name]

            # Try to match with parameter name
            if args_count < len(args) and args_count < len(param_names):
                name = param_names[args_count]
                args_count += 1
            else:
                # Use the original tensor name
                name = tensor_spec.name

            self._input_names.append(name)
            self._input_shapes[name] = list(tensor.shape)
            self._input_dtypes[name] = tensor.dtype

    def _run_inference(self, model: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run inference on the model with the given inputs.

        Args:
            model: PyTorch model
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Model outputs
        """
        logger.debug("Running inference for model analysis")
        with torch.no_grad():
            return model(*args, **kwargs)

    def _analyze_outputs(self, outputs: Any) -> None:
        """Analyze output tensor information.

        Args:
            outputs: Model outputs
        """
        logger.debug("Analyzing model outputs")
        # Initialize output data structures
        self._initialize_output_data()

        # Determine output format and extract tensor information
        if isinstance(outputs, torch.Tensor):
            self._analyze_tensor_output(outputs)
        elif type(outputs) is tuple:
            self._analyze_tuple_output(outputs)
        elif type(outputs) is list:
            self._analyze_list_output(outputs)
        elif type(outputs) is dict:
            self._analyze_dict_output(outputs)
        else:
            self._analyze_object_output(outputs)

        logger.debug("Detected output format: %s", self._output_format.value)

    def _initialize_output_data(self) -> None:
        """Initialize output data structures."""
        self._output_names = []
        self._output_shapes = {}
        self._output_dtypes = {}
        self._output_format = OutputFormat.UNKNOWN

    def _analyze_tensor_output(self, tensor: torch.Tensor) -> None:
        """Analyze single tensor output.

        Args:
            tensor: Output tensor from the model
        """
        self._output_format = OutputFormat.TENSOR
        name = "output_0"
        self._output_names = [name]
        self._output_shapes[name] = list(tensor.shape)
        self._output_dtypes[name] = tensor.dtype
        logger.debug("Analyzed single tensor output with shape %s", tensor.shape)

    def _analyze_tuple_output(self, tensors: tuple) -> None:
        """Analyze tuple of tensors output.

        Args:
            tensors: Tuple of output tensors from the model
        """
        self._output_format = OutputFormat.TUPLE
        self._analyze_iterable_output(tensors)
        logger.debug("Analyzed tuple output with %s tensors", len(self._output_names))

    def _analyze_list_output(self, tensors: list) -> None:
        """Analyze list of tensors output.

        Args:
            tensors: List of output tensors from the model
        """
        self._output_format = OutputFormat.LIST
        self._analyze_iterable_output(tensors)
        logger.debug("Analyzed list output with %s tensors", len(self._output_names))

    def _analyze_iterable_output(self, tensors: tuple | list) -> None:
        """Analyze iterable (list/tuple) of tensors output.

        Args:
            tensors: Iterable of output tensors from the model
        """
        for i, out in enumerate(tensors):
            if isinstance(out, torch.Tensor):
                name = f"output_{i}"
                self._output_names.append(name)
                self._output_shapes[name] = list(out.shape)
                self._output_dtypes[name] = out.dtype

    def _analyze_dict_output(self, outputs: dict) -> None:
        """Analyze dictionary of tensors output.

        Args:
            outputs: Dictionary of output tensors from the model
        """
        self._output_format = OutputFormat.DICT
        self._output_names = list(outputs.keys())
        for name, out in outputs.items():
            if isinstance(out, torch.Tensor):
                self._output_shapes[name] = list(out.shape)
                self._output_dtypes[name] = out.dtype
        logger.debug("Analyzed dictionary output with keys: %s", self._output_names)

    def _analyze_object_output(self, obj: Any) -> None:
        """Analyze complex object output.

        Args:
            obj: Complex object output from the model
        """
        # Extract tensor attributes from the object
        for name, value in inspect.getmembers(obj):
            if isinstance(value, torch.Tensor) and not name.startswith("_"):
                self._output_names.append(name)
                self._output_shapes[name] = list(value.shape)
                self._output_dtypes[name] = value.dtype

        # If we found tensor attributes, assume it's a custom object format
        if self._output_names:
            self._output_format = OutputFormat.OBJECT
            logger.debug("Analyzed object output with tensor attributes: %s", self._output_names)
        else:
            logger.debug("No tensor attributes found in object output")

    @property
    def input_names(self) -> list[str]:
        """List of input tensor names.

        Returns:
            List of input tensor names
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """List of output tensor names.

        Returns:
            List of output tensor names
        """
        return self._output_names

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        """Dictionary mapping input names to their shapes.

        Returns:
            Dictionary of input shapes
        """
        return self._input_shapes

    @property
    def output_shapes(self) -> dict[str, list[int]]:
        """Dictionary mapping output names to their shapes.

        Returns:
            Dictionary of output shapes
        """
        return self._output_shapes

    @property
    def input_dtypes(self) -> dict[str, torch.dtype]:
        """Dictionary mapping input names to their data types.

        Returns:
            Dictionary of input data types
        """
        return self._input_dtypes

    @property
    def output_dtypes(self) -> dict[str, torch.dtype]:
        """Dictionary mapping output names to their data types.

        Returns:
            Dictionary of output data types
        """
        return self._output_dtypes

    @property
    def output_class(self):
        """Output class type.

        Returns:
            The class type of the model output
        """
        return self._output_class

    @property
    def output_format(self) -> OutputFormat:
        """Output format type.

        Returns:
            Enum indicating the output format: TENSOR, TUPLE, LIST, DICT, OBJECT, or UNKNOWN
        """
        return self._output_format
