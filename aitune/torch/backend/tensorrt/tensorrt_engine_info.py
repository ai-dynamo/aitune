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
"""TensorRT Engine Info module for extracting information from TensorRT engines."""

import logging
from typing import Any

import torch
from wrapt import lazy_import

trt = lazy_import("tensorrt")

# Setup logger
logger = logging.getLogger(__name__)


def _get_dtype_mapping():
    return {
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int,
        trt.DataType.INT64: torch.long,
        trt.DataType.FP8: torch.float8_e4m3fn,
        trt.DataType.HALF: torch.half,
        trt.DataType.FLOAT: torch.float,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.BF16: torch.bfloat16,
    }


class TensorRTEngineInfo:
    """Helper class to extract and store information from a TensorRT engine.

    This class takes a deserialized TensorRT engine, extracts needed information,
    and makes it accessible through properties. The class doesn't own the engine
    and doesn't delete it.

    All extracted information is accessible as properties.
    """

    def __init__(self, engine: Any):
        """Initialize the TensorRT engine info.

        Args:
            engine: Deserialized TensorRT engine
        """
        try:
            logger.debug("Extracting information from TensorRT")

            # Extract all information
            self._input_names = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
            ]

            self._output_names = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
            ]
            # Create output tensors info
            dtype_mapping = _get_dtype_mapping()
            self._output_dtypes = {
                output_name: dtype_mapping[engine.get_tensor_dtype(output_name)] for output_name in self._output_names
            }
            self._output_shapes = {
                output_name: engine.get_tensor_shape(output_name) for output_name in self._output_names
            }

            logger.debug("Extracted information for model:")
            logger.debug("  Inputs: %s", self._input_names)
            logger.debug("  Outputs: %s", self._output_names)

        except Exception as e:
            logger.debug("Failed to extract information from TensorRT engine: %s", e)
            raise e

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
    def num_io_tensors(self) -> int:
        """Number of input and output tensors.

        Returns:
            Number of input and output tensors
        """
        return len(self._input_names) + len(self._output_names)

    @property
    def output_dtypes(self) -> dict[str, torch.dtype]:
        """List of output tensor dtypes.

        Returns:
            List of output tensor dtypes
        """
        return self._output_dtypes

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """List of output tensor shapes.

        Returns:
            List of output tensor shapes
        """
        return self._output_shapes
