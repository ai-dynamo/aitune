# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""ONNX Model Info module for extracting information from ONNX models."""

import logging
from pathlib import Path

import onnx

# Setup logger
logger = logging.getLogger(__name__)


class ONNXModelInfo:
    """Helper class to load ONNX model and provide access to its information.

    This class loads an ONNX model once during initialization, extracts all needed
    information, and then deletes the loaded model to free resources.

    All extracted information is accessible as properties.
    """

    def __init__(self, model_path: Path):
        """Initialize the ONNX model info.

        Args:
            model_path: Path to the ONNX model file
        """
        self._model_path = model_path

        try:
            logger.info("Loading ONNX model: %s", model_path)
            # Load the ONNX model
            model = onnx.load(model_path)

            # Extract all information
            self._input_names = [input_data.name for input_data in model.graph.input]
            self._output_names = [output_data.name for output_data in model.graph.output]

            # Extract opset versions
            self._opset_version = None
            if model.opset_import:
                # Get the main opset version (typically the first one)
                self._opset_version = model.opset_import[0].version

            # Extract producer info if available
            self._producer_name = model.producer_name if hasattr(model, "producer_name") else None
            self._producer_version = model.producer_version if hasattr(model, "producer_version") else None

            # Extract model version if available
            self._model_version = model.model_version if hasattr(model, "model_version") else None

            # Extract doc string if available
            self._doc_string = model.doc_string if hasattr(model, "doc_string") else None

            logger.info("Extracted information for model:")
            logger.info("  Inputs: %s", self._input_names)
            logger.info("  Outputs: %s", self._output_names)
            logger.info("  OpSet version: %s", self._opset_version)
            logger.info("  Producer: %s %s", self._producer_name, self._producer_version)

            # Delete the model to free resources
            del model

        except ImportError as e:
            logger.error("ONNX package not installed. Install with 'pip install onnx'")
            raise e
        except Exception as e:
            logger.error("Failed to extract information from ONNX model: %s", e)
            raise e

    @property
    def model_path(self) -> Path:
        """Path to the ONNX model file.

        Returns:
            Model path
        """
        return self._model_path

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
    def opset_version(self) -> int:
        """OpSet version of the model.

        Returns:
            OpSet version
        """
        return self._opset_version

    @property
    def producer_name(self) -> str:
        """Name of the producer/framework that created the model.

        Returns:
            Producer name
        """
        return self._producer_name

    @property
    def producer_version(self) -> str:
        """Version of the producer/framework that created the model.

        Returns:
            Producer version
        """
        return self._producer_version

    @property
    def model_version(self) -> int:
        """Version of the model.

        Returns:
            Model version
        """
        return self._model_version

    @property
    def doc_string(self) -> str:
        """Documentation string of the model.

        Returns:
            Documentation string
        """
        return self._doc_string
