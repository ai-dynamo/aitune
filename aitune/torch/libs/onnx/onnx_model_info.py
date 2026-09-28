# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX Model Info module for extracting information from ONNX models."""

import logging
from enum import Enum
from pathlib import Path

import onnx

# Setup logger
logger = logging.getLogger(__name__)


class ONNXPrecision(str, Enum):
    """Post-export quantization precision for the ONNX model.

    All modes use standard ONNX operators and are compatible with the ONNX
    Runtime CUDA Execution Provider without TensorRT.

    +----------+-------------------+---------------------------------------------+
    | Precision| Required ORT ver. | Notes                                       |
    +==========+===================+=============================================+
    | FP16     | Any               | Converts model weights & ops to FP16 via    |
    |          |                   | ``onnxconverter_common``                    |
    +----------+-------------------+---------------------------------------------+
    | INT8     | Any               | Dynamic weight-only INT8 quantization via   |
    |          |                   | ``onnxruntime.quantization`` — no           |
    |          |                   | calibration data required                   |
    +----------+-------------------+---------------------------------------------+
    | FP8      | >= 1.16           | Dynamic weight-only FP8 E4M3 quantization   |
    |          |                   | via ``onnxruntime.quantization``            |
    +----------+-------------------+---------------------------------------------+
    | INT4     | >= 1.16           | Dynamic weight-only INT4 quantization via   |
    |          |                   | ``onnxruntime.quantization``                |
    +----------+-------------------+---------------------------------------------+
    """

    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"


# Maps ONNX TensorProto data_type integers to ONNXPrecision values.
ONNX_DTYPE_TO_PRECISION: dict[int, "ONNXPrecision"] = {
    10: ONNXPrecision.FP16,  # TensorProto.FLOAT16
    3: ONNXPrecision.INT8,  # TensorProto.INT8
    17: ONNXPrecision.FP8,  # TensorProto.FLOAT8E4M3FN  (onnx >= 1.14)
    22: ONNXPrecision.INT4,  # TensorProto.INT4           (onnx >= 1.15)
}


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
            # Graph metadata and initializer dtypes do not require external weight data.
            model = onnx.load(model_path, load_external_data=False)
            if not model.HasField("graph"):
                raise ValueError(f"ONNX model {model_path} does not contain a graph")

            # Extract all information
            self._input_names = [input_data.name for input_data in model.graph.input]
            self._output_names = [output_data.name for output_data in model.graph.output]
            self._input_shapes = self._get_tensor_shapes(model.graph.input)
            self._output_shapes = self._get_tensor_shapes(model.graph.output)
            self._input_dtypes = self._get_tensor_dtypes(model.graph.input)
            self._output_dtypes = self._get_tensor_dtypes(model.graph.output)
            self._operators = self._get_operators(model.graph)
            self._precision = self._precision(model)

            # Extract opset versions
            self._opsets = {entry.domain: entry.version for entry in model.opset_import}
            # ONNX uses the empty domain for its standard operator set:
            # https://onnx.ai/onnx/repo-docs/Versioning.html#operator-sets
            self._opset_version = self._opsets.get("")

            # Extract producer info if available
            self._producer_name = model.producer_name if hasattr(model, "producer_name") else None
            self._producer_version = model.producer_version if hasattr(model, "producer_version") else None

            # Extract model version if available
            self._model_version = model.model_version if hasattr(model, "model_version") else None

            # Extract doc string if available
            self._doc_string = model.doc_string if hasattr(model, "doc_string") else None

            logger.info("Extracted information for model:")
            logger.info("  Inputs: %s", self._input_names)
            logger.info("  Input shapes: %s", self._input_shapes)
            logger.info("  Precision: %s", self._precision)
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
    def input_shapes(self) -> dict[str, list[int | str | None]]:
        """Input name -> shape (list of dims; int for static, str for dynamic/symbolic).

        Returns:
            Dict mapping each input name to its shape. Dynamic dimensions
            appear as dim_param string (e.g. "batch"); static as int; unknown as None.
        """
        return self._input_shapes

    @property
    def output_shapes(self) -> dict[str, list[int | str | None]]:
        """Declared output shapes, keyed by tensor name."""
        return self._output_shapes

    @property
    def input_dtypes(self) -> dict[str, str]:
        """Declared ONNX input tensor data types, keyed by tensor name."""
        return self._input_dtypes

    @property
    def output_dtypes(self) -> dict[str, str]:
        """Declared ONNX output tensor data types, keyed by tensor name."""
        return self._output_dtypes

    @property
    def operators(self) -> frozenset[str]:
        """Operator types used by graph nodes, qualified with their domain when nonstandard."""
        return self._operators

    @property
    def opsets(self) -> dict[str, int]:
        """Imported opset versions keyed by ONNX domain; the standard domain is ``""``."""
        return self._opsets

    @staticmethod
    def _get_tensor_shapes(value_info_list) -> dict[str, list[int | str | None]]:
        """Extract shapes from ONNX graph ValueInfoProto list (e.g. model.graph.input)."""
        shapes = {}
        for val in value_info_list:
            if not val.type.HasField("tensor_type") or not val.type.tensor_type.HasField("shape"):
                continue
            dims = []
            for d in val.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
            shapes[val.name] = dims
        return shapes

    @staticmethod
    def _get_tensor_dtypes(value_info_list) -> dict[str, str]:
        """Extract declared tensor element types where available."""
        return {
            value.name: onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
            for value in value_info_list
            if value.type.HasField("tensor_type")
        }

    @staticmethod
    def _get_operators(graph: onnx.GraphProto) -> frozenset[str]:
        """Collect operator types from a graph and any nested control-flow graphs."""
        operators = set()
        for node in graph.node:
            operators.add(f"{node.domain}::{node.op_type}" if node.domain else node.op_type)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    operators.update(ONNXModelInfo._get_operators(attribute.g))
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        operators.update(ONNXModelInfo._get_operators(subgraph))
        return frozenset(operators)

    @property
    def output_names(self) -> list[str]:
        """List of output tensor names.

        Returns:
            List of output tensor names
        """
        return self._output_names

    @property
    def opset_version(self) -> int | None:
        """OpSet version of the model.

        Returns:
            OpSet version
        """
        return self._opset_version

    @property
    def precision(self) -> "ONNXPrecision | None":
        """Precision of the model.

        Returns:
            Precision
        """
        return self._precision

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

    def _precision(self, model: onnx.ModelProto) -> "ONNXPrecision | None":
        found = {
            ONNX_DTYPE_TO_PRECISION[init.data_type]
            for init in model.graph.initializer
            if init.data_type in ONNX_DTYPE_TO_PRECISION
        }

        # QDQ (Quantize-Dequantize) models keep weights as FP32 in initializers;
        # precision is instead encoded in the zero-point type of QuantizeLinear nodes.
        if not found:
            initializer_dtypes = {init.name: init.data_type for init in model.graph.initializer}
            for node in model.graph.node:
                if node.op_type == "QuantizeLinear" and len(node.input) > 2:
                    zp_dtype = initializer_dtypes.get(node.input[2])
                    if zp_dtype in ONNX_DTYPE_TO_PRECISION:
                        found.add(ONNX_DTYPE_TO_PRECISION[zp_dtype])

        if not found:
            return None

        # Return the lowest precision (most compressed) present in the model.
        precision_order = [ONNXPrecision.INT4, ONNXPrecision.FP8, ONNXPrecision.INT8, ONNXPrecision.FP16]
        return min(found, key=lambda p: precision_order.index(p))
