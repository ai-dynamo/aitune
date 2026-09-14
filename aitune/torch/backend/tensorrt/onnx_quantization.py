# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA ModelOpt ONNX quantization module for TensorRT backend."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import modelopt
import modelopt.onnx.quantization as moq
import onnx
from packaging.version import Version

if Version(modelopt.__version__) < Version("0.40.0"):
    from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

    # Note: https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.onnx.quantization.qdq_utils.html#modelopt.onnx.quantization.qdq_utils.fp4qdq_to_2dq
    modelopt_fp4_exporter = fp4qdq_to_2dq
else:
    # Note: https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.onnx.export.html#modelopt.onnx.export.NVFP4QuantExporter
    from modelopt.onnx.export import NVFP4QuantExporter

    modelopt_fp4_exporter = NVFP4QuantExporter.process_model

from aitune.torch.backend.tensorrt.modelopt_calibration import prepare_calibration_data
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_store import Sample
from aitune.utils.monitoring import annotate

# Setup logger
logger = logging.getLogger(__name__)

QuantizationPrecision = Literal["int8", "int4", "fp8"]
CalibrationMethod = Literal["max", "entropy", "awq_clip", "awq_lite", "awq_full", "rtn_dq"]


def _validate_precision(precision: str) -> None:
    """Validate that precision is a supported value."""
    if precision not in get_args(QuantizationPrecision):
        raise ValueError(f"Unsupported precision: {precision!r}. Supported values: {get_args(QuantizationPrecision)}")


def _validate_calibration_method(method: str | None) -> None:
    """Validate that calibration_method is a supported value."""
    if method is not None and method not in get_args(CalibrationMethod):
        raise ValueError(f"Unsupported calibration_method: {method!r}. Supported values: {get_args(CalibrationMethod)}")


@dataclass
class ONNXQuantizationConfig:
    """Configuration for ONNX quantization.

    Args:
        precision: Quantization precision ("int8", "fp8", "int4")
        calibration_method: Calibration method to use ("max", "entropy", "awq_clip", "awq_lite", "awq_full", "rtn_dq")
        use_model_opt_post_processing: Whether to apply ModelOpt post-processing to the quantized model
    """

    precision: QuantizationPrecision
    calibration_method: CalibrationMethod | None = None
    use_model_opt_post_processing: bool = False

    def __post_init__(self):
        """Validate precision and calibration_method after initialization."""
        _validate_precision(self.precision)
        _validate_calibration_method(self.calibration_method)

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to ONNXQuantizationConfig."""
        return cls(**state_dict)


class ONNXQuantizer:
    """NVIDIA ModelOpt ONNX quantizer for TensorRT backend.

    This class provides functionality to quantize ONNX models using NVIDIA ModelOpt
    for deployment with TensorRT. It supports various quantization modes including
    INT8, FP8, and INT4 quantization with different calibration methods.

    The quantizer can be used in two ways:
    1. Standalone for quantizing existing ONNX models
    2. Integrated with TensorRTBackend for end-to-end quantization

    Example:
        # Standalone usage
        quantizer = ONNXQuantizer()
        quantized_path = quantizer.quantize(
            onnx_path="model.onnx",
            name="my_model",
            precision="int8",
            calibration_method="max"
        )

        # Integrated usage (automatic when using TensorRTBackend)
        config = TensorRTBackendConfig(
            quantization_type="onnx",
            precision="int8",
            onnx_quantization_calibration_method="awq_lite"
        )
        backend = TensorRTBackend(config=config)

    Supported quantization modes:
        - int8: 8-bit integer quantization
        - fp8: 8-bit floating point quantization
        - int4: 4-bit integer quantization (with AWQ support)

    Supported calibration methods:
        - max: Max calibration (default, INT8/FP8 only)
        - entropy: Entropy-based calibration (INT8/FP8 only)
        - awq_lite: Activation-aware Weight Quantization (INT4 only)
        - awq_clip: Activation-aware Weight Quantization (INT4 only)
        - awq_full: Activation-aware Weight Quantization (INT4 only)
        - rtn_dq: RTN-DQ calibration (INT4 only)
    """

    @staticmethod
    def apply_model_opt_post_processing(onnx_path: Path):
        """Apply ModelOpt post-processing optimizations to the exported ONNX model.

        Args:
            onnx_path: Path to the ONNX model file
        """
        logger.info("Applying ModelOpt post-processing optimizations to ONNX model")

        # Load the exported ONNX model
        onnx_model = onnx.load(onnx_path)

        # Used for FP4 nodes
        logger.info("Applying nvfp4 quant exporter transformation")
        post_processed_model = modelopt_fp4_exporter(onnx_model)

        #  Workaround for missing ir_version and opset, this is not needed for TensorRT
        if not hasattr(post_processed_model, "ir_version"):
            post_processed_model.ir_version = onnx_model.ir_version
        if not hasattr(post_processed_model, "opset_import"):
            post_processed_model.opset_import = onnx_model.opset_import

        # Save the post-processed model back to the same path
        onnx.save(post_processed_model, onnx_path)

        # self.verify_model(onnx_path)

        logger.info("ModelOpt post-processing completed successfully")

    def quantize(
        self,
        input_onnx_path: str | Path,
        output_path: str | Path,
        config: ONNXQuantizationConfig,
        samples: Sequence[Sample] | None = None,
        graph_spec: GraphSpec | None = None,
    ) -> Path:
        """Quantize the ONNX model using NVIDIA ModelOpt.

        Args:
            input_onnx_path: Path to the input ONNX model
            output_path: Path to the output ONNX model
            config: ONNXQuantizationConfig
            samples: Optional calibration dataset for quantization.
                If None, random data will be used for calibration
            graph_spec: Graph specification containing input names mapping

        Returns:
            Path to the quantized ONNX file

        Raises:
            ImportError: If ModelOpt is not installed
            ValueError: If unsupported precision is specified
            RuntimeError: If quantization fails
        """
        if samples is not None and graph_spec is not None:
            calibration_data = prepare_calibration_data(samples, graph_spec)
        else:
            calibration_data = None

        input_onnx_path = Path(input_onnx_path)
        output_path = Path(output_path)

        logger.info("Starting ONNX quantization: %s -> %s", input_onnx_path, output_path)
        logger.info("Quantization precision: %s, calibration method: %s", config.precision, config.calibration_method)

        with annotate(f"build: ONNX {config.precision} quantization"):
            # Perform quantization
            logger.info("Performing %s quantization", config.precision.upper())

            quantize_kwargs = {
                "onnx_path": input_onnx_path.as_posix(),
                "output_path": output_path.as_posix(),
                "quantize_mode": config.precision,
                "calibration_method": config.calibration_method,
            }

            if calibration_data is not None:
                quantize_kwargs["calibration_data"] = calibration_data

            moq.quantize(**quantize_kwargs)

            # Apply post-processing optimizations
            if config.use_model_opt_post_processing:
                self.apply_model_opt_post_processing(output_path)

        # Verify the quantized model exists
        if not output_path.exists():
            raise RuntimeError(f"Quantization failed: output file not created at {output_path}")

        logger.info("Successfully quantized ONNX model: %s", output_path)
        return output_path
