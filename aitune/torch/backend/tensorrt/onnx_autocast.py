# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA ModelOpt ONNX quantization module for TensorRT backend."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modelopt
import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision
from packaging.version import Version

from aitune.torch.backend.tensorrt.modelopt_calibration import prepare_calibration_data
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.utils.monitoring import annotate

# Setup logger
logger = logging.getLogger(__name__)

MixedPrecision = Literal["fp16", "bf16"]


@dataclass
class ONNXAutoCastConfig:
    """Configuration for mixed precision quantization.

    Args:
        precision: Mixed precision quantization precision ("fp16", "bf16")
    """

    precision: MixedPrecision = "fp16"
    keep_io_types: bool = True

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to MixedPrecisionConfig."""
        return cls(**state_dict)


class ONNXAutoCast:
    """NVIDIA ModelOpt ONNX autocast for TensorRT backend.

    This class provides functionality to autocast ONNX models using NVIDIA ModelOpt.
    """

    def autocast(
        self,
        input_onnx_path: str | Path,
        output_path: str | Path,
        config: ONNXAutoCastConfig,
        samples: list[Sample] | None = None,
        graph_spec: GraphSpec | None = None,
    ) -> Path:
        """Autocast the ONNX model using NVIDIA ModelOpt.

        Args:
            input_onnx_path: Path to the input ONNX model
            output_path: Path to the output ONNX model
            config: ONNXAutoCastConfig
            samples: Optional calibration dataset for quantization.
                If None, random data will be used for calibration
            graph_spec: Graph specification containing input names mapping


        Returns:
            Path to the autocasted ONNX file

        Raises:
            ValueError: If unsupported precision is specified
            RuntimeError: If autocast fails
        """
        # Should be fixed in ModelOpt 0.43.0 or later (https://github.com/NVIDIA/Model-Optimizer/pull/815)
        use_calibration = (
            Version(modelopt.__version__) > Version("0.42.0") and samples is not None and graph_spec is not None
        )
        if use_calibration:
            calibration_data = prepare_calibration_data(samples, graph_spec)
        else:
            calibration_data = None

        input_onnx_path = Path(input_onnx_path)
        output_path = Path(output_path)

        logger.info("Starting ONNX autocast: %s -> %s", input_onnx_path, output_path)
        logger.info("Autocast precision: %s", config.precision)

        with annotate(f"build: ONNX {config.precision} autocast"):
            # Perform autocast
            logger.info("Performing %s autocast", config.precision.upper())

            autocast_kwargs = {
                "onnx_path": input_onnx_path.as_posix(),
                "low_precision_type": config.precision,
                "keep_io_types": config.keep_io_types,
                "providers": ["cuda"],
            }

            if calibration_data is not None:
                autocast_kwargs["calibration_data"] = calibration_data

            converted_model = convert_to_mixed_precision(**autocast_kwargs)

            # Save the autocasted model
            onnx.save(converted_model, output_path)

        # Verify the quantized model exists
        if not output_path.exists():
            raise RuntimeError(f"Autocast failed: output file not created at {output_path}")

        logger.info("Successfully autocasted ONNX model: %s", output_path)
        return output_path
