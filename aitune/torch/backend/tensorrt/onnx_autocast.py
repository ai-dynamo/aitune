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
"""NVIDIA ModelOpt ONNX quantization module for TensorRT backend."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import onnx
import torch
from modelopt.onnx.autocast import convert_to_mixed_precision

from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.recording_module import Sample
from aitune.utils.system_monitor import SystemMonitor

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

    def __init__(self):
        """Initialize the ONNX quantizer."""
        self.system_monitor = SystemMonitor()

    def _prepare_calibration_data(self, data: list[Sample], graph_spec: GraphSpec) -> list[dict[str, torch.Tensor]]:
        """Prepare calibration data with proper input names mapping.

        Args:
            data: List of Sample objects containing calibration data
            graph_spec: Graph specification containing input names mapping

        Returns:
            List of dictionaries mapping input names to tensors for calibration
        """
        logger.debug("Preparing calibration data with proper input names for %d samples", len(data))
        calibration_data_with_names = []

        for sample in data:
            # Use the graph spec to flatten the sample and get proper input names
            input_dict = graph_spec.input_spec.flatten_sample(sample)
            calibration_data_with_names.append(input_dict)
            logger.debug("Mapped sample to input names: %s", list(input_dict.keys()))

        logger.debug(
            "Successfully prepared %d calibration samples with proper input names", len(calibration_data_with_names)
        )
        return calibration_data_with_names

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
            RuntimeError: If quantization fails
        """
        # TODO: Temporarily disable, re-enable when proper calibration data handling is implemented.
        # if samples is not None:
        #     calibration_data = self._prepare_calibration_data(samples, graph_spec)
        # else:
        #     calibration_data = None

        input_onnx_path = Path(input_onnx_path)
        output_path = Path(output_path)

        logger.debug("Starting ONNX autocast: %s -> %s", input_onnx_path, output_path)
        logger.debug("Autocast precision: %s", config.precision)

        with self.system_monitor.system_stats_context(log_label=f"ONNX {config.precision} autocast"):
            # Perform autocast
            logger.debug("Performing %s autocast", config.precision.upper())

            autocast_kwargs = {
                "onnx_path": input_onnx_path.as_posix(),
                "low_precision_type": config.precision,
                "keep_io_types": config.keep_io_types,
                "providers": ["cuda"],
            }

            # if calibration_data is not None:
            #     quantize_kwargs["calibration_data"] = calibration_data

            converted_model = convert_to_mixed_precision(**autocast_kwargs)

            # Save the autocasted model
            onnx.save(converted_model, output_path)

        # Verify the quantized model exists
        if not output_path.exists():
            raise RuntimeError(f"Autocast failed: output file not created at {output_path}")

        logger.debug("Successfully autocasted ONNX model: %s", output_path)
        return output_path
