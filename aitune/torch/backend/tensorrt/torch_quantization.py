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
"""NVIDIA ModelOpt PyTorch quantization module for TensorRT backend."""

import gc
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn

from aitune.torch.module.recording_module import Sample
from aitune.utils.system_monitor import SystemMonitor

# Setup logger
logger = logging.getLogger(__name__)

QuantizationConfig = Literal[
    "NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG", "INT8_DEFAULT_CFG", "INT8_SMOOTHQUANT_CFG", "INT4_AWQ_CFG"
]


@dataclass
class TorchQuantizationConfig:
    """Configuration for Torch quantization.

    Args:
        quantization_config: ModelOpt quantization configuration name.
            Available options: "NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG",
            "INT8_DEFAULT_CFG", "INT8_SMOOTHQUANT_CFG", "INT4_AWQ_CFG"
        device: Target device for quantization (default: "cuda")
        verbose: Enable verbose logging
    """

    quantization_config: QuantizationConfig = "FP8_DEFAULT_CFG"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to TorchQuantizationConfig."""
        return cls(**state_dict)


class TorchQuantizer:
    """NVIDIA ModelOpt PyTorch quantizer for TensorRT backend.

    This class provides functionality to quantize PyTorch models using NVIDIA ModelOpt
    for deployment with TensorRT. It supports various quantization modes including
    INT8, FP8, and INT4 quantization with different calibration methods.
    """

    def __init__(self):
        """Initialize the Torch quantizer."""
        self.system_monitor = SystemMonitor()

    def quantize(
        self,
        module: nn.Module,
        sample: Sample,
        config: TorchQuantizationConfig,
    ) -> nn.Module:
        """Quantize the module using NVIDIA ModelOpt.

        This method performs quantization of a PyTorch model using NVIDIA ModelOpt.
        It supports various quantization algorithms including FP8, INT8, and INT4.

        Args:
            module: PyTorch module to export
            sample: Example input for the model
            config: TorchQuantizationConfig

        Returns:
            Quantized model

        Raises:
            ImportError: If ModelOpt is not installed
            RuntimeError: If quantization or export fails
        """
        logger.info("Quantizing PyTorch module")

        # Note: Create copy because ModelOpt quantizer modifies the model during quantization
        # https://nvidia.github.io/TensorRT-Model-Optimizer/guides/6_save_load.html
        logger.info("Creating a deep copy of the model for quantization")
        with self.system_monitor.system_stats_context(log_label="Model deep copy"):
            model_copy = deepcopy(module)

            # Offload model to CPU
            module.to("cpu")
            self._clean_memory()

            # Move model to target device
            model_copy = model_copy.to(config.device)

        # Prepare calibration data
        calibration_sample = self._prepare_calibration_sample(sample=sample, device=config.device)

        # Get quantization configuration
        quant_config = self._get_quantization_config(config.quantization_config)

        # Create forward loop for calibration
        forward_loop = self._create_forward_loop(calibration_samples=[calibration_sample])

        # Quantize the model
        logger.info("Starting model quantization with config: %s", config.quantization_config)
        with self.system_monitor.system_stats_context(log_label="Model quantization"):
            quantized_model = mtq.quantize(model_copy, quant_config, forward_loop)

        logger.info("Model quantization completed successfully")

        # Configure quantizers for ONNX export
        self._configure_quantizers_for_onnx_export(quantized_model)

        # Clean up GPU memory and references
        del model_copy
        self._clean_memory()

        return quantized_model

    def _create_forward_loop(self, calibration_samples: list[Any]):
        """Create forward loop function for model calibration.

        Args:
            calibration_samples: List of calibration samples

        Returns:
            Forward loop function for calibration
        """

        def forward_loop(model):
            """Forward loop for model calibration."""
            for cal_sample in calibration_samples:
                with torch.no_grad():
                    if isinstance(cal_sample, tuple) and len(cal_sample) == 2:
                        # Handle Sample tuple format (args, kwargs)
                        args, kwargs = cal_sample
                        model(*args, **kwargs)
                    elif isinstance(cal_sample, (list, tuple)):
                        model(*cal_sample)
                    elif isinstance(cal_sample, dict):
                        model(**cal_sample)
                    else:
                        model(cal_sample)

        return forward_loop

    def _prepare_calibration_sample(
        self,
        sample: Sample,
        device: str,
    ) -> Sample:
        """Move sample to target device and return as Sample tuple.

        Args:
            sample: Example input sample
            device: Target device for calibration

        Returns:
            List containing the sample converted to target device
        """
        args, kwargs = sample

        # Convert args to target device
        device_args = ()
        if args:
            device_args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)

        # Convert kwargs to target device
        device_kwargs = {}
        if kwargs:
            device_kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # Return as Sample tuple
        return (device_args, device_kwargs)

    def _get_quantization_config(self, config_name: str) -> dict[str, Any]:
        """Get quantization configuration by name.

        Args:
            config_name: Name of the quantization configuration

        Returns:
            Quantization configuration dictionary

        Raises:
            ImportError: If ModelOpt is not available
            ValueError: If configuration name is not supported
        """
        config_mapping = {
            "NVFP4_DEFAULT_CFG": mtq.NVFP4_DEFAULT_CFG,
            "FP8_DEFAULT_CFG": mtq.FP8_DEFAULT_CFG,
            "INT8_DEFAULT_CFG": mtq.INT8_DEFAULT_CFG,
            "INT8_SMOOTHQUANT_CFG": mtq.INT8_SMOOTHQUANT_CFG,
            "INT4_AWQ_CFG": mtq.INT4_AWQ_CFG,
        }

        if config_name not in config_mapping:
            available_configs = list(config_mapping.keys())
            raise ValueError(
                "Unsupported quantization config: %s] Available options: %s", config_name, available_configs
            )

        return config_mapping[config_name]

    def _configure_quantizers_for_onnx_export(self, model: nn.Module, verbose: bool = False):
        """Configure quantizers in the model for ONNX export.

        Sets appropriate attributes on input and weight quantizers to ensure proper ONNX export.
        For input quantizers, sets dynamic quantization mode and half precision.
        For weight quantizers, sets static quantization mode.

        Args:
            model: The quantized PyTorch model with quantizers to configure
            verbose: Whether to log detailed information about each configured quantizer
        """
        logger.info("Configuring quantizers for ONNX export")

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                if hasattr(module, "input_quantizer") and module.input_quantizer is not None:
                    logger.info("Setting quantizer attributes for %s.input_quantizer", name)
                    module.input_quantizer._onnx_quantizer_type = "dynamic"
                    module.input_quantizer._trt_high_precision_dtype = "Half"

                if hasattr(module, "weight_quantizer") and module.weight_quantizer is not None:
                    logger.info("Setting quantizer attributes for %s.weight_quantizer", name)
                    module.weight_quantizer._onnx_quantizer_type = "static"

    def _clean_memory(self):
        """Clean up memory."""
        torch.cuda.empty_cache()
        gc.collect()
