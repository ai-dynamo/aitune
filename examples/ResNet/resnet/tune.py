# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune ResNet model."""

import os
from logging import basicConfig, getLogger

import torch
from PIL import Image

from aitune.torch import HighestThroughputStrategy, LocalTorchStorage, Module, save, tune
from aitune.torch.backend import (
    ONNXAutoCastConfig,
    ONNXQuantizationConfig,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchEagerBackend,
    TorchInductorBackend,
    TorchInductorBackendConfig,
)
from resnet.cmd_args import get_parser
from resnet.model import get_model, get_transform

logger = getLogger(__name__)


def tune_model(
    model_name,
    image_path,
    tuned_model_path,
    max_batch_size,
):
    """Tunes ResNet model.

    Args:
        model_name: Name of the model to tune.
        image_path: Path to the input image file.
        tuned_model_path: Path to save the tuned model.
        max_batch_size: Maximum batch size.
    """
    batch_sizes = [2**n for n in range(max_batch_size.bit_length())]
    logger.info("Tuning with batch sizes: %s", batch_sizes)

    model = get_model(model_name=model_name, pretrained=True)
    transform = get_transform(model)

    img = Image.open(image_path)
    dataset = transform(img).to("cuda")

    module_name = f"example-{model_name}"

    module = Module(
        model,
        module_name,
        strategy=HighestThroughputStrategy(
            backends=[
                TensorRTBackend(
                    config=TensorRTBackendConfig(
                        quantization_config=ONNXQuantizationConfig(
                            precision="int8",
                            calibration_method="max",
                        ),
                    ),
                ),
                TensorRTBackend(
                    config=TensorRTBackendConfig(
                        quantization_config=ONNXQuantizationConfig(
                            precision="int8",
                            calibration_method="max",
                        ),
                        use_dynamo=False,
                    ),
                ),
                TensorRTBackend(config=TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(precision="fp16"))),
                TensorRTBackend(
                    config=TensorRTBackendConfig(
                        quantization_config=ONNXAutoCastConfig(precision="fp16"), use_dynamo=False
                    )
                ),
                # Gives 3x TRT throughput but after load if fails
                TorchAOBackend(config=TorchAOBackendConfig(quantization="int8wo")),
                TorchInductorBackend(
                    config=TorchInductorBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)
                ),
                TorchEagerBackend(),
            ]
        ).enable_find_max_batch_size(False),
    )

    logger.info("Tuning module: %s", model_name)
    tune(module, dataset, batch_sizes=batch_sizes)
    logger.info("Tuning completed.")

    save(module, tuned_model_path, storage=LocalTorchStorage(remove_checkpoint_after_tune=True))
    logger.info("Model saved to %s", tuned_model_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = get_parser().parse_args()

    tune_model(
        model_name=args.model_name,
        image_path=args.image_path,
        tuned_model_path=args.tuned_model_path,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
