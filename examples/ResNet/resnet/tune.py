# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune ResNet model."""

import os
from logging import basicConfig, getLogger

import torch
from aitune_examples_common.checkpoint import copy_checkpoint_to_tmp
from PIL import Image

from aitune.torch import (
    BatchDim,
    DynamicDim,
    LatencyBudgetStrategy,
    LocalTorchStorage,
    MaxThroughputStrategy,
    Module,
    save,
    tune,
)
from aitune.torch.backend import (
    ONNXAutoCastConfig,
    ONNXQuantizationConfig,
    ONNXRuntimeBackend,
    TensorRTBackend,
    TensorRTBackendConfig,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchInductorJitBackendConfig,
)
from aitune.torch.task.profiling import ProfilingConfig
from resnet.cmd_args import get_parser
from resnet.model import get_model, get_transform
from resnet.triton import TRITON_LATENCY_BUDGET_MS

logger = getLogger(__name__)


def _strategy(target, max_batch_size, batch_sizes):
    """Select backends compatible with the requested deployment target."""
    if target == "triton":
        backends = [
            TensorRTBackend(),
            ONNXRuntimeBackend(),
            TorchInductorAotBackend(),
        ]
        return LatencyBudgetStrategy(
            latency_budget_ms=TRITON_LATENCY_BUDGET_MS,
            backends=backends,
            profiling_config=ProfilingConfig(batch_sizes=batch_sizes) if max_batch_size is not None else None,
        ).enable_find_max_batch_size(max_batch_size is None)
    else:
        backends = [
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
                config=TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(precision="fp16"), use_dynamo=False)
            ),
            TorchAOBackend(config=TorchAOBackendConfig(quantization="int8wo")),
            TorchInductorJitBackend(
                config=TorchInductorJitBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)
            ),
        ]
    # TODO: Replace target-specific lists when AITune can select backends by deployment capability.
    return MaxThroughputStrategy(backends=backends).enable_find_max_batch_size(False)


def tune_model(
    model_name,
    image_path,
    tuned_model_path,
    max_batch_size,
    dynamic_shapes,
    target,
):
    """Tunes ResNet model.

    Args:
        model_name: Name of the model to tune.
        image_path: Path to the input image file.
        tuned_model_path: Path to save the tuned model.
        max_batch_size: Optional batch-size ceiling. Triton discovers a ceiling when omitted.
        dynamic_shapes: Whether to use user-provided dynamic shapes.
        target: Runtime family for which the package is tuned.
    """
    if max_batch_size is not None and max_batch_size < 2:
        raise ValueError("max_batch_size must be at least 2 to identify the batch axis")
    if target == "triton" and dynamic_shapes and max_batch_size is None:
        raise ValueError("dynamic shapes require an explicit max_batch_size for Triton tuning")

    initial_max_batch_size = max_batch_size if max_batch_size is not None else 4
    batch_sizes = [2**n for n in range(initial_max_batch_size.bit_length())]
    if initial_max_batch_size not in batch_sizes:
        batch_sizes.append(initial_max_batch_size)
    logger.info("Tuning with batch sizes: %s", batch_sizes)

    model = get_model(model_name=model_name, pretrained=True)
    transform = get_transform(model)

    img = Image.open(image_path)
    dataset = transform(img).to(device="cuda", dtype=torch.float16)

    module_name = f"example-{model_name}"

    shape_definitions = None
    if dynamic_shapes:
        batch = BatchDim("batch", min=1, opt=initial_max_batch_size, max=initial_max_batch_size)
        height = DynamicDim("spatial", min=224, opt=224, max=256)
        width = DynamicDim("spatial", min=224, opt=224, max=256)
        shape_definitions = {"x": (batch, 3, height, width)}

    module = Module(
        model,
        module_name,
        strategy=_strategy(target, max_batch_size, batch_sizes),
        dynamic_shapes=shape_definitions,
    )

    logger.info("Tuning module: %s", model_name)
    tune(module, dataset, batch_sizes=batch_sizes, ignore_failing_modules=target != "triton")
    logger.info("Tuning completed.")

    save(module, tuned_model_path, storage=LocalTorchStorage(remove_checkpoint_after_tune=True))
    logger.info("Model saved to %s", tuned_model_path)
    relocated_path = copy_checkpoint_to_tmp(tuned_model_path)
    logger.info("Checkpoint copied to %s", relocated_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    parser = get_parser()
    parser.add_argument(
        "--target",
        choices=("python", "triton"),
        default="python",
        help="Select backends for Python or direct Triton deployment (default: python)",
    )
    args = parser.parse_args()

    tune_model(
        model_name=args.model_name,
        image_path=args.image_path,
        tuned_model_path=args.tuned_model_path,
        max_batch_size=args.max_batch_size,
        dynamic_shapes=args.dynamic_shapes,
        target=args.target,
    )


if __name__ == "__main__":
    main()
