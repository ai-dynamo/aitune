# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune YOLOv10n from ONNX and save an AITune checkpoint."""

import argparse
import os
from logging import basicConfig
from pathlib import Path

from aitune.torch import MaxThroughputStrategy, Module, PerformanceValidationMode, save, tune
from aitune.torch.backend import ONNXRuntimeBackend, TensorRTBackend
from aitune.torch.dataloader import DynamicShapeDataset
from yolo.cmd_args import add_tuned_model_path_arg
from yolo.model import onnx_model, sample_input


def tune_model(checkpoint: Path) -> None:
    """Tune the ONNX model and save its deployment-capable artifact."""

    onnx_module = onnx_model()
    input_name = "images"
    images = sample_input()

    strategy = MaxThroughputStrategy([
        ONNXRuntimeBackend(),
        TensorRTBackend(),
    ])
    strategy.enable_find_max_batch_size(False)
    strategy.enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
    module = Module(onnx_module, "yolov10n", strategy=strategy)
    try:
        tune(
            module,
            DynamicShapeDataset([{input_name: images[0]}]),
            batch_sizes=[1],
            device="cuda",
            ignore_failing_modules=False,
        )

        save(module, checkpoint)
        print(f"AITune checkpoint: {checkpoint}")
    finally:
        if module.state.name == "TUNED":
            module.deactivate()
        onnx_module.deactivate()


def main() -> None:
    """Parse arguments and start YOLO tuning."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("triton",), default="triton")
    add_tuned_model_path_arg(parser)
    args = parser.parse_args()
    tune_model(args.tuned_model_path)


if __name__ == "__main__":
    main()
