# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune BERT from Torch or ONNX and save an AITune checkpoint."""

import argparse
import os
from logging import basicConfig
from pathlib import Path

import torch
from transformers import AutoConfig

from aitune.torch import MaxThroughputStrategy, Module, save, tune
from aitune.torch.backend import (
    ONNXRuntimeBackend,
    TensorRTBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
)
from aitune.torch.dataloader import DynamicShapeDataset
from aitune.torch.module import OnnxModule
from bert.cmd_args import add_model_name_arg, add_output_path_arg
from bert.model import onnx_model, torch_model

BATCH_SIZES = [1, 2, 4]
SEQUENCE_LENGTHS = [64, 128, 256]


def tune_model(source_kind: str, target: str, checkpoint: Path, model_name: str) -> None:
    """Tune the selected source for Python or Triton inference."""
    vocab_size = AutoConfig.from_pretrained(model_name).vocab_size
    generator = torch.Generator().manual_seed(0)
    tokens = {
        length: torch.randint(vocab_size, (max(BATCH_SIZES), length), generator=generator)
        for length in SEQUENCE_LENGTHS
    }
    source = (
        torch_model(model_name).cuda()
        if source_kind == "torch"
        else onnx_model(checkpoint.parent / "model.onnx", tokens[128], model_name)
    )
    requests = {length: values.cuda() for length, values in tokens.items()}

    backends = [TensorRTBackend()]
    if source_kind == "torch":
        backends.append(TorchInductorAotBackend() if target == "triton" else TorchInductorJitBackend())
    else:
        backends.append(ONNXRuntimeBackend())

    strategy = MaxThroughputStrategy(backends)
    module = Module(source, "bert", strategy=strategy)
    try:
        tune(
            module,
            # The first sample supplies values for the single representative deployment input.
            DynamicShapeDataset([{"input_ids": requests[length][0]} for length in (128, 64, 256)]),
            batch_sizes=BATCH_SIZES,
            device="cuda",
            ignore_failing_modules=False,
        )

        save(module, checkpoint)
        print(f"AITune checkpoint: {checkpoint}")
    finally:
        if module.state.name == "TUNED":
            module.deactivate()
        if isinstance(source, OnnxModule):
            source.deactivate()


def main() -> None:
    """Parse arguments and start BERT tuning."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("torch", "onnx"), required=True)
    parser.add_argument("--target", choices=("python", "triton"), default="triton")
    add_model_name_arg(parser)
    add_output_path_arg(parser)
    args = parser.parse_args()
    tune_model(args.source, args.target, args.output_path, args.model_name)


if __name__ == "__main__":
    main()
