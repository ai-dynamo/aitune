# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a saved BERT checkpoint through Python inference."""

import argparse
from pathlib import Path
from typing import cast

import torch
from transformers import AutoConfig

from aitune.torch import Module, load
from aitune.torch.module import OnnxModule
from bert.cmd_args import add_model_name_arg, add_tuned_model_path_arg
from bert.model import torch_model
from bert.tune import BATCH_SIZES, SEQUENCE_LENGTHS


def run_inference(checkpoint: Path, model_name: str, target: str) -> None:
    """Check that the compiled checkpoint handles every supported input shape."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}; run bert-tune first")

    generator = torch.Generator().manual_seed(1)
    vocab_size = AutoConfig.from_pretrained(model_name).vocab_size
    samples = [
        torch.randint(vocab_size, (batch_size, length), generator=generator).cuda()
        for length in SEQUENCE_LENGTHS
        for batch_size in [*BATCH_SIZES, 3]
    ]
    # A Torch JIT checkpoint needs its original module; Triton-capable artifacts are self-contained.
    source = torch_model(model_name).cuda() if target == "python" else OnnxModule.for_checkpoint()
    tuned_model = cast(Module, load(source, checkpoint))
    try:
        for sample in samples:
            result = tuned_model(input_ids=sample)
            outputs = list(result.values()) if isinstance(result, dict) else list(result)
            if len(outputs) != 2 or outputs[0].shape[:2] != sample.shape or outputs[1].shape[0] != sample.shape[0]:
                raise RuntimeError(f"Unexpected BERT output shapes for input {tuple(sample.shape)}")
            if not all(torch.isfinite(output).all() for output in outputs):
                raise RuntimeError(f"Non-finite BERT output for input {tuple(sample.shape)}")
            print(f"Validated Python inference: batch={sample.shape[0]}, sequence={sample.shape[1]}", flush=True)
    finally:
        tuned_model.deactivate()


def main() -> None:
    """Parse arguments and run Python inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("python", "triton"), default="triton")
    add_model_name_arg(parser)
    add_tuned_model_path_arg(parser)
    args = parser.parse_args()
    run_inference(args.tuned_model_path, args.model_name, args.target)


if __name__ == "__main__":
    main()
