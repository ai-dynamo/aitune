# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a saved BERT checkpoint through Python inference."""

import argparse
from pathlib import Path
from typing import cast

import torch

from aitune.torch import Module, load
from bert.tune import BATCH_SIZES, SEQUENCE_LENGTHS


@torch.inference_mode()
def main() -> None:
    """Check that the compiled checkpoint handles every supported input shape."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory used by bert-tune")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "artifacts"
    checkpoint = output_dir / "bert.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run bert-tune first")

    generator = torch.Generator().manual_seed(1)
    samples = [
        torch.randint(30522, (batch_size, length), generator=generator).cuda()
        for length in SEQUENCE_LENGTHS
        for batch_size in [*BATCH_SIZES, 3]
    ]
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
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


if __name__ == "__main__":
    main()
