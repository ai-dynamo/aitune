# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a tuned BERT deployment through Triton gRPC."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tritonclient.grpc as grpcclient

from aitune.torch import Module, load
from bert.tune import BATCH_SIZES, SEQUENCE_LENGTHS


@torch.inference_mode()
def main() -> None:
    """Compare Triton outputs with the restored compiled checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--triton-url", default="localhost:8001")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "artifacts"
    checkpoint = output_dir / "bert.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run bert-tune first")
    generator = torch.Generator().manual_seed(1)
    samples = [
        torch.randint(30522, (batch_size, length), generator=generator)
        for length in SEQUENCE_LENGTHS
        for batch_size in [*BATCH_SIZES, 3]
    ]
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
    try:
        client = grpcclient.InferenceServerClient(url=args.triton_url)
        model_name = "bert"
        metadata = client.get_model_metadata(model_name)
        if len(metadata.inputs) != 1 or len(metadata.outputs) != 2:
            raise RuntimeError("BERT Triton deployment must have one input and two outputs")
        for tokens in samples:
            reference = tuned_model(input_ids=tokens.cuda())
            expected = list(reference.values()) if isinstance(reference, dict) else list(reference)
            model_input = grpcclient.InferInput(
                metadata.inputs[0].name, list(tokens.shape), metadata.inputs[0].datatype
            )
            model_input.set_data_from_numpy(tokens.numpy())
            result = client.infer(
                model_name,
                inputs=[model_input],
                outputs=[grpcclient.InferRequestedOutput(output.name) for output in metadata.outputs],
            )
            for output, baseline in zip(metadata.outputs, expected, strict=True):
                actual = result.as_numpy(output.name)
                np.testing.assert_allclose(actual, baseline.cpu().numpy(), rtol=1e-2, atol=1e-2)
            print(f"Validated Triton inference: batch={tokens.shape[0]}, sequence={tokens.shape[1]}", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
