# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a tuned YOLO deployment through Triton gRPC."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tritonclient.grpc as grpcclient

from aitune.torch import Module, load
from aitune.torch.module import onnx_checkpoint_placeholder
from yolo.cmd_args import add_tuned_model_path_arg
from yolo.model import sample_input


def run_inference(checkpoint: Path, triton_url: str) -> None:
    """Check that Triton accepts the checkpoint's input dtype and returns finite detections."""
    tuned_model = cast(Module, load(onnx_checkpoint_placeholder(), checkpoint))
    try:
        artifact = tuned_model.artifact()
        if len(artifact.inputs) != 1:
            raise RuntimeError("YOLO checkpoint must have one image input")
        images = sample_input(dtype=getattr(torch, artifact.inputs[0].dtype.value))
        artifact_input_name = artifact.input_names[0]
        if artifact_input_name not in {"images", "input_x"}:
            raise RuntimeError(f"Unexpected YOLO input name: {artifact_input_name}")
        client = grpcclient.InferenceServerClient(url=triton_url)
        model_name = "yolov10n"
        metadata = client.get_model_metadata(model_name)
        if len(metadata.inputs) != 1 or len(metadata.outputs) != 1:
            raise RuntimeError("YOLO Triton deployment must have one input and one output")
        model_input = grpcclient.InferInput(metadata.inputs[0].name, list(images.shape), metadata.inputs[0].datatype)
        model_input.set_data_from_numpy(images.cpu().numpy())
        output_name = metadata.outputs[0].name
        result = client.infer(model_name, inputs=[model_input], outputs=[grpcclient.InferRequestedOutput(output_name)])
        output = result.as_numpy(output_name)
        if output is None or output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
            raise RuntimeError("Unexpected YOLO Triton detection output shape")
        predictions = output[0]
        if not np.isfinite(predictions).all():
            raise RuntimeError("YOLO Triton detections contain non-finite values")
        print(f"Validated Triton inference: output shape {predictions.shape}", flush=True)
    finally:
        tuned_model.deactivate()


def main() -> None:
    """Parse arguments and check Triton inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_tuned_model_path_arg(parser)
    parser.add_argument("--triton-url", default="localhost:8001")
    args = parser.parse_args()
    run_inference(args.tuned_model_path, args.triton_url)


if __name__ == "__main__":
    main()
