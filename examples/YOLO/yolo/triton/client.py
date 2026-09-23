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
from yolo.tune import _image, _processor


@torch.inference_mode()
def main() -> None:
    """Compare Triton detections with the restored compiled checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--triton-url", default="localhost:8001")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(__file__).resolve().parents[2] / "artifacts"
    checkpoint = output_dir / "yolov10n.ait"
    if not checkpoint.is_file():
        parser.error(f"Missing {checkpoint}; run yolo-tune first")
    tuned_model = cast(Module, load(torch.nn.Identity(), checkpoint))
    images = _image(_processor())
    try:
        artifact_input_name = tuned_model.artifact().input_names[0]
        if artifact_input_name not in {"images", "input_x"}:
            raise RuntimeError(f"Unexpected YOLO input name: {artifact_input_name}")
        input_name = "images" if artifact_input_name == "images" else "x"
        result = tuned_model(**{input_name: images})
        output = next(iter(result.values())) if isinstance(result, dict) else result
        reference = output[0]
        reference = reference[reference[:, 4] >= 0.4].cpu().numpy()
        if not len(reference):
            raise RuntimeError("The street image produced no detections above the 0.4 confidence threshold")

        client = grpcclient.InferenceServerClient(url=args.triton_url)
        model_name = "yolov10n"
        metadata = client.get_model_metadata(model_name)
        if len(metadata.inputs) != 1 or len(metadata.outputs) != 1:
            raise RuntimeError("YOLO Triton deployment must have one input and one output")
        model_input = grpcclient.InferInput(metadata.inputs[0].name, list(images.shape), metadata.inputs[0].datatype)
        model_input.set_data_from_numpy(images.cpu().numpy())
        output_name = metadata.outputs[0].name
        result = client.infer(model_name, inputs=[model_input], outputs=[grpcclient.InferRequestedOutput(output_name)])
        predictions = result.as_numpy(output_name)[0]
        predictions = predictions[predictions[:, 4] >= 0.4]
        np.testing.assert_allclose(predictions, reference, rtol=1e-2, atol=1e-2)
        print(f"Validated {len(predictions)} detections above confidence 0.4", flush=True)
    finally:
        tuned_model.deactivate()


if __name__ == "__main__":
    main()
