# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ResNet inference through Triton Inference Server."""

import argparse
from logging import basicConfig, getLogger

import torch
import tritonclient.grpc as grpcclient
from PIL import Image

from resnet.cmd_args import add_common_args
from resnet.model import get_model, get_transform

logger = getLogger(__name__)


def add_args(parser):
    """Add Triton client arguments."""
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--expected-class-id", type=int, default=207, help="Expected ImageNet class ID")
    return parser


def main():
    """Send an image to Triton and validate the predicted class."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    parser = add_common_args(argparse.ArgumentParser(description="Run ResNet inference through Triton"))
    args = add_args(parser).parse_args()

    transform = get_transform(get_model(model_name=args.model_name, pretrained=False))
    image = Image.open(args.image_path)
    batch = transform(image).unsqueeze(0)
    if args.dynamic_shapes:
        batch = torch.nn.functional.interpolate(batch, size=(256, 256), mode="bilinear", align_corners=False).repeat(
            2, 1, 1, 1
        )
    batch = batch.half().numpy()

    client = grpcclient.InferenceServerClient(url=args.triton_url)
    metadata = client.get_model_metadata(args.model_name)
    if len(metadata.inputs) != 1 or len(metadata.outputs) != 1:
        raise RuntimeError("The ResNet Triton example expects one input and one output")

    model_input = grpcclient.InferInput(metadata.inputs[0].name, batch.shape, metadata.inputs[0].datatype)
    model_input.set_data_from_numpy(batch)
    output_name = metadata.outputs[0].name
    result = client.infer(
        args.model_name,
        inputs=[model_input],
        outputs=[grpcclient.InferRequestedOutput(output_name)],
    )
    scores = result.as_numpy(output_name)
    class_id = int(torch.from_numpy(scores[0]).argmax())
    logger.info("Class ID: %s", class_id)
    if class_id != args.expected_class_id:
        raise ValueError(f"Predicted class {class_id} does not match expected class {args.expected_class_id}")


if __name__ == "__main__":
    main()
