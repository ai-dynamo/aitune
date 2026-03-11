# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference script for ResNet models."""

from logging import basicConfig, getLogger

import torch
from PIL import Image

from aitune.torch import load
from resnet.cmd_args import get_parser
from resnet.model import get_model, get_transform

logger = getLogger(__name__)


def add_args(parser):
    """Add additional arguments to the parser."""
    parser.add_argument(
        "--expected-class-id",
        type=int,
        help="Expected class ID to verify the prediction against",
        default=207,  # golden retriever
    )
    return parser


def do_inference(model_name, tuned_model_path, image_path, expected_class_id=None):
    """Do inference on a tuned ResNet model.

    Args:
        model_name: Name of the model to tune.
        image_path: Path to the input image file.
        tuned_model_path: Path to the tuned model.
        expected_class_id: Expected class ID to verify the prediction against.
    """
    model = get_model(model_name=model_name, pretrained=False)
    transform = get_transform(model)
    tuned_model = load(model, tuned_model_path)

    img = Image.open(image_path)
    x = transform(img).to("cuda")
    batch = x.unsqueeze(0)  # during tuning model sees batches, we have to unsqueeze to see single sample
    out = tuned_model(batch)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    class_id = torch.argmax(actual_probs)
    logger.info("Class ID: %s", class_id.item())

    if expected_class_id is not None:
        if class_id == expected_class_id:
            logger.info("Predicted class matches expected class %s", expected_class_id)
        else:
            raise ValueError(f"Predicted class {class_id} does not match expected class {expected_class_id}")


def main():
    """Entry point for the script."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = add_args(get_parser()).parse_args()

    do_inference(
        model_name=args.model_name,
        image_path=args.image_path,
        tuned_model_path=args.tuned_model_path,
        expected_class_id=args.expected_class_id,
    )


if __name__ == "__main__":
    main()
