# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import datasets
from PIL import Image
from transformers import AutoImageProcessor  # pytype: disable=import-error

from aitune.torch.dataloader import samples_generator


def random_image_generator():
    """Generates random RGB image."""
    arr = (random.randint(0, 255) for _ in range(224 * 224 * 3))
    img = Image.frombytes("RGB", (224, 224), bytes(arr))
    return img


def test_resnet_dataset():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    def process_images(example):
        # Load image from file as numpy array
        # Handle both RGB and RGBA images
        image = example["image"]
        if hasattr(image, "mode") and image.mode == "RGBA":
            # Convert RGBA to RGB if needed
            image = image.convert("RGB")

        processed = processor(images=image, input_data_format="channels_last", return_tensors="pt")
        return {
            "pixel_values": processed.pixel_values.squeeze(0),
        }

    random_dataset = [{"image": random_image_generator()} for _ in range(12)]

    dataset = datasets.Dataset.from_list(random_dataset).map(process_images, remove_columns=["image"])

    samples = [*samples_generator(dataset, [4])]

    assert len(samples) == 3

    bs, args, kwargs = samples[0]
    assert bs == 4
    assert len(args) == 0
    assert len(kwargs) == 1

    assert len(kwargs["pixel_values"]) == 4
    assert kwargs["pixel_values"].shape == (4, 3, 224, 224)


if __name__ == "__main__":
    test_resnet_dataset()
