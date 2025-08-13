# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
from transformers import AutoImageProcessor  # pytype: disable=import-error

from aitune.torch.dataloader import samples_generator


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

    dataset = datasets.load_dataset("huggingchat/models-logo", split="train").map(
        process_images, remove_columns=["image"]
    )

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
