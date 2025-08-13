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
"""Tune ResNet model."""

import os
from logging import basicConfig, getLogger

from PIL import Image

from aitune.torch.checkpoint.local_torch_storage import LocalTorchStorage
from aitune.torch.module.wrapper_module import Module
from aitune.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy
from aitune.torch.tuning import save, tune
from resnet.cmd_args import get_parser
from resnet.model import get_model, get_transform

logger = getLogger(__name__)


def tune_model(
    model_name,
    image_path,
    tuned_model_path,
):
    """Tunes ResNet model.

    Args:
        model_name: Name of the model to tune.
        image_path: Path to the input image file.
        tuned_model_path: Path to save the tuned model.
    """
    model = get_model(model_name=model_name, pretrained=True)
    transform = get_transform(model)

    img = Image.open(image_path)
    dataset = transform(img).to("cuda")

    module_name = f"example-{model_name}"
    module = Module(model, module_name, strategy=HighestThroughputStrategy())

    logger.info("Tuning module: %s", model_name)
    tune(module, dataset)
    logger.info("Tuning completed.")

    save(module, tuned_model_path, storage=LocalTorchStorage(remove_checkpoint_after_tune=True))
    logger.info("Model saved to %s", tuned_model_path)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("AITUNE_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = get_parser().parse_args()

    tune_model(
        model_name=args.model_name,
        image_path=args.image_path,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
