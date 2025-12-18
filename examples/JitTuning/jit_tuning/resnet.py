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

import argparse
from logging import basicConfig, getLogger
from time import perf_counter

import numpy as np
import timm
import torch

logger = getLogger(__name__)


def get_cmd_args():
    parser = argparse.ArgumentParser(description="ResNet inference benchmark")
    parser.add_argument(
        "--max-batch-size", type=int, default=512, help="Maximum batch size for inference (default: 512)"
    )
    parser.add_argument(
        "--num-batches", type=int, default=10, help="Number of batches to run for benchmarking (default: 10)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    return args


def initialize_logger(log_level):
    basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def warm_up(resnet, max_batch_size):
    """Warmup phase for JIT tuning.

    We run model with bs=1 and max_batch_size so that JIT know min/max shapes.
    """
    logger.info("Starting warmup")
    start_time = perf_counter()
    resnet(torch.randn(1, 3, 224, 224, device="cuda"))  # min shapes
    resnet(torch.randn(max_batch_size, 3, 224, 224, device="cuda"))  # max shapes
    logger.info("Warmup took %.1f seconds", perf_counter() - start_time)


def run(
    max_batch_size: int = 512,
    num_batches: int = 10,
    log_level: str = "INFO",
):
    initialize_logger(log_level)
    resnet = timm.create_model("resnet50", pretrained=False).to("cuda")
    resnet.eval()
    warm_up(resnet, max_batch_size)

    durations = []
    for _ in range(num_batches):
        start_time = perf_counter()
        with torch.no_grad():
            resnet(torch.randn(max_batch_size, 3, 224, 224, device="cuda"))
        durations.append(perf_counter() - start_time)

    durations = np.array(durations)
    logger.info("Average batch generation time: %.2f ± %.2f seconds", durations.mean(), durations.std())


def main():
    args = get_cmd_args()
    run(max_batch_size=args.max_batch_size, num_batches=args.num_batches, log_level=args.log_level)


if __name__ == "__main__":
    main()
