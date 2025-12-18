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
"""Inspecting and patching modules."""

import time
from collections.abc import Callable
from logging import getLogger

import torch

from aitune.torch.dataloader import DataLoaderFactory, DatasetLike, ensure_enough_samples, samples_generator
from aitune.torch.inspecting.module_info import InspectedModulesInfo
from aitune.torch.inspecting.module_inspector import ModuleInspector
from aitune.torch.utils.cuda import synchronize
from aitune.utils.logging import setup_logging

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

DEFAULT_INSPECT_ITERATIONS = 10
DEFAULT_WARMUP_ITERATIONS = 5


def inspect(
    obj: Callable | torch.nn.Module,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    inference_function: Callable | None = None,
    number_of_iterations: int = DEFAULT_INSPECT_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    min_depth: int = 0,
) -> InspectedModulesInfo:
    """Inspect provided callable object searching for nn.Module members executed as part of forward pass.

    Args:
        obj: Callable object to inspect.
        dataset: List of tuples with batch size and input.
        inference_function: Custom inference function to use for inspection, obj is used by default.
        number_of_iterations: Number of iterations to run for inference.
        warmup_iterations: Number of iterations to run for warmup.
        min_depth: Minimum depth of the modules to inspect, if root level modules is not working, try to increase this value
    Returns:
        InspectedModulesInfo object.
    """
    setup_logging(format_string=LOG_FORMAT)

    logger.info("Inspecting object searching for executed nn.Module members.")
    model_inspector = ModuleInspector(min_depth=min_depth)
    model_inspector.inspect(obj)

    # If no inference function is provided, use the obj
    if inference_function is None:
        inference_function = obj

    dataset = ensure_enough_samples(dataset, max(number_of_iterations, warmup_iterations))

    # Warmup, run the model on a few samples, to make sure the model is compiled and the cache is warm
    _warmup(inference_function, dataset, warmup_iterations)
    _reset_total_execution_time(model_inspector)

    # Run the model on the dataset for the number of iterations
    total_execution_time = 0.0
    for _, args, kwargs in samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations):
        synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            inference_function(*args, **kwargs)
        synchronize()
        end_time = time.perf_counter()
        total_execution_time += end_time - start_time

    modules = model_inspector.get_modules()

    logger.info("Inspection done. Found %d candidate modules for tuning.", len(modules))

    inspected_modules_info = InspectedModulesInfo(total_execution_time, number_of_iterations)
    for module in modules:
        inspected_modules_info.add_module(module)

    model_inspector.reset()

    return inspected_modules_info


def _warmup(obj: Callable, dataset: DatasetLike | DataLoaderFactory | torch.Tensor, number_of_iterations: int):
    """Warmup the model by running it on a few samples."""
    for _, args, kwargs in samples_generator(dataset, [1], max_num_batches_per_batch_size=number_of_iterations):
        synchronize()
        with torch.no_grad():
            obj(*args, **kwargs)
        synchronize()


def _reset_total_execution_time(inspector: ModuleInspector):
    """Reset the timings of the modules."""
    for module in inspector.get_modules():
        module.total_execution_time = 0.0
        module.execution_count = 0
