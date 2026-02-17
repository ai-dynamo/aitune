# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Measuring strategies for profiling."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from aitune.torch.task.profiling.events import ProfilingResultEvent


class MeasuringStrategy(ABC):
    """Strategy how do the measurement and create a ProfilingResultEvent(s)."""

    @abstractmethod
    def do_measurement(
        self, batch_size: int, model: Callable, sample: tuple[list, dict], **kwargs
    ) -> list[ProfilingResultEvent]:
        """Do the measurement and create a ProfilingResultEvent(s).

        Args:
            batch_size: Batch size of the measurement.
            model: Model to measure.
            sample: Sample to measure.
            **kwargs: Additional keyword arguments.

        Returns:
            List of ProfilingResultEvent(s).
        """
        pass


class ModelExecutionTimeMeasuringStrategy(MeasuringStrategy):
    """Strategy to measure execution time."""

    counter: int = 0

    def __init__(self):
        """Initialize the measuring strategy."""

    def do_measurement(
        self, batch_size: int, model: Callable, sample: tuple[list, dict], **measurement_kwargs
    ) -> list[ProfilingResultEvent]:
        """Do the measurement and create a ProfilingResultEvent(s) for the model."""
        args, kwargs = sample

        start = time.monotonic_ns()
        model(*args, **kwargs)
        end = time.monotonic_ns()

        ModelExecutionTimeMeasuringStrategy.counter += 1

        return [
            ProfilingResultEvent(
                measurement_id=ModelExecutionTimeMeasuringStrategy.counter,
                timestamp=start,
                batch_size=batch_size,
                phase="inference",
                execution_time=end - start,
                backend_details=measurement_kwargs.get("backend_details", None),
                model_name=measurement_kwargs.get("model_name", None),
            )
        ]
