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
"""Task module."""

from aitune.torch.task.profiling.config import ProfilingConfig
from aitune.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from aitune.torch.task.profiling.measuring_stop_strategy import (
    MeasuringStopStrategy,
    NumStepsMeasuringStopStrategy,
    StableWindowMeasuringStopStrategy,
)
from aitune.torch.task.profiling.measuring_strategy import MeasuringStrategy, ModelExecutionTimeMeasuringStrategy
from aitune.torch.task.profiling.profiling import ProfilingResults, ProfilingStatus, profile

__all__ = [
    "MeasuringStrategy",
    "ModelExecutionTimeMeasuringStrategy",
    "MeasuringStopStrategy",
    "NumStepsMeasuringStopStrategy",
    "StableWindowMeasuringStopStrategy",
    "ProfilingConfig",
    "ProfilingResultEvent",
    "get_inference_events",
    "ProfilingResults",
    "ProfilingStatus",
    "profile",
]
