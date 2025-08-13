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
"""Profiling result events."""

from dataclasses import dataclass


@dataclass
class ProfilingResultEvent:
    """Profiling result event.

    Attributes:
        timestamp: Timestamp of the measurement.
        backend_details: Details of the backend.
        batch_size: Batch size of the measurement.
        phase: Phase of the measurement.
        measurement_id: Unique identifier for the measurement.
        execution_time: Execution time of the measurement.
    """

    timestamp: float
    model_name: str
    backend_details: str
    batch_size: int
    phase: str

    measurement_id: int = 0
    execution_time: float = 0.0


def get_inference_events(events: list[ProfilingResultEvent]) -> list[ProfilingResultEvent]:
    """Get inference events."""
    return [e for e in events if e.phase.startswith("inference")]
