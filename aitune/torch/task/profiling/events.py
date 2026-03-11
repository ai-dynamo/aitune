# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
