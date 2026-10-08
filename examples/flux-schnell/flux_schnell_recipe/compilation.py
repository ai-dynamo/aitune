# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compilation-frame wall time for the pinned PyTorch runtime compilation path."""

from .records import write_json


def interval_seconds(records):
    """Count overlapping/nested compiler intervals once, excluding runtime events."""
    intervals = []
    for record in records:
        if record["is_runtime"]:
            continue
        start, end = record["start_time_us"], record["end_time_us"]
        if start is None or end is None or end < start:
            raise ValueError("Missing or invalid compiler timing interval")
        intervals.append((start, end))
    total, previous_end = 0, None
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end if previous_end is not None else start))
        previous_end = max(end, previous_end if previous_end is not None else end)
    return total / 1e6


class CompilationRecorder:
    # AITune resets Dynamo while loading JIT backends. This record queue survives
    # torch._dynamo.reset(), unlike callbacks and compilation_time_metrics.
    limit = 100000

    def __init__(self, directory):
        from torch._dynamo.utils import clear_compilation_metrics, set_compilation_metrics_limit
        self.path = directory / "compilation.json"
        set_compilation_metrics_limit(self.limit)
        clear_compilation_metrics()

    def snapshot(self):
        from torch._dynamo.utils import get_compilation_metrics
        rows = get_compilation_metrics()
        if len(rows) >= self.limit:
            raise RuntimeError("Compilation timing records may have been truncated")
        fields = ("compile_id", "co_name", "start_time_us", "end_time_us", "duration_us",
                  "dynamo_cumulative_compile_time_us", "aot_autograd_cumulative_compile_time_us",
                  "inductor_cumulative_compile_time_us", "is_runtime", "fail_type", "fail_reason")
        return [{key: getattr(row, key) for key in fields} for row in rows]

    def finish(self):
        records = self.snapshot()
        result = {"compilation_s": interval_seconds(records), "compilation_missing_reason": None,
                  "compilation_method": "Union of PyTorch compiler frame wall-time intervals during model loading and first use; includes tracing, lowering and backend builds; excludes prebuilt artifact loading and ordinary inference",
                  "raw_compilation": str(self.path)}
        write_json(self.path, {**result, "records": records})
        return result
