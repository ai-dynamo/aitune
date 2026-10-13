# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup, compiler timing, and device-wide NVML sampling for Python and Dynamo."""

import json
import logging
import threading
import time
from pathlib import Path

from .records import write_json

logger = logging.getLogger(__name__)


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
        fields = (
            "compile_id",
            "co_name",
            "start_time_us",
            "end_time_us",
            "duration_us",
            "dynamo_cumulative_compile_time_us",
            "aot_autograd_cumulative_compile_time_us",
            "inductor_cumulative_compile_time_us",
            "is_runtime",
            "fail_type",
            "fail_reason",
        )
        return [{key: getattr(row, key) for key in fields} for row in rows]

    def finish(self):
        records = self.snapshot()
        result = {
            "compilation_s": interval_seconds(records),
            "compilation_missing_reason": None,
            "compilation_method": "Union of PyTorch compiler frame wall-time intervals during model loading and first use; includes tracing, lowering and backend builds; excludes prebuilt artifact loading and ordinary inference",
            "raw_compilation": str(self.path),
        }
        write_json(self.path, {**result, "records": records})
        return result


def compilation_counters():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    return {
        "dynamo_unique_graphs": counters["stats"]["unique_graphs"],
        "inductor_generated_kernels": metrics.generated_kernel_count,
    }


def prepare(pipe, cfg, records, compilation):
    from .model import generate, synchronize

    logger.info("First-use pass over %d inputs; lazy compilation may take several minutes", len(records))
    synchronize()
    start = time.perf_counter()
    for item in records:
        generate(pipe, cfg, item)
    synchronize()
    first_use_s = time.perf_counter() - start
    first_use_start, first_use_end = start, time.perf_counter()
    compile_timing = compilation.finish()
    compile_records = compilation.snapshot()
    logger.info(
        "First-use pass complete in %.1fs; warming up with %d passes over %d inputs",
        first_use_s,
        cfg["benchmark"]["warmup_iterations"],
        len(records),
    )
    start = time.perf_counter()
    before = compilation_counters()
    for _ in range(cfg["benchmark"]["warmup_iterations"]):
        for item in records:
            generate(pipe, cfg, item)
    synchronize()
    warmup_s = time.perf_counter() - start
    warmup_end = time.perf_counter()
    if compilation_counters() != before or compilation.snapshot() != compile_records:
        raise RuntimeError("Compilation continued during warmup; inspect logs before accepting measurements")
    logger.info("Warmup complete in %.1fs", warmup_s)
    return {
        "first_use_s": first_use_s,
        "warmup_s": warmup_s,
        **compile_timing,
        "first_use_start_monotonic_s": first_use_start,
        "first_use_end_monotonic_s": first_use_end,
        "warmup_start_monotonic_s": start,
        "warmup_end_monotonic_s": warmup_end,
        "first_use_policy": "One unmeasured pass over every input before post-compilation warmup",
        "cache_state": "new per-process compiler caches; artifact retains tuning-time state",
    }


class Sampler:
    def __init__(self, path, interval, stream=False):
        self.path, self.interval = Path(path), interval
        self.stream = stream
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()

    def start(self):
        import pynvml as nv
        import torch

        self.nv = nv
        nv.nvmlInit()
        try:
            self.uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            self.handle = nv.nvmlDeviceGetHandleByUUID(self.uuid)
        except Exception:
            nv.nvmlShutdown()
            raise
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        output = self.path.open("w") if self.stream else None
        try:
            while not self.stop_event.is_set():
                self.samples.append({
                    "monotonic_s": time.perf_counter(),
                    "unix_s": time.time(),
                    "memory_gib": self.nv.nvmlDeviceGetMemoryInfo(self.handle).used / 2**30,
                    "utilization_pct": self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                })
                if output:
                    output.write(json.dumps(self.samples[-1]) + "\n")
                    output.flush()
                    self.samples.clear()
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = str(exc)
            if output:
                output.write(json.dumps({"error": self.error}) + "\n")
        finally:
            if output:
                output.close()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.nv.nvmlShutdown()
        if not self.stream:
            self.path.write_text("".join(json.dumps(row) + "\n" for row in self.samples))

    def summarize(self, start, end):
        rows = [r for r in self.samples if start <= r["monotonic_s"] <= end]
        reason = self.error or ("No NVML sample inside the measurement window" if not rows else None)
        utilization = None
        if rows and not reason:
            # Piecewise-constant integration, extended to both window edges.
            boundaries = [start] + [r["monotonic_s"] for r in rows[1:]] + [end]
            utilization = sum(
                r["utilization_pct"] * (boundaries[i + 1] - boundaries[i]) for i, r in enumerate(rows)
            ) / (end - start)
        return {
            "gpu_uuid": self.uuid,
            "peak_gpu_memory_gib": max(r["memory_gib"] for r in rows) if rows and not reason else None,
            "gpu_utilization_pct": utilization,
            "missing_reason": reason,
            "method": "NVML device-wide sampled peak and time-weighted busy percentage",
            "sampling_interval_s": self.interval,
            "samples": len(rows),
            "raw": str(self.path),
        }


def phase_duration(ranks, phase):
    return max(r[f"{phase}_end_monotonic_s"] for r in ranks) - min(r[f"{phase}_start_monotonic_s"] for r in ranks)
