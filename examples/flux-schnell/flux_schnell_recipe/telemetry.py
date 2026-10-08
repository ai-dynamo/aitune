# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-wide NVML sampling, including allocations outside PyTorch."""

import json
from pathlib import Path
import threading
import time


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
                self.samples.append({"monotonic_s": time.perf_counter(), "unix_s": time.time(),
                    "memory_gib": self.nv.nvmlDeviceGetMemoryInfo(self.handle).used / 2**30,
                    "utilization_pct": self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu})
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
            utilization = sum(r["utilization_pct"] * (boundaries[i+1] - boundaries[i])
                              for i, r in enumerate(rows)) / (end - start)
        return {"gpu_uuid": self.uuid, "peak_gpu_memory_gib": max(r["memory_gib"] for r in rows) if rows and not reason else None,
                "gpu_utilization_pct": utilization, "missing_reason": reason,
                "method": "NVML device-wide sampled peak and time-weighted busy percentage",
                "sampling_interval_s": self.interval, "samples": len(rows), "raw": str(self.path)}
