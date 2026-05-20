---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Profiling and Hardware Metrics"
---

NVIDIA AITune provides two complementary observability features:

- **NVTX annotations** — mark key operations as colored regions visible in NVIDIA Nsight Systems
- **Hardware metrics** — continuously sample GPU/CPU utilization, memory, and power per module and backend

Both are disabled by default to avoid overhead in production.

## NVTX Profiling

NVTX (NVIDIA Tools Extension) annotations mark key operations in the AITune lifecycle, making them visible as colored timeline regions in NVIDIA Nsight Systems.

### Enabling NVTX

```bash
export AITUNE_NVTX_EVENTS=1
python your_script.py
```

### Using with Nsight Systems

```bash
AITUNE_NVTX_EVENTS=1 nsys profile -o output.nsys-rep python your_script.py
```

The NVTX annotations will appear as colored regions in the timeline, helping you identify:

* Backend inference calls (TensorRT, Torch-TensorRT, TorchAO, etc.)
* Tuning operations
* Performance bottlenecks

## Hardware Metrics

AITune can collect hardware metrics during tuning and inference, giving you visibility into resource utilization per module and backend. Metrics are collected in a background process and reported at program exit.

### Enabling Hardware Metrics

```bash
export AITUNE_HARDWARE_METRICS=1
python your_script.py
```

### Collected Metrics

The following metrics are sampled continuously (every 100 ms) and aggregated per module and backend:

| Category | Metrics |
|---|---|
| **GPU memory** (per device) | `cuda:N` used memory [GB] |
| **GPU utilization** (per device) | `cuda:N` utilization mean / max [%] |
| **GPU power** (per device) | `cuda:N` power mean / max [W] |
| **Host CPU** | CPU utilization [%] |
| **Host memory** | Used / free system memory |
| **PyTorch allocator** | Allocated and reserved CUDA memory |

GPU metrics require NVML (available on systems with NVIDIA drivers). If NVML is unavailable, only host and PyTorch metrics are collected.

### Output

At program exit, AITune logs a summary table and writes a CSV file to the working directory.

By default a timestamped filename is used:

```
hardware_metrics_20260402_153012.csv
```

To write to a fixed path instead, set `AITUNE_HARDWARE_METRICS_PATH`:

```bash
export AITUNE_HARDWARE_METRICS_PATH=hardware_metrics.csv
```

### Runtime Control

Collection can also be toggled at runtime from Python, independent of the environment variable:

```python
from aitune.utils.monitoring import enable_hardware_metrics, disable_hardware_metrics

enable_hardware_metrics()   # start collecting
# ... run workload ...
disable_hardware_metrics()  # stop collecting, dump accumulated metrics to CSV
```

`disable_hardware_metrics()` performs a graceful shutdown: it dumps the currently accumulated metrics (same CSV output as at program exit) and stops the background collection process. Calling `enable_hardware_metrics()` afterwards starts a fresh session.

### Mid-run Snapshots

`snapshot()` writes the currently accumulated metrics to a file without stopping collection:

```python
from pathlib import Path
from aitune.utils.monitoring import snapshot

snapshot(Path("phase1_metrics.csv"))                          # dump + reset accumulator (default)
snapshot(Path("phase1_metrics.csv"), reset_metrics=False)     # dump without resetting
```

By default (`reset_metrics=True`) the accumulator is cleared after writing, so the next snapshot contains only metrics from that point forward. Set `reset_metrics=False` to capture a cumulative view.

`snapshot()` is a no-op (with a warning) when collection is not active.

## Annotating Ahead-of-time Inference Functions

For **Just-in-time tuning**, inference is already annotated automatically.

For **Ahead-of-time tuning**, you load a saved model and call it from your own inference function. To get an NVTX region and hardware metrics for that call, annotate the function manually using `annotate` from `aitune.utils.monitoring`:

```python
from aitune.utils.monitoring import annotate

@annotate(name="inference", color="green")
def do_inference(model, inputs):
    return model(inputs)
```

This is the same pattern used in all AITune examples (ResNet, FLUX, StableDiffusion, ParakeetRNNT, etc.). The annotation creates an NVTX range visible in Nsight Systems and triggers hardware metrics collection for the duration of the function call.

## Combining NVTX and Hardware Metrics

Both features can be enabled together for a full profiling run:

```bash
AITUNE_HARDWARE_METRICS=1 AITUNE_NVTX_EVENTS=1 nsys profile -o output.nsys-rep python your_script.py
```
