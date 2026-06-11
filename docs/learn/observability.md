---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Observability"
---

AITune exposes observability through logs, build logs, tuning telemetry, NVTX ranges, and optional hardware metrics. Use these signals to understand which graph was tuned, which backend won, why candidates failed, and how much hardware each phase used.

## Logging level

AITune emits progress messages at **INFO** level (strategy steps, backend selection, build status, validation, etc.). If the root or `aitune` logger is left at its default level (often WARNING), those messages are not shown.

**Enable INFO for better verbosity:**

```python
import logging

logging.basicConfig(level=logging.INFO, force=True)
```

With INFO (or DEBUG for more detail), you will see messages such as:

- Which strategy and backend are running
- Build and validation progress
- Selected backend and high-level timing

## Backend build output

During tuning, backend builds (e.g. TensorRT) run inside an **output control** context. By default, their stdout, stderr, and logging are **not** printed to the console; they are redirected to a build log file (e.g. `build.log` in the backend cache directory). That keeps the terminal quiet and avoids flooding it with compiler/build logs.

To see that backend output **live on the console** as well, set:

```bash
export AITUNE_CONSOLE_OUTPUT=1
```

Accepted values (case-insensitive): `1`, `true`, `yes`. If unset or any other value, console output from the build phase remains disabled (output only in the log file).

**Behavior summary:**

| `AITUNE_CONSOLE_OUTPUT` | Console during backend build | Build log file        |
|-------------------------|-----------------------------|------------------------|
| Unset / 0 / false        | No backend output           | Yes (e.g. `build.log`) |
| 1 / true / yes           | Backend output shown        | Yes (tee: console + file) |

So:

- **Default (unset):** Cleaner terminal; inspect `build.log` under the backend cache dir if you need build details.
- **Set to 1:** More verbose terminal; same content is also written to the log file.

## Tuning telemetry

AITune always collects tuning telemetry for each run, graph, and backend attempt. The report includes backend build status, throughput measurements, baseline throughput, selected backends, and failure information.

By default, telemetry is written to:

```text
~/.cache/aitune/tuning_data/report.json
```

Override the path with `AITUNE_TUNING_DATA_PATH`:

```bash
export AITUNE_TUNING_DATA_PATH=/tmp/aitune-report.json
```

or in Python:

```python
import aitune.torch as ait

ait.config.tuning_data_output_path = "/tmp/aitune-report.json"
```

You can also snapshot telemetry during a long-running process:

```python
from aitune.torch import snapshot_tuning_data

snapshot_tuning_data()
```

## NVTX and hardware metrics

Set `AITUNE_NVTX_EVENTS=1` to emit NVTX annotations for Nsight Systems. Set `AITUNE_HARDWARE_METRICS=1` to collect GPU/CPU utilization, memory, and power metrics during tuning and annotated inference windows.

Hardware metrics are written to a timestamped CSV file by default. Set `AITUNE_HARDWARE_METRICS_PATH` to choose a fixed output path.

See [Profiling and Hardware Metrics](../guides/advanced/profiling_and_hardware_metrics.md) for the full workflow.

## Runtime attribution

After tuning, `aitune.torch.profile(...)` produces a structured per-run report that measures wall time per run and attributes CPU and device time across both AITune-managed regions and untuned module regions (including method-style entry points like `vae.decode`), with an explicit residual covering time spent outside any region. The profile combines PyTorch Profiler data with AITune wrapper annotations and returns in-memory data plus a Markdown renderer. A raw Chrome trace is written only when you pass an explicit `trace_file`.

```python
import json
from pathlib import Path

import aitune.torch as ait

profile = ait.profile(
    obj=model,
    input_data=inputs,
    trace_file="trace.json",
)

Path("profile.json").write_text(json.dumps(profile.data, indent=2) + "\n")
Path("profile.md").write_text(profile.markdown())
```

See [Performance Profile](../guides/advanced/performance_profile.md) for the full workflow.

## Maximum verbosity

For the most visible tuning process:

1. Set the logging level to INFO (or DEBUG) in your script so AITune’s own messages are shown.
2. Set `AITUNE_CONSOLE_OUTPUT=1` so backend build output is shown on the console and still captured in the build log.
3. Set `AITUNE_TUNING_DATA_PATH` when you want a predictable telemetry report path.
4. Set `AITUNE_NVTX_EVENTS=1` and `AITUNE_HARDWARE_METRICS=1` for profiling runs.

```bash
export AITUNE_CONSOLE_OUTPUT=1
export AITUNE_TUNING_DATA_PATH=/tmp/aitune-report.json
python your_tuning_script.py
```

```python
import logging

logging.basicConfig(level=logging.INFO, force=True)

# ... your tuning code ...
```

See also `aitune.utils.setup_logging`, `control_output`, and the tuning-data reporting API.
