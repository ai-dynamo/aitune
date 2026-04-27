---
name: aitune-benchmark
description: Use when measuring inference performance of a PyTorch model before and after AITune optimization — capturing baseline, compilation time, tuned throughput, and speedup.
license: Apache-2.0
---
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark workflow

## Step 1: Baseline benchmark (run once, before first backend)

Write and execute a baseline benchmark script. Record `baseline_results` for all subsequent comparisons.

Model must not be wrapped with AITune yet for baseline benchmark.
Model must be in eager mode.
Baseline benchmark must be run before any backend tuning, and should be done once, results should be captured and saved.

```python
import json, time, torch, numpy as np
from pathlib import Path

def benchmark(model, input_data, warmup=5, runs=50):

    # Warmups
    with torch.no_grad():
        for _ in range(warmup):
            model(input_data)
            torch.cuda.synchronize()

    # Time it
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(input_data)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    mem = torch.cuda.memory_allocated() / 1024**2

    return {
        "avg_ms": float(latencies.mean()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_rps": 1000.0 / float(latencies.mean()),
        "gpu_memory_mb": float(mem),
    }

# model must be in eager mode here
results = {"baseline": benchmark(model, input_data)}
print(json.dumps(results))

Path("baseline_benchmark.json").write_text(json.dumps(results))
```

## Step 2: Compile with current backend and measure compilation time

Write and execute a tuning script for the current (i.e. TensorRTBackend-fp16) backend. Measure compilation time. Capture any errors.

```python
import json, time, aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, ONNXAutoCastConfig
from aitune.torch.tune_strategy import OneBackendStrategy

# configure backend (varies per trial)
backend_config = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig(), use_cuda_graphs=True)
backend = TensorRTBackend(backend_config)
strategy = OneBackendStrategy(backend)

modules = ait.inspect(model, input_data).get_modules()
model = ait.wrap(model, modules, tune_strategy=strategy)

t0 = time.perf_counter()
try:
    ait.tune(model, input_data)
    compile_ok = True
    compile_error = None
except Exception as e:
    compile_ok = False
    compile_error = str(e)
compile_time_s = time.perf_counter() - t0

print(json.dumps({
    "backend": "TensorRTBackend-fp16",
    "compile_ok": compile_ok,
    "compile_time_s": compile_time_s,
    "compile_error": compile_error,
}))
```

If `compile_ok` is false: log the error, advance to the next backend. Do not attempt benchmark or correctness check.

## Step 3: Benchmark tuned model

Re-run the benchmark script from Step 1 against the tuned model. Compute speedup:

```python
baseline = json.loads(Path("baseline_benchmark.json").read_text())

# i.e. TensorRTBackend-fp16, depth=0
backend_name = "TensorRTBackend-fp16"
depth = 0

benchmark_tuned = benchmark(model, input_data)
print(json.dumps(benchmark_tuned))

Path(f"benchmark_{backend_name}_d={depth}.json").write_text(json.dumps(benchmark_tuned))

speedup = benchmark_tuned["throughput_rps"] / baseline["throughput_rps"]
```

If `speedup < 1.1`: log as "marginal improvement", advance to next backend.
