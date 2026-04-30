<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Just-in-Time (JIT) Tuning

No code changes required. Activated via environment variable.

**Limitations**: no artifact persistence, no benchmarking, no caching, limited strategy support. Uses only batch sizes seen at runtime.

**Use JIT when**:
- Zero code changes are required
- Quick experimentation or profiling
- Batch sizes are unknown or variable
- Full AOT control is not needed

## JIT Modes

| Mode | Trigger | When to Use |
|---|---|---|
| **Eager** (default) | After `min_samples` collected per module | Simple pipelines, single forward call per step |
| **Deferred** | Explicit `tune.deferred()` call | Image/video generation pipelines where modules are called variable number of times per step |

---

## Eager Mode (default)

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
python your_inference_script.py
```

Tuning fires automatically once each module has seen enough samples.

---

## Deferred Mode

Use when modules are called a variable number of times per pipeline step (e.g. diffusion denoisers called N times per image). Eager mode can't know when the full step is done; deferred mode lets you decide.

```python
from aitune.torch.jit.config import config, JITMode
from aitune.torch.jit import tune

# Must be set before bootstrap runs (before model/pipeline is created)
config.mode = JITMode.TUNE_DEFERRED

# ... create model/pipeline with AUTOWRAPT_BOOTSTRAP active ...

# Run at least one full forward pass to collect samples from all modules
pipe(prompt="a photo of a cat", height=1024, width=1024, num_inference_steps=20)

# Explicitly trigger tuning — replaces the automatic trigger
tune.deferred()

# Continue inference with tuned model
pipe(prompt="a photo of a dog", ...)
```

**Rules**:
- Set `config.mode = JITMode.TUNE_DEFERRED` before any module is wrapped (before `AUTOWRAPT_BOOTSTRAP` activates).
- Call `tune.deferred()` **after** at least one complete forward pass so all modules have samples.
- `tune.deferred()` raises `ValueError` if mode is not `TUNE_DEFERRED`.

---

## JIT Configuration (optional)

```python
from aitune.torch.jit.config import config
from aitune.torch.backend import TensorRTBackend
from aitune.torch.tune_strategy import FirstWinsStrategy

config.max_depth_level = 1
config.detect_graph_breaks = False
config.strategy = FirstWinsStrategy(backends=[TensorRTBackend()])
```

`config.strategy` accepts any `TuneStrategy` (e.g. `FirstWinsStrategy`, `MaxThroughputStrategy`). Default (`None`) uses `FirstWinsStrategy` over TensorRT and TorchInductorJit.