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

## JIT Workflow

```bash
export AUTOWRAPT_BOOTSTRAP=aitune_enable_jit_tuning
python your_inference_script.py
```

## JIT Configuration (optional)

```python
from aitune.torch.jit.config import config
from aitune.torch.backend import TensorRTBackend

config.max_depth_level = 1
config.detect_graph_breaks = False
config.backends = [TensorRTBackend()]
```