---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Ahead-of-Time (AOT) Tuning

Requires code changes. Full control over the tuning process.

**Capabilities**: batch detection, dynamic axes, benchmarking, save/load checkpoints, caching, deterministic backend selection.

**Use AOT when**:
- Production deployment
- Benchmarking is required
- Models must be persisted and reloaded
- Batch sizes and dynamic shapes must be explicitly controlled

## AOT Workflow

```python
import aitune.torch as ait

# 1. Inspect — analyze model structure and identify tunable modules
modules_info = ait.inspect(model, input_data)
modules_info.describe()

# 2. Wrap — prepare selected modules for tuning
modules = modules_info.get_modules()
model = ait.wrap(model, modules)

# 3. Tune — run tuning with configured strategy
ait.tune(model, input_data)

# 4. Save / Load
ait.save(model, "tuned_model.ait")
ait.load(model, "tuned_model.ait")
```