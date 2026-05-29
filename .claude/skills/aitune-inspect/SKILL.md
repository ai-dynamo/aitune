---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: aitune-inspect
description: Use when inspecting a PyTorch model or pipeline to identify tunable submodules, detect dynamic shapes, and determine the recommended tuning mode before optimization.
license: Apache-2.0
---

# Model Inspection

Load the model, run `ait.inspect()`, collect module names/types/depths, detect dynamic shapes, identify graph break risks. Capture the output and use it to populate the Model Analysis Summary.

```python
import json, sys
import aitune.torch as ait

# --- user-specific: replace with actual model loading # from transformers import ...
# model = ...
# sample_input = ...
# -----------------------------------------------------
modules_info = ait.inspect(model, sample_input, min_depth=0)
modules_info.describe()  # prints human-readable to stderr, usage percentage can be found in the output
modules = modules_info.get_modules(min_execution_percentage=0.0)

results = {
    "modules": [{"name": m.name, "type": type(m.module).__name__, "depth": m.depth} for m in modules],
    "total_tunable": len(modules),
}
print(json.dumps(results))
```

## Model Analysis Summary

After running the script, report the Model Analysis Summary:

```
## Model Analysis Summary

**Model**: [class name from output]
**Input shape**: [shape and dtype]
**Tunable modules identified**: [count and types from JSON output]
**Dynamic shapes detected**: [yes/no]
**Graph break risks**: [none / low / high — list specific module names if high]

**Recommended tuning mode**: AOT / JIT
**Reasoning**: [based on actual inspection output]

```

Wait for user confirmation before proceeding if graph break risk is HIGH.
