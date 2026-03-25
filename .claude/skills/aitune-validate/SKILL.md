---
name: aitune-validate
description: Use when verifying that a tuned model's outputs match the baseline eager model within numerical tolerance — run after each backend compilation before accepting it as a deployment candidate.
license: Apache-2.0
---
<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Correctness Validation

Write and execute a correctness script using a held-out sample (different from tuning input):

```python
import json, torch

with torch.no_grad():
    baseline_out = eager_model(validation_input)
    tuned_out = tuned_model(validation_input)

max_diff = (baseline_out - tuned_out).abs().max().item()
rel_error = ((baseline_out - tuned_out).abs() / baseline_out.abs().clamp(min=1e-8)).max().item()

# thresholds: fp32 → atol=1e-3, fp16 → atol=1e-2
atol = 1e-2  # adjust per precision
correctness_pass = max_diff <= atol

print(json.dumps({
    "max_abs_diff": max_diff,
    "max_rel_error": rel_error,
    "atol_threshold": atol,
    "correctness_pass": correctness_pass,
}))
```

If `correctness_pass` is false: log as "correctness failure", advance to next backend.
