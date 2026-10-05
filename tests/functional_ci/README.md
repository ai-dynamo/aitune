---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
# Functional CI

This suite runs deterministic local model cases on the required GPU runner.
It does not download models or access public hubs.

Run the suite with:

```bash
python -m pytest tests/functional_ci/ -ra --junitxml=functional-ci.test-report.xml
```

The first case covers the explicit Torch Eager lifecycle: inspect and wrap,
tune, infer, save, load into a fresh model, and infer again. CUDA is required.
The `Functional CI GPU` check joins the existing PR CI route. Repository rules
should require that check; the repository owner manages that setting.
