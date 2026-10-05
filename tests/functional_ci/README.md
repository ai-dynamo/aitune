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
tune, infer, save, load into a fresh model, and infer again. The PR profile
checks the required CUDA runner and backend packages. Missing requirements or
advertised capabilities fail the run. Optional exclusions retain their case ID
and reason in the JUnit report.

Run metadata is written before pytest starts. It includes the selected profile,
GPU and CUDA details, installed library versions, commit, and container image.
Failed cases also upload their AITune cache contents under
`functional-ci-logs/<case-id>/`.

For a local run, omit `AITUNE_FUNCTIONAL_PROFILE` or set it to `local`. To
validate the exact CI profile, set it to `pr`; that profile requires one SM120
GPU and the full declared dependency stack.
The `Functional CI GPU` check joins the existing PR CI route. Repository rules
should require that check; the repository owner manages that setting.
