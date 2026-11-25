---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---
# Benchmarks

Benchmarks measure performance separately from correctness tests under `tests/unit/` and functional test suites.
They are excluded from default pytest discovery.

- [`micro/`](micro/README.md): isolated operation costs, such as tensor flattening
  and output reconstruction. These measurements exclude model execution and do not
  establish end-to-end inference latency or GPU throughput improvements.
