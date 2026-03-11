<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Known Issues and Limitations

- **Multi-GPU support.** AITune currently only supports single-GPU configurations.
- Just-in-Time tuning does not support `transformers>=5` due to `@capture_outputs` decorator
