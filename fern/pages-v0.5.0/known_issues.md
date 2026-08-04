---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Known Issues and Limitations"
---

- **Multi-GPU support.** AITune currently only supports single-GPU configurations.
- **ONNX Runtime GPU wheels.** Use an ONNX Runtime GPU package that matches your CUDA, cuDNN, and TensorRT environment. The default PyPI `onnxruntime-gpu` package targets CUDA 12.x; CUDA 13.x currently uses ONNX Runtime nightly wheels. See the official install matrix: https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-or-tensorrt
