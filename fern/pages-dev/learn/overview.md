---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "NVIDIA AITune"
---

import { BadgeLinks } from "../_components/BadgeLinks";

<BadgeLinks
  badges={[
    {
      href: "https://github.com/ai-dynamo/aitune/blob/main/LICENSE",
      src: "https://img.shields.io/badge/License-Apache_2.0-blue",
      alt: "License",
    },
    {
      href: "https://www.python.org/downloads/",
      src: "https://img.shields.io/badge/python-3.10%2B-blue",
      alt: "Python",
    },
    {
      href: "https://pytorch.org/",
      src: "https://img.shields.io/badge/PyTorch-2.8%2B-red",
      alt: "PyTorch",
    },
  ]}
/>



**NVIDIA AITune** is an inference toolkit designed for tuning and deploying Deep Learning models with a focus on NVIDIA GPUs. It provides model tuning capabilities through compilation and conversion paths that can significantly improve inference speed and efficiency across various AI workloads including Computer Vision, Natural Language Processing, Speech Recognition, and Generative AI.

The toolkit enables seamless tuning of PyTorch models and pipelines using various backends such as TensorRT, Torch-TensorRT, TorchAO, Torch Inductor, and ONNX Runtime through a single Python API. The resulting tuned models are ready for deployment in production environments.

NVIDIA AITune works with your environment — relying first on your software versions — and selects the best-performing backend for your software and hardware setup, guiding you to supported technologies.

## When to Use AITune

AITune provides compute graph optimizations for PyTorch models at the `nn.Module` level. Use AITune when you want automated inference optimization with minimal code changes.

If your model is supported by a dedicated serving framework and benefits from runtime optimizations (e.g. continuous batching, speculative decoding), use frameworks like TensorRT-LLM, vLLM, or SGLang for best performance. Use AITune for general PyTorch models and pipelines that lack such specialized tooling.

## Features at Glance

The distinct capabilities of NVIDIA AITune are summarized in the feature matrix:

| Feature                     | Description                                                                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Ease-of-use                 | Single line of code to run all possible tuning paths directly from your source code                                                                      |
| Wide Backend Support        | Compatible with various tuning backends including TensorRT, Torch-TensorRT, TorchAO, Torch Inductor, and ONNX Runtime                                    |
| Model Tuning                | Enhance the performance of models such as ResNET and BERT for efficient inference deployment                                                             |
| Pipeline Tuning             | Streamline Python code pipelines for models such as Stable Diffusion and Flux using seamless model wrapping and tuning                                   |
| Model Export and Conversion | Automate the process of exporting and converting models between various formats with focus on TensorRT, Torch-TensorRT, Torch Inductor, and ONNX Runtime |
| Correctness Testing         | Ensures tuned models produce correct outputs by validating on provided data samples                                                                      |
| Performance Profiling       | Profiles models to select the optimal backend based on performance metrics such as latency and throughput                                                |
| Model Persistence           | Save and load tuned models for production deployment with flexible storage options                                                                       |
| JIT tuning                  | Just-in-time tuning of a model or a pipeline without any code changes required                                                                           |
