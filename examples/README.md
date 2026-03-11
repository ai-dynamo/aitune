<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# NVIDIA AITune Examples

This directory contains practical examples demonstrating how to use NVIDIA AITune to tune different types of AI models for inference performance.

## ResNet

### Computer Vision - Image Classification

Shows how to tune ResNet models for image classification tasks. This example demonstrates model tuning and inference tuning for convolutional neural networks.

- **Location**: [`ResNet`](./ResNet/README.md)
- **Model**: ResNet50 image classification
- **Use Case**: Optimizing CNN models for computer vision tasks
- **Key Features**:
  - Model tuning with AITune
  - Image classification inference
  - Performance comparison before/after tuning
- **More Info**:
  - <https://huggingface.co/microsoft/resnet-50>

## StableDiffusion

### Generative AI - Text-to-Image

Demonstrates tuning of Stable Diffusion models for text-to-image generation. This example shows how to tune diffusion models for faster and more efficient image generation.

- **Location**: [`StableDiffusion`](./StableDiffusion/README.md)
- **Model**: Stable Diffusion 3 from HuggingFace
- **Use Case**: Optimizing text-to-image diffusion models
- **Key Features**:
  - Diffusion pipeline tuning
  - Customizable image generation parameters
  - Text prompt-based image synthesis
- **More Info**:
  - <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>

## FLUX

### Generative AI - Advanced Text-to-Image

Shows tuning of the FLUX text-to-image model, demonstrating advanced diffusion model tuning techniques for high-quality image generation.

- **Location**: [`FLUX`](./FLUX/README.md)
- **Model**: FLUX.1-dev from Black Forest Labs
- **Use Case**: Optimizing state-of-the-art text-to-image models
- **Key Features**:
  - Advanced diffusion model tuning
  - High-quality image generation
  - Efficient inference pipeline tuning
- **More Info**:
  - <https://huggingface.co/black-forest-labs/FLUX.1-dev>

## ParakeetCTC

### Speech AI - Automatic Speech Recognition

Demonstrates tuning of ASR (Automatic Speech Recognition) models using NVIDIA's Parakeet CTC model for speech-to-text conversion.

- **Location**: [`ParakeetCTC`](./ParakeetCTC/README.md)
- **Model**: NVIDIA Parakeet CTC 0.6B
- **Use Case**: Optimizing speech recognition models
- **Key Features**:
  - ASR model tuning
  - Audio-to-text transcription
  - NVIDIA NeMo framework integration
- **More Info**:
  - <https://huggingface.co/nvidia/parakeet-ctc-0.6b>
  - <https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html>

## ParakeetRNNT

### Speech AI - Automatic Speech Recognition

Demonstrates tuning of ASR (Automatic Speech Recognition) models using NVIDIA's Parakeet RNNT model for speech-to-text conversion.

- **Location**: [`ParakeetRNNT`](./ParakeetRNNT/README.md)
- **Model**: NVIDIA Parakeet RNNT 1.1B
- **Use Case**: Optimizing speech recognition models
- **Key Features**:
  - ASR model tuning
  - Audio-to-text transcription
  - NVIDIA NeMo framework integration
- **More Info**:
  - <https://huggingface.co/nvidia/parakeet-rnnt-1.1b>
  - <https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html>

## ESM2

### Text AI - Advanced Text Embedding

Demonstrates tuning of ESM2 model for text embedding tasks.

- **Location**: [`ESM2`](./ESM2/README.md)
- **Model**: ESM2 from HuggingFace
- **Use Case**: Optimizing text embedding models
- **Key Features**:
  - Text embedding tuning
  - Text embedding inference
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/esm/esm2-t12-100M-UR50S>

## E5Large

### Text AI - Advanced Text Embedding

Demonstrates tuning of E5Large model for text embedding tasks.

- **Location**: [`E5Large`](./E5Large/README.md)
- **Model**: E5Large from HuggingFace
- **Use Case**: Optimizing text embedding models
- **Key Features**:
  - Text embedding tuning
  - Text embedding inference
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/intfloat/e5-large-v2>

## LLM

### Large Language Models - Text Generation

Demonstrates tuning of Large Language Models for text generation tasks. This example shows how to optimize LLMs for efficient inference with KV cache support.

- **Location**: [`LLM`](./LLM/README.md)
- **Model**: Microsoft Phi-3-mini-4k-instruct from HuggingFace
- **Use Case**: Optimizing LLMs for text generation and inference
- **Key Features**:
  - LLM model tuning with AITune
  - Static and dynamic KV cache optimization
  - Prefill and decode phase optimization
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>

## JIT Tuning

### Just-In-Time Model Tuning

Demonstrates how to use NVIDIA AITune with JIT (Just-In-Time) tuning that requires no code changes. This example shows how to enable automatic tuning through environment variables without modifying existing code.

- **Location**: [`JitTuning`](./JitTuning/README.md)
- **Models**: Various models including ResNet, Stable Diffusion 3, Stable Diffusion XL, and FLUX
- **Use Case**: Zero-code-change automatic model optimization
- **Key Features**:
  - No-code JIT tuning via environment variables
  - Automatic tuning without imports or code modifications
  - Support for multiple model types (ResNet, diffusion models)
  - Simple enable/disable through `AUTOWRAPT_BOOTSTRAP` variable
- **More Info**:
  - <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>
  - <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
  - <https://huggingface.co/black-forest-labs/FLUX.1-dev>

---

Each example includes:

- Complete setup instructions
- Usage examples with CLI commands
- Model-specific tuning parameters
- AI Dynamo deployment instructions

To get started, navigate to any example directory and follow the README instructions for that specific model type.
