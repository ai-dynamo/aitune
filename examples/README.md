<!--
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# NVIDIA AITune Examples

This directory contains practical examples demonstrating how to use NVIDIA AITune to tune different types of AI models for inference performance.

## ResNet

**Computer Vision - Image Classification**

Shows how to tune ResNet models for image classification tasks. This example demonstrates model tuning and inference tuning for convolutional neural networks.

- **Location**: [`ResNet/`](./ResNet/)
- **Model**: ResNet50 image classification
- **Use Case**: Optimizing CNN models for computer vision tasks
- **Key Features**:
  - Model tuning with AITune
  - Image classification inference
  - Performance comparison before/after tuning
- **More Info**:
  - https://huggingface.co/microsoft/resnet-50


## StableDiffusion

**Generative AI - Text-to-Image**

Demonstrates tuning of Stable Diffusion models for text-to-image generation. This example shows how to tune diffusion models for faster and more efficient image generation.

- **Location**: [`StableDiffusion/`](./StableDiffusion/)
- **Model**: Stable Diffusion 2.1 from HuggingFace
- **Use Case**: Optimizing text-to-image diffusion models
- **Key Features**:
  - Diffusion pipeline tuning
  - Customizable image generation parameters
  - Text prompt-based image synthesis
- **More Info**:
  - https://huggingface.co/stabilityai/stable-diffusion-2-1


## FLUX

**Generative AI - Advanced Text-to-Image**

Shows tuning of the FLUX text-to-image model, demonstrating advanced diffusion model tuning techniques for high-quality image generation.

- **Location**: [`FLUX/`](./FLUX/)
- **Model**: FLUX.1-dev from Black Forest Labs
- **Use Case**: Optimizing state-of-the-art text-to-image models
- **Key Features**:
  - Advanced diffusion model tuning
  - High-quality image generation
  - Efficient inference pipeline tuning
- **More Info**:
  - https://huggingface.co/black-forest-labs/FLUX.1-dev


## ParakeetCTC

**Speech AI - Automatic Speech Recognition**

Demonstrates tuning of ASR (Automatic Speech Recognition) models using NVIDIA's Parakeet CTC model for speech-to-text conversion.

- **Location**: [`ParakeetCTC/`](./ParakeetCTC/)
- **Model**: NVIDIA Parakeet CTC 0.6B
- **Use Case**: Optimizing speech recognition models
- **Key Features**:
  - ASR model tuning
  - Audio-to-text transcription
  - NVIDIA NeMo framework integration
- **More Info**:
  - https://huggingface.co/nvidia/parakeet-ctc-0.6b
  - https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html


---

Each example includes:
- Complete setup instructions
- Usage examples with CLI commands
- Model-specific tuning parameters
- Performance benchmarking guidance

To get started, navigate to any example directory and follow the README instructions for that specific model type.

