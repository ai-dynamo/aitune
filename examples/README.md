---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Model Recipes"
---

# Model Recipes

Browse ready-to-use recipes for tuning, validating, benchmarking, and deploying models. Choose a model below
to find configurations and commands you can adapt with your own compatible checkpoint and data.

## Find a model

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Model</th>
      <th>Hub</th>
      <th>Precision</th>
      <th>Offloading</th>
      <th>Single GPU</th>
      <th>Multi-GPU</th>
      <th>Deployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Image classification</th>
      <td><a href="./ResNet/README.md">ResNet-50</a></td>
      <td><a href="https://huggingface.co/timm">timm</a></td>
      <td>FP16</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>Dynamo + Triton</td>
    </tr>
    <tr>
      <th scope="rowgroup" rowSpan="3">Image generation</th>
      <td><a href="./FLUX/README.md">FLUX.1-dev</a></td>
      <td><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">Hugging Face</a></td>
      <td>BF16, FP8 and NVFP4</td><td>-</td>
      <td>Runtime</td><td>Runtime</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./StableDiffusion/README.md">Stable Diffusion 3 Medium</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">Hugging Face</a></td>
      <td>FP16</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./JitTuning/README.md">Stable Diffusion XL</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <th scope="row">Protein language modeling</th>
      <td><a href="./ESM2/README.md">ESM-2 650M</a></td>
      <td><a href="https://huggingface.co/facebook/esm2_t33_650M_UR50D">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="rowgroup" rowSpan="2">Speech recognition</th>
      <td><a href="./ParakeetCTC/README.md">Parakeet CTC 0.6B</a></td>
      <td><a href="https://huggingface.co/nvidia/parakeet-ctc-0.6b">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./ParakeetRNNT/README.md">Parakeet RNNT 1.1B</a></td>
      <td><a href="https://huggingface.co/nvidia/parakeet-rnnt-1.1b">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Ahead of time, Runtime</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="row">Text embeddings</th>
      <td><a href="./E5Large/README.md">E5 Large V2</a></td>
      <td><a href="https://huggingface.co/intfloat/e5-large-v2">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Ahead of time</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="row">Text generation</th>
      <td><a href="./LLM/README.md">Qwen3 0.6B</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B">Hugging Face</a></td>
      <td>-</td><td>-</td>
      <td>Runtime</td><td>Eager only</td><td>-</td>
    </tr>
    <tr>
      <th scope="row">Video generation</th>
      <td><a href="./WAN/README.md">Wan 2.1 T2V 1.3B</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">Hugging Face</a></td>
      <td>BF16; FP32 VAE</td><td>-</td>
      <td>Ahead of time, Runtime</td>
      <td>Ahead of time, Runtime</td><td>Dynamo</td>
    </tr>
  </tbody>
</table>

- **Model:** links to the recipe and usage instructions.
- **Hub:** links to the source model card or collection.
- **Precision:** numerical formats available in the recipe, such as FP16, BF16, FP8, and NVFP4.
- **Offloading:** whether the recipe supports moving model weights between GPU and CPU memory.
- **Single GPU / Multi-GPU:** compilation options for each setup. **Ahead of time** uses prebuilt backend artifacts;
  **Runtime** compiles during loading or inference; **Eager only** runs without a compiled backend.
- **Deployment:** documented serving targets, such as Triton and Dynamo.
- **-:** no option is documented in the current recipe.

Compilation options describe the main tuned component. Open the model page for supported combinations,
hardware requirements, and results.
