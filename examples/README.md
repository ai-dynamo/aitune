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
      <th>Model Source</th>
      <th>Model Card</th>
      <th>Precision</th>
      <th>Single-GPU tuning</th>
      <th>Multi-GPU tuning</th>
      <th>Deployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Image classification</th>
      <td><a href="./ResNet/README.md">ResNet-50</a></td>
      <td><a href="https://huggingface.co/docs/timm/index">timm</a></td>
      <td><a href="https://huggingface.co/timm/resnet50.a1_in1k">Hugging Face</a></td>
      <td>FP16; INT8 for Python/Dynamo</td>
      <td>AOT</td><td>-</td><td>Dynamo + Triton</td>
    </tr>
    <tr>
      <th scope="rowgroup" rowSpan="3">Image generation</th>
      <td><a href="./FLUX/README.md">FLUX.1-dev</a></td>
      <td><a href="https://huggingface.co/docs/diffusers/index">Diffusers</a></td>
      <td><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">Hugging Face</a></td>
      <td>BF16, FP8 and NVFP4</td>
      <td>AOT</td><td>AOT</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./StableDiffusion/README.md">Stable Diffusion 3 Medium</a></td>
      <td><a href="https://huggingface.co/docs/diffusers/index">Diffusers</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">Hugging Face</a></td>
      <td>FP16</td>
      <td>AOT</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./JitTuning/README.md">Stable Diffusion XL</a></td>
      <td><a href="https://huggingface.co/docs/diffusers/index">Diffusers</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">Hugging Face</a></td>
      <td>FP16</td>
      <td>JIT</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <th scope="row">Object detection</th>
      <td><a href="./YOLO/README.md">YOLOv10n</a></td>
      <td><a href="https://onnx.ai/">ONNX file</a></td>
      <td><a href="https://huggingface.co/onnx-community/yolov10n">Hugging Face</a></td>
      <td>FP16, INT8</td>
      <td>AOT</td><td>-</td><td>Triton</td>
    </tr>
    <tr>
      <th scope="row">Protein language modeling</th>
      <td><a href="./ESM2/README.md">ESM-2 650M</a></td>
      <td><a href="https://huggingface.co/docs/transformers/index">Transformers</a></td>
      <td><a href="https://huggingface.co/facebook/esm2_t33_650M_UR50D">Hugging Face</a></td>
      <td>FP32</td>
      <td>AOT</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="rowgroup" rowSpan="2">Speech recognition</th>
      <td><a href="./ParakeetCTC/README.md">Parakeet CTC 0.6B</a></td>
      <td><a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html">NeMo</a></td>
      <td><a href="https://huggingface.co/nvidia/parakeet-ctc-0.6b">Hugging Face</a></td>
      <td>FP32</td>
      <td>AOT</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <td><a href="./ParakeetRNNT/README.md">Parakeet RNNT 1.1B</a></td>
      <td><a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html">NeMo</a></td>
      <td><a href="https://huggingface.co/nvidia/parakeet-rnnt-1.1b">Hugging Face</a></td>
      <td>FP16 (autocast)</td>
      <td>AOT</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="row">Text encoding</th>
      <td><a href="./BERT/README.md">BERT-base-uncased</a></td>
      <td><a href="https://huggingface.co/docs/transformers/index">Transformers</a> / <a href="https://onnx.ai/">ONNX file</a></td>
      <td><a href="https://huggingface.co/google-bert/bert-base-uncased">Hugging Face</a></td>
      <td>FP32</td>
      <td>AOT</td><td>-</td><td>Triton</td>
    </tr>
    <tr>
      <th scope="row">Text embeddings</th>
      <td><a href="./E5Large/README.md">E5 Large V2</a></td>
      <td><a href="https://sbert.net/">Sentence Transformers</a></td>
      <td><a href="https://huggingface.co/intfloat/e5-large-v2">Hugging Face</a></td>
      <td>FP32</td>
      <td>AOT</td><td>-</td><td>Dynamo</td>
    </tr>
    <tr>
      <th scope="row">Text generation</th>
      <td><a href="./LLM/README.md">Qwen3 0.6B</a></td>
      <td><a href="https://huggingface.co/docs/transformers/index">Transformers</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B">Hugging Face</a></td>
      <td>BF16</td>
      <td>AOT</td><td>AOT</td><td>-</td>
    </tr>
    <tr>
      <th scope="row">Video generation</th>
      <td><a href="./WAN/README.md">Wan 2.1 T2V 1.3B</a></td>
      <td><a href="https://huggingface.co/docs/diffusers/index">Diffusers</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">Hugging Face</a></td>
      <td>BF16</td>
      <td>AOT</td>
      <td>AOT</td><td>Dynamo</td>
    </tr>
  </tbody>
</table>

- **Model:** links to the recipe and usage instructions.
- **Model Source:** the library providing the model or the exported file used as the recipe's starting point.
  Links open the official library or format documentation.
- **Model Card:** links to the model description and pretrained checkpoint on Hugging Face.
- **Precision:** numerical formats for the main tuned component, including its lowest configured precision.
  Other pipeline components may use higher precision. Autocast uses mixed precision without changing checkpoint weights.
- **Single-GPU tuning / Multi-GPU tuning:** AITune tuning flows used by the linked recipe for each GPU setup.
- **Deployment:** documented serving targets, such as Triton and Dynamo.
- **-:** no option is documented in the current recipe.

The tuning columns describe how the linked recipe integrates with AITune:

- **AOT (ahead of time):** the recipe explicitly selects or wraps modules and calls `aitune.torch.tune()`
  before tuned inference. Inspection can help select modules to wrap.
- **JIT (just in time):** AITune automatically discovers and tunes modules during model execution
  with JIT tuning enabled.

The tuning flow is independent of backend compilation behavior. For example, an AOT tuning flow can evaluate
`TorchInductorJitBackend`; selecting that backend does not make the recipe a JIT tuning flow.

Additional single-GPU JIT examples for ResNet, Stable Diffusion 3, and FLUX are available in
[JIT Tuning](./JitTuning/README.md). The table describes the primary recipe linked in each row.
Open the model page for supported combinations, hardware requirements, and results.
