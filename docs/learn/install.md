---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Install"
---

## Prerequisites

Before installing NVIDIA AITune, make sure your system meets these requirements:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Versions 3.11 through 3.14
* **PyTorch**: Version 2.8 or later on Python 3.11 through 3.13. Use PyTorch 2.14 or later on Python 3.14.
* **TensorRT**: Version `10.3` or higher (for TensorRT backend)
* **NVIDIA GPU**: Required for GPU-accelerated tuning

## Installing from PyPI (Recommended)

You can install NVIDIA AITune from PyPI.

```bash
pip install --extra-index-url https://pypi.nvidia.com aitune
```

### Pinning to a specific PyTorch version

If you need to pin AITune's dependencies (`torch`, `torch-tensorrt`, `torchao`) to a specific PyTorch minor version, use one of the version extras:

| Extra | PyTorch version |
|---|---|
| `torch28` | 2.8.x |
| `torch29` | 2.9.x |
| `torch210` | 2.10.x |
| `torch211` | 2.11.x |
| `torch212` | 2.12.x |
| `torch213` | 2.13.x |
| `torch214` | 2.14.x |
| `torch215` | 2.15.x |

On Python 3.14, choose `torch214` or `torch215`.

```bash
pip install --extra-index-url https://pypi.nvidia.com "aitune[torch211]"
```

### Selecting a CUDA version

To target a specific CUDA version, add the matching PyTorch index:

```bash
pip install --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 "aitune[torch211]"
```

### ONNX Runtime CUDA version

In a source checkout, uv reads the CUDA-specific ONNX Runtime package feeds configured in `pyproject.toml`.
Select one extra that matches your CUDA stack. For CUDA 13:

```bash
uv pip install --torch-backend=cu130 ".[torch210,onnxruntime-gpu-cuda13]"
```

For CUDA 12, use `onnxruntime-gpu-cuda12` on Linux x86-64 with Python 3.11 through 3.13.

```bash
uv pip install --torch-backend=cu128 ".[torch29,onnxruntime-gpu-cuda12]"
```

The repository lockfile currently selects ONNX Runtime 1.29.0 for CUDA 12 and 1.30.0 for CUDA 13. Install
with `uv sync --locked` and the matching extra to reproduce the locked dependency set.

**pip** does not read the feed selection from `pyproject.toml`. For CUDA 13, install AITune first and then
reinstall ONNX Runtime from the CUDA 13 feed. This example uses 1.26.0, the pin in the NGC compatibility CI job:

```bash
pip install onnxruntime-gpu==1.26.0 --no-deps --force-reinstall --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
```

If ORT is already installed, use UV's targeted reinstall option with the CUDA 13 extra:

```bash
uv pip install --torch-backend=cu130 --reinstall-package onnxruntime-gpu ".[torch210,onnxruntime-gpu-cuda13]"
```

### NGC container install

The compatibility CI matrix includes these NVIDIA NGC PyTorch container versions:

* [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

| NGC PyTorch container | Torch version |
|---|---|
| `nvcr.io/nvidia/pytorch:25.08-py3` | 2.8a0 |
| `nvcr.io/nvidia/pytorch:25.10-py3` | 2.9a0 |
| `nvcr.io/nvidia/pytorch:26.01-py3` | 2.10a0 |
| `nvcr.io/nvidia/pytorch:26.03-py3` | 2.11a0 |
| `nvcr.io/nvidia/pytorch:26.05-py3` | 2.12a0 |
| `nvcr.io/nvidia/pytorch:26.06-py3` | 2.13a0 |
| `nvcr.io/nvidia/pytorch:26.08-py3` | 2.14a0 |
| `nvcr.io/nvidia/pytorch:26.10-py3` | 2.15a0 |

See the [NVIDIA PyTorch container release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
for the CUDA, driver, and library requirements for each image.

Inside an NGC container, use `pip` and the package name:

```bash
pip install --extra-index-url https://pypi.nvidia.com aitune
```

## Installing from Source

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/aitune
cd aitune
pip install --extra-index-url https://pypi.nvidia.com .
```

or use editable mode for development:

```bash
pip install --extra-index-url https://pypi.nvidia.com -e .
```
