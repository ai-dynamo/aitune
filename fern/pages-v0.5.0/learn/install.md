---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Install"
---

## Prerequisites

Before installing NVIDIA AITune, make sure your system meets these requirements:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Version `3.10` or newer
* **PyTorch**: Version `2.8` or newer
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

```bash
pip install --extra-index-url https://pypi.nvidia.com "aitune[torch211]"
```

### Selecting a CUDA version

To target a specific CUDA version, add the matching PyTorch index:

```bash
pip install --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu130 "aitune[torch211]"
```

### NGC container install

You can use NGC PyTorch containers, which include the required NVIDIA dependencies:

* [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

| NGC PyTorch container | Torch version |
|---|---|
| `nvcr.io/nvidia/pytorch:25.08-py3` | 2.8a0 |
| `nvcr.io/nvidia/pytorch:25.10-py3` | 2.9a0 |
| `nvcr.io/nvidia/pytorch:26.01-py3` | 2.10a0 |
| `nvcr.io/nvidia/pytorch:26.03-py3` | 2.11a0 |
| `nvcr.io/nvidia/pytorch:26.05-py3` | 2.12a0 |

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
