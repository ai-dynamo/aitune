---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Install"
---

## Prerequisites

Before proceeding with the installation of NVIDIA AITune, ensure your system meets the following criteria:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Version `3.10` or newer
* **PyTorch**: Version `2.7` or newer
* **TensorRT**: Version `10.5.0` or higher (for TensorRT backend)
* **NVIDIA GPU**: Required for GPU-accelerated tuning

You can use NGC Containers for PyTorch which contain all necessary dependencies:

* [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## Installing from PyPI (Recommended)

NVIDIA AITune can be installed from `pypi.org`.

```bash
pip install --extra-index-url https://pypi.nvidia.com aitune
```

### PyTorch 2.10 + CUDA 13 (torch210 extra)

To use PyTorch 2.10 with CUDA 13 support, install the `torch210` optional dependency group:

```bash
pip install --extra-index-url https://pypi.nvidia.com --index-url https://download.pytorch.org/whl/cu130 "aitune[torch210]"
```

When using `uv`, add `--torch-backend=cu130` so that uv resolves the CUDA 13 variants of the PyTorch packages:

```bash
uv pip install --torch-backend=cu130 --extra-index-url https://pypi.nvidia.com "aitune[torch210]"
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

To install the `torch210` extra from source with `uv`:

```bash
uv pip install --torch-backend=cu130 --extra-index-url https://pypi.nvidia.com -e ".[torch210]"
```
