<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

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
# Install

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

## Installing from Source

```bash
# Clone the repository
## Internal NVIDIA use
git clone https://gitlab-master.nvidia.com/dl/JoC/bermuda/ai-tune.git
cd ai-tune
pip install --extra-index-url https://pypi.nvidia.com .

## Official (valid after first release)
git clone https://github.com/ai-dynamo/aitune
cd aitune
pip install --extra-index-url https://pypi.nvidia.com .
```

or use editable mode for development:

```bash
pip install --extra-index-url https://pypi.nvidia.com -e .
```
