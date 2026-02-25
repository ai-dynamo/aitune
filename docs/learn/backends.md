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
# Backends

NVIDIA AITune supports multiple tuning backends, each with different characteristics and use cases. The backends align with a common interface for the build and inference process.

## TensorRT Backend

The TensorRT backend provides highly optimized inference using NVIDIA's TensorRT engine. It offers the best performance for production deployments. The backend integrates [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) in a seamless flow.

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig

config = TensorRTBackendConfig(precision="fp16")
backend = TensorRTBackend(config)
```

### CUDA Graphs Support

The TensorRT backend supports CUDA Graphs for reduced CPU overhead and improved inference performance. CUDA Graphs automatically capture and replay GPU operations, eliminating kernel launch overhead for repeated inference calls. This feature is disabled by default.

Keep in mind that graphs are automatically recaptured when input shapes change.

```python
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig

# Enable CUDA Graphs for optimized inference
config = TensorRTBackendConfig(use_cuda_graphs=True)
backend = TensorRTBackend(config)
```

## Torch-TensorRT Backend (JIT)

Torch-TensorRT JIT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through
`torch.compile`.

```python
from aitune.torch.backend import TorchTensorRTJitBackend, TorchTensorRTJitBackendConfig

config = TorchTensorRTJitBackendConfig(precision="fp16")
backend = TorchTensorRTJitBackend(config)
```

## Torch-TensorRT Backend (AOT)

Torch-TensorRT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through `torch_tensorrt.compile`.

```python
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig

config = TorchTensorRTAotBackendConfig(precision="fp16")
backend = TorchTensorRTAotBackend(config)
```

## TorchAO Backend

TorchAO backend leverages PyTorch's AO (Accelerated Optimization) framework for model tuning.

```python
from aitune.torch.backend import TorchAOBackend

backend = TorchAOBackend()
```

## Torch Inductor Backend

Torch Inductor backend uses PyTorch's Inductor compiler for model tuning.

```python
from aitune.torch.backend import TorchInductorBackend

backend = TorchInductorBackend()
```
