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
from aitune.torch.backend import TensorRTBackend, TensorRTBackendConfig, ONNXAutoCastConfig

config = TensorRTBackendConfig(quantization_config=ONNXAutoCastConfig()) # FP16 autocast through ModelOpt
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

The Torch-TensorRT JIT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through
`torch.compile`.

```python
import torch
from aitune.torch.backend import TorchTensorRTJitBackend, TorchTensorRTJitBackendConfig, TorchTensorRTConfig

config = TorchTensorRTJitBackendConfig(compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}))
backend = TorchTensorRTJitBackend(config)
```

## Torch-TensorRT Backend (AOT)

The Torch-TensorRT backend integrates TensorRT tuning directly into PyTorch, providing seamless tuning without model conversion through `torch_tensorrt.compile`.

```python
import torch
from aitune.torch.backend import TorchTensorRTAotBackend, TorchTensorRTAotBackendConfig, TorchTensorRTConfig

config = TorchTensorRTAotBackendConfig(compile_config=TorchTensorRTConfig(enabled_precisions={torch.float16}))
backend = TorchTensorRTAotBackend(config)
```

## TorchAO Backend

The TorchAO backend leverages PyTorch's AO (Accelerated Optimization) framework for model tuning.

```python
from aitune.torch.backend import TorchAOBackend

backend = TorchAOBackend()
```

## Torch Inductor Backend

The Torch Inductor backend uses PyTorch's Inductor compiler for model tuning.

```python
from aitune.torch.backend import TorchInductorBackend

backend = TorchInductorBackend()
```
