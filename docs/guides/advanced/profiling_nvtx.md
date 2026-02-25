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
# Profiling with NVTX

NVIDIA AITune includes NVTX (NVIDIA Tools Extension) annotations for profiling and debugging. NVTX marks key operations in the code, making them visible in profiling tools like NVIDIA Nsight Systems.

**Note**: NVTX annotations are disabled by default to avoid overhead in production environments.

## Enabling NVTX

To enable NVTX profiling, set the environment variable before running your script:

```bash
export NVTX_ENABLE=1
python your_script.py
```

## Using with Nsight Systems

Once enabled, you can profile your application with Nsight Systems:

```bash
NVTX_ENABLE=1 nsys profile -o output.nsys-rep python your_script.py
```

The NVTX annotations will appear as colored regions in the timeline, helping you identify:

* Backend inference calls (TensorRT, Torch-TensorRT, TorchAO, etc.)
* Tuning operation
* Performance bottlenecks
