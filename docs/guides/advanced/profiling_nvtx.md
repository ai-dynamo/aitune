<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
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
