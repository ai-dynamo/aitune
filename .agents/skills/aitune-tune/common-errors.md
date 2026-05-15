<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Common Errors and Diagnostics

| Error | Likely Cause | Fix |
|---|---|---|
| `ImportError: tensorrt` | TensorRT not installed | Install from `https://pypi.nvidia.com` |
| `Graph break detected` | Data-dependent control flow | Try `TorchTensorRTJitBackend` or `TorchAOBackend` instead |
| `CUDA out of memory` | Model too large for GPU | Reduce batch size or try fp16 |
| `Unsupported op: ...` | Op not supported by backend | Advance to next backend in trial order |
| `RuntimeError: CUDA error` | GPU driver/CUDA mismatch | Check `nvidia-smi` and CUDA toolkit version |
| Checkpoint load fails | `.ait` file corrupt or wrong GPU arch | Re-tune on target hardware |
| Cache readonly | Cache directory based in home directory can be mounted read-only in sandboxed environments | Change cache directory with `AITUNE_CACHE_DIR` environment variable |


