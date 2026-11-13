# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the optional provider needed by the native FlexTensor workflows."""

import torch


def require_native_compile_support():
    from flextensor import custom_ops

    pre_compute = getattr(torch.ops.flextensor, "pre_compute", None)
    if not callable(getattr(custom_ops, "unit_index_tensor", None)) or not hasattr(pre_compute, "unit"):
        raise RuntimeError(
            "Native FlexTensor tuning requires per-unit compilation and replay support "
            "(unit_index_tensor and pre_compute.unit)."
        )
