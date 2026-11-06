# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the optional provider needed by the native FlexTensor workflows."""

import torch


def require_native_compile_support():
    from flextensor import custom_ops

    pre_compute = getattr(torch.ops.flextensor, "pre_compute", None)
    if not callable(getattr(custom_ops, "unit_index_tensor", None)) or not hasattr(pre_compute, "unit"):
        raise RuntimeError(
            "Native FlexTensor tuning requires a development/source build containing the per-unit compilation "
            "and replay fixes. The stock FlexTensor 0.5.0 release does not contain them."
        )
