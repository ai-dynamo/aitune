# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diffusers integrations."""

try:
    import diffusers
    import torch
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm
    from packaging.version import Version

    if Version(diffusers.__version__) >= Version("0.35"):
        # For diffusers>=0.35 models use RMSNorm which is not support on ONNX Trace path as is limited to opset 20.
        # Patching RMSNorm for using Diffusers version fix that issue making models working with trace path.
        # WAR adapted from ModelOpt which faced similar issue:
        # https://github.com/NVIDIA/Model-Optimizer/blob/2b8defc14b601491bb1479117181048912e6fdfc/examples/diffusers/quantization/diffusion_trt.py#L27
        torch.nn.RMSNorm = DiffuserRMSNorm
        torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm
except ImportError:
    pass
