# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model utilities."""

import timm


def get_model(model_name: str = "resnet50", pretrained: bool = True):
    """Get a pretrained resnet50 model."""
    model = timm.create_model(model_name, pretrained=pretrained)
    model.to("cuda")
    model.eval()
    return model


def get_transform(model):
    """Get image transform for the model."""
    config = timm.data.resolve_data_config(model.pretrained_cfg)
    return timm.data.create_transform(**config)
