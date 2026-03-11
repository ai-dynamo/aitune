# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toy models package for testing."""

from .onnx_models import ToyOnnxModel
from .torch_models import ToyTorchModel

__all__ = ["ToyOnnxModel", "ToyTorchModel"]
