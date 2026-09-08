# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX artifact records."""

from dataclasses import dataclass
from enum import Enum

from aitune.records.artifacts.base import Artifact


class ONNXExecutionProvider(str, Enum):
    """ONNX Runtime execution provider selected during tuning."""

    CUDA = "cuda"
    TENSORRT = "tensorrt"


@dataclass(frozen=True, kw_only=True)
class ONNXArtifact(Artifact):
    """An ONNX model finalized as a tuning result for ONNX Runtime."""

    execution_provider: ONNXExecutionProvider = ONNXExecutionProvider.CUDA
