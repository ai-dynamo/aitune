# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable deployment records for tuning frontends and runtime adapters.

The package deliberately depends only on the Python standard library. Frontends
map native values into these records, and publishers consume them without either
side importing the other.

``DeploymentArtifact`` composes model files, bounded tensor specifications, and
runtime settings. Adapters dispatch on ``model.format`` and ``runtime.name``;
custom formats and runtimes use the same records without subclassing.

    >>> from pathlib import Path
    >>> inputs = (
    ...     BoundedTensorSpec(
    ...         name="input_ids",
    ...         dtype=DType.INT64,
    ...         min_shape=(1, 8),
    ...         max_shape=(8, 512),
    ...         batch_axis=0,
    ...     ),
    ...     BoundedTensorSpec(
    ...         name="attention_mask",
    ...         dtype=DType.BOOL,
    ...         min_shape=(1, 8),
    ...         max_shape=(8, 512),
    ...         batch_axis=0,
    ...     ),
    ... )
    >>> outputs = (
    ...     BoundedTensorSpec(
    ...         name="embedding",
    ...         dtype=DType.FLOAT32,
    ...         min_shape=(1, 8, 768),
    ...         max_shape=(8, 512, 768),
    ...         batch_axis=0,
    ...     ),
    ... )
    >>> artifact = DeploymentArtifact(
    ...     model=ModelFiles(format="onnx", path=Path("encoder.onnx")),
    ...     inputs=inputs,
    ...     outputs=outputs,
    ...     runtime=RuntimeConfig(name="onnxruntime", options={"execution_provider": "cuda"}),
    ... )
    >>> artifact.input_names
    ('input_ids', 'attention_mask')
    >>> artifact.max_batch_size
    8
"""

from aitune.records.artifact import DeploymentArtifact, ModelFiles, RuntimeConfig, TensorSample
from aitune.records.dtypes import DType
from aitune.records.shapes import BoundedTensorSpec

__all__ = [
    "BoundedTensorSpec",
    "DType",
    "DeploymentArtifact",
    "ModelFiles",
    "RuntimeConfig",
    "TensorSample",
]
