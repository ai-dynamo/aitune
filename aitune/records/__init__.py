# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral values shared by tuning frontends and publishers.

The package deliberately depends only on the Python standard library. Frontends
map native values into these records, and publishers consume them without either
side importing the other.

``BoundedTensorSpec`` records the executable tensor interface together with the
concrete bounds and batch interpretation established by tuning.

    >>> from pathlib import Path
    >>> inputs = (
    ...     BoundedTensorSpec("input_ids", DType.INT64, ("batch", "sequence"), (1, 8), (8, 512), 0),
    ...     BoundedTensorSpec("attention_mask", DType.BOOL, ("batch", "sequence"), (1, 8), (8, 512), 0),
    ... )
    >>> outputs = (
    ...     BoundedTensorSpec("embedding", DType.FLOAT32, ("batch", "sequence", 768), (1, 8, 768), (8, 512, 768), 0),
    ... )
    >>> artifact = ONNXArtifact(
    ...     path=Path("encoder.onnx"),
    ...     fingerprint="0" * 64,
    ...     inputs=inputs,
    ...     outputs=outputs,
    ... )
    >>> artifact.input_names
    ('input_ids', 'attention_mask')
    >>> artifact.max_batch_size
    8
"""

from aitune.records.artifact import (
    Artifact,
    ArtifactFile,
    ArtifactIntegrityError,
    ONNXArtifact,
    ONNXExecutionProvider,
    PT2Artifact,
    TensorRTPlanArtifact,
)
from aitune.records.dtypes import DType
from aitune.records.shapes import BoundedTensorSpec

__all__ = [
    "Artifact",
    "ArtifactFile",
    "ArtifactIntegrityError",
    "BoundedTensorSpec",
    "DType",
    "ONNXArtifact",
    "ONNXExecutionProvider",
    "PT2Artifact",
    "TensorRTPlanArtifact",
]
