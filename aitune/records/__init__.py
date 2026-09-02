# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-neutral values shared by tuning frontends and publishers.

The package deliberately depends only on the Python standard library. Frontends
map native values into these records, and publishers consume them without either
side importing the other.

``TensorSpec`` is the static interface declared by an executable. Tuning adds
concrete bounds and batch interpretation in ``TunedTensorSpec``; only tuned
specifications are stored on an ``Artifact``.

    >>> from pathlib import Path
    >>> input_ids = TensorSpec("input_ids", DType.INT64, ("batch", "sequence"))
    >>> attention_mask = TensorSpec("attention_mask", DType.BOOL, ("batch", "sequence"))
    >>> output_spec = TensorSpec("embedding", DType.FLOAT32, ("batch", "sequence", 768))
    >>> inputs = (
    ...     TunedTensorSpec.from_spec(input_ids, min_shape=(1, 8), max_shape=(8, 512), batch_axis=0),
    ...     TunedTensorSpec.from_spec(attention_mask, min_shape=(1, 8), max_shape=(8, 512), batch_axis=0),
    ... )
    >>> outputs = (TunedTensorSpec.from_spec(output_spec, min_shape=(1, 8, 768), max_shape=(8, 512, 768), batch_axis=0),)
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

from aitune.records.artifact import Artifact, ArtifactIntegrityError, ONNXArtifact, TensorRTPlanArtifact
from aitune.records.dtypes import DType
from aitune.records.shapes import TensorSpec, TunedTensorSpec

__all__ = [
    "Artifact",
    "ArtifactIntegrityError",
    "DType",
    "ONNXArtifact",
    "TensorRTPlanArtifact",
    "TensorSpec",
    "TunedTensorSpec",
]
