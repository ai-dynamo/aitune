# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for exact sample metadata."""

import torch

from aitune.torch.module.exact_sample_metadata import ExactSampleMetadata
from aitune.torch.module.sample_metadata import SampleMetadata


def test_exact_sample_metadata_requires_exact_tensor_shapes():
    metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1, 1)})
    different_shape_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1, 2)})
    same_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1, 1)})

    assert metadata == same_metadata
    assert hash(metadata) == hash(same_metadata)
    assert metadata != different_shape_metadata
    assert hash(metadata) != hash(different_shape_metadata)


def test_exact_sample_metadata_uses_tensor_location():
    metadata = ExactSampleMetadata.from_inputs({"first": torch.randn(1), "second": torch.randn(2)})
    reordered_metadata = ExactSampleMetadata.from_inputs({"second": torch.randn(2), "first": torch.randn(1)})
    swapped_metadata = ExactSampleMetadata.from_inputs({"first": torch.randn(2), "second": torch.randn(1)})
    renamed_metadata = ExactSampleMetadata.from_inputs({"renamed": torch.randn(1), "second": torch.randn(2)})

    assert metadata == reordered_metadata
    assert hash(metadata) == hash(reordered_metadata)
    assert metadata != swapped_metadata
    assert hash(metadata) != hash(swapped_metadata)
    assert metadata != renamed_metadata
    assert hash(metadata) != hash(renamed_metadata)


def test_exact_sample_metadata_tracks_non_tensor_inputs():
    metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "label": "same", "flag": True})
    same_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "label": "same", "flag": True})
    different_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "label": "same", "flag": False})

    assert metadata == same_metadata
    assert hash(metadata) == hash(same_metadata)
    assert metadata != different_metadata
    assert hash(metadata) == hash(same_metadata)


def test_exact_sample_metadata_with_unhashable_inputs_is_a_valid_dict_key():
    metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "options": {"fast", "stable"}})
    same_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "options": {"stable", "fast"}})
    different_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1), "options": {"fast"}})

    cache = {metadata: "matching", different_metadata: "different"}

    assert cache[same_metadata] == "matching"
    assert cache[different_metadata] == "different"


def test_exact_sample_metadata_handles_nested_structures():
    metadata = ExactSampleMetadata.from_inputs({
        "values": [torch.randn(1, 2), {"x": torch.randn(3)}],
        "nested": {"y": torch.randn(4, 5)},
    })
    same_metadata = ExactSampleMetadata.from_inputs({
        "values": [torch.randn(1, 2), {"x": torch.randn(3)}],
        "nested": {"y": torch.randn(4, 5)},
    })
    different_nested_shape_metadata = ExactSampleMetadata.from_inputs({
        "values": [torch.randn(1, 2), {"x": torch.randn(333)}],
        "nested": {"y": torch.randn(4, 5)},
    })

    assert metadata == same_metadata
    assert hash(metadata) == hash(same_metadata)
    assert metadata != different_nested_shape_metadata
    assert hash(metadata) != hash(different_nested_shape_metadata)


def test_exact_sample_metadata_ignores_tensor_dtype():
    float_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1, 2, dtype=torch.float32)})
    int_metadata = ExactSampleMetadata.from_inputs({"input": torch.ones(1, 2, dtype=torch.int64)})

    assert float_metadata == int_metadata
    assert hash(float_metadata) == hash(int_metadata)


def test_exact_sample_metadata_is_not_equal_to_base_sample_metadata():
    exact_metadata = ExactSampleMetadata.from_inputs({"input": torch.randn(1, 2)})
    base_metadata = SampleMetadata.from_inputs({"input": torch.randn(1, 2)})

    assert exact_metadata != base_metadata
    assert hash(exact_metadata) != hash(base_metadata)
