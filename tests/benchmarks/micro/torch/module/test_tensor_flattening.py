# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tensor extraction costs across recorded structures and tensor sizes."""

import pytest

from aitune.torch.module.sample_metadata import SampleMetadata
from tests.benchmarks.micro.reporting import record_timings
from tests.benchmarks.micro.torch.module.samples import make_timing_sample


@pytest.mark.parametrize("gc_enabled", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("tensor_elements", [1, 4096], ids=["scalar-sized", "4096-elements"])
@pytest.mark.parametrize(
    "structure,tensor_count",
    [
        pytest.param("tensor", 1, id="root-tensor"),
        pytest.param("dict", 2, id="dict-2"),
        pytest.param("dict", 8, id="dict-8"),
        pytest.param("nested_dict", 8, id="nested-dict-depth-4"),
        pytest.param("tuple", 16, id="tuple-16"),
        pytest.param("tuple", 64, id="tuple-64"),
        pytest.param("tuple", 256, id="tuple-256"),
        pytest.param("cache", 65, id="cache-32-layers"),
        pytest.param("metadata", 8, id="dict-with-mutable-metadata"),
    ],
)
def test_tensor_flattening(structure, tensor_count, tensor_elements, gc_enabled, record_property):
    value = make_timing_sample(structure, tensor_count, tensor_elements)
    metadata = SampleMetadata.from_outputs(value)
    original_tensors = metadata.flatten_tensors(value)
    assert len(original_tensors) == tensor_count

    record_timings({"flatten": lambda: metadata.flatten_tensors(value)}, gc_enabled, record_property)

    assert all(
        actual is original for actual, original in zip(metadata.flatten_tensors(value), original_tensors, strict=True)
    )
