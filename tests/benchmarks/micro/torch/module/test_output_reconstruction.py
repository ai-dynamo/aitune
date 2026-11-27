# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU output reconstruction costs, excluding model execution and recipe preparation."""

from time import perf_counter_ns

import pytest
import torch

from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.output_reconstruction import OutputReconstruction
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.benchmarks.micro.reporting import record_timings
from tests.benchmarks.micro.torch.module.samples import make_timing_sample


@pytest.mark.parametrize("gc_enabled", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("tensor_template", [False, True], ids=["tensor-free-template", "tensor-template"])
@pytest.mark.parametrize("tensor_elements", [1, 4096], ids=["scalar-sized", "4096-elements"])
@pytest.mark.parametrize(
    "structure,tensor_count,preserve_cache",
    [
        pytest.param("tensor", 1, False, id="root-tensor"),
        pytest.param("dict", 2, False, id="dict-2"),
        pytest.param("dict", 8, False, id="dict-8"),
        pytest.param("nested_dict", 8, False, id="nested-dict-depth-4"),
        pytest.param("tuple", 16, False, id="tuple-16"),
        pytest.param("tuple", 64, False, id="tuple-64"),
        pytest.param("tuple", 256, False, id="tuple-256"),
        pytest.param("cache", 65, False, id="cache-32-layers"),
        pytest.param("cache", 65, True, id="preserved-cache-32-layers"),
        pytest.param("metadata", 8, False, id="dict-with-mutable-metadata"),
    ],
)
def test_output_reconstruction(
    structure, tensor_count, preserve_cache, tensor_elements, tensor_template, gc_enabled, record_property
):
    """Report CPU microseconds per call, without machine-dependent performance assertions.

    Run on the same idle machine for comparisons.
    Setup, allocation, metadata discovery, and correctness checks are outside the timed regions.
    Exercise both cyclic GC settings and both template forms. These measurements exclude model
    execution and GPU scheduling, so they do not establish end-to-end inference impact.
    """
    value = make_timing_sample(structure, tensor_count, tensor_elements)
    metadata = SampleMetadata.from_outputs(value)
    original_tensors = metadata.flatten_tensors(value)
    tensors = tuple(torch.empty_like(tensor) for tensor in original_tensors)
    preserved_values = ()
    preserved_containers = ()
    if preserve_cache:
        preserved_containers = (value["cache"],)
        preserved_values = ((Locator((("cache", ObjectType.DICT),), is_output=True), value["cache"]),)
    template = value if tensor_template else metadata.without_tensors(value, preserved_containers)
    preparation_start = perf_counter_ns()
    reconstruction = None if preserved_values else OutputReconstruction.prepare(metadata, template)
    record_property("preparation_us", (perf_counter_ns() - preparation_start) / 1000)
    record_property("prepared_reconstruction", reconstruction is not None)

    restored = metadata.unflatten_tensors(tensors, template, preserved_values)
    expected_tensors = (tensors[0], *original_tensors[1:]) if preserve_cache else tensors
    assert len(original_tensors) == tensor_count
    assert all(
        actual is expected
        for actual, expected in zip(metadata.flatten_tensors(restored), expected_tensors, strict=True)
    )
    if preserve_cache:
        assert restored["cache"] is value["cache"]
    if structure == "metadata":
        assert restored["metadata"] == value["metadata"]
        restored["metadata"][0]["tokens"].append(-1)
        assert template["metadata"][0]["tokens"] == list(range(32))
        assert value["metadata"][0]["tokens"] == list(range(32))

    operations = {
        "unflatten": lambda: metadata.unflatten_tensors(tensors, template, preserved_values),
    }
    if reconstruction is not None:
        prepared_result = reconstruction.unflatten_tensors(tensors)
        assert all(
            actual is expected
            for actual, expected in zip(metadata.flatten_tensors(prepared_result), tensors, strict=True)
        )
        if structure == "metadata":
            prepared_result["metadata"][0]["tokens"].append(-1)
            assert template["metadata"][0]["tokens"] == list(range(32))
        operations["prepared_unflatten"] = lambda: reconstruction.unflatten_tensors(tensors)
    record_timings(operations, gc_enabled, record_property)

    assert all(
        actual is original for actual, original in zip(metadata.flatten_tensors(value), original_tensors, strict=True)
    )
    if not tensor_template:
        assert template == metadata.without_tensors(value, preserved_containers)


@pytest.mark.parametrize("gc_enabled", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("metadata_kind", ["list", "dict"])
@pytest.mark.parametrize("scalar_count", [32, 256, 1024, 4096])
def test_output_reconstruction_metadata_size(scalar_count, metadata_kind, gc_enabled, record_property):
    """Report absolute reconstruction cost as scalar metadata containers grow."""
    scalars = (
        list(range(scalar_count)) if metadata_kind == "list" else {str(index): index for index in range(scalar_count)}
    )
    value = {"tensor": torch.ones(1), "metadata": scalars}
    metadata = SampleMetadata.from_outputs(value)
    template = metadata.without_tensors(value)
    start = perf_counter_ns()
    reconstruction = OutputReconstruction.prepare(metadata, template)
    record_property("preparation_us", (perf_counter_ns() - start) / 1000)
    assert reconstruction is not None
    tensors = (torch.zeros(1),)

    first = reconstruction.unflatten_tensors(tensors)
    assert first["tensor"] is tensors[0]
    assert first["metadata"] == scalars
    assert type(first["metadata"]) is type(scalars)
    first["metadata"].clear()
    assert len(scalars) == scalar_count
    assert len(template["metadata"]) == scalar_count

    record_timings(
        {
            "unflatten": lambda: metadata.unflatten_tensors(tensors, template),
            "prepared_unflatten": lambda: reconstruction.unflatten_tensors(tensors),
        },
        gc_enabled,
        record_property,
    )
    second = reconstruction.unflatten_tensors(tensors)
    assert second["metadata"] == template["metadata"] == scalars
