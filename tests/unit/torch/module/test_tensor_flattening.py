# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flattening reads current tensors at recorded paths, independent of container insertion order."""

import pickle
from collections import UserDict, namedtuple
from dataclasses import dataclass

import pytest
import torch

from aitune.torch.module.sample_metadata import SampleMetadata


@pytest.mark.parametrize("tensor_count", [1, 16, 256])
@pytest.mark.parametrize("container_type", [tuple, list])
@pytest.mark.parametrize("restore_metadata", [False, True])
def test_flatten_indexed_sequences_preserves_order_across_calls_and_checkpoint_loading(
    tensor_count, container_type, restore_metadata
):
    original = container_type(torch.full((2,), float(index)) for index in range(tensor_count))
    metadata = SampleMetadata.from_outputs(original)
    assert all(
        actual is expected for actual, expected in zip(metadata.flatten_tensors(original), original, strict=True)
    )
    if restore_metadata:
        metadata = pickle.loads(pickle.dumps(metadata))
        metadata = SampleMetadata.from_dict(metadata.to_dict())
    replacements = container_type(torch.zeros(2) for _ in range(tensor_count))

    flattened = metadata.flatten_tensors(replacements)

    assert type(flattened) is tuple
    assert len(flattened) == tensor_count
    assert all(actual is expected for actual, expected in zip(flattened, replacements, strict=True))
    for index, tensor in enumerate(original):
        torch.testing.assert_close(tensor, torch.full((2,), float(index)))


@pytest.mark.parametrize("container_type", [dict, UserDict])
def test_flatten_mapping_reads_recorded_keys_instead_of_runtime_insertion_order(container_type):
    first, second = torch.ones(2), torch.zeros(2)
    metadata = SampleMetadata.from_inputs(container_type({"first": first, "second": second, "constant": 3}))
    runtime = container_type({"second": first, "extra": torch.empty(2), "constant": 4, "first": second})

    flattened = metadata.flatten_tensors(runtime)

    assert len(flattened) == 2
    assert flattened[0] is second
    assert flattened[1] is first


def test_flatten_sequence_reads_recorded_indices_and_ignores_new_tensor_leaves():
    metadata = SampleMetadata.from_outputs(("constant", torch.ones(2), None, torch.zeros(2)))
    first, second = torch.zeros(2), torch.ones(2)

    flattened = metadata.flatten_tensors((torch.empty(2), first, "changed", second, torch.empty(2)))

    assert len(flattened) == 2
    assert flattened[0] is first
    assert flattened[1] is second


@dataclass
class State:
    tensor: torch.Tensor
    labels: list


def test_flatten_root_nested_and_attribute_paths_keep_their_behavior():
    Pair = namedtuple("Pair", ("first", "second"))
    first, second = torch.ones(2), torch.zeros(2)
    value = {"state": State(first, []), "nested": [Pair(second, "constant")]}
    metadata = SampleMetadata.from_outputs(value)

    flattened = metadata.flatten_tensors(value)

    assert flattened[0] is first
    assert flattened[1] is second
    assert SampleMetadata.from_outputs(first).flatten_tensors(first)[0] is first
    assert SampleMetadata.from_outputs({"labels": []}).flatten_tensors({"labels": [1]}) == ()
    state_metadata = SampleMetadata.from_outputs(value["state"])
    assert state_metadata.flatten_tensors(State(second, []))[0] is second


@pytest.mark.parametrize(
    "recorded,runtime,error,message",
    [
        ((torch.ones(2),), (), IndexError, ""),
        ((torch.ones(2), torch.ones(2)), (None,), IndexError, ""),
        ({"value": torch.ones(2)}, {}, KeyError, "value"),
        ({"first": torch.ones(2), "second": torch.zeros(2)}, {"first": None}, KeyError, "second"),
        ((torch.ones(2),), (None,), TypeError, "resolved to NoneType, not a tensor"),
        ({"value": torch.ones(2)}, {"value": []}, TypeError, "resolved to list, not a tensor"),
    ],
)
def test_flatten_indexed_paths_preserves_missing_path_and_non_tensor_errors(recorded, runtime, error, message):
    metadata = SampleMetadata.from_outputs(recorded)
    with pytest.raises(error, match=message):
        metadata.flatten_tensors(runtime)
