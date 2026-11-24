# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness and CPU baseline measurements for the complete restoration contract."""

from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch

from aitune.torch.module.forward_effects_spec import ForwardEffectsCapture
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata
from tests.unit.torch.module.reconstruction_benchmark import record_timings


@dataclass
class Result:
    values: object
    metadata: list


@pytest.mark.parametrize("tensor_template", [False, True])
def test_reconstruction_preserves_types_and_isolates_consecutive_calls(tensor_template):
    Pair = namedtuple("Pair", ("first", "second"))
    original = (torch.ones(2), torch.ones(3))
    value = Result(Pair(original[0], [{"tensor": original[1]}]), [{"tokens": []}])
    metadata = SampleMetadata.from_outputs(value)
    template = value if tensor_template else metadata.without_tensors(value)
    replacements = tuple(torch.zeros_like(tensor) for tensor in original)

    first = metadata.unflatten_tensors(replacements, template)
    second = metadata.unflatten_tensors(replacements, template)

    for restored in (first, second):
        assert type(restored) is Result
        assert type(restored.values) is Pair
        assert type(restored.values.second) is list
        assert type(restored.values.second[0]) is dict
        assert restored.values.first is replacements[0]
        assert restored.values.second[0]["tensor"] is replacements[1]
    first.metadata[0]["tokens"].append(1)
    first.values.second.append("changed")
    assert second.metadata == template.metadata == value.metadata == [{"tokens": []}]
    assert len(second.values.second) == len(template.values.second) == len(value.values.second) == 1
    assert value.values.first is original[0]
    assert value.values.second[0]["tensor"] is original[1]


@pytest.mark.parametrize("parent_first", [False, True])
def test_preserved_parent_takes_precedence_over_child_and_tensor_replacements(parent_first):
    value = {"cache": {"tensor": torch.ones(2)}, "metadata": []}
    metadata = SampleMetadata.from_outputs(value)
    parent = Locator((("cache", ObjectType.DICT),), is_output=True)
    child = Locator((("cache", ObjectType.DICT), ("tensor", ObjectType.DICT)), is_output=True)
    preserved_cache = {"tensor": torch.full((2,), 7.0)}
    preserved = ((parent, preserved_cache), (child, torch.zeros(2)))
    if not parent_first:
        preserved = tuple(reversed(preserved))
    template = metadata.without_tensors(value, (value["cache"],))

    restored = metadata.unflatten_tensors((torch.empty(2),), template, preserved)

    assert restored["cache"] is preserved_cache
    torch.testing.assert_close(preserved_cache["tensor"], torch.full((2,), 7.0))
    assert template == {"cache": None, "metadata": []}


@pytest.mark.parametrize("preserve_root", [False, True])
@pytest.mark.parametrize("tensor_template", [False, True])
@pytest.mark.parametrize("result_container", [None, tuple, list], ids=["tensor", "tuple", "list"])
def test_root_tensor_reconstruction_respects_preserved_identity(preserve_root, tensor_template, result_container):
    original, replacement, preserved = torch.ones(2), torch.zeros(2), torch.full((2,), 3.0)
    metadata = SampleMetadata.from_outputs(original)
    aliases = ((Locator((), is_output=True), preserved),) if preserve_root else ()

    flat_result = replacement if result_container is None else result_container((replacement,))
    result = metadata.unflatten_tensors(flat_result, original if tensor_template else None, aliases)

    assert result is (preserved if preserve_root else replacement)
    torch.testing.assert_close(original, torch.ones(2))


@pytest.mark.parametrize(
    "replacements,error,message",
    [
        ((), ValueError, "Expected 1 tensors, got 0"),
        ((torch.ones(2), torch.ones(2)), ValueError, "Expected 1 tensors, got 2"),
        ((None,), TypeError, "received NoneType, not a tensor"),
    ],
)
def test_root_tensor_reconstruction_validates_results(replacements, error, message):
    metadata = SampleMetadata.from_outputs(torch.ones(2))
    with pytest.raises(error, match=message):
        metadata.unflatten_tensors(replacements, None)


class CacheModel(torch.nn.Module):
    def __init__(self, mutation, alias_count):
        super().__init__()
        self.mutation = mutation
        self.alias_count = alias_count

    def forward(self, x, state):
        for cache in state:
            if self.mutation == "copy":
                cache["tensor"].copy_(x)
            elif self.mutation == "replace":
                cache["tensor"] = x + 1
        return {"value": x * 2, "aliases": state[: self.alias_count], "metadata": []}


@pytest.mark.parametrize("gc_enabled", [False, True], ids=["gc-off", "gc-on"])
@pytest.mark.parametrize("alias_count", [0, 1, 8])
@pytest.mark.parametrize("mutation", ["none", "copy", "replace"])
def test_restore_outputs_timing(mutation, alias_count, gc_enabled, record_property):
    """Time GraphSpec restoration including mutation handling and live alias lookup.

    The local model records the contract outside timing; this measures CPU restoration alone,
    without compilation, tensor allocation, or model execution in the measured region.
    """
    model = CacheModel(mutation, alias_count)
    signature = ForwardSignature.from_callable(model.forward)
    value = torch.ones(16)
    state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    arguments = signature.normalize((value, state), {}).arguments
    input_spec = SampleMetadata.from_inputs(arguments)
    capture = ForwardEffectsCapture.before_call(arguments)
    output = model(value, state)
    output_spec = SampleMetadata.from_outputs(output)
    effects = capture.after_call(arguments, output, output_spec)
    graph = GraphSpec("reconstruction-baseline", input_spec, output_spec, signature, effects)
    flat_outputs = output_spec.flatten_tensors(output) + tuple(
        mutation.locator.get_value(arguments) for mutation in graph.extra_tensor_mutations()
    )
    runtime_state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    runtime_arguments = {"x": value, "state": runtime_state}
    original_tensors = tuple(cache["tensor"] for cache in runtime_state)

    def check_restoration():
        restored = graph.restore_outputs(flat_outputs, runtime_arguments)
        assert restored["value"] is flat_outputs[0]
        assert type(restored) is dict
        assert type(restored["aliases"]) is list
        assert len(restored["aliases"]) == alias_count
        for index, cache in enumerate(restored["aliases"]):
            assert cache is runtime_state[index]
        for index, cache in enumerate(runtime_state):
            expected = value + 1 if mutation == "replace" else value if mutation == "copy" else torch.zeros_like(value)
            torch.testing.assert_close(cache["tensor"], expected)
            if mutation != "replace":
                assert cache["tensor"] is original_tensors[index]
        return restored

    first = check_restoration()
    first["metadata"].append("changed")
    record_timings(
        {"restore_outputs": lambda: graph.restore_outputs(flat_outputs, runtime_arguments)}, gc_enabled, record_property
    )
    second = check_restoration()
    assert second["metadata"] == effects.output_template["metadata"] == []
    assert second is not first
    # The same recorded graph must resolve aliases from each call's arguments. A prepared
    # strategy must never keep the previous caller's containers or mutation targets alive.
    previous_state = runtime_state
    runtime_state = [{"tensor": torch.zeros_like(value)} for _ in range(8)]
    runtime_arguments["state"] = runtime_state
    original_tensors = tuple(cache["tensor"] for cache in runtime_state)
    third = check_restoration()
    for index, cache in enumerate(third["aliases"]):
        assert cache is not previous_state[index]
