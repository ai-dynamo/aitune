# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared reconstruction preserves the observable output contract across calls and graphs."""

import gc
import pickle
import weakref
from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch

from aitune.torch.module.forward_effects_spec import ForwardEffectsSpec, OutputInputAlias
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.output_reconstruction import OutputReconstruction
from aitune.torch.module.sample_metadata import SampleMetadata


def _forward(x):
    return x


def _graph(value):
    output_spec = SampleMetadata.from_outputs(value)
    return GraphSpec(
        name="output-reconstruction",
        input_spec=SampleMetadata.from_inputs({"x": torch.ones(2)}),
        output_spec=output_spec,
        forward_signature=ForwardSignature.from_callable(_forward),
        forward_effects_spec=ForwardEffectsSpec(output_spec.without_tensors(value)),
    )


@pytest.mark.parametrize("tensor_template", [False, True])
@pytest.mark.parametrize("restore_recipe", [False, True])
def test_prepared_cache_reconstruction_preserves_types_order_and_call_isolation(tensor_template, restore_recipe):
    original = tuple(torch.full((2,), float(index)) for index in range(5))
    value = {
        "logits": original[0],
        "cache": ((original[1], original[2]), (original[3], original[4])),
        "metadata": {"tokens": [], "step": 0},
    }
    metadata = SampleMetadata.from_outputs(value)
    template = value if tensor_template else metadata.without_tensors(value)
    recipe = OutputReconstruction.prepare(metadata, template)
    assert recipe is not None
    if restore_recipe:
        recipe = pickle.loads(pickle.dumps(recipe))
    replacements = tuple(torch.zeros_like(tensor) for tensor in original)
    next_replacements = tuple(torch.ones_like(tensor) for tensor in original)

    first = recipe.unflatten_tensors(replacements)
    second = recipe.unflatten_tensors(next_replacements)

    for output, expected in ((first, replacements), (second, next_replacements)):
        assert type(output) is dict
        assert type(output["cache"]) is tuple
        assert all(type(layer) is tuple for layer in output["cache"])
        assert type(output["metadata"]) is dict
        assert type(output["metadata"]["tokens"]) is list
        assert all(actual is tensor for actual, tensor in zip(metadata.flatten_tensors(output), expected, strict=True))
    first["metadata"]["tokens"].append(7)
    first["cache"] = ()
    assert second["metadata"] == template["metadata"] == value["metadata"] == {"tokens": [], "step": 0}
    assert len(second["cache"]) == len(template["cache"]) == len(value["cache"]) == 2
    assert all(actual is tensor for actual, tensor in zip(metadata.flatten_tensors(value), original, strict=True))


def test_prepared_reconstruction_uses_recorded_paths_instead_of_template_leaf_order():
    first, second = torch.ones(2), torch.zeros(2)
    metadata = SampleMetadata.from_outputs({"first": first, "second": second})
    recipe = OutputReconstruction.prepare(metadata, {"second": None, "first": None})
    assert recipe is not None

    restored = recipe.unflatten_tensors([first, second])

    assert list(restored) == ["second", "first"]
    assert restored["first"] is first
    assert restored["second"] is second


@pytest.mark.parametrize("scalar_count", [0, 32, 1024])
@pytest.mark.parametrize("container_type", [list, dict])
@pytest.mark.parametrize("restore_recipe", [False, True])
def test_scalar_metadata_containers_remain_isolated_between_calls(scalar_count, container_type, restore_recipe):
    tokens = (
        list(range(scalar_count)) if container_type is list else {str(index): index for index in range(scalar_count)}
    )
    value = {"tensor": torch.ones(2), "metadata": [{"tokens": tokens, "tags": ["first", "second"]}]}
    metadata = SampleMetadata.from_outputs(value)
    template = metadata.without_tensors(value)
    recipe = OutputReconstruction.prepare(metadata, template)
    assert recipe is not None
    if restore_recipe:
        recipe = pickle.loads(pickle.dumps(recipe))
    first_tensor, second_tensor = torch.zeros(2), torch.ones(2)

    first = recipe.unflatten_tensors((first_tensor,))
    if container_type is list:
        first["metadata"][0]["tokens"].append(-1)
    else:
        first["metadata"][0]["tokens"]["changed"] = -1
    first["metadata"][0]["tags"].append("changed")
    second = recipe.unflatten_tensors((second_tensor,))

    assert first["tensor"] is first_tensor
    assert second["tensor"] is second_tensor
    assert type(second["metadata"]) is list
    assert type(second["metadata"][0]) is dict
    assert type(second["metadata"][0]["tokens"]) is container_type
    assert second["metadata"] == template["metadata"] == value["metadata"]
    assert second["metadata"][0]["tokens"] is not tokens


def test_scalar_none_containers_do_not_hide_tensor_placeholders():
    first, second = torch.ones(2), torch.zeros(2)
    value = {"tensors": [first, second], "constants": [None, None]}
    metadata = SampleMetadata.from_outputs(value)
    template = metadata.without_tensors(value)
    recipe = OutputReconstruction.prepare(metadata, template)
    assert recipe is not None

    restored = recipe.unflatten_tensors((second, first))

    assert restored["tensors"][0] is second
    assert restored["tensors"][1] is first
    assert restored["constants"] == [None, None]
    restored["constants"].append("changed")
    assert template == {"tensors": [None, None], "constants": [None, None]}
    assert value["constants"] == [None, None]


@pytest.mark.parametrize("value", [list(range(32)), {str(index): index for index in range(32)}])
def test_scalar_only_outputs_are_copied_without_tensor_results(value):
    metadata = SampleMetadata.from_outputs(value)
    recipe = OutputReconstruction.prepare(metadata, value)
    assert recipe is not None

    first = recipe.unflatten_tensors(())
    first.clear()
    second = recipe.unflatten_tensors(())

    assert type(second) is type(value)
    assert second == value
    assert len(second) == 32
    assert second is not value


def test_shared_scalar_metadata_keeps_general_reconstruction():
    shared = [1, 2, 3]
    graph = _graph({"tensor": torch.ones(2), "first": shared, "second": shared})
    template = graph.forward_effects_spec.output_template
    assert OutputReconstruction.prepare(graph.output_spec, template) is None
    tensor = torch.zeros(2)

    restored = graph.restore_outputs((tensor,), {})

    assert restored["tensor"] is tensor
    assert restored["first"] is restored["second"]
    assert restored["first"] is not shared
    restored["first"].append(4)
    assert restored["second"] == [1, 2, 3, 4]
    assert shared == template["first"] == [1, 2, 3]


def test_unrecorded_tensors_in_metadata_keep_general_reconstruction():
    metadata = SampleMetadata.from_outputs({"tensor": torch.ones(2)})
    template = {"tensor": None, "metadata": [torch.zeros(2)]}

    assert OutputReconstruction.prepare(metadata, template) is None


def test_prepared_recipe_does_not_retain_sample_tensor_storage():
    tensor = torch.ones(2)
    reference = weakref.ref(tensor)
    metadata = SampleMetadata.from_outputs({"result": (tensor,)})
    recipe = OutputReconstruction.prepare(metadata, {"result": (tensor,)})
    assert recipe is not None

    del tensor
    gc.collect()

    assert reference() is None
    replacement = torch.zeros(2)
    assert recipe.unflatten_tensors((replacement,))["result"][0] is replacement


@pytest.mark.parametrize("result_container", [None, tuple, list], ids=["tensor", "tuple", "list"])
def test_prepared_single_tensor_container_accepts_supported_result_formats(result_container):
    tensor = torch.ones(2)
    recipe = OutputReconstruction.prepare(SampleMetadata.from_outputs({"result": tensor}), {"result": None})
    assert recipe is not None
    flat_result = tensor if result_container is None else result_container((tensor,))

    assert recipe.unflatten_tensors(flat_result)["result"] is tensor


@pytest.mark.parametrize(
    "results,error,message",
    [((), ValueError, "Expected 1 tensors, got 0"), ((None,), TypeError, "received NoneType, not a tensor")],
)
def test_prepared_reconstruction_validates_results(results, error, message):
    recipe = OutputReconstruction.prepare(SampleMetadata.from_outputs({"result": torch.ones(2)}), {"result": None})
    assert recipe is not None
    with pytest.raises(error, match=message):
        recipe.unflatten_tensors(results)


@dataclass
class State:
    tensor: torch.Tensor
    labels: list


@pytest.mark.parametrize("container_type", [State, namedtuple("StateTuple", ("tensor", "labels"))])
def test_custom_containers_keep_general_reconstruction(container_type):
    value = {"state": container_type(torch.ones(2), [])}
    graph = _graph(value)
    template = graph.forward_effects_spec.output_template
    assert OutputReconstruction.prepare(graph.output_spec, template) is None
    tensor = torch.zeros(2)

    restored = graph.restore_outputs((tensor,), {})

    assert type(restored["state"]) is container_type
    assert restored["state"].tensor is tensor
    restored["state"].labels.append("changed")
    assert template["state"].labels == value["state"].labels == []


def test_shared_containers_keep_their_identity_and_isolation():
    shared = {"tensor": torch.ones(2), "labels": []}
    graph = _graph((shared, shared))
    template = graph.forward_effects_spec.output_template
    assert OutputReconstruction.prepare(graph.output_spec, template) is None
    tensor = torch.zeros(2)

    restored = graph.restore_outputs((tensor, tensor), {})

    assert restored[0] is restored[1]
    assert restored[0] is not shared
    assert restored[0]["tensor"] is tensor
    restored[0]["labels"].append("changed")
    assert restored[1]["labels"] == ["changed"]
    assert template[0]["labels"] == shared["labels"] == []


def test_graph_reconstruction_resolves_aliases_from_each_calls_arguments():
    value = {"result": torch.ones(2), "cache": {"tensor": torch.ones(2)}}
    graph = _graph(value)
    output_locator = Locator((("cache", ObjectType.DICT),), is_output=True)
    input_locator = Locator((("state", ObjectType.DICT),))
    graph.forward_effects_spec = ForwardEffectsSpec(
        graph.output_spec.without_tensors(value, (value["cache"],)),
        output_input_aliases=(OutputInputAlias(output_locator, input_locator),),
    )
    first_state, second_state = {"tensor": torch.zeros(2)}, {"tensor": torch.ones(2)}
    results = (torch.zeros(2), torch.empty(2))

    first = graph.restore_outputs(results, {"state": first_state})
    second = graph.restore_outputs(results, {"state": second_state})

    assert first["result"] is second["result"] is results[0]
    assert first["cache"] is first_state
    assert second["cache"] is second_state


def test_graph_recipe_is_rebuilt_after_serialization_and_kept_separate_between_graphs():
    first_graph = _graph({"result": (torch.ones(2),), "labels": []})
    second_graph = _graph({"result": [torch.zeros(2)]})
    first_tensor, second_tensor = torch.ones(2), torch.zeros(2)
    first = first_graph.restore_outputs((first_tensor,), {})
    first["labels"].append("changed")
    restored_graph = GraphSpec.from_dict(first_graph.to_dict())

    restored = restored_graph.restore_outputs((second_tensor,), {})
    separate = second_graph.restore_outputs((first_tensor,), {})

    assert type(first["result"]) is type(restored["result"]) is tuple
    assert type(separate["result"]) is list
    assert restored["labels"] == first_graph.forward_effects_spec.output_template["labels"] == []
    assert first["result"][0] is first_tensor
    assert restored["result"][0] is second_tensor
    assert separate["result"][0] is first_tensor
