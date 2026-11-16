# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sample metadata."""

from dataclasses import dataclass
from statistics import median
from timeit import repeat
from typing import Any

import pytest
import torch

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.module.locator import Locator, ObjectType
from aitune.torch.module.sample_metadata import SampleMetadata, batch_tensor
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec


def make_timing_sample(structure, tensor_count, tensor_elements):
    tensors = tuple(torch.empty(tensor_elements, device="cpu") for _ in range(tensor_count))
    if structure == "tensor":
        return tensors[0]
    if structure == "tuple":
        return tensors
    if structure == "cache":
        return {"logits": tensors[0], "cache": tuple(zip(tensors[1::2], tensors[2::2], strict=True))}

    value: dict[str, Any] = {f"tensor_{index}": tensor for index, tensor in enumerate(tensors)}
    if structure == "nested_dict":
        for _ in range(3):
            value = {"nested": value}
    elif structure == "metadata":
        value["metadata"] = [{"tokens": list(range(32)), "label": "decode"} for _ in range(32)]
    else:
        assert structure == "dict"
    return value


@pytest.fixture
def simple_sample():
    return {
        "values": [1, "abc", torch.randn(1)],
        "t": torch.randn(4, 4),
        "other": "other",
    }


@pytest.mark.parametrize(
    "strict,expected",
    [
        pytest.param(False, "Tensors: values[2], t", id="non-strict"),
        pytest.param(
            True,
            "Tensors: values[2], t Others: values[0]=1, values[1]=abc, other=other",
            id="strict",
        ),
    ],
)
def test_from_inputs(simple_sample, strict, expected):
    res = SampleMetadata.from_inputs(simple_sample, strict=strict)
    assert res.describe(InfoLevel.SHORT) == expected


@pytest.mark.parametrize(
    "strict,expected",
    [
        pytest.param(False, 'Tensors: output["values"][2], output["t"]', id="non-strict"),
        pytest.param(
            True,
            'Tensors: output["values"][2], output["t"] Others: output["values"][0]=1, '
            'output["values"][1]=abc, output["other"]=other',
            id="strict",
        ),
    ],
)
def test_from_outputs(simple_sample, strict, expected):
    res = SampleMetadata.from_outputs(simple_sample, strict=strict)
    assert res.describe(InfoLevel.SHORT) == expected


def test_describe():
    @dataclass
    class TestClass:
        t: torch.Tensor
        other: str

    inputs = {
        "values": [
            "first",
            torch.randn(1),
            (torch.randn(2), torch.randn(3)),
            {"t": torch.randn(4)},
            TestClass(t=torch.randn(5), other="abc"),
        ],
        "t1": torch.randn(1, 1),
        "t2": [torch.randn(2, 2), torch.randn(3, 3)],
        "t3": TestClass(t=torch.randn(4, 4), other="xyz"),
        "last": "other",
    }

    res = SampleMetadata.from_inputs(inputs, strict=True)
    assert (
        res.describe(InfoLevel.SHORT)
        == 'Tensors: values[1], values[2][0], values[2][1], values[3]["t"], values[4].t, t1, t2[0], t2[1], '
        "t3.t Others: values[0]=first, values[4].other=abc, t3.other=xyz, last=other"
    )

    assert res.describe(InfoLevel.MEDIUM)  # check only string is not empty
    assert res.describe(InfoLevel.FULL)  # check only string is not empty


@pytest.mark.parametrize("strict", [True, False])
def test_to_from_dict(simple_sample, strict):
    metadata = SampleMetadata.from_inputs(simple_sample, strict=strict)
    result = SampleMetadata.from_dict(metadata.to_dict())
    assert metadata == result

    metadata = SampleMetadata.from_outputs(simple_sample, strict=strict)
    result = SampleMetadata.from_dict(metadata.to_dict())
    assert metadata == result
    assert str(metadata) == str(result)


def test_flatten_and_unflatten_tensors_preserve_structure():
    first = torch.ones(2)
    second = torch.ones(3)
    value = {"first": first, "nested": ["static", second]}
    metadata = SampleMetadata.from_inputs(value)

    flattened = metadata.flatten_tensors(value)
    replacements = (torch.zeros(2), torch.zeros(3))
    result = metadata.unflatten_tensors(replacements, value)

    assert flattened[0] is first
    assert flattened[1] is second
    assert result["nested"][0] == "static"
    assert result["first"] is replacements[0]
    assert result["nested"][1] is replacements[1]
    assert value["first"] is first
    assert value["nested"][0] == "static"
    assert value["nested"][1] is second


def test_unflatten_tensors_isolates_mutable_non_tensor_branches():
    value = {"tensor": torch.ones(2), "metadata": {"items": []}}
    metadata = SampleMetadata.from_inputs(value)

    result = metadata.unflatten_tensors(torch.zeros(2), value)
    result["metadata"]["items"].append("updated")

    assert value["metadata"] == {"items": []}


def test_tensor_free_template_does_not_copy_tensor_subclasses():
    class NonDeepcopyTensor(torch.Tensor):
        def __deepcopy__(self, memo):
            raise AssertionError("tensor must not be deep-copied")

    value = {
        "tensor": torch.ones(2).as_subclass(NonDeepcopyTensor),
        "metadata": {"items": []},
    }
    metadata = SampleMetadata.from_outputs(value)

    result = metadata.without_tensors(value)

    assert result == {"tensor": None, "metadata": {"items": []}}
    assert result["metadata"] is not value["metadata"]


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
def test_flatten_unflatten_timing(structure, tensor_count, preserve_cache, tensor_elements, record_property):
    """Report CPU microseconds per call, without machine-dependent performance assertions.

    Run with ``pytest -s -k test_flatten_unflatten_timing`` on the same idle machine for comparisons.
    Setup, allocation, metadata discovery, and correctness checks are outside the timed regions.
    Like timeit's default, cyclic garbage collection is disabled during timing. These measurements
    exclude model execution and GPU scheduling, so they do not establish end-to-end inference impact.
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
    template = metadata.without_tensors(value, preserved_containers)

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

    timing_globals = {
        "metadata": metadata,
        "value": value,
        "tensors": tensors,
        "template": template,
        "preserved_values": preserved_values,
    }
    operations = {
        "flatten": "metadata.flatten_tensors(value)",
        "unflatten": "metadata.unflatten_tensors(tensors, template, preserved_values)",
    }
    iterations = 100
    repeats = 5
    for operation, statement in operations.items():
        # Warm each path before collecting repeated batches of independent calls.
        repeat(statement, globals=timing_globals, number=20, repeat=1)
        samples_us = [
            elapsed * 1e6 / iterations
            for elapsed in repeat(statement, globals=timing_globals, number=iterations, repeat=repeats)
        ]
        record_property(f"{operation}_median_us", median(samples_us))
        record_property(f"{operation}_samples_us", samples_us)
        print(
            f"{structure}, tensors={tensor_count}, elements={tensor_elements}, preserved_cache={preserve_cache}: "
            f"{operation} median={median(samples_us):.3f} us/call, min={min(samples_us):.3f}, "
            f"max={max(samples_us):.3f} ({repeats} batches of {iterations})"
        )

    assert all(
        actual is original for actual, original in zip(metadata.flatten_tensors(value), original_tensors, strict=True)
    )
    assert template == metadata.without_tensors(value, preserved_containers)


def test_make_batch():
    # the following tensors should not be modified
    constant_tensor = torch.ones(2, 2)
    constant_tensor_scalar = torch.tensor(3.14)

    # given input data, where 3rd axis is batch axis with multipliers 1, 2, 3
    inputs_bs2 = {
        "values": (
            torch.randn(1, 2, 2),  # tensor, multiplier 1
            constant_tensor,
            constant_tensor_scalar,
            [torch.randn(4, 5, 4)],  # tensor list, multiplier 2
            {"t": torch.randn(7, 8, 6)},  # tensor dict, multiplier 3
        ),  # this is on purpose a tuple
        "t": torch.randn(2),
        "z": "abc",
    }

    inputs_bs3 = {
        "values": [
            torch.randn(1, 2, 3),  # tensor, multiplier 1
            constant_tensor,
            constant_tensor_scalar,
            [torch.randn(4, 5, 6)],  # tensor list, multiplier 2
            {"t": torch.randn(7, 8, 9)},  # tensor dict, multiplier 3
        ],  # this is on purpose a list
        "t": torch.randn(3),
        "z": "abc",
    }

    with global_context:
        global_context.set(BATCH_SIZE_KEY, 3)
        metadata1 = SampleMetadata.from_inputs(inputs_bs3)
        global_context.set(BATCH_SIZE_KEY, 2)
        metadata2 = SampleMetadata.from_inputs(inputs_bs2)
        metadata1.update_shapes_seen(metadata2)

    # test making batch either from first or second sample
    for inputs in [inputs_bs3, inputs_bs2]:
        result = metadata1.make_batch(inputs, batch_size=10)
        values = result["values"]
        assert values[0].shape == (1, 2, 10)
        assert values[1].shape == constant_tensor.shape
        assert values[2].shape == constant_tensor_scalar.shape
        assert values[3][0].shape == (4, 5, 20)
        assert values[4]["t"].shape == (7, 8, 30)
        assert result["t"].shape == (10,)
        assert result["z"] == "abc"

        # note that input sample should not  be modified
        assert inputs_bs2["values"][0].shape == (1, 2, 2)
        assert inputs_bs3["values"][0].shape == (1, 2, 3)
        assert inputs_bs2["t"].shape == (2,)
        assert inputs_bs3["t"].shape == (3,)


@pytest.mark.parametrize(
    "from_sample", [SampleMetadata.from_inputs, SampleMetadata.from_outputs], ids=["inputs", "outputs"]
)
@pytest.mark.parametrize("max_batch_size", [1, 8])
def test_update_max_batch_size_extends_only_known_batch_bounds(from_sample, max_batch_size):
    metadata = from_sample({"scaled": torch.zeros(2, 7), "static": torch.zeros(5)}, batch_size=1)
    metadata.update_shapes_seen(from_sample({"scaled": torch.zeros(4, 11), "static": torch.zeros(5)}, batch_size=2))

    metadata.update_max_batch_size(max_batch_size)

    scaled, static = metadata.tensor_specs
    assert scaled.shape == ["batch0", "dim1"]
    assert scaled.min_shape == [2, 7]
    assert scaled.max_shape == [2 * max(2, max_batch_size), 11]
    assert scaled.get_batch_axis_multipliers() == {0: 2}
    assert static.min_shape == static.max_shape == [5]


def test_update_shapes_seen_matches_tensors_by_path():
    metadata = SampleMetadata.from_inputs({"x": torch.randn(1, 2), "y": torch.randn(3)})
    other = SampleMetadata.from_inputs({"y": torch.randn(5), "x": torch.randn(4, 2)})

    metadata.update_shapes_seen(other)

    tensor_specs = {locator.path: tensor_spec for locator, tensor_spec in metadata.tensor_data}
    assert tensor_specs["x"].min_shape == [1, 2]
    assert tensor_specs["x"].max_shape == [4, 2]
    assert tensor_specs["y"].min_shape == [3]
    assert tensor_specs["y"].max_shape == [5]


def test_update_shapes_seen_rank_error_includes_metadata():
    metadata = SampleMetadata.from_inputs({"inputs": {"tokens": torch.randn(2, 3)}})
    different_rank = SampleMetadata.from_inputs({"inputs": {"tokens": torch.randn(2)}})

    with pytest.raises(ValueError) as exc_info:
        metadata.update_shapes_seen(different_rank)

    message = str(exc_info.value)
    assert metadata.describe(InfoLevel.FULL) in message
    assert different_rank.describe(InfoLevel.FULL) in message


def test_hash_non_strict():
    # same tensor ranks
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "t": torch.randn(2, 2)})
    assert hash(m1) == hash(m2)

    # tensor rank same, different shapes
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs({"x": torch.randn(2, 2), "t": torch.randn(2, 2)})
    assert hash(m1) == hash(m2)

    # different rank
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1), "t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "t": torch.randn(2)})
    assert hash(m1) != hash(m2)

    # different parameter name
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "other": torch.randn(2, 2)})
    assert hash(m1) != hash(m2)


def test_hash_strict():
    # same tensor ranks
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "label": "abc", "t": torch.randn(2, 2)}, strict=True)
    m2 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "label": "abc", "t": torch.randn(2, 2)}, strict=True)
    assert hash(m1) == hash(m2)

    # different parameter value
    m1 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "label": "abc", "t": torch.randn(2, 2)}, strict=True)
    m2 = SampleMetadata.from_inputs({"x": torch.randn(1, 1), "label": "xyz", "t": torch.randn(2, 2)}, strict=True)
    assert hash(m1) != hash(m2)

    # different option value
    m1 = SampleMetadata.from_inputs(
        {"x": torch.randn(1, 1), "label": "a", "t": torch.randn(2, 2), "option": 1}, strict=True
    )
    m2 = SampleMetadata.from_inputs(
        {"x": torch.randn(1, 1), "label": "a", "t": torch.randn(2, 2), "option": 2}, strict=True
    )
    assert hash(m1) != hash(m2)


def test_hash_strict_with_unhashable_inputs_is_a_valid_dict_key():
    metadata = SampleMetadata.from_inputs({"options": {"fast", "stable"}}, strict=True)
    same_metadata = SampleMetadata.from_inputs({"options": {"stable", "fast"}}, strict=True)
    different_metadata = SampleMetadata.from_inputs({"options": {"fast"}}, strict=True)

    cache = {metadata: "matching", different_metadata: "different"}

    assert cache[same_metadata] == "matching"
    assert cache[different_metadata] == "different"


def test_hash_llm_dynamic():
    m1 = SampleMetadata.from_inputs({"hidden_states": "a", "cache_position": torch.randn(5)}, strict=True)
    m2 = SampleMetadata.from_inputs({"hidden_states": "b", "cache_position": torch.randn(5)}, strict=True)
    m3 = SampleMetadata.from_inputs({"hidden_states": "b", "cache_position": torch.randn(6)}, strict=True)
    assert hash(m1) == hash(m2), "other inputs should be ignored"
    assert hash(m2) == hash(m3), "cache position should not affect hash"

    m1 = SampleMetadata.from_inputs({"hidden_states": torch.randn(1), "cache_position": torch.randn(5)})
    m2 = SampleMetadata.from_inputs({"hidden_states": torch.randn(1, 2), "cache_position": torch.randn(5)})
    assert hash(m1) == hash(m2), "other tensor inputs should be ignored"


def test_hash_llm_static():
    m1 = SampleMetadata.from_inputs({"hidden_states": "a", "cache_position": torch.tensor([0])}, strict=True)
    m2 = SampleMetadata.from_inputs({"hidden_states": "b", "cache_position": torch.tensor([0])}, strict=True)
    assert hash(m1) == hash(m2), "other inputs should be ignored"

    m1 = SampleMetadata.from_inputs({"hidden_states": torch.randn(1), "cache_position": torch.tensor([0])})
    m2 = SampleMetadata.from_inputs({"hidden_states": torch.randn(1, 2), "cache_position": torch.tensor([0])})
    assert hash(m1) == hash(m2), "other tensor inputs should be ignored"

    m1 = SampleMetadata.from_inputs({"hidden_states": "a", "cache_position": torch.tensor([0])})
    m2 = SampleMetadata.from_inputs({"hidden_states": "a", "cache_position": torch.tensor([1])})
    assert hash(m1) != hash(m2), "cache position should affect hash"


def test_batch_tensor():
    # Case 1: tensor needs to be sliced to match batch size
    tensor = torch.arange(15).reshape(3, 5)
    spec = TensorSpec(
        shape=[3, 5],
        min_shape=[3, 5],
        max_shape=[3, 5],
        dtype=torch.float32,
        _bs_multipliers=[1, 5],
    )
    # Mark axis 0 as batch axis with multiplier 1
    spec.shape[0] = "batch0"
    batched = batch_tensor(tensor, spec, batch_size=2)
    assert batched.shape == (2, 5)
    assert torch.equal(batched, tensor[:2, :])

    # Case 2: tensor needs to be repeated to match batch size
    tensor = torch.arange(15).reshape(3, 5)
    spec = TensorSpec(
        shape=["batch0", 5],
        min_shape=[3, 5],
        max_shape=[3, 5],
        dtype=torch.float32,
        _bs_multipliers=[1, 5],
    )
    batched = batch_tensor(tensor, spec, batch_size=5)
    assert batched.shape == (5, 5)
    assert torch.equal(batched, tensor[:1, :].repeat(5, 1))

    # Case 3: multiple batch axes
    tensor = torch.arange(24).reshape(2, 3, 4)
    spec = TensorSpec(
        shape=["batch0", 3, "batch2"],
        min_shape=[2, 3, 4],
        max_shape=[2, 3, 4],
        dtype=torch.float32,
        _bs_multipliers=[1, 3, 2],
    )
    batched = batch_tensor(tensor, spec, batch_size=3)
    # Should slice axis 0 to 1, axis 2 to 2
    assert batched.shape == (3, 3, 6)
    assert torch.equal(batched, tensor[:1, :, :1].repeat(3, 1, 6))

    # Case 4: tensor does not have batch axis
    tensor = torch.arange(15).reshape(3, 5)
    spec = TensorSpec(
        shape=[3, 5],
        min_shape=[3, 5],
        max_shape=[3, 5],
        dtype=torch.float32,
        _bs_multipliers=[float("nan"), 1.5],
    )
    batched = batch_tensor(tensor, spec, batch_size=2)
    assert batched.shape == (3, 5)
    assert torch.equal(batched, tensor)


def test_input_identity_uses_parameter_paths():
    inputs = {
        "hidden_states": torch.randn(2, 2),
        "inputs": {"tokens": torch.randn(3, 3)},
    }

    metadata = SampleMetadata.from_inputs(inputs)

    assert [locator.path for locator, _ in metadata.tensor_data] == [
        "hidden_states",
        ("inputs", "tokens"),
    ]


def test_input_identity_does_not_depend_on_dictionary_order():
    x = torch.randn(2, 2)
    y = torch.randn(3, 3)

    first = SampleMetadata.from_inputs({"x": x, "y": y})
    second = SampleMetadata.from_inputs({"y": y, "x": x})

    assert first == second
    assert hash(first) == hash(second)


def test_detected_dynamic_axis():
    # Test case 1: No dynamic or batch axes
    tensor1 = torch.randn(2, 3)
    metadata1 = SampleMetadata.from_outputs(tensor1)
    assert not metadata1.detected_dynamic_axis()

    # Test case 2: Has dynamic axis
    tensor2 = torch.randn(2, 3)
    metadata2 = SampleMetadata.from_outputs(tensor2)
    # Manually modify tensor spec to have dynamic axis
    metadata2.tensor_specs[0].shape[1] = "dim1"
    assert metadata2.detected_dynamic_axis()

    # Test case 3: Has batch axis
    tensor3 = torch.randn(2, 3)
    metadata3 = SampleMetadata.from_outputs(tensor3)
    # Manually modify tensor spec to have batch axis
    metadata3.tensor_specs[0].shape[0] = "batch0"
    assert metadata3.detected_dynamic_axis()


def test_to_json_dict_non_strict():
    metadata = SampleMetadata.from_inputs(
        {"x": torch.randn(2, 3), "inputs": {"mask": torch.randn(4, 5)}},
        strict=False,
    )

    result = metadata.to_json_dict()

    assert result["strict"] is False
    assert result["llm_phase"] is None
    assert result["other_data"] == []
    assert len(result["tensor_data"]) == 2

    assert {entry["access_path"]: entry["semantic_path"] for entry in result["tensor_data"]} == {
        "x": "x",
        'inputs["mask"]': ["inputs", "mask"],
    }
    for entry in result["tensor_data"]:
        assert isinstance(entry["shape"], list)
        assert isinstance(entry["min_shape"], list)
        assert isinstance(entry["max_shape"], list)
        assert isinstance(entry["dtype"], str)


def test_to_json_dict_strict():
    metadata = SampleMetadata.from_inputs({"count": 1, "x": torch.randn(2, 3)}, strict=True)

    result = metadata.to_json_dict()

    assert result["strict"] is True
    assert len(result["tensor_data"]) == 1
    assert result["other_data"] == [{"access_path": "count", "semantic_path": "count", "value": 1}]


def test_to_json_dict_is_json_serializable():
    import json

    metadata = SampleMetadata.from_inputs({"x": torch.randn(2, 3), "mask": torch.randn(4, 5)})

    result = metadata.to_json_dict()
    serialized = json.dumps(result)  # should not raise
    assert isinstance(serialized, str)
