# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sample metadata."""

from dataclasses import dataclass

import pytest
import torch

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.module.sample_metadata import SampleMetadata, batch_tensor
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec


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
