# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for sample metadata."""

from dataclasses import dataclass

import pytest
import torch

from aitune.global_context import BATCH_SIZE_KEY, global_context
from aitune.torch.module.sample_metadata import SampleMetadata, batch_tensor
from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec


@pytest.fixture
def sample():
    return (
        1,
        3.14,
        True,
        b"bytes",
        "string",
        torch.randn(1, 1),
        [torch.randn(2, 2), torch.randn(3, 3)],
        {"t": torch.randn(4, 4)},
    )


@pytest.fixture
def simple_sample():
    args = [1, "abc", torch.randn(1)]
    kwargs = {"t": torch.randn(4, 4), "xyz": "other"}
    return args, kwargs


@pytest.mark.parametrize(
    "strict,expected",
    [
        pytest.param(False, "Tensors: args_2, kwargs_t", id="non-strict"),
        pytest.param(True, "Tensors: args_2, kwargs_t Others: args_0=1, args_1=abc, kwargs_xyz=other", id="strict"),
    ],
)
def test_from_inputs(simple_sample, strict, expected):
    args, kwargs = simple_sample
    res = SampleMetadata.from_inputs(args, kwargs, strict=strict)
    assert res.describe(InfoLevel.SHORT) == expected


@pytest.mark.parametrize(
    "strict,expected",
    [
        pytest.param(False, "Tensors: outputs_0_2, outputs_1_t", id="non-strict"),
        pytest.param(
            True,
            "Tensors: outputs_0_2, outputs_1_t Others: outputs_0_0=1, outputs_0_1=abc, outputs_1_xyz=other",
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

    args = [
        "first",
        torch.randn(1),
        (torch.randn(2), torch.randn(3)),
        {"t": torch.randn(4)},
        TestClass(t=torch.randn(5), other="abc"),
    ]
    kwargs = {
        "t1": torch.randn(1, 1),
        "t2": [torch.randn(2, 2), torch.randn(3, 3)],
        "t3": TestClass(t=torch.randn(4, 4), other="xyz"),
        "last": "other",
    }

    res = SampleMetadata.from_inputs(args, kwargs, strict=True)
    assert (
        res.describe(InfoLevel.SHORT)
        == "Tensors: args_1, args_2_0, args_2_1, args_3_t, args_4.t, kwargs_t1, kwargs_t2_0, kwargs_t2_1, kwargs_t3.t "
        "Others: args_0=first, args_4.other=abc, kwargs_last=other, kwargs_t3.other=xyz"
    )

    assert res.describe(InfoLevel.MEDIUM)  # check only string is not empty
    assert res.describe(InfoLevel.FULL)  # check only string is not empty


@pytest.mark.parametrize("strict", [True, False])
def test_to_from_dict(simple_sample, strict):
    args, kwargs = simple_sample
    metadata = SampleMetadata.from_inputs(args, kwargs, strict=strict)
    result = SampleMetadata.from_dict(metadata.to_dict())
    assert metadata == result

    metadata = SampleMetadata.from_outputs(simple_sample, strict=strict)
    result = SampleMetadata.from_dict(metadata.to_dict())
    assert metadata == result


def test_make_batch():
    # the following tensors should not be modified
    constant_tensor = torch.ones(2, 2)
    constant_tensor_scalar = torch.tensor(3.14)

    # given input data, where 3rd axis is batch axis with multipliers 1, 2, 3
    inputs_bs2_args = (
        torch.randn(1, 2, 2),  # tensor, multiplier 1
        constant_tensor,
        constant_tensor_scalar,
        [torch.randn(4, 5, 4)],  # tensor list, multiplier 2
        {"t": torch.randn(7, 8, 6)},  # tensor dict, multiplier 3
    )  # this is on purpose a tuple
    inputs_bs2_kwargs = {
        "t": torch.randn(2),
        "z": "abc",
    }

    inputs_bs3_args = [
        torch.randn(1, 2, 3),  # tensor, multiplier 1
        constant_tensor,
        constant_tensor_scalar,
        [torch.randn(4, 5, 6)],  # tensor list, multiplier 2
        {"t": torch.randn(7, 8, 9)},  # tensor dict, multiplier 3
    ]  # this is on purpose a list

    inputs_bs3_kwargs = {
        "t": torch.randn(3),
        "z": "abc",
    }

    with global_context:
        global_context.set(BATCH_SIZE_KEY, 3)
        metadata1 = SampleMetadata.from_inputs(inputs_bs3_args, inputs_bs3_kwargs)
        global_context.set(BATCH_SIZE_KEY, 2)
        metadata2 = SampleMetadata.from_inputs(inputs_bs2_args, inputs_bs2_kwargs)
        metadata1.update_shapes_seen(metadata2)

    # test making batch either from first or second sample
    for args, kwargs in [(inputs_bs3_args, inputs_bs3_kwargs), (inputs_bs2_args, inputs_bs2_kwargs)]:
        res_args, res_kwargs = metadata1.make_batch(args, kwargs, batch_size=10)
        assert res_args[0].shape == (1, 2, 10)
        assert res_args[1].shape == constant_tensor.shape
        assert res_args[2].shape == constant_tensor_scalar.shape
        assert res_args[3][0].shape == (4, 5, 20)
        assert res_args[4]["t"].shape == (7, 8, 30)
        assert res_kwargs["t"].shape == (10,)
        assert res_kwargs["z"] == "abc"

        # note that input sample should not  be modified
        assert inputs_bs2_args[0].shape == (1, 2, 2)
        assert inputs_bs3_args[0].shape == (1, 2, 3)
        assert inputs_bs2_kwargs["t"].shape == (2,)
        assert inputs_bs3_kwargs["t"].shape == (3,)


def test_hash_non_strict():
    # same tensor ranks
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1),), kwargs={"t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1),), kwargs={"t": torch.randn(2, 2)})
    assert hash(m1) == hash(m2)

    # tensor rank same, different shapes
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1),), kwargs={"t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs(args=(torch.randn(2, 2),), kwargs={"t": torch.randn(2, 2)})
    assert hash(m1) == hash(m2)

    # different rank
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1),), kwargs={"t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1),), kwargs={"t": torch.randn(2)})
    assert hash(m1) != hash(m2)

    # different kwarg name
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1),), kwargs={"t": torch.randn(2, 2)})
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1)), kwargs={"other": torch.randn(2, 2)})
    assert hash(m1) != hash(m2)


def test_hash_strict():
    # same tensor ranks
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "abc"), kwargs={"t": torch.randn(2, 2)}, strict=True)
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "abc"), kwargs={"t": torch.randn(2, 2)}, strict=True)
    assert hash(m1) == hash(m2)

    # different other data in args
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "abc"), kwargs={"t": torch.randn(2, 2)}, strict=True)
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "xyz"), kwargs={"t": torch.randn(2, 2)}, strict=True)
    assert hash(m1) != hash(m2)

    # different other data in wargs
    m1 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "a"), kwargs={"t": torch.randn(2, 2), "x": 1}, strict=True)
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 1), "a"), kwargs={"t": torch.randn(2, 2), "x": 2}, strict=True)
    assert hash(m1) != hash(m2)


def test_hash_llm_dynamic():
    m1 = SampleMetadata.from_inputs(args=("a",), kwargs={"cache_position": torch.randn(5)}, strict=True)
    m2 = SampleMetadata.from_inputs(args=("b",), kwargs={"cache_position": torch.randn(5)}, strict=True)
    m3 = SampleMetadata.from_inputs(args=("b",), kwargs={"cache_position": torch.randn(6)}, strict=True)
    assert hash(m1) == hash(m2), "args should be ignored"
    assert hash(m2) == hash(m3), "cache position should not affect hash"

    m1 = SampleMetadata.from_inputs(args=(torch.randn(1),), kwargs={"cache_position": torch.randn(5)}, strict=False)
    m2 = SampleMetadata.from_inputs(args=(torch.randn(1, 2),), kwargs={"cache_position": torch.randn(5)}, strict=False)
    assert hash(m1) == hash(m2), "tensor args should be ignored"


def test_hash_llm_static():
    m1 = SampleMetadata.from_inputs(args=("a",), kwargs={"cache_position": torch.tensor([0])}, strict=True)
    m2 = SampleMetadata.from_inputs(args=("b",), kwargs={"cache_position": torch.tensor([0])}, strict=True)
    assert hash(m1) == hash(m2), "args should be ignored"

    m1 = SampleMetadata.from_inputs(args=(torch.randn(1),), kwargs={"cache_position": torch.tensor([0])}, strict=False)
    m2 = SampleMetadata.from_inputs(
        args=(torch.randn(1, 2),), kwargs={"cache_position": torch.tensor([0])}, strict=False
    )
    assert hash(m1) == hash(m2), "tensor args should be ignored"

    m1 = SampleMetadata.from_inputs(args=("a",), kwargs={"cache_position": torch.tensor([0])})
    m2 = SampleMetadata.from_inputs(args=("a",), kwargs={"cache_position": torch.tensor([1])})
    assert hash(m1) != hash(m2), "cache position should affect hash"


def test_batch_tensor():
    # Case 1: tensor needs to be sliced to match batch size
    tensor = torch.arange(15).reshape(3, 5)
    spec = TensorSpec(
        name="x",
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
        name="x",
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
        name="x",
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
        name="x",
        shape=[3, 5],
        min_shape=[3, 5],
        max_shape=[3, 5],
        dtype=torch.float32,
        _bs_multipliers=[float("nan"), 1.5],
    )
    batched = batch_tensor(tensor, spec, batch_size=2)
    assert batched.shape == (3, 5)
    assert torch.equal(batched, tensor)


def test_get_names_mapping():
    """Test get_names_mapping method with various input types.

    This test verifies that get_names_mapping correctly extracts tensor names
    from different metadata structures including:
    1. Simple tensor
    2. Sequence of tensors
    3. Dictionary with tensors
    4. Mixed structure with tensors and primitives
    5. Nested structures
    #"""
    # Test case 1: Single tensor
    tensor = torch.randn(2, 2)
    metadata = SampleMetadata.from_inputs(args=(tensor,), kwargs={}, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["args_0"]
    assert kwargs_mapping == {}

    # Test case 2: Sequence of tensors
    tensors = [torch.randn(2, 2), torch.randn(3, 3)]
    metadata = SampleMetadata.from_inputs(args=(tensors,), kwargs={}, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["args_0_0", "args_0_1"]
    assert kwargs_mapping == {}

    # Test case 3: Dictionary with tensors
    tensor_dict = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
    metadata = SampleMetadata.from_inputs(args=(), kwargs=tensor_dict, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {"a": ["kwargs_a"], "b": ["kwargs_b"]}

    # Test case 4: Mixed structure with tensors and primitives
    args = (1, torch.randn(2, 2), "str")
    kwargs = {"t": torch.randn(3, 3)}
    metadata = SampleMetadata.from_inputs(args=args, kwargs=kwargs, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["args_1"]
    assert kwargs_mapping == {"t": ["kwargs_t"]}

    # Test case 5: Nested structure
    nested = {
        "a": [torch.randn(2, 2), torch.randn(3, 3)],
        "b": {"c": torch.randn(4, 4), "d": torch.randn(5, 5)},
        "e": torch.randn(6, 6),
    }
    metadata = SampleMetadata.from_inputs(args=(), kwargs=nested, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {"a": ["kwargs_a_0", "kwargs_a_1"], "b": ["kwargs_b_c", "kwargs_b_d"], "e": ["kwargs_e"]}

    # Test case 6: Empty structure
    metadata = SampleMetadata.from_inputs(args=(), kwargs={}, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {}

    # Test case 7: Structure with only primitives
    primitives = (1, 2.0, "str", True)
    metadata = SampleMetadata.from_inputs(args=primitives, kwargs={}, strict=True)
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {}


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
