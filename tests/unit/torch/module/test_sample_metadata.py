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


def test_from_sample_sequence_with_prefix(sample):
    res = SampleMetadata.from_sample(sample, prefix="test")
    assert (
        str(res._metadata)
        == "(1, 3.14, True, b'bytes', 'string', test__0[1, 1], [test__1[2, 2], test__2[3, 3]], {'t': test__3[4, 4]})"
    )


def test_from_sample_dict_with_prefix(sample):
    keys = list(range(len(sample)))
    sample = dict(zip(keys, sample, strict=False))
    res = SampleMetadata.from_sample(sample, prefix="test")
    assert (
        str(res._metadata)
        == "{0: 1, 1: 3.14, 2: True, 3: b'bytes', 4: 'string', 5: test__0[1, 1], 6: [test__1[2, 2], test__2[3, 3]], 7: {'t': test__3[4, 4]}}"
    )


def test_from_sample_sequence_with_names(sample):
    res = SampleMetadata.from_sample(sample, names=list("abcd"))
    assert str(res._metadata) == "(1, 3.14, True, b'bytes', 'string', a[1, 1], [b[2, 2], c[3, 3]], {'t': d[4, 4]})"


def test_from_sample_dict_with_names(sample):
    keys = list(range(len(sample)))
    sample = dict(zip(keys, sample, strict=False))
    res = SampleMetadata.from_sample(sample, names=list("abcd"))
    assert (
        str(res._metadata)
        == "{0: 1, 1: 3.14, 2: True, 3: b'bytes', 4: 'string', 5: a[1, 1], 6: [b[2, 2], c[3, 3]], 7: {'t': d[4, 4]}}"
    )


def test_sample_with_arrays_tuples():
    sample = SampleMetadata.from_sample(([1, 2], (3, 4), (5,)), prefix="test")
    assert str(sample._metadata) == "([1, 2], (3, 4), (5,))"


def test_to_from_dict(sample):
    metadata = SampleMetadata.from_sample(sample, prefix="test")
    result = SampleMetadata.from_dict(metadata.to_dict())
    assert metadata == result


def test_sample_metadata_info(sample):
    metadata = SampleMetadata.from_sample(sample, prefix="t")
    assert metadata.describe(InfoLevel.SHORT) == "(1, 3.14, True, b'bytes', string, t__0, [t__1, t__2], {t: t__3})"
    assert (
        metadata.describe(InfoLevel.MEDIUM)
        == "(1, 3.14, True, b'bytes', string, t__0[1, 1], [t__1[2, 2], t__2[3, 3]], {t: t__3[4, 4]})"
    )
    assert (
        metadata.describe(InfoLevel.FULL)
        == "(1, 3.14, True, b'bytes', string, t__0[1, 1] min_shape=[1, 1] max_shape=[1, 1] dtype=torch.float32, [t__1[2, 2] min_shape=[2, 2] max_shape=[2, 2] dtype=torch.float32, t__2[3, 3] min_shape=[3, 3] max_shape=[3, 3] dtype=torch.float32], {t: t__3[4, 4] min_shape=[4, 4] max_shape=[4, 4] dtype=torch.float32})"
    )


def test_flatten_and_unflatten(sample):
    metadata = SampleMetadata.from_sample(sample, prefix="test")
    flattened = metadata.flatten_sample(sample)

    assert list(flattened.keys()) == ["test__0", "test__1", "test__2", "test__3"]

    result = metadata.unflatten_sample(flattened)
    final_metadata = SampleMetadata.from_sample(result, prefix="test")
    assert metadata == final_metadata
    assert str(metadata) == "(1, 3.14, True, b'bytes', string, test__0, [test__1, test__2], {t: test__3})"


def test_make_batch():
    constant_tensor = torch.ones(2, 2)
    constant_tensor_scalar = torch.tensor(3.14)
    # given input data, where 3rd axis is batch axis with multipliers 1, 2, 3
    input_sample_bs3 = (
        torch.randn(1, 2, 3),  # tensor, multiplier 1
        constant_tensor,
        constant_tensor_scalar,
        [torch.randn(4, 5, 6)],  # tensor list, multiplier 2
        {"t": torch.randn(7, 8, 9)},  # tensor dict, multiplier 3
    )
    input_sample_bs2 = (
        torch.randn(1, 2, 2),  # tensor, multiplier 1
        constant_tensor,
        constant_tensor_scalar,
        [torch.randn(4, 5, 4)],  # tensor list, multiplier 2
        {"t": torch.randn(7, 8, 6)},  # tensor dict, multiplier 3
    )

    with global_context:
        global_context.set(BATCH_SIZE_KEY, 3)
        metadata1 = SampleMetadata.from_sample(input_sample_bs3, prefix="t")
        global_context.set(BATCH_SIZE_KEY, 2)
        metadata2 = SampleMetadata.from_sample(input_sample_bs2, prefix="t")
        metadata1.update_shapes_seen(metadata2)

    for inputs in [input_sample_bs3, input_sample_bs2]:
        res = metadata1.make_batch(inputs, batch_size=10)
        assert res[0].shape == (1, 2, 10)
        assert res[1].shape == constant_tensor.shape
        assert res[2].shape == constant_tensor_scalar.shape
        assert res[3][0].shape == (4, 5, 20)
        assert res[4]["t"].shape == (7, 8, 30)


def test_hash(sample):
    metadata1 = SampleMetadata.from_sample(sample, prefix="test")
    metadata2 = SampleMetadata.from_sample(sample, prefix="test")
    assert hash(metadata1) == hash(metadata2)

    metadata1 = SampleMetadata.from_sample(torch.randn(1), prefix="test")
    metadata2 = SampleMetadata.from_sample(torch.randn(1), prefix="test")
    assert hash(metadata1) == hash(metadata2)

    metadata1 = SampleMetadata.from_sample(torch.randn(1), names=["a"])
    metadata2 = SampleMetadata.from_sample(torch.randn(1), names=["b"])
    assert hash(metadata1) != hash(metadata2)


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
    metadata = SampleMetadata.from_sample(tensor, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["test__0"]
    assert kwargs_mapping == {}

    # Test case 2: Sequence of tensors
    tensors = [torch.randn(2, 2), torch.randn(3, 3)]
    metadata = SampleMetadata.from_sample(tensors, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["test__0", "test__1"]
    assert kwargs_mapping == {}

    # Test case 3: Dictionary with tensors
    tensor_dict = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
    metadata = SampleMetadata.from_sample(tensor_dict, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {"a": ["test__0"], "b": ["test__1"]}

    # Test case 4: Mixed structure with tensors and primitives
    mixed = (1, torch.randn(2, 2), "str", {"t": torch.randn(3, 3)})
    metadata = SampleMetadata.from_sample(mixed, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == ["test__0"]
    assert kwargs_mapping == {"t": ["test__1"]}

    # Test case 5: Nested structure
    nested = {
        "a": [torch.randn(2, 2), torch.randn(3, 3)],
        "b": {"c": torch.randn(4, 4), "d": torch.randn(5, 5)},
        "e": torch.randn(6, 6),
    }
    metadata = SampleMetadata.from_sample(nested, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {"a": ["test__0", "test__1"], "b": ["test__2", "test__3"], "e": ["test__4"]}

    # Test case 6: Empty structure
    empty = {}
    metadata = SampleMetadata.from_sample(empty, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {}

    # Test case 7: Structure with only primitives
    primitives = (1, 2.0, "str", True)
    metadata = SampleMetadata.from_sample(primitives, prefix="test")
    args_mapping, kwargs_mapping = metadata.get_names_mapping()
    assert args_mapping == []
    assert kwargs_mapping == {}


def test_detected_dynamic_axis():
    # Test case 1: No dynamic or batch axes
    tensor1 = torch.randn(2, 3)
    metadata1 = SampleMetadata.from_sample(tensor1, prefix="test")
    assert not metadata1.detected_dynamic_axis()

    # Test case 2: Has dynamic axis
    tensor2 = torch.randn(2, 3)
    metadata2 = SampleMetadata.from_sample(tensor2, prefix="test")
    # Manually modify tensor spec to have dynamic axis
    metadata2._tensor_specs[0].shape[1] = "dim1"
    assert metadata2.detected_dynamic_axis()

    # Test case 3: Has batch axis
    tensor3 = torch.randn(2, 3)
    metadata3 = SampleMetadata.from_sample(tensor3, prefix="test")
    # Manually modify tensor spec to have batch axis
    metadata3._tensor_specs[0].shape[0] = "batch0"
    assert metadata3.detected_dynamic_axis()
