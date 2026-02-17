# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for tensor spec."""

import pytest
import torch

from aitune.torch.module.tensor_spec import InfoLevel, TensorSpec


def test_tensor_spec_from_tensor():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert spec.name == "test"
    assert spec.shape == [1, 2, 3]
    assert spec.min_shape == [1, 2, 3]
    assert spec.max_shape == [1, 2, 3]
    assert spec.get_batch_axis_multipliers() == {}


def test_tensor_spec_hash():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2, 1), batch_size=1)
    assert hash(spec1) == hash(spec2)
    assert spec1 == spec2
    spec3 = TensorSpec.from_tensor("test123", torch.randn(4, 2, 4), batch_size=1)
    assert hash(spec1) != hash(spec3)
    assert spec1 != spec3
    spec4 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    assert hash(spec1) != hash(spec4)
    assert spec1 != spec4


def test_tensor_spec_info():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert spec.describe(InfoLevel.SHORT) == "test"
    assert spec.describe(InfoLevel.MEDIUM) == "test[1, 2, 3]"
    assert spec.describe(InfoLevel.FULL) == "test[1, 2, 3] min_shape=[1, 2, 3] max_shape=[1, 2, 3] dtype=torch.float32"
    assert str(spec) == "test"
    assert repr(spec) == "test[1, 2, 3]"


def test_tensor_spec_update_shapes_seen():
    # case 1
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2, 6), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.shape == ["batch0", 2, "dim2"]
    assert spec1.min_shape == [1, 2, 3]
    assert spec1.max_shape == [3, 2, 6]
    assert spec1.get_batch_axis_multipliers() == {0: 1}

    # case 2
    spec3 = TensorSpec.from_tensor("test", torch.randn(4, 2, 4), batch_size=1)
    spec1.update_shapes_seen(spec3)
    assert spec1.shape == ["dim0", 2, "dim2"]
    assert spec1.min_shape == [1, 2, 3]
    assert spec1.max_shape == [4, 2, 6]
    assert spec1.get_batch_axis_multipliers() == {}

    # case 3
    spec4 = TensorSpec.from_tensor("test", torch.randn(5, 4, 10), batch_size=5)
    spec2.update_shapes_seen(spec4)
    assert spec2.shape == ["batch0", "dim1", "batch2"]
    assert spec2.min_shape == [3, 2, 6]
    assert spec2.max_shape == [5, 4, 10]
    assert spec2.get_batch_axis_multipliers() == {0: 1, 2: 2}
    # case 4
    spec5 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    with pytest.raises(ValueError):
        spec1.update_shapes_seen(spec5)


def test_tensor_spec_to_dict_from_dict():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    wrong_dict = spec.to_dict()
    assert wrong_dict["type"] == "TensorSpec"
    assert wrong_dict["name"] == "test"
    assert wrong_dict["shape"] == [1, 2, 3]
    assert wrong_dict["min_shape"] == [1, 2, 3]
    assert wrong_dict["max_shape"] == [1, 2, 3]
    spec_from_dict = TensorSpec.from_dict(wrong_dict)
    assert spec == spec_from_dict

    # wrong type
    wrong_dict = {"a": 1}
    with pytest.raises(ValueError):
        TensorSpec.from_dict(wrong_dict)


def test_has_batch_axis():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.has_batch_axis()


def test_has_dynamic_axis():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(2, 2), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.has_dynamic_axis()


def test_matches():
    """Test the matches function with various scenarios."""
    # Test case 1: Same shapes with integer dimensions
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 2: Same shapes with symbolic dimensions (dim1)
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2.shape[1] = "dim1"  # Replace second dimension with symbolic
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 3: Same shapes with symbolic dimensions (batch1)
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2.shape[1] = "batch1"  # Replace second dimension with batch symbolic
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 4: Different integer dimensions - should not match
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 3), batch_size=1)
    assert not spec1.matches(spec2)
    assert not spec2.matches(spec1)

    # Test case 5: Different ranks - should not match
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert not spec1.matches(spec2)
    assert not spec2.matches(spec1)

    # Test case 6: Scalars i.e. empty shapes
    spec1 = TensorSpec.from_tensor("test", torch.randn(1), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1), batch_size=1)
    spec1.shape = []
    spec2.shape = []
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)
