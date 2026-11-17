# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from aitune.torch.dataloader import InputConfig, MinMaxRandomDataset, samples_generator


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize("bounds", [{}, {"min_value": 0}, {"max_value": None}])
def test_discrete_inputs_require_explicit_upper_bound(dtype, bounds):
    with pytest.raises(ValueError, match="max_value must be specified for integer and boolean dtypes"):
        InputConfig((4,), dtype=dtype, **bounds)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_discrete_inputs_preserve_explicit_zero_one_range(dtype, include_min_max_shapes):
    dataset = MinMaxRandomDataset(
        3,
        [InputConfig((4,), (8,), dtype=dtype, min_value=0, max_value=1)],
        include_min_max_shapes=include_min_max_shapes,
    )

    for index in range(len(dataset)):
        tensor = dataset[index][0]
        assert tensor.dtype == dtype
        assert torch.count_nonzero(tensor) == 0


@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_boolean_inputs_with_explicit_upper_bound_sample_both_values(include_min_max_shapes):
    torch.manual_seed(42)
    dataset = MinMaxRandomDataset(
        3,
        [InputConfig((128,), (256,), dtype=torch.bool, max_value=2)],
        include_min_max_shapes=include_min_max_shapes,
    )

    values = torch.cat([dataset[index][0] for index in range(len(dataset))])
    assert torch.any(values)
    assert torch.any(~values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_floating_inputs_preserve_default_upper_bound(dtype, include_min_max_shapes):
    np.random.seed(21)
    torch.manual_seed(42)
    expected = MinMaxRandomDataset(
        3,
        [InputConfig((4,), (8,), dtype=dtype, max_value=1)],
        include_min_max_shapes=include_min_max_shapes,
    )
    np.random.seed(21)
    torch.manual_seed(42)
    actual = MinMaxRandomDataset(
        3,
        [InputConfig((4,), (8,), dtype=dtype)],
        include_min_max_shapes=include_min_max_shapes,
    )

    for index in range(len(actual)):
        assert torch.equal(actual[index][0], expected[index][0])


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool])
@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_integer_random_inputs(dtype, include_min_max_shapes):
    upper_bound = 2 if dtype == torch.bool else 17
    dataset = MinMaxRandomDataset(
        4,
        [InputConfig((8,), (16,), dtype=dtype, kwarg_name="input_ids", max_value=upper_bound)],
        include_min_max_shapes=include_min_max_shapes,
    )

    for _, args, kwargs in samples_generator(dataset, [2]):
        tensor = kwargs["input_ids"]
        assert not args
        assert tensor.dtype == dtype
        assert tensor.shape[0] == 2
        assert 8 <= tensor.shape[1] <= 16
        assert torch.all(tensor >= 0)
        assert torch.all(tensor < upper_bound)

    if include_min_max_shapes:
        assert dataset[0]["input_ids"].shape == (8,)
        assert dataset[1]["input_ids"].shape == (16,)


def test_integer_random_inputs_can_feed_embedding():
    dataset = MinMaxRandomDataset(
        3,
        [InputConfig((4,), (8,), dtype=torch.int64, kwarg_name="input", min_value=3, max_value=23)],
    )
    embedding = torch.nn.Embedding(23, 6)

    for _, _, kwargs in samples_generator(dataset, [2]):
        assert torch.all(kwargs["input"] >= 3)
        output = embedding(**kwargs)
        assert output.shape == (*kwargs["input"].shape, 6)


@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_random_inputs_are_seeded(include_min_max_shapes):
    config = InputConfig((4,), (8,), dtype=torch.int64, min_value=-7, max_value=11)
    np.random.seed(21)
    torch.manual_seed(42)
    first = MinMaxRandomDataset(4, [config], include_min_max_shapes=include_min_max_shapes)
    np.random.seed(21)
    torch.manual_seed(42)
    second = MinMaxRandomDataset(4, [config], include_min_max_shapes=include_min_max_shapes)

    for index in range(4):
        assert torch.equal(first[index][0], second[index][0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_floating_random_inputs_preserve_uniform_sampling(dtype):
    config = InputConfig((4,), dtype=dtype, min_value=-0.75, max_value=1.25)
    torch.manual_seed(42)
    expected = [torch.empty((4,), dtype=dtype).uniform_(-0.75, 1.25) for _ in range(3)]
    torch.manual_seed(42)
    actual = MinMaxRandomDataset(3, [config], include_min_max_shapes=False)

    for index in range(3):
        assert torch.equal(actual[index][0], expected[index])


@pytest.mark.parametrize("bounds", [(0.5, 17), (0, 17.5)])
def test_integer_random_inputs_reject_fractional_bounds(bounds):
    config = InputConfig((4,), dtype=torch.int64, min_value=bounds[0], max_value=bounds[1])
    with pytest.raises(TypeError):
        MinMaxRandomDataset(1, [config], include_min_max_shapes=False)


@pytest.mark.parametrize("include_min_max_shapes", [False, True])
def test_iterating_yields_all_samples(include_min_max_shapes):
    dataset = MinMaxRandomDataset(
        5, [InputConfig((4,), (8,), kwarg_name="input")], include_min_max_shapes=include_min_max_shapes
    )

    samples = list(dataset)

    assert len(samples) == len(dataset) == 5
    for index, sample in enumerate(samples):
        assert sample is dataset[index]


def test_assigned_sample_is_seen_by_indexing_and_iteration():
    dataset = MinMaxRandomDataset(3, [InputConfig((4,), (8,), kwarg_name="input")])
    replacement = {"input": torch.zeros(5)}

    dataset[0] = replacement

    assert dataset[0] is replacement
    assert next(iter(dataset)) is replacement
    assert dataset.samples[0] is replacement
