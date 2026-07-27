# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for explicit dynamic shape definitions."""

import pytest
import torch

import aitune.torch as ait
from aitune.exceptions import AITuneUserInputError
from aitune.torch.dynamic_shapes import dynamic_shapes_to_json, validate_dynamic_shape_definitions


def test_public_dimension_types_and_default_opt():
    sequence = ait.DynamicDim("sequence", min=1, max=512)
    batch = ait.BatchDim("batch", min=1, opt=8, max=32)

    assert sequence.opt is None
    assert batch.opt == 8


@pytest.mark.parametrize(
    ("dimension", "expected_type"),
    [
        (ait.DynamicDim("sequence", min=1, max=512), "DynamicDim"),
        (ait.BatchDim("batch", min=1, opt=8, max=32), "BatchDim"),
    ],
)
def test_dynamic_dimension_to_dict(dimension, expected_type):
    assert dimension.to_dict() == {
        "type": expected_type,
        "name": dimension.name,
        "min": dimension.min,
        "max": dimension.max,
        "opt": dimension.opt,
    }


def test_dynamic_shapes_to_json():
    dynamic_shapes = {
        "x": (ait.BatchDim("batch", min=1, opt=2, max=4), 128),
        ("options", "mask"): (ait.DynamicDim("sequence", min=1, max=512),),
    }

    assert dynamic_shapes_to_json(dynamic_shapes) == [
        {
            "path": "x",
            "shape": [
                {"type": "BatchDim", "name": "batch", "min": 1, "max": 4, "opt": 2},
                128,
            ],
        },
        {
            "path": ["options", "mask"],
            "shape": [
                {"type": "DynamicDim", "name": "sequence", "min": 1, "max": 512, "opt": None},
            ],
        },
    ]


def test_dynamic_shapes_to_json_preserves_none():
    assert dynamic_shapes_to_json(None) is None


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"name": "", "min": 1, "max": 2}, AITuneUserInputError, "non-empty string"),
        ({"name": "sequence", "min": 0, "max": 2}, ValueError, "must be positive"),
        ({"name": "sequence", "min": 2, "max": 2}, AITuneUserInputError, "min < max"),
        (
            {"name": "sequence", "min": 1, "opt": 3, "max": 2},
            AITuneUserInputError,
            "min <= opt <= max",
        ),
    ],
)
def test_invalid_dynamic_dimensions_are_rejected(kwargs, error, message):
    with pytest.raises(error, match=message):
        ait.DynamicDim(**kwargs)


def test_validate_dynamic_shape_definitions_accepts_valid_shared_dimensions():
    validate_dynamic_shape_definitions({
        "x": (
            ait.BatchDim("batch", min=1, opt=2, max=4),
            ait.DynamicDim("sequence", min=1, opt=8, max=16),
        ),
        ("options", "mask"): (
            ait.BatchDim("batch", min=1, opt=2, max=4),
            ait.DynamicDim("sequence", min=1, opt=8, max=16),
        ),
    })


@pytest.mark.parametrize(
    ("dynamic_shapes", "error", "message"),
    [
        (None, AITuneUserInputError, "must be a dictionary"),
        ({("", 0): (1,)}, AITuneUserInputError, "Forward input path"),
        ({"x": [1]}, AITuneUserInputError, "must be a tuple"),
        ({"x": (-1,)}, ValueError, "must not be negative"),
        ({"x": (True,)}, AITuneUserInputError, "static integer or dynamic dimension"),
        (
            {
                "x": (ait.DynamicDim("shared", min=1, max=4),),
                "y": (ait.BatchDim("shared", min=1, max=4),),
            },
            AITuneUserInputError,
            "conflicting definitions",
        ),
        (
            {
                "x": (ait.DynamicDim("shared", min=1, opt=2, max=4),),
                "y": (ait.DynamicDim("shared", min=1, opt=3, max=4),),
            },
            AITuneUserInputError,
            "conflicting definitions",
        ),
    ],
)
def test_validate_dynamic_shape_definitions_rejects_invalid_input(dynamic_shapes, error, message):
    with pytest.raises(error, match=message):
        validate_dynamic_shape_definitions(dynamic_shapes)


def test_conflicting_dimension_names_are_rejected():
    with pytest.raises(AITuneUserInputError, match="conflicting definitions"):
        ait.Module(
            torch.nn.Identity(),
            dynamic_shapes={
                "x": (ait.DynamicDim("sequence", min=1, max=512),),
                "y": (ait.DynamicDim("sequence", min=1, max=256),),
            },
        )
