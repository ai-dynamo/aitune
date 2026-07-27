# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aitune.exceptions import AITuneUserInputError
from aitune.torch.module.forward_signature import ForwardSignature, validate_forward_input_path


@pytest.mark.parametrize("path", ["input", ("input", 0), ("input", "nested")])
def test_validate_forward_input_path_accepts_valid_paths(path):
    assert validate_forward_input_path(path) is None


@pytest.mark.parametrize(
    "path",
    [None, "", ("input",), ("", 0), (0, "nested"), ("input", True), ("input", None)],
)
def test_validate_forward_input_path_rejects_invalid_paths(path):
    with pytest.raises(AITuneUserInputError, match="Forward input path"):
        validate_forward_input_path(path)


def test_equivalent_calls_have_same_normalized_layout():
    def forward(x, y):
        return x + y

    signature = ForwardSignature.from_callable(forward)

    for args, kwargs in (
        ((1, 2), {}),
        ((1,), {"y": 2}),
        ((), {"x": 1, "y": 2}),
        ((), {"y": 2, "x": 1}),
    ):
        normalized = signature.normalize(args, kwargs)
        assert normalized.args == (1, 2)
        assert normalized.kwargs == {}


def test_normalize_supports_all_parameter_kinds():
    def forward(positional, /, optional=None, *args, keyword=None, **kwargs):
        return positional, optional, args, keyword, kwargs

    signature = ForwardSignature.from_callable(forward)
    normalized = signature.normalize(
        (1,),
        {"optional": 2, "keyword": 3, "extra": 4},
    )

    assert normalized.args == (1, 2)
    assert normalized.kwargs == {"keyword": 3, "extra": 4}
    assert forward(*normalized.args, **normalized.kwargs) == (1, 2, (), 3, {"extra": 4})


def test_forward_signature_serialization_round_trip():
    def forward(x, optional=None, *args, keyword=None, **kwargs):
        pass

    signature = ForwardSignature.from_callable(forward)

    assert ForwardSignature.from_dict(signature.to_dict()) == signature
