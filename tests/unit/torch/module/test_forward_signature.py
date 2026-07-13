# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aitune.torch.module.forward_signature import ForwardSignature


def test_equivalent_calls_have_same_normalized_layout():
    def forward(x, y):
        return x + y

    signature = ForwardSignature.from_callable(forward)

    expected = ((1, 2), {})
    assert signature.normalize((1, 2), {}) == expected
    assert signature.normalize((1,), {"y": 2}) == expected
    assert signature.normalize((), {"x": 1, "y": 2}) == expected
    assert signature.normalize((), {"y": 2, "x": 1}) == expected


def test_normalize_supports_all_parameter_kinds():
    def forward(positional, /, optional=None, *args, keyword=None, **kwargs):
        return positional, optional, args, keyword, kwargs

    signature = ForwardSignature.from_callable(forward)
    normalized = signature.normalize(
        (1,),
        {"optional": 2, "keyword": 3, "extra": 4},
    )

    assert normalized == ((1, 2), {"keyword": 3, "extra": 4})
    assert forward(*normalized[0], **normalized[1]) == (1, 2, (), 3, {"extra": 4})


def test_forward_signature_serialization_round_trip():
    def forward(x, optional=None, *args, keyword=None, **kwargs):
        pass

    signature = ForwardSignature.from_callable(forward)

    assert ForwardSignature.from_dict(signature.to_dict()) == signature
