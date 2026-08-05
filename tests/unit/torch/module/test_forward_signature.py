# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect

import pytest
import wrapt

from aitune.exceptions import AITuneUserInputError
from aitune.torch.module.forward_signature import ForwardSignature, validate_forward_input_path


def _forward(x, y=2, *, scale=1):
    return (x + y) * scale


def _module_forward(module, x, y=2, *, scale=1):
    return module, (x + y) * scale


def _plain_forward():
    return _forward


def _bound_method():
    class Module:
        def forward(self, x, y=2, *, scale=1):
            return (x + y) * scale

    return Module().forward


def _functools_wrapped_forward():
    @functools.wraps(_forward)
    def wrapper(*args, **kwargs):
        return _forward(*args, **kwargs)

    return wrapper


def _functools_wrapped_bound_method():
    def decorator(forward):
        @functools.wraps(forward)
        def wrapper(self, *args, **kwargs):
            return forward(self, *args, **kwargs)

        return wrapper

    class Module:
        @decorator
        def forward(self, x, y=2, *, scale=1):
            return (x + y) * scale

    return Module().forward


def _partial_forward():
    return functools.partial(_module_forward, object())


def _wrapped_partial_forward():
    return functools.update_wrapper(_partial_forward(), _module_forward)


def _wrapt_forward():
    return wrapt.FunctionWrapper(
        _forward,
        lambda wrapped, _, args, kwargs: wrapped(*args, **kwargs),
    )


def _wrapt_functools_wrapped_bound_method():
    return wrapt.FunctionWrapper(
        _functools_wrapped_bound_method(),
        lambda wrapped, _, args, kwargs: wrapped(*args, **kwargs),
    )


def _wrapt_wrapped_partial_forward():
    return wrapt.FunctionWrapper(
        _wrapped_partial_forward(),
        lambda wrapped, _, args, kwargs: wrapped(*args, **kwargs),
    )


def _nested_wrapped_partial_forward():
    wrapped_partial = _wrapt_wrapped_partial_forward()

    @functools.wraps(wrapped_partial)
    def outer(*args, **kwargs):
        return wrapped_partial(*args, **kwargs)

    return outer


def _explicit_signature_forward():
    def wrapper(*args, **kwargs):
        return _forward(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(_forward)
    return wrapper


def _bound_partial_method():
    class Module:
        def forward(self, x, y=2, *, scale=1):
            return (x + y) * scale

        inference = functools.partialmethod(forward, y=2)

    return Module().inference


def _callable_object():
    class Forward:
        def __call__(self, x, y=2, *, scale=1):
            return (x + y) * scale

    return Forward()


@pytest.mark.parametrize(
    "forward_factory",
    [
        pytest.param(_plain_forward, id="plain-function"),
        pytest.param(_bound_method, id="bound-method"),
        pytest.param(_functools_wrapped_forward, id="functools-wraps"),
        pytest.param(_functools_wrapped_bound_method, id="functools-wraps-bound-method"),
        pytest.param(_partial_forward, id="partial"),
        pytest.param(_wrapped_partial_forward, id="update-wrapper-partial"),
        pytest.param(_wrapt_forward, id="wrapt-function-wrapper"),
        pytest.param(_wrapt_functools_wrapped_bound_method, id="wrapt-over-functools-wraps-bound-method"),
        pytest.param(_wrapt_wrapped_partial_forward, id="wrapt-over-wrapped-partial"),
        pytest.param(_nested_wrapped_partial_forward, id="nested-functools-wrapt-partial"),
        pytest.param(_explicit_signature_forward, id="explicit-signature"),
        pytest.param(_bound_partial_method, id="bound-partial-method"),
        pytest.param(_callable_object, id="callable-object"),
    ],
)
def test_wrapping_compatibility_matrix(forward_factory):
    signature = ForwardSignature.from_callable(forward_factory())

    assert tuple(parameter.name for parameter in signature.parameters) == ("x", "y", "scale")

    normalized = signature.normalize((1,), {"y": 2, "scale": 3})
    assert normalized.arguments == {"x": 1, "y": 2, "scale": 3}


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
