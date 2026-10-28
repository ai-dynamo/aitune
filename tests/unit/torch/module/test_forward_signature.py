# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Any, cast

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

    cast(Any, wrapper).__signature__ = inspect.signature(_forward)
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


@pytest.mark.parametrize("prefer_keyword_arguments", [False, True])
def test_equivalent_calls_have_same_normalized_layout(prefer_keyword_arguments):
    def forward(x, y):
        return x + y

    signature = ForwardSignature.from_callable(forward, prefer_keyword_arguments=prefer_keyword_arguments)

    for args, kwargs in (
        ((1, 2), {}),
        ((1,), {"y": 2}),
        ((), {"x": 1, "y": 2}),
        ((), {"y": 2, "x": 1}),
    ):
        normalized = signature.normalize(args, kwargs)
        assert normalized.args == (() if prefer_keyword_arguments else (1, 2))
        assert normalized.kwargs == ({"x": 1, "y": 2} if prefer_keyword_arguments else {})


@pytest.mark.parametrize("prefer_keyword_arguments", [False, True])
def test_normalize_supports_all_parameter_kinds(prefer_keyword_arguments):
    def forward(positional, /, optional=None, *args, keyword=None, **kwargs):
        return positional, optional, args, keyword, kwargs

    signature = ForwardSignature.from_callable(forward, prefer_keyword_arguments=prefer_keyword_arguments)
    normalized = signature.normalize(
        (1,),
        {"optional": 2, "keyword": 3, "extra": 4},
    )

    assert normalized.args == ((1,) if prefer_keyword_arguments else (1, 2))
    expected_kwargs = {"keyword": 3, "extra": 4}
    if prefer_keyword_arguments:
        expected_kwargs["optional"] = 2
    assert normalized.kwargs == expected_kwargs
    assert forward(*normalized.args, **normalized.kwargs) == (1, 2, (), 3, {"extra": 4})

    normalized = signature.normalize((1, 2, 5, 6), {"keyword": 3, "extra": 4})
    assert normalized.args == (1, 2, 5, 6)
    assert normalized.kwargs == {"keyword": 3, "extra": 4}
    assert forward(*normalized.args, **normalized.kwargs) == (1, 2, (5, 6), 3, {"extra": 4})


@pytest.mark.parametrize("prefer_keyword_arguments", [False, True])
def test_forward_signature_serialization_round_trip(prefer_keyword_arguments):
    def forward(x, optional=None, *args, keyword=None, **kwargs):
        pass

    signature = ForwardSignature.from_callable(forward, prefer_keyword_arguments=prefer_keyword_arguments)

    assert ForwardSignature.from_dict(signature.to_dict()) == signature


def test_normalized_call_supports_decorators_requiring_keyword_arguments():
    def typed(forward):
        @functools.wraps(forward)
        def wrapped(self, *args, **kwargs):
            if args:
                raise TypeError("All arguments must be passed by kwargs only for typed methods")
            return forward(self, **kwargs)

        return wrapped

    class Model:
        @typed
        def forward(self, input_signal, input_signal_length):
            return input_signal + input_signal_length

    model = Model()
    signature = ForwardSignature.from_callable(model.forward, prefer_keyword_arguments=True)
    normalized = signature.normalize((), {"input_signal": 2, "input_signal_length": 3})

    assert model.forward(*normalized.args, **normalized.kwargs) == 5


def test_default_signature_keeps_positional_call_layout():
    inspected = ForwardSignature.from_callable(_forward)
    for signature in (inspected, ForwardSignature(inspected.parameters)):
        assert not signature.prefer_keyword_arguments
        normalized = signature.normalize((), {"x": 1, "y": 2, "scale": 3})
        assert normalized.args == (1, 2)
        assert normalized.kwargs == {"scale": 3}


def test_legacy_checkpoint_keeps_positional_compiled_call_layout():
    signature = ForwardSignature.from_callable(_forward, prefer_keyword_arguments=True)
    legacy = signature.to_dict()
    del legacy["prefer_keyword_arguments"]

    restored = ForwardSignature.from_dict(legacy)
    normalized = restored.normalize((), {"x": 1, "y": 2, "scale": 3})

    assert normalized.args == (1, 2)
    assert normalized.kwargs == {"scale": 3}
    assert _forward(*normalized.args, **normalized.kwargs) == 9
    assert ForwardSignature.from_dict(restored.to_dict()) == restored
