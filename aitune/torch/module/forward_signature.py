# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind module inputs to a forward signature."""

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any

from aitune.exceptions import AITuneUserInputError

_PARAMETER_KINDS = {
    kind.name: kind
    for kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.VAR_KEYWORD,
    )
}

ForwardInputPath = str | tuple[str | int, ...]


def _is_partial(obj: Any) -> bool:
    """Return whether the object itself, rather than a transparent proxy, is a partial."""
    return issubclass(type(obj), functools.partial)


def _find_partial(obj: Any) -> functools.partial | None:
    """Return the first concrete partial in a callable's wrapper chain.

    ``inspect.signature`` normally follows ``__wrapped__``. For a wrapped partial, that loses the partial's pre-bound
    arguments and exposes the original callable's signature. An explicit ``__signature__`` remains authoritative and
    therefore stops traversal before any deeper partial is considered.

    For example, Diffusers Context Parallelism effectively installs
    ``update_wrapper(partial(context_parallel_forward, module), context_parallel_forward)``. Following
    ``__wrapped__`` reports ``(module, x, ...)`` even though ``module`` is already bound; inspecting the concrete
    partial reports the callable interface correctly as ``(x, ...)``.
    """
    seen = set()
    # Wrapper metadata is user-controlled and may contain cycles. Track identities so malformed chains cannot hang
    # signature inspection.
    while id(obj) not in seen:
        seen.add(id(obj))
        if _is_partial(obj):
            return obj
        if hasattr(obj, "__signature__"):
            # Match inspect.signature semantics: a caller-provided public signature takes precedence over unwrapping.
            return None
        try:
            obj = obj.__wrapped__
        except AttributeError:
            return None
    return None


def validate_forward_input_path(path: object) -> None:
    """Validate a forward parameter name or nested path rooted at one."""
    if isinstance(path, str):
        if path:
            return
    elif isinstance(path, tuple):
        if (
            len(path) >= 2
            and isinstance(path[0], str)
            and path[0]
            and all(isinstance(component, (str, int)) and not isinstance(component, bool) for component in path[1:])
        ):
            return
    raise AITuneUserInputError(
        f"Forward input path must be a non-empty parameter name or a nested path tuple, got {path!r}."
    )


@dataclass(frozen=True, slots=True)
class ForwardParameter:
    """Serializable description of a forward parameter."""

    name: str
    kind: str
    has_default: bool

    @staticmethod
    def from_parameter(parameter: inspect.Parameter) -> "ForwardParameter":
        """Create a description from an inspected parameter."""
        return ForwardParameter(
            name=parameter.name,
            kind=parameter.kind.name,
            has_default=parameter.default is not inspect.Parameter.empty,
        )

    def to_parameter(self) -> inspect.Parameter:
        """Create an inspect parameter suitable for call binding."""
        default = None if self.has_default else inspect.Parameter.empty
        return inspect.Parameter(self.name, _PARAMETER_KINDS[self.kind], default=default)


@dataclass(frozen=True, slots=True)
class ForwardSignature:
    """Serializable forward signature used to normalize module calls.

    An inspected signature is stored as ``ForwardParameter`` records so it can be serialized with an AITune
    checkpoint. An ``inspect.Signature`` is reconstructed from those records for argument binding.
    """

    parameters: tuple[ForwardParameter, ...]
    _signature: inspect.Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the inspect signature used for call binding."""
        object.__setattr__(
            self,
            "_signature",
            inspect.Signature([parameter.to_parameter() for parameter in self.parameters]),
        )

    @staticmethod
    def from_callable(forward: Any) -> "ForwardSignature":
        """Inspect a forward callable."""
        partial = _find_partial(forward)
        signature = (
            inspect.signature(partial, follow_wrapped=False) if partial is not None else inspect.signature(forward)
        )
        return ForwardSignature(
            tuple(ForwardParameter.from_parameter(parameter) for parameter in signature.parameters.values())
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ForwardSignature":
        """Restore a signature from a dictionary."""
        return ForwardSignature(tuple(ForwardParameter(**parameter) for parameter in data["parameters"]))

    def to_dict(self) -> dict[str, Any]:
        """Convert the signature to a serializable dictionary."""
        return {
            "parameters": [
                {"name": parameter.name, "kind": parameter.kind, "has_default": parameter.has_default}
                for parameter in self.parameters
            ]
        }

    def normalize(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments:
        """Match args and kwargs to forward parameters in a consistent layout."""
        return self._signature.bind(*args, **kwargs)
