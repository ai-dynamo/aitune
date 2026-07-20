# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind module inputs to a forward signature."""

import inspect
from dataclasses import dataclass, field
from typing import Any

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
        signature = inspect.signature(forward)
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
