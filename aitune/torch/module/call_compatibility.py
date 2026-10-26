# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility policies for replaying third-party module calls."""

import torch

# NeMo's @typecheck() rejects positional inputs when input types are defined, even though the wrapped forward
# signature permits them. Apply keyword replay to the Typing family regardless of the current typecheck toggle.
# Qualified base-class names avoid importing optional model libraries and also cover user-defined subclasses.
_KEYWORD_ARGUMENT_MODULE_BASES = frozenset({"nemo.core.classes.common.Typing"})


def prefers_keyword_arguments(module: torch.nn.Module) -> bool:
    """Return whether a module belongs to a family that needs keyword replay."""
    return any(
        f"{base.__module__}.{base.__qualname__}" in _KEYWORD_ARGUMENT_MODULE_BASES for base in type(module).__mro__
    )
