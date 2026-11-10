# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules for what the call recorder instruments and what it treats as transparent."""

import inspect
from typing import Any

import torch
from torch import nn

# Library packages whose classes' methods are not recorded as nodes; their ``forward`` still is, through the
# adapters' forward wrappers.
SKIP_LIBRARY_PREFIXES = (
    "torch.",
    "transformers.",
    "diffusers.",
    "accelerate.",
    "peft.",
    "timm.",
    "sentence_transformers.",
    "huggingface_hub.",
)

SKIP_METHODS = frozenset({
    "extra_repr",
    "train",
    "eval",
    "state_dict",
    "load_state_dict",
    "to",
    "cuda",
    "cpu",
    "float",
    "half",
    "named_parameters",
    "parameters",
    "buffers",
    "children",
    "modules",
    "named_modules",
    "apply",
})


def user_defined_methods(obj: Any) -> list[str]:
    """Return the names of the user-defined methods of ``obj``, without ``forward``.

    Methods defined by model code are kept, including single-underscore ones (``_forward_flow``). Methods of library
    base classes, of ``nn.Module`` and ``object``, dunder methods and ``SKIP_METHODS`` are left out.

    Args:
        obj: Module or plain object.

    Returns:
        Method names in class-hierarchy order.

    Example:
        >>> class Model(nn.Module):
        ...     def forward(self, x):
        ...         return self._step(x)
        ...     def _step(self, x):
        ...         return x
        >>> user_defined_methods(Model())
        ['_step']
    """
    base = set(dir(nn.Module)) if isinstance(obj, nn.Module) else set(dir(object))
    names: list[str] = []
    for cls in type(obj).__mro__:
        if cls in (object, nn.Module) or cls.__module__.startswith(SKIP_LIBRARY_PREFIXES):
            continue
        for name, attr in vars(cls).items():
            if name.startswith("__") or name == "forward" or name in SKIP_METHODS or name in names or name in base:
                continue
            if inspect.isfunction(attr):
                names.append(name)
    return names


def is_transparent_container(obj: Any) -> bool:
    """Whether ``obj`` only passes calls on, so its calls are not recorded as nodes.

    Only the exact classes ``nn.Sequential``, ``nn.ModuleList`` and ``nn.ModuleDict``, never subclasses: JIT mode does
    not patch these classes (``_DEFAULT_PATCH_EXCLUDE`` matches the exact qualified name), so skipping them gives both
    modes the same tree. Subclasses must stay: ``SentenceTransformer`` subclasses ``nn.Sequential`` and its ``forward``
    is a unit of its own.

    The ``OptimizedModule`` that ``torch.compile(module)`` returns is transparent as well: it only calls the original
    module (``_orig_mod``), which is recorded instead, so one call does not appear as two nested nodes.

    Example:
        >>> class Encoder(nn.Sequential):
        ...     pass
        >>> is_transparent_container(nn.Sequential()), is_transparent_container(Encoder())
        (True, False)
        >>> is_transparent_container(torch.compile(nn.Linear(1, 1), backend="eager"))
        True
    """
    if type(obj) in (nn.Sequential, nn.ModuleList, nn.ModuleDict):
        return True
    from torch._dynamo.eval_frame import OptimizedModule

    return isinstance(obj, OptimizedModule)


def is_transparent_call(args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> bool:
    """Whether a call runs a callable passed as an argument, like ``_apply_checkpoint(func, x)``.

    Recording such helpers would merge unrelated work under one node, so they are not recorded. Positional and keyword
    arguments both count.

    Example:
        >>> is_transparent_call((len, 1)), is_transparent_call((1,), {"fn": len}), is_transparent_call((nn.Linear(1, 1), 1))
        (True, True, False)
    """
    values = (*args, *(kwargs or {}).values())
    return any(callable(a) and not isinstance(a, (nn.Module, torch.Tensor, type)) for a in values)
