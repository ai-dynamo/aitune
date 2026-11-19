# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rules for what the call recorder instruments and what it treats as transparent."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

# Library packages, and aitune's own wrappers, whose classes' methods are not recorded as nodes; their ``forward``
# still is, through the adapters' forward wrappers.
SKIP_LIBRARY_PREFIXES = (
    "torch.",
    "aitune.torch.module.",
    "aitune.torch.jit.",
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


@functools.cache
def optimized_module_type() -> type:
    """The ``OptimizedModule`` class that ``torch.compile(module)`` returns, imported once.

    The recorders check it on every wrapped call; a function-level import there costs about 0.2 us per call, which
    adds to the times the timing pass measures.
    """
    from torch._dynamo.eval_frame import OptimizedModule

    return OptimizedModule


def is_compiled_call(obj: Any, method: str) -> bool:
    """Whether calling ``obj.<method>`` enters compiled code.

    ``torch.compile(module)``'s ``OptimizedModule`` compiles every call; ``nn.Module.compile()`` compiles the call
    itself, so only its ``forward`` counts; and a ``torch.compile``'d method counts — the last is found on the
    method's ``__wrapped__`` chain, so wrappers in between must keep ``__wrapped__`` (``functools.wraps``) for it.

    Args:
        obj: The object the method belongs to.
        method: The method's name.

    Returns:
        Whether a call of ``obj.<method>`` enters compiled code.

    Example:
        >>> class M(nn.Module):
        ...     def forward(self, x):
        ...         return self.lin(x)
        ...     def encode(self, x):
        ...         return x
        >>> is_compiled_call(torch.compile(M(), backend="eager"), "forward")
        True
        >>> compiled = M()
        >>> compiled.compile(backend="eager")
        >>> is_compiled_call(compiled, "forward"), is_compiled_call(compiled, "encode")
        (True, False)
        >>> m = M()
        >>> m.forward = torch.compile(m.forward, backend="eager")
        >>> is_compiled_call(m, "forward")
        True
        >>> is_compiled_call(M(), "forward")
        False
    """
    if isinstance(obj, optimized_module_type()):  # torch.compile(module)
        return True
    if method == "forward" and getattr(obj, "_compiled_call_impl", None) is not None:
        return True  # module.compile() compiles the call only
    return runs_compiled(getattr(obj, method, None))  # a compiled forward or other method


def runs_compiled(function: Callable[..., Any] | None) -> bool:
    """Whether ``function`` is a ``torch.compile``'d callable.

    Found by following the ``__wrapped__`` chain for ``_torchdynamo_orig_callable``, since not every wrapper
    copies the marker. False otherwise: an unmarked callable, or one marked ``torch.compiler.disable``'d, which
    runs eagerly.

    Args:
        function: The callable to check for a compile mark.

    Returns:
        Whether a call of ``function`` enters compiled code.

    Example:
        >>> def f(x):
        ...     return x
        >>> runs_compiled(f)
        False
        >>> runs_compiled(torch.compile(f, backend="eager"))
        True
        >>> runs_compiled(torch.compiler.disable(f))
        False
    """
    callable_ = function
    for _ in range(10):  # bound the __wrapped__ walk
        if callable_ is None:
            return False
        if hasattr(callable_, "_torchdynamo_orig_callable"):
            return not getattr(callable_, "_torchdynamo_disable", False)
        callable_ = getattr(callable_, "__wrapped__", None)
    return False


def is_compiled_module(module: Any, forward: Callable[..., Any] | None = None) -> bool:
    """Whether ``module`` runs compiled.

    That is ``nn.Module.compile()`` (``_compiled_call_impl``), ``torch.compile()`` (an ``OptimizedModule``), or a
    ``torch.compile``'d ``forward`` (``module.forward = torch.compile(module.forward)``). A
    ``torch.compiler.disable``'d ``forward`` is not compiled: it runs eagerly.

    ``forward`` defaults to ``module.forward``; the inspector passes the original it kept, because its wrapper
    shadows it during inspection. The check follows the ``__wrapped__`` chain, since not every wrapper copies the
    compiled callable's ``_torchdynamo_orig_callable`` marker.

    Args:
        module: The object to check.
        forward: The callable to check for a compile mark, defaults to ``module.forward``.

    Returns:
        Whether ``module`` is an ``nn.Module`` compiled in any of the three ways.

    Example:
        >>> class M(nn.Module):
        ...     def forward(self, x):
        ...         return x
        >>> is_compiled_module(M()), is_compiled_module(object())
        (False, False)
        >>> compiled = M()
        >>> compiled.compile()
        >>> is_compiled_module(compiled)
        True
        >>> is_compiled_module(torch.compile(M(), backend="eager"))
        True
        >>> m = M()
        >>> m.forward = torch.compile(m.forward, backend="eager")
        >>> is_compiled_module(m)
        True
        >>> class D(nn.Module):
        ...     @torch.compiler.disable
        ...     def forward(self, x):
        ...         return x
        >>> d = M()
        >>> d.forward = torch.compiler.disable(d.forward)
        >>> is_compiled_module(D()), is_compiled_module(d)
        (False, False)
    """
    if not isinstance(module, nn.Module):
        return False
    if isinstance(module, optimized_module_type()) or getattr(module, "_compiled_call_impl", None) is not None:
        return True
    callable_ = module.forward if forward is None else forward
    return runs_compiled(callable_)


def is_transparent_container(obj: Any) -> bool:
    """Whether ``obj`` only passes calls on, so its calls are not recorded as nodes.

    Only the exact classes ``nn.Sequential``, ``nn.ModuleList`` and ``nn.ModuleDict``, never subclasses: JIT mode does
    not patch these classes (``_DEFAULT_PATCH_EXCLUDE`` matches the exact qualified name), so skipping them gives both
    modes the same tree. Subclasses must stay: ``SentenceTransformer`` subclasses ``nn.Sequential`` and its ``forward``
    is a unit of its own.

    A container that is compiled itself with ``nn.Module.compile()`` is not transparent: compiled, it is a unit of
    its own, and the pass records and times its call as one node. The ``OptimizedModule`` that ``torch.compile``
    returns is not transparent either — it is the compiled call's node; its ``_orig_mod``'s first entry is
    suppressed by the inspector instead.

    Example:
        >>> class Encoder(nn.Sequential):
        ...     pass
        >>> is_transparent_container(nn.Sequential()), is_transparent_container(Encoder())
        (True, False)
        >>> compiled = nn.Sequential()
        >>> compiled.compile()
        >>> is_transparent_container(compiled)
        False
        >>> is_transparent_container(torch.compile(nn.Linear(1, 1), backend="eager"))
        False
    """
    return (
        type(obj) in (nn.Sequential, nn.ModuleList, nn.ModuleDict) and getattr(obj, "_compiled_call_impl", None) is None
    )


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
