# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch module utilities for parameter counting, device management, and memory offloading."""

import inspect
import weakref
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from aitune.torch.integrations import is_integration_distributed_module
from aitune.torch.module.locator import Locator
from aitune.torch.module.onnx_module import OnnxModule
from aitune.torch.utils.memory import cleanup_memory
from aitune.utils.monitoring import annotate

if TYPE_CHECKING:
    from aitune.torch.backend.backend import Backend


def format_num_parameters(num: int) -> str:
    """Formats a number into human-readable format with appropriate suffixes.

    Args:
        num: The number to format.

    Returns:
        A string representation of the number in human-readable format
        (e.g., "1.2B", "500M", "100K", "50").

    Examples:
        >>> format_num_parameters(1_200_000_000)
        '1.2B'
        >>> format_num_parameters(500_000)
        '500.0K'
        >>> format_num_parameters(50)
        '50'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num}"


def count_parameters(module: nn.Module) -> int:
    """Counts the total number of parameters and returns the count as an integer.

    Args:
        module: The PyTorch module to count parameters for.

    Returns:
       The total number of parameters as an integer (e.g., 1200000, 500000, 100).

    Examples:
        >>> import torch.nn as nn
        >>> module = nn.Linear(1000, 100)
        >>> count_parameters(module)
        100100
    """
    num_params = sum(p.numel() for p in module.parameters())
    return num_params


def get_forward_arguments_names(forward_func: Callable) -> tuple[list[str], list[str]]:
    """Get the forward_func signature arguments names.

    Args:
        forward_func: PyTorch module forward function
    Returns:
        Tuples of forward positional only arguments and forward keyword arguments (and positional that could be passed as kwargs)
    """
    forward_signature = inspect.signature(forward_func)
    params_list = list(forward_signature.parameters.values())

    if len(params_list) == 0:
        raise ValueError(f"Forward function '{forward_func.__name__}' has no parameters!")

    allowed_kinds = {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    forward_args = [p.name for p in params_list if p.kind == inspect.Parameter.POSITIONAL_ONLY]
    forward_kwargs = [p.name for p in params_list if p.kind in allowed_kinds]

    return forward_args, forward_kwargs


def get_module_device(module: nn.Module) -> Optional["torch.device"]:
    """Get the device of the given module.

    Args:
        module: Module to get the device of.

    Returns:
        The device of module based on parameters. If not parameters, returns None.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def is_distributed_module(module: nn.Module) -> bool:
    """Return whether a module requires distributed backend execution.

    DTensor state, modules implemented by ``torch.distributed``, and structures
    recognized by enabled integrations are detected automatically.
    """
    if isinstance(module, DistributedDataParallel):
        return True

    tensors = (*module.parameters(recurse=True), *module.buffers(recurse=True))
    if any(
        isinstance(tensor, DTensor) or type(tensor).__module__.startswith("torch.distributed") for tensor in tensors
    ):
        return True

    if any(type(child).__module__.startswith("torch.distributed") for child in module.modules()):
        return True

    return is_integration_distributed_module(module)


def move_module_to_device(module: nn.Module, device: str | torch.device) -> None:
    """Move an ordinary module while preserving distributed module placement."""
    if not is_distributed_module(module):
        module.to(device)


def move_tensors_to_device(value: Any, device: str | torch.device) -> Any:
    """Recursively move ordinary tensor leaves while preserving distributed tensors."""
    for locator, tensor in Locator.find_leaves(value, only_tensors=True):
        if isinstance(tensor, DTensor):
            continue

        value = locator.set_value(value, tensor.to(device))
    return value


# Weights that offloading must leave where they are, because something outside the offloaded module still uses or
# moves them (``ait.wrap(..., call_tree=...)``). Weak, so the set does not keep models alive.
# Keyed by ``id``: a WeakSet would compare tensors with ``==``, which is element-wise.
# Each entry has owners (the ids of the wrapped modules that keep it, ``None`` for unowned): a tensor stays kept
# while any owner keeps it, so releasing one wrapped module does not drop what another still needs.
_KEPT_ON_DEVICE: dict[int, tuple["weakref.ref[torch.Tensor]", set[int | None]]] = {}


def _owner_key(owner: object | None) -> int | None:
    # An AITune wrapper proxy stands for the module it wraps.
    return None if owner is None else id(getattr(owner, "__wrapped__", owner))


def keep_on_device(tensors: Iterable[torch.Tensor], owner: object | None = None) -> None:
    """Exclude ``tensors`` from every later ``offload``; they stay the same objects on the same device.

    Args:
        tensors: The tensors to keep.
        owner: The wrapped module that keeps them, so ``release_kept_on_device(owner)`` can drop its share; None
            keeps them until ``clear_kept_on_device``.
    """
    for tensor in tensors:
        entry = _KEPT_ON_DEVICE.get(id(tensor))
        owners = entry[1] if entry is not None and entry[0]() is tensor else set()
        owners.add(_owner_key(owner))
        _KEPT_ON_DEVICE[id(tensor)] = (weakref.ref(tensor), owners)


def is_kept_on_device(tensor: torch.Tensor, owner: object | None = None) -> bool:
    """Whether ``tensor`` is kept and still alive; with ``owner``, whether that owner keeps it."""
    entry = _KEPT_ON_DEVICE.get(id(tensor))
    if entry is None or entry[0]() is not tensor:
        return False
    return owner is None or _owner_key(owner) in entry[1]


def release_kept_on_device(owner: object) -> None:
    """Drop ``owner``'s share of every kept tensor; a tensor no other owner keeps is no longer kept."""
    key = _owner_key(owner)
    for tensor_id in list(_KEPT_ON_DEVICE):
        owners = _KEPT_ON_DEVICE[tensor_id][1]
        owners.discard(key)
        if not owners:
            del _KEPT_ON_DEVICE[tensor_id]


def clear_kept_on_device() -> None:
    """Forget the tensors registered with ``keep_on_device``."""
    _KEPT_ON_DEVICE.clear()


def offload(model: nn.Module, device: str | torch.device = "meta") -> None:
    """Offload an ordinary module while preserving distributed placement.

    Parameters and buffers registered with ``keep_on_device`` are not moved, so modules that share them keep working.

    Args:
        model: Model to offload, including any nested ONNX Runtime sessions.
        device: Device to offload to. ONNX modules use CPU for meta.
    """
    if is_distributed_module(model):
        return

    with annotate("Offloading module"):
        kept = {id(t) for t in (*model.parameters(), *model.buffers()) if is_kept_on_device(t)}
        if kept:
            # Same conversion as ``model.to(device)``, which goes through ``_apply`` too, minus the kept tensors.
            model._apply(lambda t: t if id(t) in kept else t.to(device))
        else:
            model.to(device)

        if isinstance(model, OnnxModule):
            model.offload(device)
            return

        # TODO(kn): go deeper?
        for module in model.modules():
            if isinstance(module, OnnxModule):
                module.offload(device)
        cleanup_memory()


def offload_after_tuning(
    model: nn.Module,
    backends: Iterable["Backend"],
    device: str | torch.device,
) -> None:
    """Offload a tuned ordinary module when no selected backend builds just in time."""
    from aitune.torch.backend.backend import BuildMode

    backends = tuple(backends)
    if not backends or any(backend.build_mode == BuildMode.JUST_IN_TIME for backend in backends):
        return

    offload(model, device=device)


def get_default_backend_for_module(module: nn.Module) -> "Backend":
    """Gets the default backend for a module."""
    from aitune.torch.backend import ONNXRuntimeBackend, TorchEagerBackend

    return ONNXRuntimeBackend() if isinstance(module, OnnxModule) else TorchEagerBackend()
