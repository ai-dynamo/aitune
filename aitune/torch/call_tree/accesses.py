# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight access recording: which recorded calls read or look up the device of a weight.

Owned by ``CallRecorder``: during ``recording()`` a ``TorchFunctionMode`` sees every torch op and attributes each op
whose arguments touch a parameter's or buffer's storage to the current call stack. Ops inside ``nn.Module._apply``
(``.to()`` internals) are moves, not reads, and are skipped by the recorder's ``_apply_depth`` counter.

Ops are classified by name: metadata an offloaded ``meta`` weight still answers (shape, dtype, strides) is ignored;
device lookups (``.device``, ``is_cuda``, ``get_device``) are recorded as ``"device"`` accesses; every other op —
unknown names included — counts as a read. The default is deliberate: a misclassification only keeps more memory, it
never frees a weight that is read.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aitune.torch.call_tree.recorder import CallRecorder

import torch
from torch.overrides import TorchFunctionMode

IGNORED = {
    "dtype",
    "shape",
    "size",
    "dim",
    "ndim",
    "numel",
    "nelement",
    "stride",
    "storage_offset",
    "layout",
    "is_contiguous",
    "element_size",
    "itemsize",
    "nbytes",
    "is_floating_point",
    "is_complex",
    "is_signed",
    "requires_grad",
    "grad",
    "grad_fn",
    "is_leaf",
    "_version",
    "names",
    "is_sparse",
    "is_quantized",
    "is_nested",
    "__len__",
    "__hash__",
    "_is_view",
    "data",
}

DEVICE = {"device", "is_cuda", "is_cpu", "is_meta", "is_xpu", "is_mps", "get_device"}


def op_name(func: Any) -> str:
    """The op name of a ``__torch_function__`` call; property getters arrive as ``Tensor.device.__get__``."""
    name = getattr(func, "__name__", None) or str(func)
    if name == "__get__":
        name = getattr(getattr(func, "__self__", None), "__name__", name)
    return name


class WeightAccessMode(TorchFunctionMode):
    """Record ``(node key, "read"|"device")`` for each op that touches a parameter or buffer storage.

    Built by the recorder at ``recording()`` start over all registered modules' named parameters and buffers; the
    caller name is filled in by the recorder (current node's label, or the unrecorded caller).

    Args:
        recorder: The owning ``CallRecorder``.
    """

    def __init__(self, recorder: "CallRecorder") -> None:
        """Record weight accesses against ``recorder``'s registered modules."""
        super().__init__()
        self._recorder = recorder
        # tracking key -> ids of the tensors that share it; a tied weight is one entry, not two
        self._storage: dict[Hashable, set[int]] = {}
        for _, module in recorder._modules:
            self.register(module)
        # tracking key -> deduplicated (node key, kind, caller) records
        self.accesses: dict[Hashable, set[tuple[tuple[str, ...], str, str]]] = {}

    def register(self, module: torch.nn.Module) -> None:
        """Track the parameters and buffers of ``module`` and its submodules under their current storage.

        Called for every registered module when recording starts, and again after ``nn.Module._apply`` gave a
        module's tensors new storage (a move or a dtype cast during recording). Reading the storage must not count
        as an access, so it runs with torch functions, and this mode, disabled.
        """
        with torch._C.DisableTorchFunction():
            for _, t in (
                *module.named_parameters(remove_duplicate=False),
                *module.named_buffers(remove_duplicate=False),
            ):
                self._storage.setdefault(_tracking_key(t), set()).add(id(t))

    def tensor_ids(self, key: Hashable) -> set[int]:
        """The tensor ids sharing ``key``, for expanding the deduplicated records in ``result()``."""
        return self._storage.get(key, set())

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        """Classify each op on a tracked storage as a read or device lookup, then run it."""
        kwargs = kwargs or {}
        self._record(func, args, kwargs)
        return func(*args, **kwargs)

    def _record(self, func: Any, args: tuple, kwargs: dict) -> None:
        if self._recorder._apply_depth:
            return  # a move inside nn.Module._apply, recorded by the recorder's probe
        name = op_name(func)
        if name in IGNORED:
            return
        kind = "device" if name in DEVICE else "read"
        node_key = self._recorder._stack[-1] if self._recorder._stack else ()
        for t in _leaves((args, kwargs)):
            key = _tracking_key(t)
            if key not in self._storage:
                continue
            self.accesses.setdefault(key, set()).add((node_key, kind, self._recorder._access_caller(node_key)))


def _tracking_key(t: torch.Tensor) -> Hashable:
    """The storage address of ``t``, or its identity when it has no readable storage.

    Sparse tensors and wrapper subclasses have no storage to read, and empty or ``meta`` tensors have address 0.
    They are tracked as the object itself: a direct use of the parameter is still seen, a view of it is not.
    """
    try:
        ptr = t.untyped_storage().data_ptr()
    except (NotImplementedError, RuntimeError, TypeError):
        ptr = 0
    return ptr if ptr else ("object", id(t))


def _leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
