# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight accesses: which code reads each weight, checks its device, or moves it during call-tree recording.

A weight is a parameter or a buffer. While a ``CallRecorder`` records (``with recorder:``), ``WeightAccessMode`` (a
``TorchFunctionMode``) sees every torch op. Each op whose arguments touch a weight's storage is recorded as an access
by the innermost recorded call, or by the calling code if no recorded call is running.

Ops are classified by name:

- Metadata that an offloaded ``meta`` weight still answers, such as ``shape``, ``dtype`` and ``stride``, is ignored.
- Device lookups (``.device``, ``is_cuda``, ``get_device()``) are ``"device"`` accesses.
- Every other op, including ops with unknown names, is a ``"read"``. A wrong guess therefore only keeps a weight on the
  device that could have been offloaded; it never lets a weight that is read be offloaded.

Moves are recorded separately. ``.to()`` and casts run their ops inside ``nn.Module._apply``, where the mode is paused,
so those ops are not reads. Instead, the recorder's ``_apply`` probe records one ``"move"`` access for each weight it
reaches.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Literal, NamedTuple

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from aitune.torch.call_tree.callers import unrecorded_caller
from aitune.torch.call_tree.nodes import short_label

METADATA_OPS = {
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

DEVICE_OPS = {"device", "is_cuda", "is_cpu", "is_meta", "is_xpu", "is_mps", "get_device"}


class WeightAccess(NamedTuple):
    """One access to a weight: which call or code did it, and how.

    Attributes:
        by: The path of the recorded call it ran in (``forward > llm.embed.forward``, see ``CallNode.path``); outside
            any recorded call, the code found by ``unrecorded_caller()`` (``<unrecorded: Pipe.encode>``) or
            ``<outside any call>``. ``shared_weights_of`` reports unrecorded code by its qualified name
            (``Pipe.encode``).
        kind: ``"read"`` (an op uses its data), ``"device"`` (its device is checked: ``.device``, ``is_cuda``,
            ``get_device()``), or ``"move"`` (``nn.Module._apply``: ``.to()``, ``.cuda()``, casts, reached it).
    """

    by: str
    kind: Literal["read", "device", "move"]


class WeightAccessLog:
    """Weight accesses recorded by ``WeightAccessMode``, kept by weight name and accumulated over recording passes.

    The log keeps two tables, rebuilt by ``start()`` at the beginning of each pass:

    - the owner table: each weight name to the ``_Owner`` attribute that holds it;
    - the storage map: each storage key (device and address) to the names of the weights on it. A tied weight has
      one key with two names.

    An op is credited to a weight when one of its tensors has a key in the storage map. Records are kept by name, not
    by tensor, so they stay valid when a tensor is replaced, and a weight read only in an earlier pass still counts.
    ``snapshot()`` returns all records so far.

    The storage map is rebuilt from the owner table whenever a weight may have moved: at ``start()``, after each
    ``nn.Module._apply`` (``.to()``, casts), when an op hits an address that one of its weights has left (storage
    swapped outside ``_apply``, such as ``param.data = ...``), and when an untracked ``nn.Parameter`` is seen.

    Limits:

    - Weights of modules attached after ``start()`` are not tracked, since they have no recorded name.
    - A module replaced during recording (``model.head = NewHead()``) is picked up at the next ``start()``; until
      then, the replaced module is tracked under the name.
    - A buffer replaced outside ``_apply`` is missed until the storage map is next rebuilt, because an op on the new
      buffer cannot be told apart from one on an activation.
    """

    def __init__(self) -> None:
        """Create an empty log."""
        # storage key -> names of the weights that share it
        self._storage: dict[Hashable, set[str]] = {}
        # weight name -> the module attribute that holds it
        self._owners: dict[str, _Owner] = {}
        # weight name -> deduplicated records over all passes
        self._accesses: dict[str, set[WeightAccess]] = {}
        # Parameters found untracked since the last start() or move, so a miss on them re-scans only once
        self._untracked: dict[int, weakref.ref[torch.Tensor]] = {}

    def start(self, modules: Iterable[tuple[str, nn.Module]]) -> None:
        """Start a recording pass: rebuild the owner table and the storage map from ``modules``."""
        self._owners = {}
        seen: set[str] = set()
        for prefix, module in modules:
            for name, owner, _ in _weight_owners(prefix, module, seen):
                self._owners[name] = owner
        self.retrack()

    def retrack(self) -> None:
        """Rebuild the storage map and forget the parameters found untracked so far.

        Called once per pass by ``start()`` and after ``nn.Module._apply`` gave a module's tensors new storage (a
        move or a dtype cast during recording): every weight keeps its recorded name, so a tied alias living
        outside the moved subtree follows it to the new storage. Reading the storage must not count as an access,
        so it runs with torch functions, and this mode, disabled.
        """
        self._untracked.clear()
        self._rebuild_storage_map()

    def _rebuild_storage_map(self) -> None:
        """Rebuild the storage map from the tensors the owners hold now, keeping the untracked cache.

        A weight whose attribute is now empty (``module.weight = None``) is left out, so a tied weight that still uses
        the storage no longer credits it, and an address no weight uses any more drops out.
        """
        with torch._C.DisableTorchFunction():
            storage: dict[Hashable, set[str]] = {}
            for name, owner in self._owners.items():
                tensor = owner.tensor()
                if tensor is not None:
                    storage.setdefault(storage_key(tensor), set()).add(name)
            self._storage = storage

    def record(self, key: Hashable, access: WeightAccess) -> None:
        """Add ``access`` to the records of every weight name on the storage ``key``."""
        for name in self._storage.get(key, ()):
            self._accesses.setdefault(name, set()).add(access)

    def lookup(self, tensor: torch.Tensor, key: Hashable) -> bool:
        """Whether ``key`` is tracked; on a miss for an ``nn.Parameter``, rebuild the storage map once and look again.

        A weight given new storage outside ``nn.Module._apply`` (``param.data = ...``, a new ``nn.Parameter`` or
        buffer on its owner, as accelerate's ``set_module_tensor_to_device`` does) leaves its name under the old
        key: a hit there rebuilds the storage map first, so the stale address neither credits it nor hides its new
        storage.
        """
        if key in self._storage:
            if self._is_current(key):
                return True
            self._rebuild_storage_map()  # a weight left this storage outside nn.Module._apply
            if key in self._storage:
                return True
        if not isinstance(tensor, nn.Parameter):
            return False
        ref = self._untracked.get(id(tensor))
        if ref is not None and ref() is tensor:
            return False
        self._rebuild_storage_map()
        if key in self._storage:
            return True
        self._untracked[id(tensor)] = weakref.ref(tensor)
        return False

    def _is_current(self, key: Hashable) -> bool:
        """Whether every weight filed under ``key`` still has its storage there; False means the entry is stale.

        All names must still be there, not just one: a tied weight where one name moved away must not keep crediting
        that name. Reads the owners' tensors with torch functions disabled, like ``_rebuild_storage_map``.
        """
        with torch._C.DisableTorchFunction():
            for name in self._storage.get(key, ()):
                tensor = self._owners[name].tensor()
                if tensor is None or storage_key(tensor) != key:
                    return False
            return True

    def snapshot(self) -> dict[str, set[WeightAccess]]:
        """Accesses of all recording passes so far, keyed by weight name, as ``WeightAccess`` records."""
        return {name: set(records) for name, records in self._accesses.items()}


class WeightAccessMode(TorchFunctionMode):
    """Record a ``"read"`` or ``"device"`` access for each torch op that touches a weight's storage.

    The recorder enters it for one recording pass (its ``with`` block). For each op, it skips metadata ops
    (``METADATA_OPS``), classifies device lookups (``DEVICE_OPS``) and reads, and records a ``WeightAccess`` in ``log``
    for every tracked weight among the op's tensors. The access is credited to the innermost recorded call, or to the
    code found by ``unrecorded_caller()`` when no recorded call is running. Inside ``paused()`` it records nothing.

    Args:
        log: The ``WeightAccessLog`` to record into.
        current_call_path: Returns the path of the innermost recorded call, or ``""`` outside any call.
    """

    def __init__(self, log: WeightAccessLog, current_call_path: Callable[[], str]) -> None:
        """Record weight accesses into ``log``."""
        super().__init__()
        self._log = log
        self._current_call_path = current_call_path
        self._paused = 0

    @contextmanager
    def paused(self) -> Generator[None]:
        """Suppress recording while inside ``nn.Module._apply``; nestable because ``_apply`` recurses."""
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        """Classify each op on a tracked storage as a read or device lookup, then run it."""
        kwargs = kwargs or {}
        self._record(func, args, kwargs)
        return func(*args, **kwargs)

    def _record(self, func: Any, args: tuple, kwargs: dict) -> None:
        if self._paused:
            return  # a move inside nn.Module._apply, recorded by the recorder's probe
        name = op_name(func)
        if name in METADATA_OPS:
            return
        kind: Literal["read", "device"] = "device" if name in DEVICE_OPS else "read"
        by = self._current_call_path()
        for t in _tensors_in((args, kwargs)):
            key = storage_key(t)
            if not self._log.lookup(t, key):
                continue
            by = by or unrecorded_caller()
            self._log.record(key, WeightAccess(by, kind))


def named_weights(prefix: str, module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Each parameter and buffer of ``module`` with its full name.

    The full name is the adapter's label for the module without the leading ``root.`` plus the parameter or buffer
    name: prefix ``root`` gives ``weight``, prefix ``root.vlm.lm_head`` gives ``vlm.lm_head.weight``. Tied or aliased
    tensors yield each of their names.
    """
    for name, _, tensor in _weight_owners(prefix, module):
        yield name, tensor


def op_name(func: Any) -> str:
    """The op name of a ``__torch_function__`` call; property getters arrive as ``Tensor.device.__get__``."""
    name = getattr(func, "__name__", None) or str(func)
    if name == "__get__":
        name = getattr(getattr(func, "__self__", None), "__name__", name)
    return name


def storage_key(t: torch.Tensor) -> Hashable:
    """The key of ``t``'s storage in the log's storage map: ``(device, data address)``.

    Views of a weight (``w.t()``, ``w[0]``) share its storage, so they get the same key and count as the weight.

    A tensor without a readable address gets ``("object", id(t))`` instead: sparse tensors and wrapper subclasses,
    which have no storage to read, and empty or ``meta`` tensors, whose address is 0. Such a weight is then seen only
    when the tensor itself is used, not through a view.
    """
    try:
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        if ptr:
            return (storage.device, ptr)
    except (NotImplementedError, RuntimeError, TypeError):
        return ("object", id(t))
    return ("object", id(t))


class _Owner(NamedTuple):
    """The module attribute that holds a weight, e.g. ``lm_head.weight`` as ``_Owner(lm_head, "weight", False)``.

    The log keeps this rather than the tensor, so it can read the attribute again and find the tensor that is there
    now, after ``.to()``, ``param.data = ...`` or a new ``nn.Parameter`` replaced it.
    """

    module: nn.Module
    attribute: str
    is_buffer: bool

    def tensor(self) -> torch.Tensor | None:
        """The tensor the attribute holds now, or None if it is empty."""
        return (self.module._buffers if self.is_buffer else self.module._parameters).get(self.attribute)


def _full_name(module_path: str, local: str) -> str:
    """The weight name: the module path without the leading ``root.`` plus the local parameter or buffer name."""
    base = short_label(module_path) if "." in module_path else ""
    return f"{base}.{local}" if base else local


def _weight_owners(
    prefix: str, module: nn.Module, seen: set[str] | None = None
) -> Iterator[tuple[str, _Owner, torch.Tensor]]:
    """Yield every parameter and buffer under ``module`` with its owner.

    A submodule reachable through two attributes is visited under each path, so a tied weight is yielded once per
    name. Empty attributes (``None``) are skipped. All parameters come before all buffers, as with
    ``named_parameters`` followed by ``named_buffers``.

    Args:
        prefix: The adapter's label for ``module``, e.g. ``root.vlm``.
        module: The module to walk, including its submodules.
        seen: Module paths already walked, shared across calls. When the registered modules are walked one after the
            other, a path inside an earlier one is not yielded again.

    Yields:
        ``(full name, owner, tensor)``, e.g.
        ``("vlm.lm_head.weight", _Owner(lm_head, "weight", False), lm_head.weight)``.
    """
    subs = []
    for sub_name, sub in module.named_modules(remove_duplicate=False):
        path = prefix if not sub_name else f"{prefix}.{sub_name}"
        if seen is not None:
            if path in seen:
                continue
            seen.add(path)
        subs.append((path, sub))
    for is_buffer in (False, True):
        for path, sub in subs:
            for attribute, tensor in (sub._buffers if is_buffer else sub._parameters).items():
                if tensor is not None:
                    yield _full_name(path, attribute), _Owner(sub, attribute, is_buffer), tensor


def _tensors_in(value: Any):
    """Yield each ``torch.Tensor`` found in ``value``, recursing through lists, tuples and dict values."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors_in(item)
