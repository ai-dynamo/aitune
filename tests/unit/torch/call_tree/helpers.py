# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test helpers: wrap methods so they report to the active call recorder, and record call trees."""

import functools
from contextlib import contextmanager

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.methods import user_defined_methods
from aitune.torch.call_tree.recording import CallRecorder, StructureRecorder


def wrap_method(obj, name: str, label: str, restore: list):
    """Replace ``obj.<name>`` with a wrapper reporting to the active recorder; remember how to restore it."""
    original = getattr(obj, name)
    was_instance = name in vars(obj)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        recorder = CallRecorder.active()
        if recorder is None:
            return original(*args, **kwargs)
        return recorder.call(original, args, kwargs, label=label, obj=obj, method=name)

    setattr(obj, name, wrapper)
    restore.append((obj, name, was_instance, original))


def restore_methods(restore: list):
    """Undo ``wrap_method``."""
    for obj, name, was_instance, original in reversed(restore):
        if was_instance:
            setattr(obj, name, original)
        else:
            delattr(obj, name)
    restore.clear()


def invisible_to_recorder():
    """The recorder does not see torch ops run inside this, like an extension kernel or a raw storage swap."""
    return torch._C.DisableTorchFunction()


@contextmanager
def instrumented(recorder: StructureRecorder, root, root_label: str = "root"):
    """Wrap ``forward`` and user methods of every module under ``root`` (and ``root``'s own methods)."""
    restore: list = []
    objects = [(root_label, root)]
    if isinstance(root, nn.Module):
        objects = [(f"{root_label}.{n}" if n else root_label, m) for n, m in root.named_modules()]
    else:
        for attr, value in vars(root).items():
            if isinstance(value, nn.Module):
                objects += [
                    (f"{root_label}.{attr}.{n}" if n else f"{root_label}.{attr}", m) for n, m in value.named_modules()
                ]
    try:
        for label, obj in objects:
            if isinstance(obj, nn.Module):
                recorder.register_module(label, obj)
                wrap_method(obj, "forward", f"{label}.forward", restore)
            for name in user_defined_methods(obj):
                wrap_method(obj, name, f"{label}.{name}", restore)
        yield
    finally:
        restore_methods(restore)


def record(root, call, num_samples: int = 2, root_label: str = "root") -> CallTree:
    """Record ``call()`` for ``num_samples`` samples with ``root`` instrumented."""
    recorder = StructureRecorder()
    with instrumented(recorder, root, root_label), recorder:
        for _ in range(num_samples):
            with recorder.record_sample():
                call()
    return recorder.result()
