# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test helpers: wrap methods so they report to the active call recorder, and build call trees by hand."""

import functools
from contextlib import contextmanager

import torch
from torch import nn

from aitune.torch.call_tree.call_tree import CallTree
from aitune.torch.call_tree.methods import user_defined_methods
from aitune.torch.call_tree.nodes import CallNode, TimedIteration
from aitune.torch.call_tree.recording import CallRecorder, StructureRecorder
from tests.toy_models.call_tree_models import Block, NoWeights


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


def record(root, call, iterations: int = 2, root_label: str = "root") -> CallTree:
    """Record ``call()`` for ``iterations`` iterations with ``root`` instrumented."""
    recorder = StructureRecorder()
    with instrumented(recorder, root, root_label), recorder:
        for _ in range(iterations):
            recorder.begin_iteration()
            call()
            recorder.end_iteration()
    return recorder.result()


def timed(ms: float, calls: int) -> list[TimedIteration]:
    """``timings`` that make a node with ``calls`` calls derive ``time_ms == ms``: one iteration timing them all."""
    return [TimedIteration(0, calls, ms)]


def make_tree(spec: dict, iterations: int = 1, modules: dict | None = None) -> CallTree:
    """Build a call tree from ``{label: (calls, time_ms, children_spec)}``, nested; labels end in ``.method``.

    ``time_ms`` becomes ``timed(time_ms, calls)``, so the node's derived ``time_ms`` is the spec's value.
    ``modules`` maps labels to the ``nn.Module`` the node belongs to.
    """
    modules = modules or {}
    tree = CallTree(iterations=iterations)
    order = [0]

    def add(parent_path, label, calls, time_ms, children):
        path = (*parent_path, label)
        method = label.rsplit(".", 1)[-1]
        module = modules.get(label)
        node = CallNode(
            path=path,
            label=label,
            owner_class=type(module).__name__ if module is not None else label.rsplit(".", 2)[-2],
            method=method,
            module=module,
            calls=calls,
            timings=timed(time_ms, calls),
            first_seen=order[0],
        )
        order[0] += 1
        tree.nodes[path] = node
        if parent_path:
            tree.nodes[parent_path].children.append(path)
        for child_label, child_spec in children.items():
            add(path, child_label, *child_spec)

    for label, node_spec in spec.items():
        add((), label, *node_spec)
    for label, module in modules.items():
        tree.modules.append((label.rsplit(".", 1)[0], module))
    tree.sequences = [[] for _ in range(iterations)]
    return tree


def linear():
    return nn.Linear(2, 2)


def vla_tree(loop_calls: int = 5):
    """root -> vision, llm (4 blocks), dit_forward x5 -> (dit, t_embedder); weight-less positions."""
    modules = {
        "root.forward": nn.Module(),
        "root.vision.forward": linear(),
        "root.llm.forward": nn.Module(),
        "root.positions.forward": NoWeights(),
        "root.dit.forward": linear(),
        "root.t_embedder.forward": linear(),
        **{f"root.llm.layers.{i}.forward": Block() for i in range(4)},
    }
    spec = {
        "root.forward": (
            1,
            100.0,
            {
                "root.vision.forward": (1, 20.0, {}),
                "root.positions.forward": (1, 5.0, {}),
                "root.llm.forward": (1, 30.0, {f"root.llm.layers.{i}.forward": (1, 7.0, {}) for i in range(4)}),
                "root.dit_forward": (
                    loop_calls,
                    45.0,
                    {"root.dit.forward": (loop_calls, 41.0, {}), "root.t_embedder.forward": (loop_calls, 4.0, {})},
                ),
            },
        )
    }
    return make_tree(spec, modules=modules)
