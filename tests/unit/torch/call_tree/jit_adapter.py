# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only JIT adapter for the call recorder.

Finds modules the JIT way, by intercepting ``nn.Module.__init__`` and filtering with ``Patcher._is_allowed_to_tune``,
and wraps each new instance's ``forward`` and user-defined methods. Nodes are named by class and creation index,
since JIT has no object paths. This is not an early version of the phase 1b ``InspectModule`` adapter; it only
checks that the recorder does not depend on AOT.
"""

from collections.abc import Generator
from contextlib import contextmanager

from torch import nn

from aitune.torch.call_tree.methods import user_defined_methods
from aitune.torch.call_tree.recorder import CallRecorder
from aitune.torch.jit.patcher import Patcher
from tests.unit.torch.call_tree.helpers import restore_methods, wrap_method


@contextmanager
def jit_recording_adapter(recorder: CallRecorder) -> Generator[None, None, None]:
    """Instrument every module created inside the context; restore everything on exit."""
    original_init = nn.Module.__init__
    restore: list = []
    created = [0]

    def patched_init(module, *args, **kwargs):
        original_init(module, *args, **kwargs)
        if not Patcher._is_allowed_to_tune(module):
            return
        prefix = f"{type(module).__qualname__}#{created[0]}"
        created[0] += 1
        recorder.register_module(prefix, module)
        wrap_method(recorder, module, "forward", f"{prefix}.forward", restore)
        for name in user_defined_methods(module):
            wrap_method(recorder, module, name, f"{prefix}.{name}", restore)

    nn.Module.__init__ = patched_init
    try:
        yield
    finally:
        nn.Module.__init__ = original_init
        restore_methods(restore)
