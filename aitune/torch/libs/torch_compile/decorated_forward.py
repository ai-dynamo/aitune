# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind decorated forward methods for torch.compile."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
import wrapt


class DecoratedForwardBindings:
    """Keep wrapt forwards bound while a compiled backend is active.

    Dynamo's FuncDefaultsGuardAccessor rejects wrapt class descriptors in newer PyTorch
    versions. Instance binding preserves the decorator, including its runtime
    checks, while giving Dynamo a bound-method source. Restore before copying the model:
    bound wrapt proxies do not support deepcopy. Checkpoints store state dictionaries, so
    every backend activation must bind the fresh model again.
    """

    def __init__(self) -> None:
        """Track the instance bindings owned by one backend."""
        self._forwards: dict[torch.nn.Module, Callable] = {}

    def bind(self, module: torch.nn.Module) -> None:
        """Bind class-defined wrapt forwards without replacing instance overrides.

        Args:
            module: Model whose decorated forwards Dynamo will trace.
        """
        for child in module.modules():
            if "forward" in vars(child):
                continue
            forward = child.forward
            if isinstance(forward, wrapt.BoundFunctionWrapper):
                self._forwards[child] = forward
                child.forward = forward

    def restore(self) -> None:
        """Restore class lookup unless another owner has replaced the bound forward."""
        for module, forward in self._forwards.items():
            # AITune can install its own proxy after compilation; leave that proxy intact.
            if vars(module).get("forward") is forward:
                del module.forward
        self._forwards.clear()

    @contextmanager
    def compiling(self, module: torch.nn.Module) -> Generator[None]:
        """Keep bindings after compilation succeeds; roll back if compilation or warmup fails.

        Args:
            module: Model to bind for compilation and subsequent inference.
        """
        try:
            self.bind(module)
            yield
        except BaseException:
            # Compilation may be interrupted; never leave temporary forwards after a failed build.
            self.restore()
            raise
