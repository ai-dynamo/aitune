# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile caller-provided prepared units without a provider integration dependency."""

# /// script
# scope = "always"
# allow_failure = false
# arguments = [{mode = "eager"}, {mode = "deferred"}]
# ///

import argparse
import logging
import tempfile
from pathlib import Path
from types import MethodType

import torch
from torch._dynamo.utils import counters

from aitune.torch import jit_config, register_for_jit_tuning
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.patcher import Patcher
from aitune.torch.jit.tune import deferred
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_strategy import OneBackendStrategy


class Unit(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer = torch.nn.Linear(16, 16)
        self.offset = (index + 1) / 100

    def forward(self, value):
        return self.layer(value).tanh()


def prepared_forward(self, value):
    # Represents a caller-installed forward that must survive recording and compilation.
    return self.layer(value + self.offset).tanh() + self.offset


def pre_hook(module, args):
    return (args[0] + 0.125,)


def post_hook(module, args, output):
    return output * 1.5


def run(mode, cache_dir):
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Ten same-frame units exceed the common eight-variant default. The caller owns this budget.
    import torch._dynamo.config as dynamo_config

    for name in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "cache_size_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, max(getattr(dynamo_config, name), 256))

    model = torch.nn.Sequential(*(Unit(index) for index in range(10))).to("cuda").eval()
    units = tuple(model)
    for unit in units:
        unit.forward = MethodType(prepared_forward, unit)
        unit.register_forward_pre_hook(pre_hook)
        unit.register_forward_hook(post_hook)
    values = [torch.randn(batch, 16, device="cuda") for batch in (1, 2)]
    with torch.no_grad():
        expected = [model(value).clone() for value in values]

    jit_config.mode = JITMode.TUNE_EAGER if mode == "eager" else JITMode.TUNE_DEFERRED
    jit_config.device = "cuda"
    jit_config.cache_dir = cache_dir
    jit_config.min_samples = 2
    jit_config.batch_axis_required = False
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(
        TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
    ).enable_performance_validation(False)
    original_constructor = torch.nn.Module.__init__
    compiled_before = counters["stats"]["unique_graphs"]
    registration = register_for_jit_tuning(units)
    # Registration leaves constructors unchanged; Dynamo may later wrap them for generation tagging.
    assert torch.nn.Module.__init__ is original_constructor

    with torch.no_grad():
        for value, baseline in zip(values, expected, strict=True):
            torch.testing.assert_close(model(value), baseline)
        if mode == "deferred":
            assert registration.state_counts == {"recording": len(units)}
            deferred()
            model(values[0])
        assert registration.all_tuned, registration.state_counts
        for value, baseline in zip(values, expected, strict=True):
            torch.testing.assert_close(model(value), baseline)

    assert registration.modules == units == tuple(model)
    for wrapper in registration._patched_modules:
        tuned_module = wrapper._wrapper
        assert isinstance(tuned_module, TunedModule)
        assert tuned_module.backends
        assert all(isinstance(backend, TorchInductorJitBackend) for backend in tuned_module.backends.values())
    assert counters["stats"]["unique_graphs"] > compiled_before, "No Dynamo graphs were compiled"
    unrelated = Unit(len(units))
    assert "forward" not in unrelated.__dict__
    assert "forward" not in unrelated.layer.__dict__
    assert not Patcher._torch_patched
    assert tuple(Patcher._patched_modules) == registration._patched_modules
    print(f"{mode}: {len(units)} prepared units, two input shapes, real Inductor graphs, preserved forwards and hooks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "deferred"), default="deferred")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aitune-prepared-units-") as directory:
        run(args.mode, Path(directory))
