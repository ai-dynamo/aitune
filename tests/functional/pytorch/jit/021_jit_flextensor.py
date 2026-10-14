# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile real FlexTensor offload units, then replan without replacing AITune forwards."""

# /// script
# dependencies = ["flextensor==0.5.0"]
# scope = "always"
# allow_failure = false
# arguments = [{mode = "eager"}, {mode = "deferred"}]
# ///

import argparse
import logging
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import flextensor as ft
import torch
from torch._dynamo.utils import counters

from aitune.torch import jit_config
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import register_flextensor_units, replan_flextensor
from aitune.torch.jit.tune import deferred
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.tune_strategy import OneBackendStrategy

WIDTH = 4096


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, value):
        return value + self.projection(value).relu() * 0.1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(Block() for _ in range(10))
        self.head = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return self.head(value)


def configure_jit(mode, cache_dir):
    # These same-frame units and their two shapes share a finite, caller-owned Dynamo budget.
    import torch._dynamo.config as dynamo_config

    for name in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "cache_size_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, max(getattr(dynamo_config, name), 256))
    jit_config.mode = JITMode.TUNE_EAGER if mode == "eager" else JITMode.TUNE_DEFERRED
    jit_config.device = "cuda"
    jit_config.cache_dir = cache_dir
    jit_config.min_samples = 2
    jit_config.batch_axis_required = False
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(
        TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
    ).enable_performance_validation(False)


def selected_backends(registration):
    backends = {}
    for wrapper in registration._patched_modules:
        tuned = wrapper._wrapper
        assert isinstance(tuned, TunedModule), "Registration silently retained an eager fallback"
        assert tuned.backends
        for backend in tuned.backends.values():
            assert type(backend) is TorchInductorJitBackend
            backends[id(backend)] = backend
    return tuple(backends.values())


def assert_offloading(manager, original_weight_bytes):
    usage = manager.get_gpu_memory_usage()
    assert usage.blocks_bytes > 0, "No GPU transfer blocks were allocated"
    assert usage.total_bytes < original_weight_bytes, "The model is entirely resident rather than offloaded"


def run(mode, cache_dir):
    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure_jit(mode, cache_dir)
    model = Model().to("cuda").eval()
    values = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    with torch.no_grad():
        expected = tuple(model(value).clone() for value in values)
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    # Approximately 704 MiB of weights allow a 352 MiB budget: two rolling slots plus
    # the resident head, above FlexTensor's 256 MiB floor after its resident reservation.
    # The whole model cannot fit. Derive the fraction from actual device capacity.
    budget_fraction = weight_bytes / (2 * torch.cuda.get_device_properties(0).total_memory)
    model = ft.offload(
        model.to("cpu"),
        ft.OffloadConfig(
            include_patterns=["layers.*"],
            external_compile=True,
            transfer_mode="allocation_block_transfer",
            num_blocks=2,
            min_blocks=2,
            max_gpu_mem_fraction=budget_fraction,
            discovery_iters=1,
            profiling_iters=3,
        ),
    )
    manager = ft.get_offload_manager()
    with torch.no_grad():
        for _ in range(manager.iters_before_inference):
            model(values[0])
        assert manager.phase.value == "inference"
        assert manager.external_compile_active
        offload_units, resident_units = manager.offload_units(), manager.resident_units()
        assert len(offload_units) == 10, offload_units
        assert tuple(name for name, _ in resident_units) == ("head",), resident_units
        units = tuple(module for _, module in offload_units)
        resident_forwards = tuple(module.forward for _, module in resident_units)
        assert_offloading(manager, weight_bytes)
        for value, baseline in zip(values, expected, strict=True):
            torch.testing.assert_close(model(value), baseline)
        compiled_before = counters["stats"]["unique_graphs"]
        registration = register_flextensor_units([manager])
        assert len(registration.modules) == len(units)
        assert all(actual is selected for actual, selected in zip(registration.modules, units, strict=True))
        for value, baseline in zip(values, expected, strict=True):
            torch.testing.assert_close(model(value), baseline)
        if mode == "deferred":
            assert registration.state_counts == {"recording": len(units)}
            deferred()
        for _ in range(8):
            for value, baseline in zip(values, expected, strict=True):
                torch.testing.assert_close(model(value), baseline)
            if registration.all_tuned:
                break
        assert registration.all_tuned, registration.state_counts
        backends = selected_backends(registration)
        assert counters["stats"]["unique_graphs"] > compiled_before, "No Inductor graphs were compiled"
        forwards = tuple(module.forward for module in units)
        with ExitStack() as stack:
            spies = {
                id(backend): stack.enter_context(patch.object(backend, "infer", wraps=backend.infer))
                for backend in backends
            }
            replan_flextensor(manager, registration, lambda: model(values[0]))
            status = manager.strategy_replan_status
            assert status.state == "done" and status.applied is True, status
            assert all(module.forward is forward for module, forward in zip(units, forwards, strict=True))
            assert all(
                module.forward == forward
                for (_, module), forward in zip(manager.resident_units(), resident_forwards, strict=True)
            )
            assert_offloading(manager, weight_bytes)
            calls_before = {identity: spy.call_count for identity, spy in spies.items()}
            for value, baseline in zip(values, expected, strict=True):
                torch.testing.assert_close(model(value), baseline)
            for wrapper in registration._patched_modules:
                assert any(
                    spies[id(backend)].call_count > calls_before[id(backend)]
                    for backend in wrapper._wrapper.backends.values()
                ), "An offload unit bypassed all of its compiled routes after replan"
    print(f"{mode}: {len(units)} live units compiled; real offload and compiled dispatch survive strategy replan")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "deferred"), default="deferred")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aitune-flextensor-") as directory:
        run(args.mode, Path(directory))
