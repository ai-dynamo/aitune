# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep ordinary eager fallback resident after a genuine fullgraph compilation failure."""

# /// script
# dependencies = ["flextensor==0.5.0"]
# scope = "always"
# allow_failure = false
# arguments = [{setup = "manual"}, {setup = "helper"}]
# ///

import argparse
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

import flextensor as ft
import torch
from _tuning_data_artifacts import collect_tuning_data

from aitune.torch import jit_config, snapshot_tuning_data
from aitune.torch.backend import TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import (
    FlexTensorUnit,
    prepare_flextensor_units,
    register_flextensor_units,
    replan_flextensor,
)
from aitune.torch.jit.tune import deferred
from aitune.torch.task.profiling import ModelExecutionTimeMeasuringStrategy
from aitune.torch.tune_strategy import OneBackendStrategy, PerformanceValidationMode

WIDTH = 4096


@torch.compiler.disable
def eager_identity(value):
    # Legal eager execution, but a real fullgraph=True candidate must reject this call.
    return value


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, value):
        return value + self.projection(value).relu() * 0.1


class UncompilableBlock(Block):
    def forward(self, value):
        return eager_identity(super().forward(value))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(UncompilableBlock() if index == 1 else Block() for index in range(10))
        self.head = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return self.head(value)


def adjust_output(module, args, output):
    return output + 0.125


def assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes):
    usage = manager.get_gpu_memory_usage()
    assert usage.blocks_bytes > 0, "No real GPU transfer blocks were allocated"
    assert usage.unmapped_tensors_bytes >= 0
    assert usage.total_bytes == usage.blocks_bytes + usage.unmapped_tensors_bytes
    assert usage.total_bytes < weight_bytes, "The entire model is resident rather than offloaded"
    assert usage.total_bytes <= managed_limit_bytes, (
        f"Managed residency {usage.total_bytes} exceeds the explicit {managed_limit_bytes}-byte weight budget"
    )
    print(f"Managed weight residency: {usage}; excluded head reservation: {resident_head_bytes}", flush=True)
    return usage


def assert_failed_compile(registration, cache_dir):
    path = snapshot_tuning_data(cache_dir / "tuning-data.json")
    assert path is not None, "No public tuning telemetry was produced"
    report = json.loads(path.read_text())
    module_id = registration.reports[0].module_id
    module = next(item for item in report["modules"] if item["module_id"] == module_id)
    assert module["exception"] is not None, "The ordinary tuning failure handler was not exercised"
    assert module["graphs"], "The selected compute target never reached ordinary AITune tuning"
    builds = [
        build
        for graph in module["graphs"]
        for build in graph["backend_builds"]
        if build["backend"].startswith("TorchInductor")
    ]
    assert len(builds) == 1, "Expected exactly one genuine Inductor candidate failure"
    build = builds[0]
    assert build["success"] is False and build["exception"] is not None
    assert "disable" in build["exception"]["message"].lower(), build["exception"]
    assert all(graph["exception"] is not None and graph["selected_backend"] is None for graph in module["graphs"])
    assert any(graph["baseline_throughput"] > 0 for graph in module["graphs"]), "Eager profiling did not run"
    return build["exception"]


@collect_tuning_data(__file__)
def run(cache_dir, setup):
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jit_config.mode = JITMode.TUNE_DEFERRED
    jit_config.device = "cuda"
    jit_config.cache_dir = cache_dir
    jit_config.min_samples = 2
    jit_config.batch_axis_required = False
    # Only this negative case bypasses preflight detection, so the real backend build fails.
    jit_config.detect_graph_breaks = False
    jit_config.strategy = OneBackendStrategy(
        TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
    ).enable_performance_validation(PerformanceValidationMode.DIAGNOSTIC)
    assert jit_config.strategy._enable_correctness_check, "Built-in correctness checks must remain enabled"

    model = Model().to("cuda").eval()
    compute = model.layers[1]
    hook = compute.register_forward_hook(adjust_output)
    original_forward = compute.forward
    values = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    held_out = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    with torch.no_grad():
        expected = tuple(model(value).clone() for value in values)
        held_out_expected = tuple(model(value).clone() for value in held_out)
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    managed_limit_bytes = weight_bytes // 2
    # The excluded head is already resident during profiling; reserve it explicitly.
    resident_head_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.head.parameters())
    strategy_budget_bytes = managed_limit_bytes - resident_head_bytes
    budget_fraction = strategy_budget_bytes / torch.cuda.get_device_properties(0).total_memory
    config = ft.OffloadConfig(
        include_patterns=["layers.*"],
        external_compile=True,
        transfer_mode="allocation_block_transfer",
        num_blocks=2,
        min_blocks=2,
        max_gpu_mem_fraction=budget_fraction,
        discovery_iters=1,
        profiling_iters=3,
    )
    model = model.to("cpu")
    if setup == "helper":
        prepared = prepare_flextensor_units(model, ft_config=config)
        model, manager = prepared.model, prepared.manager
    else:
        model.layers = torch.nn.ModuleList(FlexTensorUnit(block) for block in model.layers)
        model = ft.offload(model, config)
        manager = ft.get_offload_manager()
    with torch.no_grad():
        for _ in range(manager.iters_before_inference):
            model(values[0])
        assert manager.phase.value == "inference"
        assert manager.compiled_offload_active and manager.compiled_replan_active
        assert not manager.compiled_profile_active
        shells = tuple(module for _, module in manager.offload_units())
        assert len(shells) == 10 and all(type(shell) is FlexTensorUnit for shell in shells)
        assert tuple(name for name, _ in manager.resident_units()) == ("head",)
        assert shells[1].compute is compute and compute.forward == original_forward
        shell_forwards = tuple(shell.forward for shell in shells)
        root_forward = manager.model.forward
        assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes)
        calls = Counter()

        def completed_shell(module, args, output):
            calls[id(module)] += 1

        for shell in shells:
            shell.register_forward_hook(completed_shell)
        completed_roots = 0

        def check_forward(value, baseline):
            nonlocal completed_roots
            torch.testing.assert_close(model(value), baseline)
            completed_roots += 1
            assert all(calls[id(shell)] == completed_roots for shell in shells), (
                "Tuning or fallback replayed an outer FlexTensor residency boundary"
            )

        registration = register_flextensor_units([manager], modules=[shells[1]])
        assert registration.modules == (compute,)
        for value, baseline in zip(values, expected, strict=True):
            check_forward(value, baseline)
        assert registration.state_counts == {"recording": 1}
        measurements_before = ModelExecutionTimeMeasuringStrategy.counter
        deferred()
        # A normal whole-model call enters the selected shell, attempts real compilation,
        # and returns through ordinary AITune's native eager fallback while still resident.
        check_forward(values[0], expected[0])
        assert registration.state_counts == {"eager": 1} and not registration.all_tuned
        assert compute.forward == original_forward, "Native fallback did not restore the original compute forward"
        assert compute._forward_hooks[hook.id] is adjust_output, "Native fallback lost an original compute hook"
        assert ModelExecutionTimeMeasuringStrategy.counter > measurements_before, "No real baseline profiling ran"
        failure = assert_failed_compile(registration, cache_dir)
        usage = assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes)

        replan_calls = 0

        def unexpected_replan_forward():
            nonlocal replan_calls
            replan_calls += 1
            raise AssertionError("Native eager fallback must not request strategy replanning")

        status = manager.strategy_replan_status
        assert status.state == "idle" and status.can_request
        replan_flextensor(manager, registration, unexpected_replan_forward)
        assert replan_calls == 0
        status = manager.strategy_replan_status
        assert status.state == "idle" and status.can_request
        assert manager.model.forward == root_forward
        assert tuple(module for _, module in manager.offload_units()) == shells
        assert all(shell.forward == forward for shell, forward in zip(shells, shell_forwards, strict=True))
        for _ in range(3):
            for value, baseline in zip((*values, *held_out), (*expected, *held_out_expected), strict=True):
                check_forward(value, baseline)
        assert registration.state_counts == {"eager": 1} and not registration.all_tuned
        assert compute.forward == original_forward
        usage = assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes)
    print(
        json.dumps({
            "result": "PASS",
            "mode": "deferred",
            "setup": setup,
            "selected_unit": 1,
            "state_counts": registration.state_counts,
            "compile_failure": failure,
            "managed_gpu_bytes": usage.total_bytes,
            "managed_limit_bytes": managed_limit_bytes,
            "strategy_budget_bytes": strategy_budget_bytes,
            "resident_head_bytes": resident_head_bytes,
            "weight_bytes": weight_bytes,
            "replan_state": status.state,
            "replan_calls": replan_calls,
            "completed_whole_model_forwards": completed_roots,
        }),
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=("manual", "helper"), default="manual")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aitune-flextensor-fallback-") as directory:
        run(Path(directory), args.setup)
