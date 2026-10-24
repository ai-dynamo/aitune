# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep ordinary AITune tuning inside real FlexTensor-managed residency boundaries."""

# /// script
# dependencies = ["flextensor==0.5.0"]
# scope = "always"
# allow_failure = false
# arguments = [
#   {mode = "eager"},
#   {mode = "deferred"},
#   {mode = "deferred", performance = "enabled"},
#   {mode = "deferred", finalization = "profile"},
#   {mode = "eager", setup = "helper"},
#   {mode = "deferred", setup = "helper"},
#   {mode = "deferred", performance = "enabled", setup = "helper"},
#   {mode = "deferred", finalization = "profile", setup = "helper"},
# ]
# ///

import argparse
import json
import logging
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import flextensor as ft
import torch
from _tuning_data_artifacts import collect_tuning_data
from torch._dynamo.utils import counters

from aitune.torch import jit_config, snapshot_tuning_data
from aitune.torch.backend import TorchEagerBackend, TorchInductorJitBackend, TorchInductorJitBackendConfig
from aitune.torch.jit.config import JITMode
from aitune.torch.jit.flextensor import (
    FlexTensorUnit,
    _tuned_forward_signature,
    prepare_flextensor_units,
    register_flextensor_units,
    replan_flextensor,
)
from aitune.torch.jit.tune import deferred
from aitune.torch.module.tuned_module import TunedModule
from aitune.torch.task.profiling import ModelExecutionTimeMeasuringStrategy
from aitune.torch.tune_strategy import OneBackendStrategy, PerformanceValidationMode
from aitune.torch.tune_strategy.tune_strategy import check_inference_output_correctness

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


def configure_jit(mode, performance, cache_dir):
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
    jit_config.detect_graph_breaks = True
    jit_config.strategy = OneBackendStrategy(
        TorchInductorJitBackend(TorchInductorJitBackendConfig(fullgraph=True, mode="default"))
    ).enable_performance_validation(PerformanceValidationMode(performance))
    assert jit_config.strategy._enable_correctness_check, "Built-in correctness checks must remain enabled"


def selected_backends(registration, performance):
    backends = {}
    for wrapper in registration._patched_modules:
        tuned = wrapper._wrapper
        assert isinstance(tuned, TunedModule), "Registration silently retained an eager fallback"
        assert tuned.backends
        for backend in tuned.backends.values():
            if performance == "diagnostic":
                assert type(backend) is TorchInductorJitBackend
            else:
                # The ordinary enabled speed gate may legitimately select its measured eager baseline.
                assert type(backend) in (TorchInductorJitBackend, TorchEagerBackend)
            backends[id(backend)] = backend
    return tuple(backends.values())


def assert_offloading(manager, original_weight_bytes, managed_limit_bytes, resident_head_bytes):
    usage = manager.get_gpu_memory_usage()
    assert usage.blocks_bytes > 0, "No GPU transfer blocks were allocated"
    assert usage.unmapped_tensors_bytes >= 0
    assert usage.total_bytes == usage.blocks_bytes + usage.unmapped_tensors_bytes
    assert usage.total_bytes < original_weight_bytes, "The model is entirely resident rather than offloaded"
    assert usage.total_bytes <= managed_limit_bytes, (
        f"Managed residency {usage.total_bytes} exceeds the explicit {managed_limit_bytes}-byte weight budget"
    )
    print(f"Managed weight residency: {usage}; excluded head reservation: {resident_head_bytes}", flush=True)
    return usage


def assert_profiled_graphs(registration, cache_dir, correctness_calls):
    path = snapshot_tuning_data(cache_dir / "tuning-data.json")
    assert path is not None, "No public tuning telemetry was produced"
    report = json.loads(path.read_text())
    by_id = {module["module_id"]: module for module in report["modules"]}
    graphs = []
    for inspection in registration.reports:
        module_report = by_id[inspection.module_id]
        assert module_report["exception"] is None
        assert module_report["graphs"], "A registered block has no tuned graph evidence"
        for graph in module_report["graphs"]:
            assert graph["exception"] is None
            assert graph["baseline_throughput"] > 0, "Eager baseline profiling was skipped or failed"
            builds = [build for build in graph["backend_builds"] if build["backend"].startswith("TorchInductor")]
            assert len(builds) == 1, "Expected exactly the one configured Inductor candidate"
            assert builds[0]["success"] is True and builds[0]["exception"] is None
            assert builds[0]["throughput"] > 0, "Candidate profiling was skipped or failed"
            assert graph["selected_backend"], "No actual backend selection was recorded"
            graphs.append(graph)
    assert correctness_calls >= len(graphs), "Ordinary recorded-sample correctness checks did not run for every graph"
    return len(graphs)


def assert_outputs(run_forward, values, expected):
    for value, baseline in zip(values, expected, strict=True):
        torch.testing.assert_close(run_forward(value), baseline)


def tune_targets(registration, run_forward, values, expected, mode):
    with patch(
        "aitune.torch.tune_strategy.tune_strategy.check_inference_output_correctness",
        wraps=check_inference_output_correctness,
    ) as correctness:
        assert_outputs(run_forward, values, expected)
        if mode == "deferred":
            assert registration.state_counts == {"recording": len(registration.modules)}
            deferred()
        for _ in range(8):
            assert_outputs(run_forward, values, expected)
            if registration.all_tuned:
                break
    assert registration.all_tuned, registration.state_counts
    return correctness.call_count


@collect_tuning_data(__file__)
def run(mode, performance, finalization, setup, cache_dir):
    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure_jit(mode, performance, cache_dir)
    model = Model().to("cuda").eval()
    values = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    held_out = tuple(torch.randn(batch, WIDTH, device="cuda") for batch in (1, 2))
    with torch.no_grad():
        expected = tuple(model(value).clone() for value in values)
        held_out_expected = tuple(model(value).clone() for value in held_out)
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    managed_limit_bytes = weight_bytes // 2
    # Keep the strict total-residency limit independent of the provider's strategy budget.
    # The excluded head is already placed on CUDA during profiling, so reserve its
    # storage explicitly instead of assuming a fraction also caps preexisting residents.
    resident_head_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.head.parameters())
    strategy_budget_bytes = managed_limit_bytes - resident_head_bytes
    budget_fraction = strategy_budget_bytes / torch.cuda.get_device_properties(0).total_memory
    config = ft.OffloadConfig(
        include_patterns=["layers.*"],
        external_compile=finalization == "replan",
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
        assert prepared.unit_paths == tuple(f"layers.{index}" for index in range(10))
    else:
        # Keep the original manual option under the identical acceptance assertions.
        model.layers = torch.nn.ModuleList(FlexTensorUnit(block) for block in model.layers)
        model = ft.offload(
            model,
            config,
            compile_fn=(lambda module: module) if finalization == "profile" else None,
        )
        manager = ft.get_offload_manager()
    with torch.no_grad():
        for _ in range(manager.iters_before_inference):
            model(values[0])
        assert manager.phase.value == "inference"
        assert manager.compiled_offload_active
        assert manager.compiled_replan_active == (finalization == "replan")
        assert manager.compiled_profile_active == (finalization == "profile")
        offload_units, resident_units = manager.offload_units(), manager.resident_units()
        assert len(offload_units) == 10, offload_units
        assert tuple(name for name, _ in resident_units) == ("head",), resident_units
        shells = tuple(module for _, module in offload_units)
        assert all(type(shell) is FlexTensorUnit for shell in shells)
        units = tuple(shell.compute for shell in shells)
        # Inner targets must have no provider instrumentation to replay during ordinary tuning.
        assert all("forward" not in unit.__dict__ for unit in units)
        live_model = manager.model
        root_forward = live_model.forward
        shell_forwards = tuple(shell.forward for shell in shells)
        resident_forwards = tuple(module.forward for _, module in resident_units)
        usage = assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes)
        calls = Counter()

        def completed_unit(module, args, output):
            calls[id(module)] += 1

        for shell in shells:
            shell.register_forward_hook(completed_unit)
        completed_roots = 0

        def run_forward(value):
            nonlocal completed_roots
            result = model(value)
            completed_roots += 1
            assert all(calls[id(shell)] == completed_roots for shell in shells), (
                "AITune replayed an outer FlexTensor boundary while tuning an inner compute target"
            )
            return result

        assert_outputs(run_forward, values, expected)
        compiled_before = counters["stats"]["unique_graphs"]
        measurements_before = ModelExecutionTimeMeasuringStrategy.counter
        registration = register_flextensor_units([manager])
        assert len(registration.modules) == len(units)
        assert all(actual is selected for actual, selected in zip(registration.modules, units, strict=True))
        correctness_calls = tune_targets(registration, run_forward, values, expected, mode)
        backends = selected_backends(registration, performance)
        assert counters["stats"]["unique_graphs"] > compiled_before, "No Inductor graphs were compiled"
        measurements = ModelExecutionTimeMeasuringStrategy.counter - measurements_before
        assert measurements > 0, "No real AITune profiling measurements completed"
        graph_count = assert_profiled_graphs(registration, cache_dir, correctness_calls)
        assert_outputs(run_forward, held_out, held_out_expected)
        forward_signatures = tuple(_tuned_forward_signature(wrapper) for wrapper in registration._patched_modules)
        with ExitStack() as stack:
            spies = {
                id(backend): stack.enter_context(patch.object(backend, "infer", wraps=backend.infer))
                for backend in backends
            }
            if finalization == "replan":
                replan_flextensor(manager, registration, lambda: run_forward(values[0]))
            status = manager.strategy_replan_status
            if finalization == "replan":
                assert status.state == "done" and status.applied is True, status
            else:
                assert manager.compiled_profile_active and status.state == "idle", status
            assert manager.model is live_model and live_model.forward == root_forward
            assert tuple(module for _, module in manager.offload_units()) == shells
            assert tuple(shell.compute for shell in shells) == units
            assert all(shell.forward == forward for shell, forward in zip(shells, shell_forwards, strict=True))
            assert all(
                actual is original
                for wrapper, signature in zip(registration._patched_modules, forward_signatures, strict=True)
                for actual, original in zip(_tuned_forward_signature(wrapper), signature, strict=True)
            ), "An offload unit changed its prepared forward, AITune router or tuned dispatcher after replan"
            assert all(
                module.forward == forward
                for (_, module), forward in zip(manager.resident_units(), resident_forwards, strict=True)
            )
            usage = assert_offloading(manager, weight_bytes, managed_limit_bytes, resident_head_bytes)
            calls_before = {identity: spy.call_count for identity, spy in spies.items()}
            assert_outputs(run_forward, (*values, *held_out), (*expected, *held_out_expected))
            for wrapper in registration._patched_modules:
                tuned = wrapper._wrapper
                assert isinstance(tuned, TunedModule), "An offload unit lost its tuned routes after replan"
                assert any(
                    spies[id(backend)].call_count > calls_before[id(backend)] for backend in tuned.backends.values()
                ), "An offload unit bypassed all of its compiled routes after replan"
    print(
        json.dumps({
            "result": "PASS",
            "mode": mode,
            "performance": performance,
            "finalization": finalization,
            "setup": setup,
            "units": len(units),
            "graphs": graph_count,
            "profiling_measurements": measurements,
            "correctness_checks": correctness_calls,
            "selected_inductor_routes": sum(type(backend) is TorchInductorJitBackend for backend in backends),
            "selected_eager_routes": sum(type(backend) is TorchEagerBackend for backend in backends),
            "managed_gpu_bytes": usage.total_bytes,
            "managed_limit_bytes": managed_limit_bytes,
            "strategy_budget_bytes": strategy_budget_bytes,
            "resident_head_bytes": resident_head_bytes,
            "weight_bytes": weight_bytes,
            "replan_state": status.state,
            "replan_applied": status.applied,
            "completed_whole_model_forwards": completed_roots,
        }),
        flush=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "deferred"), default="deferred")
    parser.add_argument("--performance", choices=("diagnostic", "enabled"), default="diagnostic")
    parser.add_argument("--finalization", choices=("replan", "profile"), default="replan")
    parser.add_argument("--setup", choices=("manual", "helper"), default="manual")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aitune-flextensor-") as directory:
        run(args.mode, args.performance, args.finalization, args.setup, Path(directory))
