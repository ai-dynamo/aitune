# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd.profiler_util import EventList

import aitune.torch as ait
from aitune.torch.backend import TorchEagerBackend
from aitune.torch.performance import PerformanceProfile
from aitune.torch.performance.attribution_hooks import _CONTEXT_ATTR
from aitune.torch.performance.runtime_profile import (
    _RegionAggregate,
    _Residual,
    _RuntimeProfile,
    _RunTiming,
)


class TinyKeywordModel(torch.nn.Module):
    """Tiny model that makes keyword input handling visible in the report."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_profile_returns_in_memory_data_without_files_by_default(tmp_path):
    model = TinyKeywordModel()
    model.eval()
    input_data = {"x": torch.randn(1, 4)}

    def inference_function(**kwargs):
        with torch.inference_mode():
            return model(**kwargs)

    result = ait.profile(
        obj=model,
        input_data=input_data,
        inference_function=inference_function,
        warmup_runs=1,
        measured_runs=2,
    )

    assert isinstance(result, PerformanceProfile)
    assert result.trace_file is None
    assert "artifacts" not in result.data
    assert list(tmp_path.iterdir()) == []
    assert result.data["config"]["warmup_runs"] == 1
    assert result.data["config"]["measured_runs"] == 2
    assert result.data["config"]["uses_inference_function"] is True
    assert result.data["input"]["kwargs"] == ["x"]

    assert len(result.data["runs"]) == 2
    assert all(run["timing"]["wall_time_us"] > 0 for run in result.data["runs"])
    assert all(run["timing"]["cpu_time_us"] > 0 for run in result.data["runs"])

    # TinyKeywordModel has one direct child (`linear`) which becomes the only untuned region.
    untuned_id = "untuned_module:linear"
    for run in result.data["runs"]:
        region_ids = [region["region_id"] for region in run["regions"]]
        assert region_ids == [untuned_id]
        assert run["regions"][0]["cpu_time_us"] > 0
    assert result.data["regions"] == [
        {
            "id": untuned_id,
            "name": "linear",
            "kind": "untuned_module",
            "module_type": "torch.nn.modules.linear.Linear",
        }
    ]
    assert result.data["warnings"] == []

    key_averages = result.data["profiler"]["key_averages"]["cpu_time_total"]
    assert key_averages["sort_by"] == "cpu_time_total"
    assert key_averages["events"]
    assert any(event["key"] == "aitune.performance.profiled_run" for event in key_averages["events"])

    markdown = result.markdown()
    assert "# Performance Profile" in markdown
    assert "## Trace" not in markdown
    assert "## Regions" in markdown
    assert "`untuned_module:linear`" in markdown
    assert "No warnings." in markdown


def test_profile_runs_callable_under_no_grad():
    grad_states = []

    def inference_function():
        grad_states.append(torch.is_grad_enabled())
        return torch.ones(1)

    with torch.enable_grad():
        ait.profile(
            obj=inference_function,
            input_data=None,
            warmup_runs=1,
            measured_runs=2,
        )

    # User-requested warmup + profiler-internal warmup + measured runs.
    assert grad_states == [False, False, False, False]


def test_performance_profile_rejects_markdown_options():
    profile = PerformanceProfile(data={})

    with pytest.raises(ValueError, match="markdown options are not supported yet"):
        profile.markdown(options=object())


def test_markdown_includes_residual_for_run_without_regions():
    profile = PerformanceProfile(
        data={
            "created_at": "2026-05-25T00:00:00+00:00",
            "aitune_version": "test",
            "config": {
                "warmup_runs": 0,
                "measured_runs": 1,
                "uses_inference_function": False,
            },
            "target": {"type": "tests.Target"},
            "input": {"args_count": 0, "kwargs": []},
            "runs": [
                {
                    "run_index": 0,
                    "timing": {"wall_time_us": 100.0, "cpu_time_us": 80.0},
                    "regions": [],
                    "residual": {"cpu_time_us": 80.0, "cpu_time_fraction": 1.0},
                }
            ],
            "regions": [],
            "profiler": {"key_averages": {}},
            "warnings": [],
        }
    )

    markdown = profile.markdown()

    assert "## Per-Run Attribution" in markdown
    assert "| 0 | _(residual)_ | — | 0.080 ms | 100.0% | - | - |" in markdown


def test_profile_records_tuned_aot_module_regions(tmp_path):
    class TinyAotModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    module_name = "attributed-linear"
    model = ait.Module(
        TinyAotModel(),
        module_name,
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    sample = torch.ones(1, 4)
    model(sample)
    model.tune(device=torch.device("cpu"))

    trace_file = tmp_path / "runtime_attribution" / "trace.json"
    result = ait.profile(
        obj=model,
        input_data=sample,
        warmup_runs=1,
        measured_runs=2,
        trace_file=trace_file,
    )

    report = result.data
    markdown = result.markdown()
    trace = trace_file.read_text(encoding="utf-8")

    assert result.trace_file == trace_file.resolve()
    assert trace_file.exists()
    assert trace_file.stat().st_size > 0
    assert "artifacts" not in report
    assert len(report["runs"]) == 2
    assert all(len(run["regions"]) == 1 for run in report["runs"])
    assert all(run["regions"][0]["region_id"] == f"aot_module:{module_name}" for run in report["runs"])
    assert all(run["regions"][0]["cpu_time_us"] > 0 for run in report["runs"])
    assert all(0 < run["regions"][0]["cpu_time_fraction"] <= 1 for run in report["runs"])
    assert report["regions"] == [
        {
            "id": f"aot_module:{module_name}",
            "name": module_name,
            "kind": "aot_managed_module",
            "module_type": f"{TinyAotModel.__module__}.{TinyAotModel.__qualname__}",
            "wrapper_state": "tuned",
        }
    ]
    assert report["warnings"] == []

    aot_region_name = f"aitune.performance.aot_module:{module_name}"
    assert aot_region_name in trace
    assert any(
        event["key"] == aot_region_name for event in report["profiler"]["key_averages"]["cpu_time_total"]["events"]
    )
    assert "## Regions" in markdown
    assert f"`aot_module:{module_name}`" in markdown
    assert "## Per-Run Attribution" in markdown
    assert "Calls" in markdown
    assert "CPU Fraction" in markdown


def test_run_region_join_includes_device_share_and_unmapped_warning():
    class FakeEvent:
        def __init__(self, name, cpu_time_total, device_time_total=0.0, cpu_parent=None, device_type="DeviceType.CPU"):
            self.name = name
            self.cpu_time_total = cpu_time_total
            self.device_time_total = device_time_total
            self.cpu_parent = cpu_parent
            self.device_type = device_type

    run_event = FakeEvent("aitune.performance.profiled_run", cpu_time_total=100.0, device_time_total=50.0)
    mapped_region = FakeEvent(
        "aitune.performance.aot_module:mapped",
        cpu_time_total=25.0,
        device_time_total=10.0,
        cpu_parent=run_event,
    )
    second_mapped_region_call = FakeEvent(
        "aitune.performance.aot_module:mapped",
        cpu_time_total=15.0,
        device_time_total=15.0,
        cpu_parent=run_event,
    )
    unmapped_region = FakeEvent("aitune.performance.aot_module:unmapped", cpu_time_total=5.0)
    cuda_side_region = FakeEvent(
        "aitune.performance.aot_module:mapped",
        cpu_time_total=0.0,
        device_time_total=10.0,
        device_type="DeviceType.CUDA",
    )
    runtime_profile = _RuntimeProfile(
        run_wall_times_us=[123.4],
        events=EventList([run_event, mapped_region, second_mapped_region_call, unmapped_region, cuda_side_region]),
        key_averages=EventList(),
        activities=["CPU", "CUDA"],
    )

    runs = runtime_profile.runs()
    warnings = runtime_profile.warnings()

    assert runs == [
        {
            "run_index": 0,
            "timing": {
                "wall_time_us": 123.4,
                "cpu_time_us": 100.0,
                "device_time_us": 50.0,
            },
            "regions": [
                {
                    "region_id": "aot_module:mapped",
                    "calls": 2,
                    "cpu_time_us": 40.0,
                    "cpu_time_fraction": 0.4,
                    "device_time_us": 25.0,
                    "device_time_fraction": 0.5,
                }
            ],
            "residual": {
                "cpu_time_us": 60.0,
                "cpu_time_fraction": 0.6,
                "device_time_us": 25.0,
                "device_time_fraction": 0.5,
            },
        }
    ]
    assert warnings == [
        {
            "code": "UNMAPPED_AOT_REGION_EVENTS",
            "message": "1 AOT region profiler event(s) could not be mapped to a profiled run.",
            "source": "core",
        }
    ]


def test_run_timing_to_dict_omits_device_when_unavailable():
    timing = _RunTiming(wall_time_us=100.0, cpu_time_us=50.0)

    assert timing.to_dict() == {"wall_time_us": 100.0, "cpu_time_us": 50.0}


def test_region_aggregate_omits_cpu_time_fraction_when_run_has_no_cpu_time():
    aggregate = _RegionAggregate(region_id="aot_module:x", calls=1, cpu_time_us=20.0)
    run_timing = _RunTiming(wall_time_us=100.0, cpu_time_us=0.0)

    assert aggregate.to_dict(run_timing) == {
        "region_id": "aot_module:x",
        "calls": 1,
        "cpu_time_us": 20.0,
    }


def test_region_aggregate_omits_device_fields_for_cpu_only_region():
    aggregate = _RegionAggregate(region_id="aot_module:x", calls=1, cpu_time_us=20.0)
    run_timing = _RunTiming(wall_time_us=100.0, cpu_time_us=100.0, device_time_us=50.0)

    assert aggregate.to_dict(run_timing) == {
        "region_id": "aot_module:x",
        "calls": 1,
        "cpu_time_us": 20.0,
        "cpu_time_fraction": 0.2,
    }


def test_region_aggregate_omits_device_time_fraction_when_run_has_no_device_time():
    aggregate = _RegionAggregate(
        region_id="aot_module:x",
        calls=1,
        cpu_time_us=20.0,
        device_time_us=10.0,
    )
    run_timing = _RunTiming(wall_time_us=100.0, cpu_time_us=100.0)

    assert aggregate.to_dict(run_timing) == {
        "region_id": "aot_module:x",
        "calls": 1,
        "cpu_time_us": 20.0,
        "cpu_time_fraction": 0.2,
        "device_time_us": 10.0,
    }


def test_residual_clamps_negative_cpu_to_zero():
    aggregates = [_RegionAggregate(region_id="aot_module:x", calls=1, cpu_time_us=120.0)]
    timing = _RunTiming(wall_time_us=100.0, cpu_time_us=100.0)

    residual = _Residual.from_run(timing, aggregates)

    assert residual.cpu_time_us == 0.0
    assert residual.cpu_time_fraction == 0.0


def test_residual_for_cpu_only_run_omits_device_fields():
    aggregates = [_RegionAggregate(region_id="aot_module:x", calls=1, cpu_time_us=40.0)]
    timing = _RunTiming(wall_time_us=100.0, cpu_time_us=100.0)

    residual = _Residual.from_run(timing, aggregates)

    assert residual.to_dict() == {"cpu_time_us": 60.0, "cpu_time_fraction": 0.6}


def test_residual_with_no_regions_is_full_run_timing():
    timing = _RunTiming(wall_time_us=100.0, cpu_time_us=100.0, device_time_us=50.0)

    residual = _Residual.from_run(timing, [])

    assert residual.to_dict() == {
        "cpu_time_us": 100.0,
        "cpu_time_fraction": 1.0,
        "device_time_us": 50.0,
        "device_time_fraction": 1.0,
    }


def test_profile_mixes_aot_and_untuned_regions():
    """Diagnostic for the topmost-owned invariant.

    A model with one AITune-wrapped child and one untuned child should produce both region
    kinds in the report without overlap or double-counting.
    """

    class _MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tuned = torch.nn.Linear(4, 4)
            self.untuned = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.untuned(self.tuned(x))

    model = _MixedModel()
    model.tuned = ait.Module(
        model.tuned,
        "tuned-linear",
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    sample = torch.ones(1, 4)
    model(sample)
    model.tuned.tune(device=torch.device("cpu"))

    result = ait.profile(
        obj=model,
        input_data=sample,
        warmup_runs=1,
        measured_runs=2,
    )

    report = result.data

    region_ids_metadata = {region["id"] for region in report["regions"]}
    assert region_ids_metadata == {"aot_module:tuned-linear", "untuned_module:untuned"}

    aot_metadata = next(region for region in report["regions"] if region["kind"] == "aot_managed_module")
    untuned_metadata = next(region for region in report["regions"] if region["kind"] == "untuned_module")
    assert aot_metadata["wrapper_state"] == "tuned"
    assert "wrapper_state" not in untuned_metadata
    assert untuned_metadata["module_type"] == "torch.nn.modules.linear.Linear"

    for run in report["runs"]:
        per_run_ids = {region["region_id"] for region in run["regions"]}
        assert per_run_ids == {"aot_module:tuned-linear", "untuned_module:untuned"}
        residual = run["residual"]
        # Residual must be non-negative and strictly less than the run's CPU time,
        # since at least the two attributed regions contributed.
        assert 0.0 <= residual["cpu_time_us"] < run["timing"]["cpu_time_us"]
        assert 0.0 <= residual["cpu_time_fraction"] < 1.0

    assert report["warnings"] == []


def test_profile_captures_method_wrapped_region(tmp_path):
    """End-to-end: invoking a module via a custom method produces an untuned_module:<m>.<method> region.

    Diffusers-style pipelines call ``self.vae.decode(z)`` rather than ``self.vae(z)``.
    The forward hook on ``vae`` never fires for that path; the method wrapper does.
    This test exercises that end-to-end and verifies the parent-fallback for
    ``module_type`` resolution (the method-suffixed path is not in the discovery
    type dict; it resolves to the parent module's type).
    """

    class VaeLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):  # pragma: no cover — never invoked by the test
            return self.layer(x)

        def decode(self, x):
            return self.layer(x) + 1

    class PipelineLike:
        def __init__(self):
            self.vae = VaeLike()

    pipeline = PipelineLike()

    def inference(z: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return pipeline.vae.decode(z)

    trace_file = tmp_path / "method_region" / "trace.json"
    result = ait.profile(
        obj=pipeline,
        input_data=torch.randn(1, 4),
        inference_function=inference,
        warmup_runs=1,
        measured_runs=2,
        trace_file=trace_file,
    )

    assert result.trace_file == trace_file.resolve()
    assert trace_file.exists()
    assert trace_file.stat().st_size > 0
    report = result.data

    region_metadata = next(region for region in report["regions"] if region["id"] == "untuned_module:vae.decode")
    assert region_metadata["name"] == "vae.decode"
    assert region_metadata["kind"] == "untuned_module"
    # Parent-fallback: full path "vae.decode" isn't in untuned_module_types; "vae" is.
    assert region_metadata["module_type"].endswith("VaeLike")

    for run in report["runs"]:
        method_region = next(
            (region for region in run["regions"] if region["region_id"] == "untuned_module:vae.decode"),
            None,
        )
        assert method_region is not None
        assert method_region["calls"] == 1
        assert method_region["cpu_time_us"] > 0
        # The bare forward hook on vae never fires for this invocation path.
        forward_region_ids = {region["region_id"] for region in run["regions"]}
    assert "untuned_module:vae" not in forward_region_ids

    # Markdown surfaces the method region in its tables.
    markdown = result.markdown()
    assert "`untuned_module:vae.decode`" in markdown

    # The raw Chrome trace contains the underlying record_function annotation —
    # downstream tooling (Perfetto, chrome://tracing) sees it.
    trace = trace_file.read_text(encoding="utf-8")
    assert "aitune.performance.untuned_module:vae.decode" in trace


def _make_nested_aot_outer(name_outer: str, name_inner: str):
    """Helper: build an outer wrapper that contains an AOT-wrapped inner child.

    Both ``inner`` and ``outer`` end up registered in ``MODULE_REGISTRY``.
    """

    class _Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.inner(x)

    outer = _Outer()
    outer.inner = ait.Module(
        outer.inner,
        name_inner,
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    outer_wrapped = ait.Module(
        outer,
        name_outer,
        strategy=ait.OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),
    )
    return outer_wrapped


def test_profile_filters_nested_aot_events():
    """When outer.forward calls inner.forward, the inner AOT event is suppressed at classification.

    The per-event check walks cpu_parent up to the profiled-run marker; if an
    enclosing AOT event is found, the inner event is dropped from aggregation
    (the outer already owns its time). The additive invariant
    ``sum(fractions) + residual ≈ 1.0`` is preserved.
    """
    outer_wrapped = _make_nested_aot_outer("outer", "inner")

    sample = torch.ones(1, 4)
    outer_wrapped(sample)
    outer_wrapped.tune(device=torch.device("cpu"))

    result = ait.profile(
        obj=outer_wrapped,
        input_data=sample,
        warmup_runs=1,
        measured_runs=2,
    )
    report = result.data

    # Outer fires and is counted; inner fires inside it and is suppressed per-event.
    # Inner does not appear in the regions metadata because no non-nested event was ever observed.
    region_ids = {region["id"] for region in report["regions"]}
    assert region_ids == {"aot_module:outer"}

    for run in report["runs"]:
        per_run_ids = {region["region_id"] for region in run["regions"]}
        assert per_run_ids == {"aot_module:outer"}
        assert run["regions"][0]["cpu_time_fraction"] <= 1.0
        assert run["residual"]["cpu_time_fraction"] >= 0.0


def test_profile_attributes_inner_when_profiled_directly():
    """Profile the inner AOT wrapper directly: it is correctly attributed even though the outer is registered.

    Per-event nesting suppression must not over-fire: if the outer never runs in
    a measured iteration, the inner event has no AOT ancestor and is the legitimate
    owner of the workload. Static discovery-time filtering would wrongly drop this
    attribution because it cannot see what the workload actually invokes.
    """
    outer_wrapped = _make_nested_aot_outer("outer", "inner")
    # Reach the inner wrapper through outer.__wrapped__ so AOT machinery is engaged.
    inner_wrapped = outer_wrapped.__wrapped__.inner

    sample = torch.ones(1, 4)
    inner_wrapped(sample)
    inner_wrapped.tune(device=torch.device("cpu"))

    # Note: obj is the INNER wrapper. The outer is still in MODULE_REGISTRY but never fires.
    result = ait.profile(
        obj=inner_wrapped,
        input_data=sample,
        warmup_runs=1,
        measured_runs=2,
    )
    report = result.data

    # Inner is counted; outer is registered but never fired so does not appear.
    region_ids = {region["id"] for region in report["regions"]}
    assert region_ids == {"aot_module:inner"}

    for run in report["runs"]:
        per_run_ids = {region["region_id"] for region in run["regions"]}
        assert per_run_ids == {"aot_module:inner"}
        assert run["regions"][0]["cpu_time_us"] > 0


def test_profile_propagates_inference_exception_and_restores_state():
    """If the inference function raises, the exception propagates and instrumentation is fully restored.

    Public-API-level companion to the installer-level exception-safety tests in
    test_attribution_hooks. Verifies that a real `profile(...)` call
    leaves no leftover wrappers on the pipeline's modules and no leftover context
    attributes when the workload raises mid-run.
    """

    class _BoomVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, x):  # pragma: no cover — never invoked
            return self.layer(x)

        def decode(self, x):
            raise RuntimeError("inference failure")

    class _Pipe:
        def __init__(self):
            self.vae = _BoomVae()

    pipe = _Pipe()
    original_decode = type(pipe.vae).decode

    def inference(x: torch.Tensor) -> torch.Tensor:
        return pipe.vae.decode(x)

    with pytest.raises(RuntimeError, match="inference failure"):
        ait.profile(
            obj=pipe,
            input_data=torch.randn(1, 4),
            inference_function=inference,
            warmup_runs=1,
            measured_runs=1,
        )

    # Method wrapper is gone: instance no longer shadows the class method.
    assert "decode" not in vars(pipe.vae)
    # Class-level method is the original, not a residual wrapper somewhere up the MRO.
    assert type(pipe.vae).decode is original_decode
    # Forward-hook context attribute swept on the underlying module.
    assert not hasattr(pipe.vae, _CONTEXT_ATTR)
