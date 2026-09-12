# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# scope = "always"
# allow_failure = false
# ///

"""Exercise capture fallback and measure TensorRT graph caching per static profile.

Run this script on a CUDA/TensorRT host. It prints the benchmark report to the CI
log and writes it to --output (default: artifacts/tensorrt_cuda_graph_profiles.json).
The timing covers synchronized backend.infer calls, including input preparation,
capture/setup/replay when enabled, and output copies. Engine compilation and
capture-state checks are outside the timed regions. This small model measures the
effect of graph caching; its results are not a prediction for other models.
The overflow scenarios compare LRU recapture against aged LFU admission, both
for cyclic requests and for hot profiles interrupted by a scan of other shapes.
Performance has no hardware-dependent pass/fail threshold.
The intentional capture failure runs in a separate process because a failed CUDA
capture can leave PyTorch's CUDA RNG state unusable for subsequent tests.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorrt as trt
import torch
from torch import nn

from aitune.torch.backend.tensorrt import TensorRTBackend, TensorRTBackendConfig, TensorRTProfile
from aitune.torch.backend.tensorrt.cuda_graphs import TensorRTCudaGraphCache
from aitune.torch.dynamic_shapes import DynamicDim
from aitune.torch.module.forward_signature import ForwardSignature
from aitune.torch.module.graph_spec import GraphSpec
from aitune.torch.module.sample_metadata import SampleMetadata
from aitune.torch.module.sample_store import SampleStore

logger = logging.getLogger(__name__)
LENGTHS = (16, 32, 64, 24)
OVERFLOW_LENGTHS = tuple(range(16, 88, 8))  # Nine profiles exceed the default eight-graph cache.
WARMUP_REQUESTS = 8
MEASURED_REQUESTS = 48
ROUNDS = 4
DEFAULT_REPORT_PATH = Path("artifacts/tensorrt_cuda_graph_profiles.json")


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, x):
        return self.layers(x)


class CountingCudaGraphs(TensorRTCudaGraphCache):
    """Count actual capture attempts without mocking CUDA or TensorRT execution."""

    def __init__(self):
        super().__init__()
        self.capture_attempts = 0

    def _capture(self, profile, stream):
        self.capture_attempts += 1
        return super()._capture(profile, stream)


@contextmanager
def _built_backend(cache_dir, scenario="static_profiles"):
    torch.manual_seed(42)
    device = torch.device("cuda")
    model = SmallModel().eval().to(device)
    lengths = LENGTHS[:1] if scenario == "static_shape" else LENGTHS
    if scenario in ("static_profiles_lru", "hot_profiles_scan"):
        lengths = OVERFLOW_LENGTHS
    inputs = [torch.randn(2, length, 128, device=device) for length in lengths]
    with torch.no_grad():
        sample_outputs = [model(value) for value in inputs]
    samples = [((value,), {}) for value in inputs]
    graph_spec = GraphSpec(
        name="cuda-graph-shape-changes",
        forward_signature=ForwardSignature.from_callable(model.forward),
        input_spec=SampleMetadata.from_inputs({"x": inputs[0]}),
        output_spec=SampleMetadata.from_outputs(sample_outputs[0]),
        dynamic_shapes=(
            {"x": (2, DynamicDim("tokens", min=16, opt=32, max=64), 128)} if scenario == "dynamic_range" else {}
        ),
    )
    for value, output in zip(inputs[1:], sample_outputs[1:], strict=True):
        graph_spec.update_shapes_seen(SampleMetadata.from_inputs({"x": value}), SampleMetadata.from_outputs(output))

    # Leave use_cuda_graphs at its default to exercise the behavior introduced by the MR.
    config = TensorRTBackendConfig(use_dynamo=False, opset_version=20, enable_tf32=False)
    if scenario in ("static_profiles", "static_profiles_lru", "hot_profiles_scan"):
        config.profiles = [
            TensorRTProfile().add_input_shape("x", tuple(value.shape), tuple(value.shape), tuple(value.shape))
            for value in inputs
        ]
    backend = TensorRTBackend(config)
    backend._cuda_graphs = CountingCudaGraphs()
    started = time.perf_counter()
    sample_store = SampleStore.from_samples(samples, cache_dir, "samples")
    backend.build(model, graph_spec, sample_store, device=device, cache_dir=cache_dir)
    compilation_seconds = time.perf_counter() - started
    try:
        expected_profiles = set(range(len(inputs))) if scenario != "dynamic_range" else set()
        assert backend._cuda_graphs.static_profile_indices == expected_profiles
        assert backend._cuda_graphs.policy == "lfu"
        assert backend._cuda_graphs.capture_attempts == 0
        assert not backend._cuda_graphs.profiles
        yield backend, inputs, compilation_seconds
    finally:
        backend.deactivate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_real_cuda_graph_capture_failure_falls_back():
    """Run the destructive capture test in a fresh interpreter, including under pytest."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", "capture-failure"],
        check=True,
    )


def _run_capture_failure():
    """Invalidate a real capture, then verify inference continues without recapture."""
    with tempfile.TemporaryDirectory() as directory, _built_backend(Path(directory)) as built:
        backend, inputs, _ = built
        real_graph = torch.cuda.graph
        failure_injected = False

        @contextmanager
        def failing_capture(*args, **kwargs):
            nonlocal failure_injected
            with real_graph(*args, **kwargs):
                yield  # The backend records real TensorRT work before the injected failure.
                assert torch.cuda.is_current_stream_capturing()
                try:
                    # Device synchronization is forbidden during capture. Exercise an actual
                    # CUDA error and capture teardown, rather than raising a synthetic exception.
                    torch.cuda.synchronize()
                except RuntimeError:
                    failure_injected = True
                    raise

        with patch("torch.cuda.graph", new=failing_capture):
            for _ in range(2):
                for value in inputs:
                    backend.infer(value)
                    assert failure_injected, "The intended CUDA capture failure was not reached"
                    assert backend._cuda_graphs.capture_attempts == 1, "Capture retried after fallback"
                    assert backend._cuda_graphs.capture_failed
                    assert not backend._use_cuda_graphs
                    assert backend._cuda_graphs.active is None, "Fallback retained active graph resources"
                    assert not backend._cuda_graphs.profiles, "Fallback retained cached graphs"

        # Restoring the normal capture API must not re-enable capture for this instance.
        backend.infer(inputs[0])
        assert backend._cuda_graphs.capture_attempts == 1
        assert not torch.cuda.is_current_stream_capturing()
    logger.info("Real CUDA capture failure: fallback and subsequent shape changes passed")


def _track_profile_request(cache, index, capacity):
    """Track expected LRU order independently of the backend."""
    capture = index not in cache
    if not capture:
        cache.remove(index)
    elif len(cache) == capacity:
        cache.pop(0)
    cache.append(index)
    return capture


def _cached_resources(graphs):
    # Keep only IDs: retaining tensors or contexts here would prevent eviction from freeing memory.
    return {
        index: (
            id(entry.graph),
            id(entry.context),
            id(entry.output_allocator),
            {name: id(tensor) for name, tensor in entry.inputs.items()},
        )
        for index, entry in graphs.profiles.items()
    }


def _request_slot(index, num_inputs, scenario):
    if scenario == "hot_profiles_scan" and index >= WARMUP_REQUESTS:
        # Train two hot profiles, scan all other profiles, then return to the hot pair.
        sequence = (0, 1) * 8 + tuple(range(2, num_inputs)) + (0, 1) * 8 + tuple(range(2, num_inputs))
        return sequence[(index - WARMUP_REQUESTS) % len(sequence)]
    return index % num_inputs


def _measure_requests(backend, inputs, policy, scenario):
    # Reuse the compiled engine; activation and warmup are outside the reported latency.
    backend.deactivate()
    backend._config.use_cuda_graphs = policy != "normal"
    backend._config.cuda_graph_cache_policy = "lru" if policy == "normal" else policy
    backend.activate()
    graphs = backend._cuda_graphs
    enabled = policy != "normal" and bool(graphs.static_profile_indices)
    expected_cache = []
    expected_captures = 0
    before_warmup = graphs.capture_attempts
    for index in range(WARMUP_REQUESTS):
        slot = index % len(inputs)
        if enabled:
            expected_captures += _track_profile_request(expected_cache, slot, graphs.max_graphs)
        backend.infer(inputs[slot])
    assert not graphs.capture_failed, "Capture silently fell back during warmup"
    warmup_captures = graphs.capture_attempts - before_warmup
    assert warmup_captures == expected_captures
    assert list(graphs.profiles) == expected_cache

    resources = _cached_resources(graphs)
    before = graphs.capture_attempts
    expected_captures = 0
    rows = []
    previous_shape = inputs[(WARMUP_REQUESTS - 1) % len(inputs)].shape
    for index in range(MEASURED_REQUESTS):
        slot = _request_slot(WARMUP_REQUESTS + index, len(inputs), scenario)
        value = inputs[slot]
        if len(inputs) > 1:
            assert value.shape != previous_shape, "Every request must change shape"
        # These traces are shorter than the aging interval. Under LFU the warmed
        # residents are at least as frequent as the uncached profile, so it must
        # run normally throughout measurement. LRU admits every cache miss.
        capture = False
        if enabled and (policy == "lru" or slot in expected_cache):
            capture = _track_profile_request(expected_cache, slot, graphs.max_graphs)
        expected_captures += capture

        torch.cuda.synchronize()
        started = time.perf_counter()
        backend.infer(value)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        # Check capture counts, eviction order, and retained resources outside the timer.
        assert not graphs.capture_failed
        assert graphs.capture_attempts - before == expected_captures
        assert list(graphs.profiles) == expected_cache
        if enabled and slot in expected_cache:
            assert graphs.active is not None and graphs.active.graph is not None
        else:
            assert graphs.active is None
        current_resources = _cached_resources(graphs)
        for profile_index in resources.keys() & current_resources.keys():
            assert current_resources[profile_index] == resources[profile_index]
        resources = current_resources
        rows.append({
            "shape": list(value.shape),
            "latency_ms": latency_ms,
            "captured": capture,
            "graph_used": graphs.active is not None,
        })
        previous_shape = value.shape
    return {
        "warmup_capture_attempts": warmup_captures,
        "measured_capture_attempts": graphs.capture_attempts - before,
        "requests": rows,
    }


def _latency_summary(rows):
    latencies = np.array([row["latency_ms"] for row in rows])
    assert np.isfinite(latencies).all() and (latencies > 0).all()
    mean_ms = float(latencies.mean())
    return {
        "requests": len(rows),
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "requests_per_second": 1000 / mean_ms,
    }


def _comparison(normal_rows, capture_rows):
    normal = _latency_summary(normal_rows)
    capture = _latency_summary(capture_rows)
    return {
        "without_cuda_graphs": normal,
        "with_cuda_graphs": capture,
        "latency_ratio_graphs_over_normal": capture["mean_ms"] / normal["mean_ms"],
        "extra_latency_ms": capture["mean_ms"] - normal["mean_ms"],
    }


def _benchmark_scenario(scenario):
    """Measure warmed request latency with and without per-profile CUDA graphs."""
    trials = []
    with tempfile.TemporaryDirectory() as directory, _built_backend(Path(directory), scenario) as built:
        backend, inputs, compilation_seconds = built
        # Alternate trial order to reduce systematic bias from clock/temperature drift.
        for round_index in range(ROUNDS):
            modes = ("normal", "lru", "lfu")
            offset = round_index % len(modes)
            modes = modes[offset:] + modes[:offset]
            for policy in modes:
                trial = _measure_requests(backend, inputs, policy, scenario)
                trial.update(round=round_index, use_cuda_graphs=policy != "normal", policy=policy)
                trials.append(trial)

    rows_by_policy = {
        policy: [row for trial in trials if trial["policy"] == policy for row in trial["requests"]]
        for policy in ("normal", "lru", "lfu")
    }
    normal_rows = rows_by_policy["normal"]
    by_shape = {}
    for value in inputs:
        shape = list(value.shape)
        by_shape[str(shape)] = {
            policy: _comparison(
                [row for row in normal_rows if row["shape"] == shape],
                [row for row in rows_by_policy[policy] if row["shape"] == shape],
            )
            for policy in ("lru", "lfu")
        }
    if scenario in ("static_profiles_lru", "hot_profiles_scan"):
        assert sum(row["captured"] for row in rows_by_policy["lfu"]) == 0
        assert sum(row["captured"] for row in rows_by_policy["lru"]) > 0
    return {
        "scenario": scenario,
        "measurement": "synchronized wall-clock backend.infer latency; not isolated capture time",
        "model": "Linear(128,256) -> GELU -> Linear(256,128), float32, TF32 disabled",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "compilation_seconds": compilation_seconds,
        "max_cuda_graphs": backend._config.max_cuda_graphs,
        "shape_sequence": [list(value.shape) for value in inputs],
        "measured_profile_sequence": [
            _request_slot(WARMUP_REQUESTS + index, len(inputs), scenario) for index in range(MEASURED_REQUESTS)
        ],
        "warmup_requests_per_trial": WARMUP_REQUESTS,
        "measured_requests_per_trial": MEASURED_REQUESTS,
        "rounds": ROUNDS,
        "summary": {
            policy: {
                **_comparison(normal_rows, rows_by_policy[policy]),
                "measured_captures": sum(row["captured"] for row in rows_by_policy[policy]),
                "graph_requests": sum(row["graph_used"] for row in rows_by_policy[policy]),
            }
            for policy in ("lru", "lfu")
        },
        "latency_ratio_lfu_over_lru": (
            _latency_summary(rows_by_policy["lfu"])["mean_ms"] / _latency_summary(rows_by_policy["lru"])["mean_ms"]
        ),
        "by_shape": by_shape,
        "trials": trials,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_graph_profile_performance(output_path=DEFAULT_REPORT_PATH):
    report = {
        scenario: _benchmark_scenario(scenario)
        for scenario in ("static_shape", "static_profiles", "dynamic_range", "static_profiles_lru", "hot_profiles_scan")
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Keep summary results visible even when CI does not retain the JSON artifact.
    print(
        json.dumps(
            {
                scenario: {key: value for key, value in result.items() if key != "trials"}
                for scenario, result in report.items()
            },
            indent=2,
        )
    )
    logger.info("TensorRT CUDA graph benchmark report: %s", output_path.resolve())


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--case", choices=("all", "capture-failure", "benchmark"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, force=True)
    logger.setLevel(logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("These functional tests require a CUDA GPU and TensorRT")
    if args.case == "capture-failure":
        # Never create another CUDA graph or use CUDA RNG after this case. Some
        # PyTorch versions leave the generator in capture mode after an error:
        # https://github.com/pytorch/pytorch/issues/171263
        _run_capture_failure()
        return
    if args.case == "all":
        test_real_cuda_graph_capture_failure_falls_back()
    test_cuda_graph_profile_performance(output_path=args.output)


if __name__ == "__main__":
    _main()
