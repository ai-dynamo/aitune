# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT CUDA graph capture, reuse, fallback, and eviction across static profiles."""

from contextlib import contextmanager
from weakref import ref

import pytest
import torch

import aitune.torch as ait
from aitune.torch.backend import TensorRTBackend, TensorRTProfileMode
from aitune.torch.config import config
from tests.functional_ci.backends import tensorrt_backend
from tests.functional_ci.lifecycle import TENSORRT_FP32, assert_matches_eager, pinned_strategy, selected_backend
from tests.functional_ci.models import make_input, make_mlp


def _tune_cuda_graphs(device, monkeypatch, static_batches, **cache_options):
    monkeypatch.setattr(config, "max_num_samples_stored", len(static_batches))
    backend = tensorrt_backend(
        use_dynamo=False,
        use_cuda_graphs=True,
        opset_version=20,
        profiles=TensorRTProfileMode.SAMPLES_USED,
        **cache_options,
    )
    tuned = ait.Module(make_mlp(device), "cuda-graphs", strategy=pinned_strategy(backend))
    ait.tune(
        tuned,
        list(make_input(device, batch_size=5).unbind(0)),
        batch_sizes=list(static_batches),
        max_num_batches_per_batch_size=1,
        device=device,
        ignore_failing_modules=False,
    )

    selected = selected_backend(tuned, TensorRTBackend)
    assert isinstance(selected, TensorRTBackend)
    return tuned, selected


@pytest.mark.matrix("other.cuda-graphs-fp32")
def test_cuda_graphs(device: torch.device, monkeypatch: pytest.MonkeyPatch):
    """Record static profiles, then alternate between graph replay and uncaptured shapes."""
    static_batches = (1, 3, 5)
    tuned, selected = _tune_cuda_graphs(device, monkeypatch, static_batches)
    graphs = selected._cuda_graphs
    fallback_index = len(static_batches)
    assert len(selected._trt_optimization_profiles) == fallback_index + 1
    assert graphs.static_profile_indices == set(range(fallback_index))

    captured = {}
    reference = make_mlp(device)
    for batch_size in static_batches:
        assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(batch_size,), seeds=(11,))
        assert graphs.active is not None
        graph = graphs.active.graph
        assert graph is not None
        assert all(graph is not previous for previous in captured.values())
        captured[batch_size] = graph

    assert graphs.profiles.keys() == graphs.static_profile_indices
    # Change the input values and alternate static and dynamic profiles to detect stale inputs or graph selection.
    for batch_size in (5, 2, 1, 4, 3):
        assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(batch_size,), seeds=(12,))
        if batch_size in captured:
            assert graphs.active is not None
            assert graphs.active.graph is captured[batch_size]
        else:
            assert graphs.active is None
            assert selected._context is not None
            assert selected._context.active_optimization_profile == fallback_index
    assert graphs.profiles.keys() == graphs.static_profile_indices


@pytest.mark.matrix("other.cuda-graphs-fp32")
def test_cuda_graph_capture_failure(device: torch.device, monkeypatch: pytest.MonkeyPatch):
    """Fall back on capture failure, release cached resources, and keep capture disabled."""
    tuned, selected = _tune_cuda_graphs(device, monkeypatch, (1, 3, 5))
    # Start cold so correctness validation during tuning cannot pre-capture the failing profile.
    selected.deactivate()
    selected.activate()
    graphs = selected._cuda_graphs
    reference = make_mlp(device)
    assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(1,))
    assert graphs.active is not None
    assert graphs.active.graph is not None

    graph_context = torch.cuda.graph
    capture_attempts = 0
    profile_refs = []
    input_refs = []

    @contextmanager
    def fail_capture(*args, **kwargs):
        nonlocal capture_attempts
        capture_attempts += 1
        profile_refs.extend(ref(profile) for profile in graphs.profiles.values())
        input_refs.extend(ref(tensor) for profile in graphs.profiles.values() for tensor in profile.inputs.values())
        # Let CUDA finish capture normally so the injected error does not poison PyTorch's CUDA RNG.
        with graph_context(*args, **kwargs):
            yield
        raise RuntimeError("injected CUDA graph capture failure")

    monkeypatch.setattr(torch.cuda, "graph", fail_capture)
    assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(3,), seeds=(11,))
    assert capture_attempts == 1
    assert graphs.capture_failed
    assert graphs.active is None
    assert not graphs.profiles
    assert len(profile_refs) == len(input_refs) == 2
    assert all(resource() is None for resource in (*profile_refs, *input_refs))

    for batch_size in (3, 1, 2, 5, 3):
        assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(batch_size,), seeds=(12,))
        assert capture_attempts == 1
        assert graphs.active is None
        assert not graphs.profiles
        assert selected._context is selected._base_context
        assert selected._output_allocator is selected._base_output_allocator


@pytest.mark.matrix("other.cuda-graphs-fp32")
@pytest.mark.parametrize(
    ("policy", "expected_residents"),
    [
        ("lfu", ((1,), (1,), (1, 3), (1, 3), (1, 5), (5, 1))),
        ("lru", ((1,), (1,), (1, 3), (3, 5), (3, 5), (5, 1))),
    ],
    ids=["lfu", "lru"],
)
def test_cuda_graph_cache_eviction(device: torch.device, monkeypatch: pytest.MonkeyPatch, policy, expected_residents):
    """LRU admits every miss; LFU admits a new profile only after it becomes more frequent."""
    static_batches = (1, 3, 5)
    tuned, selected = _tune_cuda_graphs(
        device, monkeypatch, static_batches, max_cuda_graphs=2, cuda_graph_cache_policy=policy
    )
    selected.deactivate()
    selected.activate()
    graphs = selected._cuda_graphs
    reference = make_mlp(device)
    # Map batches to profile indices without relying on the order of the tuning samples.
    profile_indices = {}
    for index in graphs.static_profile_indices:
        (shape_range,) = selected._trt_optimization_profiles[index].values()
        minimum, _, _ = shape_range
        profile_indices[minimum[0]] = index
    assert profile_indices.keys() == set(static_batches)
    assert not graphs.profiles

    # At the first request for batch 5, batch 1 is least recent and batch 3 is least frequent.
    requests = (1, 1, 3, 5, 5, 1)
    for seed, (batch_size, residents) in enumerate(zip(requests, expected_residents, strict=True), start=11):
        previous = dict(graphs.profiles)
        assert_matches_eager(tuned, reference, device, TENSORRT_FP32, batch_sizes=(batch_size,), seeds=(seed,))
        assert not graphs.capture_failed
        assert tuple(graphs.profiles) == tuple(profile_indices[batch] for batch in residents)
        assert graphs.active is graphs.profiles.get(profile_indices[batch_size])
        if graphs.active is not None:
            assert graphs.active.graph is not None
        for index, profile in previous.items():
            if index in graphs.profiles:
                assert graphs.profiles[index] is profile
            else:
                assert profile.graph is None
                assert not profile.inputs
                assert not profile.output_allocator.outputs
